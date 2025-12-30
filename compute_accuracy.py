import torch
import os
import json
import argparse
import time
import csv
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

from prompt_templates import build_caption_yesno_prompt, build_scene_yesno_prompt, build_visual_yesno_prompt

# --- Configuration ---
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
BASE_DATA_PATH = "/home/kkarthikeyan/deep-learning/DeepLearningProject/Synthetic-Data/vlm_levels"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)


def _pick_model_input_device(model) -> torch.device:
    """Best-effort device for placing input tensors with device_map='auto'."""
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_rel_group(rel_type: str | None) -> str:
    """Best-effort group inference for older JSONs without rel_group."""
    if not rel_type:
        return "UNKNOWN"
    rel_type = rel_type.strip().lower()
    if rel_type in {"left_of", "right_of", "above", "below"}:
        return "PRIMARY"
    # In this codebase, CONTACT is treated as part of ADVANCED.
    if rel_type in {"touching", "overlapping", "inside", "encapsulates", "near", "far", "next_to", "beside"}:
        return "ADVANCED"
    return "UNKNOWN"

def get_prediction(outputs, processor):
    """Extract 'yes'/'no' from first-token logits (robust to prompt formatting)."""
    first_token_logits = outputs.scores[0][0]
    probs = torch.softmax(first_token_logits, dim=-1)

    tok = processor.tokenizer

    def _last_id(text: str) -> int | None:
        ids = tok.encode(text, add_special_tokens=False)
        return ids[-1] if ids else None

    yes_tokens = [tid for tid in (_last_id(v) for v in [" yes", " Yes", "yes", "Yes"]) if tid is not None]
    no_tokens = [tid for tid in (_last_id(v) for v in [" no", " No", "no", "No"]) if tid is not None]

    prob_yes = sum([probs[t_id].item() for t_id in yes_tokens if t_id < len(probs)])
    prob_no = sum([probs[t_id].item() for t_id in no_tokens if t_id < len(probs)])

    return "yes" if prob_yes > prob_no else "no"


def get_yes_no_confidence(outputs, tokenizer) -> tuple[float | None, float | None, float | None]:
    """Return (confidence, p_yes, p_no) from the first generation step.

    Confidence is computed as max(p_yes, p_no) after normalizing to p_yes + p_no.
    Returns (None, None, None) if scores are unavailable.
    """
    scores = getattr(outputs, "scores", None)
    if not scores:
        return None, None, None

    first_token_logits = scores[0][0]
    probs = torch.softmax(first_token_logits, dim=-1)

    # Token ids for yes/no. Use several variants to be robust to tokenization.
    yes_variants = ["Yes", " yes", "yes"]
    no_variants = ["No", " no", "no"]

    def _token_ids(variants: list[str]) -> set[int]:
        ids: set[int] = set()
        for s in variants:
            enc = tokenizer.encode(s, add_special_tokens=False)
            if enc:
                ids.add(int(enc[-1]))
        return ids

    yes_ids = _token_ids(yes_variants)
    no_ids = _token_ids(no_variants)

    p_yes = float(sum(probs[t].item() for t in yes_ids if t < probs.numel()))
    p_no = float(sum(probs[t].item() for t in no_ids if t < probs.numel()))

    denom = p_yes + p_no
    if denom <= 0:
        return None, None, None

    p_yes_n = p_yes / denom
    p_no_n = p_no / denom
    conf = max(p_yes_n, p_no_n)
    return conf, p_yes_n, p_no_n

def scene_to_text(annotation: dict) -> str:
    meta = annotation.get("meta", {}) or {}
    patch = int(meta.get("patch", 14))

    lines = []
    for obj in annotation.get("objects", []):
        oid = obj.get("id", None)
        color = obj.get("color", "unknown")
        shape = obj.get("shape", "unknown")

        center = obj.get("center", None)
        if isinstance(center, list) and len(center) == 2:
            cx, cy = center
            gx, gy = int(cx) // patch, int(cy) // patch
            pos = f"grid ({gx}, {gy})"
        else:
            pos = "grid (unknown, unknown)"

        lines.append(f"Object {oid}: {color} {shape} at {pos}")

    return "\n".join(lines)

def get_caption_for_qa(annotation: dict, qa_item: dict) -> str | None:
    """Fetch the caption text associated with this QA item, if available.

    Newer JSONs: qa_item has caption_id and annotation has captions_meta.
    Returns None for older JSONs or if the caption is missing/truncated.
    """
    cap_id = qa_item.get("caption_id")
    if cap_id is None:
        return None

    captions_meta = annotation.get("captions_meta")
    if not isinstance(captions_meta, list) or not captions_meta:
        return None

    try:
        cap_id_int = int(cap_id)
    except Exception:
        return None

    # captions_meta ids are sequential in generation; prefer direct indexing when possible.
    if 0 <= cap_id_int < len(captions_meta):
        c = captions_meta[cap_id_int]
        if isinstance(c, dict):
            cap_text = c.get("caption")
            return cap_text.strip() if isinstance(cap_text, str) and cap_text.strip() else None

    # Fallback scan by explicit id
    for c in captions_meta:
        if not isinstance(c, dict):
            continue
        if c.get("id") == cap_id_int:
            cap_text = c.get("caption")
            return cap_text.strip() if isinstance(cap_text, str) and cap_text.strip() else None

    return None

def get_yesno_from_generated_text(s: str) -> str:
    s = (s or "").strip().lower()
    # take first token-ish answer
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    # fallback: search
    if " yes" in f" {s}":
        return "yes"
    if " no" in f" {s}":
        return "no"
    return "unknown"

def _move_to_device(batch: dict, device: torch.device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if hasattr(v, "to") else v
    return out

def predict_text_only(model, processor, prompt: str) -> tuple[str, str, float | None, float | None, float | None]:
    tok = processor.tokenizer
    inputs = tok(prompt, return_tensors="pt")
    model_device = _pick_model_input_device(model)
    inputs = _move_to_device(inputs, model_device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            
        )

    # Decode completion
    decoded_full = tok.decode(out.sequences[0], skip_special_tokens=True)
    decoded_prompt = tok.decode(inputs["input_ids"][0], skip_special_tokens=True)
    completion = decoded_full[len(decoded_prompt):].strip()

    pred = get_yesno_from_generated_text(completion)
    conf, p_yes, p_no = get_yes_no_confidence(out, tok)
    return pred, completion, conf, p_yes, p_no

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[0,1,2,3,4,5,6])
    ap.add_argument("--output_json", type=str, default="evaluation_results.json")
    ap.add_argument("--log_file", type=str, default="model_calls_log.csv")
    ap.add_argument("--text_only", action="store_true", help="Evaluate text-only relational reasoning (no image).")
    return ap.parse_args()

def main():
    args = parse_args()

    print(f"Loading {MODEL_ID}...")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        quantization_config=bnb,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
    model.eval()

    level_results = {}

    # Initialize CSV Log File
    with open(args.log_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Timestamp",
                "Level",
                "Image_Path",
                "Rel_Group",
                "Rel_Type",
                "Prompt",
                "Completion",
                "Confidence",
                "P_Yes",
                "P_No",
                "Ground_Truth",
                "Model_Prediction",
                "Result",
                "Mode",
            ]
        )

    for level in args.levels:
        level_dir = os.path.join(BASE_DATA_PATH, f"level_{level}")
        ann_dir = os.path.join(level_dir, "ann")
        img_dir = os.path.join(level_dir, "images")

        if not os.path.exists(ann_dir):
            print(f"Skipping Level {level}: Directory not found.")
            continue

        json_files = [f for f in os.listdir(ann_dir) if f.endswith(".json")]
        correct_questions = 0
        total_questions = 0

        # Per-group breakdown
        group_stats: dict[str, dict[str, int]] = {}

        mode_label = "text_only" if args.text_only else "visual"
        print(f"\n--- Processing Level {level}: {len(json_files)} images | mode={mode_label} ---")

        for filename in tqdm(json_files, desc=f"Level {level}"):
            file_id = filename.replace(".json", "")
            ann_path = os.path.join(ann_dir, filename)

            with open(ann_path, "r") as f:
                data = json.load(f)

            image_abs_path = os.path.join(img_dir, f"{file_id}.png")

            # Load image only for visual mode
            image = None
            if not args.text_only:
                if not os.path.exists(image_abs_path):
                    continue
                image = Image.open(image_abs_path).convert("RGB")

            # For text-only mode we prefer per-question caption context.
            # Keep a fallback scene description for older JSONs.
            scene_text = scene_to_text(data) if args.text_only else None

            for qa in data.get("qa", []):
                question = (qa.get("question") or "").strip()
                gt = (qa.get("answer") or "").strip().lower()

                if not question or gt not in {"yes", "no"}:
                    continue

                rel_type = qa.get("rel_type")
                rel_group = qa.get("rel_group") or infer_rel_group(rel_type)

                if rel_group not in group_stats:
                    group_stats[rel_group] = {"correct": 0, "total": 0}

                if args.text_only:
                    caption = get_caption_for_qa(data, qa)
                    if caption is not None:
                        conversation = build_caption_yesno_prompt(caption, question)
                    else:
                        # Backward-compatible fallback when caption linkage is missing.
                        conversation = build_scene_yesno_prompt(scene_text or "", question)

                    prompt_text = processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    pred, completion, confidence, p_yes, p_no = predict_text_only(model, processor, prompt_text)
                else:
                    caption = None
                    conversation = build_visual_yesno_prompt(question)
                    prompt_text = processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    inputs = processor(text=prompt_text, images=image, return_tensors="pt")
                    inputs = _move_to_device(inputs, _pick_model_input_device(model))

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=1,
                            output_scores=True,
                            return_dict_in_generate=True,
                            pad_token_id=processor.tokenizer.pad_token_id,
                        )
                    pred = get_prediction(outputs, processor)
                    confidence, p_yes, p_no = get_yes_no_confidence(outputs, processor.tokenizer)
                    # Decode the generated completion (what the model actually output)
                    try:
                        seq = outputs.sequences[0]
                        prompt_len = inputs["input_ids"].shape[1]
                        gen_ids = seq[prompt_len:]
                        completion = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    except Exception:
                        completion = ""

                # Store a single transcript in Prompt column for easier debugging.
                transcript = f"{prompt_text}\nMODEL: {completion}" if completion else prompt_text

                total_questions += 1
                is_correct = (pred == gt)
                if is_correct:
                    correct_questions += 1

                group_stats[rel_group]["total"] += 1
                if is_correct:
                    group_stats[rel_group]["correct"] += 1

                # Append to Model Call Log
                with open(args.log_file, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            time.strftime("%Y-%m-%d %H:%M:%S"),
                            level,
                            "" if args.text_only else image_abs_path,
                            rel_group,
                            rel_type if rel_type is not None else "",
                            transcript,
                            completion,
                            "" if confidence is None else float(confidence),
                            "" if p_yes is None else float(p_yes),
                            "" if p_no is None else float(p_no),
                            gt,
                            pred,
                            "CORRECT" if is_correct else "INCORRECT",
                            mode_label,
                        ]
                    )

        accuracy = (correct_questions / total_questions) * 100 if total_questions > 0 else 0

        # Add per-group accuracies
        group_acc = {}
        for g, s in group_stats.items():
            g_total = s.get("total", 0)
            g_correct = s.get("correct", 0)
            group_acc[g] = {
                "correct": g_correct,
                "total": g_total,
                "acc": (g_correct / g_total) * 100 if g_total > 0 else 0,
            }

        level_results[level] = {
            "correct": correct_questions,
            "total": total_questions,
            "acc": accuracy,
            "by_rel_group": group_acc,
        }
        print(f"Level {level} Finished: {correct_questions}/{total_questions} correct ({accuracy:.2f}%)")

    # Save summary files
    with open(args.output_json, "w") as jf:
        json.dump(level_results, jf, indent=4)

    report_name = "evaluation_summary.txt"
    with open(report_name, "w") as f:
        f.write(f"Evaluation Summary Report | Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 45 + "\n")
        f.write(f"{'Level':<10} | {'Correct/Total':<18} | {'Accuracy %':<10}\n")
        f.write("-" * 45 + "\n")
        for lvl, stats in level_results.items():
            f.write(f"{lvl:<10} | {stats['correct']}/{stats['total']:<17} | {stats['acc']:.2f}%\n")
            by_group = stats.get("by_rel_group", {})
            if isinstance(by_group, dict) and by_group:
                for g_name, g_stats in by_group.items():
                    f.write(f"  - {g_name:<8} : {g_stats['correct']}/{g_stats['total']} ({g_stats['acc']:.2f}%)\n")
        f.write("=" * 45 + "\n")

    print(f"\nResults saved to '{args.output_json}', '{report_name}', and call logs in '{args.log_file}'.")

if __name__ == "__main__":
    main()