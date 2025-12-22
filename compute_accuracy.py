import torch
import os
import json
import argparse
import time
import csv
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# --- Configuration ---
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
BASE_DATA_PATH = "/home/kkarthikeyan/deep-learning/DeepLearningProject/Synthetic-Data/vlm_levels"


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

def get_prediction(outputs, processor, prompt_context="ASSISTANT:"):
    """Extracts 'yes' or 'no' based on logit probability."""
    first_token_logits = outputs.scores[0][0]
    probs = torch.softmax(first_token_logits, dim=-1)

    yes_tokens = [
        processor.tokenizer.encode(f"{prompt_context} Yes", add_special_tokens=False)[-1],
        processor.tokenizer.encode(f"{prompt_context} yes", add_special_tokens=False)[-1]
    ]
    no_tokens = [
        processor.tokenizer.encode(f"{prompt_context} No", add_special_tokens=False)[-1],
        processor.tokenizer.encode(f"{prompt_context} no", add_special_tokens=False)[-1]
    ]

    prob_yes = sum([probs[t_id].item() for t_id in yes_tokens if t_id < len(probs)])
    prob_no = sum([probs[t_id].item() for t_id in no_tokens if t_id < len(probs)])

    return "yes" if prob_yes > prob_no else "no"

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

def build_text_only_prompt(scene_text: str, question: str) -> str:
    return (
        "You are given a synthetic scene description.\n"
        "Answer the question with only \"yes\" or \"no\".\n\n"
        f"Scene:\n{scene_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

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

def _move_to_device(batch: dict, device: str):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if hasattr(v, "to") else v
    return out

def predict_text_only(model, processor, prompt: str, device: str) -> str:
    tok = processor.tokenizer
    inputs = tok(prompt, return_tensors="pt")
    # with device_map="auto", model tensors live on cuda; use model.device if present
    model_device = getattr(model, "device", device)
    inputs = _move_to_device(inputs, model_device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            return_dict_in_generate=False,
        )

    decoded_full = tok.decode(out[0], skip_special_tokens=True)
    decoded_prompt = tok.decode(inputs["input_ids"][0], skip_special_tokens=True)
    completion = decoded_full[len(decoded_prompt):]
    return get_yesno_from_generated_text(completion)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[0,1,2,3,4,5,6])
    ap.add_argument("--output_json", type=str, default="evaluation_results.json")
    ap.add_argument("--log_file", type=str, default="model_calls_log.csv")
    ap.add_argument("--text_only", action="store_true", help="Evaluate text-only relational reasoning (no image).")
    return ap.parse_args()

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID}...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
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

            # Build scene text once for text-only mode
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
                    prompt = build_text_only_prompt(scene_text or "", question)
                    pred = predict_text_only(model, processor, prompt, device=device)
                else:
                    prompt = f"USER: <image>\n{question}\nASSISTANT:"
                    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=1,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )
                    pred = get_prediction(outputs, processor)

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
                            question if args.text_only else prompt,
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