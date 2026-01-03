import argparse
import json
import os

import torch

from transformers import AutoProcessor, LlavaForConditionalGeneration


MODEL_ID = "llava-hf/llava-1.6-7b-hf"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_b_ann_paths(base_dir: str, level: str) -> list[str]:
    ann_dir = os.path.join(base_dir, level, "ann")
    if not os.path.isdir(ann_dir):
        return []
    paths = [os.path.join(ann_dir, f) for f in os.listdir(ann_dir) if f.endswith("_b.json")]
    return sorted(paths)


def _get_objective_caption(ann: dict) -> str | None:
    caps = ann.get("captions", []) or []
    if isinstance(caps, list) and caps:
        cap = caps[0]
        if isinstance(cap, str) and cap.strip():
            return cap.strip()
    meta = ann.get("meta") or {}
    cap = meta.get("objective_caption")
    if isinstance(cap, str) and cap.strip():
        return cap.strip()
    return None


def _get_yesno_from_generated_text(s: str) -> str:
    s = (s or "").strip().lower()
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    if " yes" in f" {s}":
        return "yes"
    if " no" in f" {s}":
        return "no"
    return "unknown"


def _get_yesno_from_scores(outputs, tokenizer) -> tuple[str | None, float | None, float | None]:
    scores = getattr(outputs, "scores", None)
    if not scores:
        return None, None, None

    first_token_logits = scores[0][0]
    probs = torch.softmax(first_token_logits, dim=-1)

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
    pred = "yes" if p_yes_n >= p_no_n else "no"
    return pred, p_yes_n, p_no_n


def _build_vlm_style_prompt(caption: str, question: str) -> str:
    return f"USER: {caption}\n{question}\nASSISTANT:"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run VLM-style prompting on objective captions.")
    ap.add_argument(
        "--base_data_path",
        type=str,
        default=os.path.join(_REPO_ROOT, "vlm_levels_objective"),
        help="Base objective dataset path (expects level_*/ann/*_b.json).",
    )
    ap.add_argument("--levels", nargs="*", default=None, help="Levels to process (e.g., level_1 level_2).")
    ap.add_argument("--max_files", type=int, default=None, help="Limit number of *_b.json files per level.")
    ap.add_argument("--max_questions", type=int, default=None, help="Limit QAs per file.")
    ap.add_argument("--model_id", type=str, default=MODEL_ID)
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_REPO_ROOT, "Text-Only", "vis_results_caption", "objective_vlm_prompt"),
    )
    ap.add_argument("--max_new_tokens", type=int, default=5)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.base_data_path):
        raise SystemExit(f"Objective dataset path not found: {args.base_data_path}")

    levels = args.levels
    if not levels:
        levels = [d for d in os.listdir(args.base_data_path) if d.startswith("level_")]
        levels = sorted(levels)

    if not levels:
        raise SystemExit(f"No level_* directories found in: {args.base_data_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model: {args.model_id}")
    if torch.cuda.is_available():
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_id,
            dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_id,
            dtype=torch.float32,
            device_map=None,
            attn_implementation="eager",
        )
        model = model.to("cpu")

    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer
    model.eval()

    for lvl in levels:
        ann_paths = _find_b_ann_paths(args.base_data_path, lvl)
        if args.max_files is not None:
            ann_paths = ann_paths[: args.max_files]
        if not ann_paths:
            continue

        level_out_dir = os.path.join(args.out_dir, lvl)
        os.makedirs(level_out_dir, exist_ok=True)
        summary_path = os.path.join(level_out_dir, "summary.jsonl")
        metrics_path = os.path.join(level_out_dir, "metrics.json")

        num_total = 0
        num_correct = 0

        with open(summary_path, "w", encoding="utf-8") as f:
            for ann_path in ann_paths:
                with open(ann_path, "r", encoding="utf-8") as jf:
                    ann = json.load(jf)

                caption = _get_objective_caption(ann)
                if not caption:
                    continue

                qa_list = ann.get("qa", []) or []
                if args.max_questions is not None:
                    qa_list = qa_list[: args.max_questions]

                for qi, qa in enumerate(qa_list):
                    question = (qa.get("question") or "").strip()
                    answer = (qa.get("answer") or "").strip().lower()
                    if not question or answer not in ("yes", "no"):
                        continue

                    prompt = _build_vlm_style_prompt(caption, question)
                    tok = tokenizer(prompt, return_tensors="pt")
                    tok = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in tok.items()}

                    with torch.no_grad():
                        gen = model.generate(
                            **tok,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                    decoded = tokenizer.decode(gen.sequences[0], skip_special_tokens=True)
                    pred_text = decoded.split("ASSISTANT:", 1)[-1].strip()
                    pred = _get_yesno_from_generated_text(pred_text)
                    score_pred, p_yes, p_no = _get_yesno_from_scores(gen, tokenizer)
                    if score_pred is not None:
                        pred = score_pred

                    is_correct = pred == answer
                    num_total += 1
                    num_correct += int(is_correct)

                    record = {
                        "ann_path": ann_path,
                        "qa_index": qi,
                        "caption": caption,
                        "question": question,
                        "answer": answer,
                        "prediction": pred,
                        "is_correct": is_correct,
                        "p_yes": p_yes,
                        "p_no": p_no,
                    }
                    f.write(json.dumps(record) + "\n")

        metrics = {
            "num_total": num_total,
            "num_correct": num_correct,
            "accuracy": (num_correct / num_total) if num_total > 0 else None,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"[{lvl}] Wrote summary to: {summary_path}")
        print(f"[{lvl}] Wrote metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
