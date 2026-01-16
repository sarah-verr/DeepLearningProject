#!/usr/bin/env python3
import argparse
import json
import os
import sys

from PIL import Image

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RESULTS_ROOT = os.path.join(_REPO_ROOT, "results_llava-hf", "llava-1.5-7b-hf")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils import prompt_llava
from utils.logit_lens import logit_lens_attribute, attribute_token_ids

SHAPES = ["square", "circle", "triangle", "star"]
COLORS = ["red", "blue", "green", "yellow", "purple", "cyan", "orange", "pink", "lime"]


def _find_ann_paths(base_dir: str, level: str) -> list[str]:
    ann_dir = os.path.join(base_dir, level, "ann")
    if not os.path.isdir(ann_dir):
        return []
    paths = [os.path.join(ann_dir, f) for f in os.listdir(ann_dir) if f.endswith(".json")]
    return sorted(paths)


def _build_image_map(images_dir: str) -> dict[str, str]:
    image_map: dict[str, str] = {}
    if not os.path.isdir(images_dir):
        return image_map
    for filename in os.listdir(images_dir):
        lower = filename.lower()
        if not lower.endswith((".png", ".jpg", ".jpeg")):
            continue
        image_id = os.path.splitext(filename)[0]
        image_map[image_id] = os.path.join(images_dir, filename)
    return image_map


def _level_number(level: str) -> int | None:
    try:
        return int(level.split("_", 1)[1])
    except Exception:
        return None


def _choice_sets(question_type: str) -> list[str]:
    if question_type == "shape":
        return SHAPES
    if question_type == "color":
        return COLORS
    return []


def _score_attribute_choices(outputs, tokenizer, options: list[str]) -> tuple[str | None, dict[str, float]]:
    scores = getattr(outputs, "scores", None)
    if not scores:
        return None, {}
    first_token_logits = scores[0][0]
    probs = first_token_logits.softmax(dim=-1)
    option_ids = attribute_token_ids(tokenizer, options)

    masses: dict[str, float] = {}
    for opt, ids in option_ids.items():
        masses[opt] = float(sum(probs[i].item() for i in ids if i < probs.numel()))
    total = sum(masses.values())
    if total > 0:
        for k in masses:
            masses[k] = masses[k] / total
    if not masses:
        return None, {}
    pred = max(masses.items(), key=lambda x: x[1])[0]
    return pred, masses


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run visual logit-lens prompting on attribute QA.")
    ap.add_argument(
        "--base_data_path",
        type=str,
        default=os.path.join(_REPO_ROOT, "data", "vlm_levels_v2"),
        help="Base dataset path (expects level_*/ann/*.json).",
    )
    ap.add_argument("--levels", nargs="*", default=None, help="Levels to process (e.g., level_1 level_2).")
    ap.add_argument("--max_files", type=int, default=None, help="Limit number of annotation files per level.")
    ap.add_argument("--max_questions", type=int, default=None, help="Limit QAs per file.")
    ap.add_argument("--model_id", type=str, default=MODEL_ID)
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_RESULTS_ROOT, "visual_logit_lens_attribute"),
    )
    ap.add_argument("--max_new_tokens", type=int, default=10)
    ap.add_argument(
        "--no_logit_lens",
        action="store_true",
        help="Disable per-layer logit-lens computation (faster).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.base_data_path):
        raise SystemExit(f"Dataset path not found: {args.base_data_path}")

    levels = args.levels
    if not levels:
        levels = [d for d in os.listdir(args.base_data_path) if d.startswith("level_")]
        levels = sorted(levels)

    if not levels:
        raise SystemExit(f"No level_* directories found in: {args.base_data_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model: {args.model_id}")

    processor, model, device = prompt_llava._init_model(args.model_id)
    tokenizer = processor.tokenizer
    include_logit_lens = not args.no_logit_lens

    for lvl in levels:
        ann_paths = _find_ann_paths(args.base_data_path, lvl)
        if args.max_files is not None:
            ann_paths = ann_paths[: args.max_files]
        if not ann_paths:
            continue

        images_dir = os.path.join(args.base_data_path, lvl, "images")
        image_map = _build_image_map(images_dir)

        level_out_dir = os.path.join(args.out_dir, lvl)
        os.makedirs(level_out_dir, exist_ok=True)
        summary_path = os.path.join(level_out_dir, "summary.jsonl")
        metrics_path = os.path.join(level_out_dir, "metrics.json")

        num_total = 0
        num_correct = 0
        level_num = _level_number(lvl)

        with open(summary_path, "w", encoding="utf-8") as f:
            for ann_path in ann_paths:
                with open(ann_path, "r", encoding="utf-8") as jf:
                    ann = json.load(jf)

                image_id = os.path.splitext(os.path.basename(ann_path))[0]
                image_path = image_map.get(image_id)
                if not image_path:
                    continue

                qa_list = ann.get("qa", []) or []
                if args.max_questions is not None:
                    qa_list = qa_list[: args.max_questions]

                with Image.open(image_path) as image:
                    for qa in qa_list:
                        question = (qa.get("question") or "").strip()
                        answer = (qa.get("answer") or "").strip().lower()
                        question_type = (qa.get("question_type") or "").strip().lower()
                        options = _choice_sets(question_type)
                        if not question or not answer or not options:
                            continue
                        if answer not in options:
                            continue

                        prompt = prompt_llava.build_prompt(
                            processor,
                            "visual_attribute",
                            question,
                            qa_item=qa,
                        )
                        inputs = prompt_llava.prepare_inputs(
                            processor,
                            prompt,
                            image=image,
                            device=device,
                        )
                        gen = prompt_llava.generate_output_for_model(
                            model,
                            inputs,
                            max_new_tokens=args.max_new_tokens,
                        )

                        pred, p_by_choice = _score_attribute_choices(
                            gen, tokenizer, options
                        )
                        if pred is None:
                            continue

                        is_correct = pred == answer
                        num_total += 1
                        num_correct += int(is_correct)

                        logit_lens = None
                        if include_logit_lens:
                            model_kwargs = {}
                            if "pixel_values" in inputs:
                                model_kwargs["pixel_values"] = inputs["pixel_values"]
                            logit_lens = logit_lens_attribute(
                                model=model,
                                tokenizer=tokenizer,
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs.get("attention_mask"),
                                options=options,
                                model_kwargs=model_kwargs if model_kwargs else None,
                            )

                        record = {
                            "level": level_num,
                            "image_id": image_id,
                            "ann_path": os.path.abspath(ann_path),
                            "question": question,
                            "question_type": question_type,
                            "answer": answer,
                            "prediction": pred,
                            "is_correct": is_correct,
                            "p_by_choice": p_by_choice,
                            "logit_lens": logit_lens,
                            "mode": "visual_attribute",
                        }
                        for key in ("subject_id", "object_id", "rel_type", "rel_group", "rel_phrase", "id"):
                            if key in qa:
                                record[key] = qa[key]
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
