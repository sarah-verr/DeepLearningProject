#!/usr/bin/env python3
import argparse
import json
import os
import sys

import torch

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RESULTS_ROOT = os.path.join(_REPO_ROOT, "results_llava-hf", "llava-1.5-7b-hf")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils import prompt_llava
from utils.logit_lens import logit_lens_yesno

REL_PHRASE_CANDIDATES = [
    "to the left of",
    "to the right of",
    "left of",
    "right of",
    "above",
    "below",
    "touching",
    "overlapping",
    "inside",
    "around",
    "next to",
    "beside",
]


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


def _find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    if not needle or not haystack or len(needle) > len(haystack):
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def _locate_span(tokenizer, full_ids: list[int], span_text: str) -> list[int]:
    span_ids_a = tokenizer(span_text, add_special_tokens=False).input_ids
    span_ids_b = tokenizer(" " + span_text, add_special_tokens=False).input_ids

    start = _find_subsequence(full_ids, span_ids_a)
    span_ids = span_ids_a
    if start is None:
        start = _find_subsequence(full_ids, span_ids_b)
        span_ids = span_ids_b
    if start is None:
        return []
    return list(range(start, start + len(span_ids)))


def _locate_rel_phrase_token_positions(tokenizer, full_ids: list[int], question_text: str, rel_phrase: str) -> list[int]:
    if not question_text or not rel_phrase:
        return []
    q_ids = tokenizer(question_text, add_special_tokens=False).input_ids
    q_start = _find_subsequence(full_ids, q_ids)

    phrase_ids_a = tokenizer(rel_phrase, add_special_tokens=False).input_ids
    phrase_ids_b = tokenizer(" " + rel_phrase, add_special_tokens=False).input_ids

    if q_start is not None:
        p_start = _find_subsequence(q_ids, phrase_ids_a)
        p_len = len(phrase_ids_a)
        if p_start is None:
            p_start = _find_subsequence(q_ids, phrase_ids_b)
            p_len = len(phrase_ids_b)
        if p_start is None:
            return []
        abs_start = q_start + p_start
        return list(range(abs_start, abs_start + p_len))

    p_start = _find_subsequence(full_ids, phrase_ids_a)
    p_len = len(phrase_ids_a)
    if p_start is None:
        p_start = _find_subsequence(full_ids, phrase_ids_b)
        p_len = len(phrase_ids_b)
    if p_start is None:
        return []
    return list(range(p_start, p_start + p_len))


def _infer_rel_phrase(question: str) -> str | None:
    q = f" {question.lower()} "
    for phrase in REL_PHRASE_CANDIDATES:
        if f" {phrase} " in q:
            return phrase
    return None


def _mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def _compute_attention_masses(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    pairs: dict[str, tuple[list[int], list[int]]],
) -> dict[str, dict]:
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    attns = out.attentions
    if not attns:
        return {}

    seq_len = int(input_ids.shape[1])
    results: dict[str, dict] = {}
    for name, (q_pos, k_pos) in pairs.items():
        q_pos = [p for p in q_pos if 0 <= p < seq_len]
        k_pos = [p for p in k_pos if 0 <= p < seq_len]
        if not q_pos or not k_pos:
            continue
        per_layer = []
        for layer_attn in attns:
            layer = layer_attn[0]  # [heads, seq, seq]
            sub = layer[:, q_pos, :][:, :, k_pos]
            mass = sub.sum(dim=-1).mean().item()
            per_layer.append(float(mass))
        results[name] = {"per_layer": per_layer, "mean": _mean(per_layer)}
    return results


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run VLM-style prompting on objective captions (text-only).")
    ap.add_argument(
        "--base_data_path",
        type=str,
        default=os.path.join(_REPO_ROOT, "data", "vlm_levels_objective"),
        help="Base objective dataset path (expects level_*/ann/*_b.json).",
    )
    ap.add_argument("--levels", nargs="*", default=None, help="Levels to process (e.g., level_1 level_2).")
    ap.add_argument("--max_files", type=int, default=None, help="Limit number of *_b.json files per level.")
    ap.add_argument("--max_questions", type=int, default=None, help="Limit QAs per file.")
    ap.add_argument("--model_id", type=str, default=MODEL_ID)
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_RESULTS_ROOT, "text_only_objective"),
    )
    ap.add_argument("--max_new_tokens", type=int, default=5)
    ap.add_argument(
        "--no_attention_summary",
        action="store_true",
        help="Disable attention summary computation (faster).",
    )
    ap.add_argument(
        "--no_logit_lens",
        action="store_true",
        help="Disable per-layer logit-lens computation (faster).",
    )
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

    processor, model, device = prompt_llava._init_model(args.model_id)
    tokenizer = processor.tokenizer

    include_attention = not args.no_attention_summary
    include_logit_lens = not args.no_logit_lens

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

                    prompt = prompt_llava.build_prompt(
                        processor,
                        "text_only",
                        question,
                        caption=caption,
                    )
                    tok = prompt_llava.prepare_inputs(processor, prompt, device=device)

                    gen = prompt_llava.generate_output_for_model(

                        model,
                        tok,
                        max_new_tokens=args.max_new_tokens,
                    )
                    decoded = tokenizer.decode(gen.sequences[0], skip_special_tokens=True)
                    pred_text = decoded.split("ASSISTANT:", 1)[-1].strip()
                    pred = _get_yesno_from_generated_text(pred_text)
                    score_pred, _, p_yes, p_no = prompt_llava.get_yes_no_probability(gen, tokenizer)
                    if score_pred is not None:
                        pred = score_pred

                    is_correct = pred == answer
                    num_total += 1
                    num_correct += int(is_correct)

                    logit_lens = None
                    if include_logit_lens:
                        logit_lens = logit_lens_yesno(
                            model=model,
                            tokenizer=tokenizer,
                            input_ids=tok["input_ids"],
                            attention_mask=tok.get("attention_mask"),
                        )

                    attention_summary = None
                    if include_attention:
                        full_ids = tok["input_ids"][0]
                        full_ids_list = full_ids.detach().cpu().tolist()
                        caption_positions = _locate_span(tokenizer, full_ids_list, caption)
                        question_positions = _locate_span(tokenizer, full_ids_list, question)
                        rel_phrase = _infer_rel_phrase(question)
                        phrase_positions = []
                        if rel_phrase:
                            phrase_positions = _locate_rel_phrase_token_positions(
                                tokenizer, full_ids_list, question_text=question, rel_phrase=rel_phrase
                            )

                        entity_positions = []
                        for obj in ann.get("objects", []):
                            color = obj.get("color")
                            shape = obj.get("shape")
                            if color and shape:
                                phrase = f"{color} {shape}"
                                entity_positions.extend(_locate_span(tokenizer, full_ids_list, phrase))

                        if caption_positions and question_positions:
                            seq_len = int(tok["input_ids"].shape[1])
                            pos_set = set(caption_positions) | set(question_positions)
                            other_positions = [i for i in range(seq_len) if i not in pos_set]

                            pairs = {
                                "caption": (question_positions, caption_positions),
                                "question": (question_positions, question_positions),
                                "other": (question_positions, other_positions),
                            }
                            if entity_positions:
                                pairs["entity"] = (question_positions, entity_positions)
                            if phrase_positions and entity_positions:
                                pairs["rel_entity"] = (phrase_positions, entity_positions)

                            masses = _compute_attention_masses(
                                model=model,
                                input_ids=tok["input_ids"],
                                attention_mask=tok.get("attention_mask"),
                                pairs=pairs,
                            )

                            attention_summary = {
                                "caption_mass_per_layer": masses.get("caption", {}).get("per_layer"),
                                "caption_mass_mean": masses.get("caption", {}).get("mean"),
                                "question_mass_per_layer": masses.get("question", {}).get("per_layer"),
                                "question_mass_mean": masses.get("question", {}).get("mean"),
                                "other_mass_per_layer": masses.get("other", {}).get("per_layer"),
                                "other_mass_mean": masses.get("other", {}).get("mean"),
                                "entity_mass_per_layer": masses.get("entity", {}).get("per_layer"),
                                "entity_mass_mean": masses.get("entity", {}).get("mean"),
                                "rel_entity_mass_per_layer": masses.get("rel_entity", {}).get("per_layer"),
                                "rel_entity_mass_mean": masses.get("rel_entity", {}).get("mean"),
                                "rel_phrase": rel_phrase,
                            }

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
                        "logit_lens": logit_lens,
                        "attention_summary": attention_summary,
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
