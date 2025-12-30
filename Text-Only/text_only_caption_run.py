import argparse
import json
import os
import sys

import torch

from transformers import AutoProcessor, LlavaForConditionalGeneration

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from text_only_phrase_attention import (  # noqa: E402
    _pick_model_input_device,
    _locate_rel_phrase_token_positions,
    _locate_span,
    _token_labels_for_positions,
    build_caption_question_prompt,
    compute_phrase_to_caption_attention,
    plot_phrase_to_caption,
)


MODEL_ID = "llava-hf/llava-1.5-7b-hf"

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


def _find_ann_paths(base_dir: str, allowed_levels: set[str] | None = None) -> list[str]:
    paths = []
    for entry in os.listdir(base_dir):
        if not entry.startswith("level_"):
            continue
        if allowed_levels is not None and entry not in allowed_levels:
            continue
        ann_dir = os.path.join(base_dir, entry, "ann")
        if not os.path.isdir(ann_dir):
            continue
        for fname in os.listdir(ann_dir):
            if fname.endswith("_b.json"):
                paths.append(os.path.join(ann_dir, fname))
    return sorted(paths)


def _get_caption_for_qa(annotation: dict, qa_item: dict) -> str | None:
    cap_id = qa_item.get("caption_id")
    captions_meta = annotation.get("captions_meta")
    if cap_id is not None and isinstance(captions_meta, list) and captions_meta:
        try:
            cap_id_int = int(cap_id)
        except Exception:
            cap_id_int = None
        if cap_id_int is not None and 0 <= cap_id_int < len(captions_meta):
            c = captions_meta[cap_id_int]
            if isinstance(c, dict):
                cap_text = c.get("caption")
                if isinstance(cap_text, str) and cap_text.strip():
                    return cap_text.strip()
        if cap_id_int is not None:
            for c in captions_meta:
                if isinstance(c, dict) and c.get("id") == cap_id_int:
                    cap_text = c.get("caption")
                    if isinstance(cap_text, str) and cap_text.strip():
                        return cap_text.strip()

    caps = annotation.get("captions", []) or []
    if isinstance(caps, list) and caps:
        cap = caps[0]
        if isinstance(cap, str) and cap.strip():
            return cap.strip()
    return None


def _infer_rel_phrase(question: str) -> str | None:
    q = f" {question.lower()} "
    for phrase in REL_PHRASE_CANDIDATES:
        if f" {phrase} " in q:
            return phrase
    return None


def _move_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if hasattr(v, "to") else v
    return out


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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run caption->question text-only eval with attention.")
    ap.add_argument(
        "--base_data_path",
        type=str,
        default=os.path.join(_REPO_ROOT, "Synthetic-Data", "vlm_levels"),
        help="Base dataset path (expects level_*/ann/*_b.json).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.abspath(os.path.dirname(__file__)), "vis_results_caption"),
        help="Output directory for summaries and plots.",
    )
    ap.add_argument("--model_id", type=str, default=MODEL_ID)
    ap.add_argument("--max_files", type=int, default=None, help="Limit number of *_b.json files.")
    ap.add_argument("--max_questions", type=int, default=None, help="Limit QAs per file.")
    ap.add_argument(
        "--levels",
        nargs="*",
        default=None,
        help="Only process these level_* directories (e.g., level_1 level_2).",
    )
    ap.add_argument(
        "--plot_examples",
        type=int,
        default=3,
        help="Plots per file (use -1 for all).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    allowed_levels = set(args.levels) if args.levels else None
    ann_paths = _find_ann_paths(args.base_data_path, allowed_levels=allowed_levels)
    if args.max_files is not None:
        ann_paths = ann_paths[: args.max_files]
    if not ann_paths:
        raise SystemExit(f"No *_b.json files found in: {args.base_data_path}")

    print(f"Loading model: {args.model_id}")
    if torch.cuda.is_available():
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.float32,
            device_map=None,
            attn_implementation="eager",
        )
        model = model.to("cpu")
    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer
    model.eval()

    summary_path = os.path.join(args.out_dir, "summary.jsonl")
    metrics_path = os.path.join(args.out_dir, "metrics.json")

    num_total = 0
    num_correct = 0
    num_no_caption = 0
    num_no_rel_phrase = 0
    num_attn_plotted = 0

    with open(summary_path, "w", encoding="utf-8") as f:
        for ann_i, ann_path in enumerate(ann_paths):
            with open(ann_path, "r", encoding="utf-8") as jf:
                ann = json.load(jf)

            qa_list = ann.get("qa", []) or []
            if args.max_questions is not None:
                qa_list = qa_list[: args.max_questions]

            per_file_plots = 0
            plot_limit = args.plot_examples

            for qi, qa in enumerate(qa_list):
                question = (qa.get("question") or "").strip()
                answer = (qa.get("answer") or "").strip().lower()
                rel_phrase = (qa.get("rel_phrase") or "").strip()

                caption = _get_caption_for_qa(ann, qa)
                if not caption or not question:
                    num_no_caption += 1
                    continue

                if not rel_phrase:
                    rel_phrase = _infer_rel_phrase(question) or ""

                prompt = build_caption_question_prompt(caption, question)
                tok = tokenizer(prompt, return_tensors="pt")
                model_device = _pick_model_input_device(model)
                tok = _move_to_device(tok, model_device)

                full_ids = tok["input_ids"][0]
                full_ids_list = full_ids.detach().cpu().tolist()

                caption_positions = _locate_span(tokenizer, full_ids_list, caption)
                phrase_positions = []
                if rel_phrase:
                    phrase_positions = _locate_rel_phrase_token_positions(
                        tokenizer, full_ids, question_text=question, rel_phrase=rel_phrase
                    )

                if not caption_positions:
                    num_no_caption += 1
                    continue
                if not phrase_positions:
                    num_no_rel_phrase += 1

                with torch.no_grad():
                    gen = model.generate(
                        **tok,
                        max_new_tokens=5,
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                decoded = tokenizer.decode(gen.sequences[0], skip_special_tokens=True)
                pred_text = decoded.split("Answer:", 1)[-1].strip()
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
                    "rel_phrase": rel_phrase or None,
                    "p_yes": p_yes,
                    "p_no": p_no,
                }

                if phrase_positions:
                    stats = compute_phrase_to_caption_attention(
                        model=model,
                        tokenizer=tokenizer,
                        input_ids=tok["input_ids"],
                        attention_mask=tok.get("attention_mask"),
                        phrase_positions=phrase_positions,
                        caption_positions=caption_positions,
                    )

                    caption_tok_labels = _token_labels_for_positions(
                        tokenizer, full_ids_list, caption_positions
                    )
                    phrase_tok_labels = _token_labels_for_positions(
                        tokenizer, full_ids_list, phrase_positions
                    )

                    record.update(
                        {
                            "caption_token_positions": caption_positions,
                            "phrase_token_positions": phrase_positions,
                            "caption_tokens": caption_tok_labels,
                            "phrase_tokens": phrase_tok_labels,
                            "token_scores_mean": stats["mean"].tolist(),
                        }
                    )

                    if plot_limit != 0 and (plot_limit < 0 or per_file_plots < plot_limit):
                        ex_dir = os.path.join(
                            args.out_dir,
                            os.path.splitext(os.path.basename(ann_path))[0],
                            f"qa_{qi:03d}",
                        )
                        title_prefix = f"Phrase→Caption | '{rel_phrase}'"
                        plot_phrase_to_caption(
                            out_dir=ex_dir,
                            caption_tokens=caption_tok_labels,
                            phrase_tokens=phrase_tok_labels,
                            scores_per_layer=stats["per_layer"],
                            scores_mean=stats["mean"],
                            title_prefix=title_prefix,
                        )
                        per_file_plots += 1
                        num_attn_plotted += 1

                f.write(json.dumps(record) + "\n")

    metrics = {
        "base_data_path": args.base_data_path,
        "num_files": len(ann_paths),
        "num_total": num_total,
        "num_correct": num_correct,
        "accuracy": (num_correct / num_total) if num_total else 0.0,
        "num_no_caption": num_no_caption,
        "num_no_rel_phrase": num_no_rel_phrase,
        "num_attn_plotted": num_attn_plotted,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote summary to: {summary_path}")
    print(f"Wrote metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
