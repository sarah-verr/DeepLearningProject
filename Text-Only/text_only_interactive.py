import argparse
import json
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoProcessor, LlavaForConditionalGeneration

from text_only_phrase_attention import (
    _pick_model_input_device,
    _locate_rel_phrase_token_positions,
    _locate_span,
    _token_labels_for_positions,
    build_caption_question_prompt,
    compute_phrase_to_caption_attention,
    plot_phrase_to_caption,
)


MODEL_ID = "llava-hf/llava-1.5-7b-hf"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def _infer_rel_phrase(question: str) -> str | None:
    q = f" {question.lower()} "
    for phrase in REL_PHRASE_CANDIDATES:
        if f" {phrase} " in q:
            return phrase
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


def _find_b_ann_paths(base_dir: str, level: str) -> list[str]:
    ann_dir = os.path.join(base_dir, level, "ann")
    if not os.path.isdir(ann_dir):
        return []
    paths = [os.path.join(ann_dir, f) for f in os.listdir(ann_dir) if f.endswith("_b.json")]
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


def _attention_mass_per_layer(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    query_positions: list[int],
    key_positions: list[int],
) -> list[float]:
    if not query_positions or not key_positions:
        return []

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
        raise RuntimeError("Model did not return attentions. Ensure attn_implementation='eager'.")

    seq_len = int(input_ids.shape[1])
    q_pos = [p for p in query_positions if 0 <= p < seq_len]
    k_pos = [p for p in key_positions if 0 <= p < seq_len]
    if not q_pos or not k_pos:
        return []

    per_layer = []
    for layer_attn in attns:
        layer = layer_attn[0]  # [heads, seq, seq]
        sub = layer[:, q_pos, :][:, :, k_pos]
        # Average attention mass from queries to the key set.
        mass = sub.sum(dim=-1).mean().item()
        per_layer.append(float(mass))
    return per_layer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Interactive text-only QA with attention summaries.")
    ap.add_argument("--caption", type=str, default=None)
    ap.add_argument("--question", type=str, default=None)
    ap.add_argument("--gt", type=str, default=None, help="Ground-truth answer: yes/no (optional).")
    ap.add_argument("--rel_phrase", type=str, default=None, help="Relational phrase to analyze (optional).")
    ap.add_argument("--model_id", type=str, default=MODEL_ID)
    ap.add_argument(
        "--base_data_path",
        type=str,
        default=os.path.join(_REPO_ROOT, "Synthetic-Data", "vlm_levels"),
        help="Base dataset path (expects level_*/ann/*_b.json).",
    )
    ap.add_argument("--level", type=str, default=None, help="Run over a level (e.g., level_1).")
    ap.add_argument("--max_files", type=int, default=None, help="Limit number of *_b.json files.")
    ap.add_argument("--max_questions", type=int, default=None, help="Limit QAs per file.")
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.abspath(os.path.dirname(__file__)), "vis_results_caption", "interactive"),
    )
    ap.add_argument("--max_new_tokens", type=int, default=5)
    ap.add_argument("--save_phrase_plots", action="store_true", help="Save phrase->caption plots if rel_phrase is found.")
    ap.add_argument(
        "--plot_examples",
        type=int,
        default=0,
        help="Plots per file when running a level (use -1 for all).",
    )
    ap.add_argument(
        "--summary_plots",
        action="store_true",
        help="Save correctness vs attention plots in batch mode.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

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

    def _run_single(*, caption: str, question: str, gt: str | None, rel_phrase: str | None) -> dict:
        prompt = build_caption_question_prompt(caption, question)
        tok = tokenizer(prompt, return_tensors="pt")
        model_device = _pick_model_input_device(model)
        tok = {k: (v.to(model_device) if hasattr(v, "to") else v) for k, v in tok.items()}

        with torch.no_grad():
            gen = model.generate(
                **tok,
                max_new_tokens=args.max_new_tokens,
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

        gt_clean = (gt or "").strip().lower() if gt is not None else None
        if gt_clean not in (None, "yes", "no"):
            gt_clean = None
        is_correct = (pred == gt_clean) if gt_clean in ("yes", "no") else None

        full_ids = tok["input_ids"][0]
        full_ids_list = full_ids.detach().cpu().tolist()

        caption_positions = _locate_span(tokenizer, full_ids_list, caption)
        question_positions = _locate_span(tokenizer, full_ids_list, question)
        if not caption_positions or not question_positions:
            raise ValueError("Could not locate caption/question tokens in the prompt.")

        seq_len = int(tok["input_ids"].shape[1])
        pos_set = set(caption_positions) | set(question_positions)
        other_positions = [i for i in range(seq_len) if i not in pos_set]

        caption_mass = _attention_mass_per_layer(
            model=model,
            input_ids=tok["input_ids"],
            attention_mask=tok.get("attention_mask"),
            query_positions=question_positions,
            key_positions=caption_positions,
        )
        question_mass = _attention_mass_per_layer(
            model=model,
            input_ids=tok["input_ids"],
            attention_mask=tok.get("attention_mask"),
            query_positions=question_positions,
            key_positions=question_positions,
        )
        other_mass = _attention_mass_per_layer(
            model=model,
            input_ids=tok["input_ids"],
            attention_mask=tok.get("attention_mask"),
            query_positions=question_positions,
            key_positions=other_positions,
        )

        def _mean(xs: list[float]) -> float | None:
            return float(sum(xs) / len(xs)) if xs else None

        attention_summary = {
            "caption_mass_per_layer": caption_mass,
            "question_mass_per_layer": question_mass,
            "other_mass_per_layer": other_mass,
            "caption_mass_mean": _mean(caption_mass),
            "question_mass_mean": _mean(question_mass),
            "other_mass_mean": _mean(other_mass),
        }

        phrase_info = None
        if rel_phrase:
            phrase_positions = _locate_rel_phrase_token_positions(
                tokenizer, full_ids, question_text=question, rel_phrase=rel_phrase
            )
            if phrase_positions:
                stats = compute_phrase_to_caption_attention(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=tok["input_ids"],
                    attention_mask=tok.get("attention_mask"),
                    phrase_positions=phrase_positions,
                    caption_positions=caption_positions,
                )
                caption_tok_labels = _token_labels_for_positions(tokenizer, full_ids_list, caption_positions)
                phrase_tok_labels = _token_labels_for_positions(tokenizer, full_ids_list, phrase_positions)
                phrase_info = {
                    "rel_phrase": rel_phrase,
                    "caption_tokens": caption_tok_labels,
                    "phrase_tokens": phrase_tok_labels,
                    "scores_per_layer": stats["per_layer"].tolist(),
                    "scores_mean": stats["mean"].tolist(),
                }

                if args.save_phrase_plots:
                    plot_phrase_to_caption(
                        out_dir=args.out_dir,
                        caption_tokens=caption_tok_labels,
                        phrase_tokens=phrase_tok_labels,
                        scores_per_layer=stats["per_layer"],
                        scores_mean=stats["mean"],
                        title_prefix=f"Phrase to Caption | '{rel_phrase}'",
                    )

        return {
            "caption": caption,
            "question": question,
            "prediction": pred,
            "p_yes": p_yes,
            "p_no": p_no,
            "gt": gt_clean,
            "is_correct": is_correct,
            "attention_summary": attention_summary,
            "phrase_analysis": phrase_info,
        }

    if args.level:
        ann_paths = _find_b_ann_paths(args.base_data_path, args.level)
        if args.max_files is not None:
            ann_paths = ann_paths[: args.max_files]
        if not ann_paths:
            raise SystemExit(f"No *_b.json files found in: {os.path.join(args.base_data_path, args.level)}")

        level_out_dir = os.path.join(args.out_dir, args.level)
        os.makedirs(level_out_dir, exist_ok=True)
        summary_path = os.path.join(level_out_dir, "summary.jsonl")
        metrics_path = os.path.join(level_out_dir, "metrics.json")

        num_total = 0
        num_correct = 0
        num_no_caption = 0
        num_no_rel_phrase = 0

        with open(summary_path, "w", encoding="utf-8") as f:
            plot_rows = []
            for ann_path in ann_paths:
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

                    try:
                        record = _run_single(
                            caption=caption,
                            question=question,
                            gt=answer,
                            rel_phrase=rel_phrase or None,
                        )
                    except Exception:
                        continue

                    record.update(
                        {
                            "ann_path": ann_path,
                            "qa_index": qi,
                            "rel_phrase": rel_phrase or None,
                        }
                    )

                    if rel_phrase and record.get("phrase_analysis") is None:
                        num_no_rel_phrase += 1

                    if record.get("is_correct") is not None:
                        num_total += 1
                        num_correct += int(record["is_correct"])
                        attn = record.get("attention_summary") or {}
                        plot_rows.append(
                            {
                                "is_correct": bool(record["is_correct"]),
                                "caption_mass_mean": attn.get("caption_mass_mean"),
                                "question_mass_mean": attn.get("question_mass_mean"),
                                "other_mass_mean": attn.get("other_mass_mean"),
                            }
                        )

                    if plot_limit != 0 and (plot_limit < 0 or per_file_plots < plot_limit):
                        if record.get("phrase_analysis") and record.get("phrase_analysis").get("caption_tokens"):
                            ex_dir = os.path.join(
                                level_out_dir,
                                os.path.splitext(os.path.basename(ann_path))[0],
                                f"qa_{qi:03d}",
                            )
                            plot_phrase_to_caption(
                                out_dir=ex_dir,
                                caption_tokens=record["phrase_analysis"]["caption_tokens"],
                                phrase_tokens=record["phrase_analysis"]["phrase_tokens"],
                                scores_per_layer=torch.tensor(
                                    record["phrase_analysis"]["scores_per_layer"]
                                ).numpy(),
                                scores_mean=torch.tensor(record["phrase_analysis"]["scores_mean"]).numpy(),
                                title_prefix=f"Phrase to Caption | '{rel_phrase}'",
                            )
                            per_file_plots += 1

                    f.write(json.dumps(record) + "\n")

        metrics = {
            "num_total": num_total,
            "num_correct": num_correct,
            "accuracy": (num_correct / num_total) if num_total > 0 else None,
            "num_no_caption": num_no_caption,
            "num_no_rel_phrase": num_no_rel_phrase,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        if args.summary_plots and plot_rows:
            _save_summary_plots(plot_rows, level_out_dir)

        print(f"Wrote summary to: {summary_path}")
        print(f"Wrote metrics to: {metrics_path}")
        return

    caption = (args.caption or "").strip()
    question = (args.question or "").strip()
    if not caption or not question:
        raise SystemExit("Provide --caption/--question or --level for batch mode.")

    rel_phrase = args.rel_phrase or _infer_rel_phrase(question)
    summary = _run_single(caption=caption, question=question, gt=args.gt, rel_phrase=rel_phrase)

    out_path = os.path.join(args.out_dir, "interactive_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Prediction: {summary['prediction']} (p_yes={summary['p_yes']}, p_no={summary['p_no']})")
    if summary["gt"] in ("yes", "no"):
        print(f"Correct: {bool(summary['is_correct'])} (gt={summary['gt']})")
    print("Attention mass from question to:")
    print(f"  caption: {summary['attention_summary']['caption_mass_mean']}")
    print(f"  question: {summary['attention_summary']['question_mass_mean']}")
    print(f"  other: {summary['attention_summary']['other_mass_mean']}")
    print(f"Saved summary to: {out_path}")


def _save_summary_plots(rows: list[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def _values(key: str, correct: bool) -> list[float]:
        vals = []
        for r in rows:
            if r.get("is_correct") != correct:
                continue
            v = r.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return vals

    def _boxplot(key: str, title: str, fname: str) -> None:
        correct_vals = _values(key, True)
        wrong_vals = _values(key, False)
        if not correct_vals and not wrong_vals:
            return
        plt.figure(figsize=(6, 4))
        plt.boxplot([correct_vals, wrong_vals], labels=["correct", "wrong"])
        plt.ylabel("Mean attention mass")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close()

    def _scatter(x_key: str, y_key: str, title: str, fname: str) -> None:
        xs = []
        ys = []
        cs = []
        for r in rows:
            x = r.get(x_key)
            y = r.get(y_key)
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                continue
            xs.append(float(x))
            ys.append(float(y))
            cs.append("tab:blue" if r.get("is_correct") else "tab:red")
        if not xs:
            return
        plt.figure(figsize=(5, 5))
        plt.scatter(xs, ys, c=cs, alpha=0.6, s=16)
        plt.xlabel(x_key)
        plt.ylabel(y_key)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close()

    _boxplot("caption_mass_mean", "Caption attention vs correctness", "attn_caption_box.png")
    _boxplot("question_mass_mean", "Question self-attention vs correctness", "attn_question_box.png")
    _boxplot("other_mass_mean", "Other-token attention vs correctness", "attn_other_box.png")

    _scatter(
        "caption_mass_mean",
        "question_mass_mean",
        "Caption vs Question attention (colored by correctness)",
        "attn_caption_vs_question.png",
    )


if __name__ == "__main__":
    main()
