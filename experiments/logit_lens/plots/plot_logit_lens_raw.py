import argparse
import json
import math
import os

import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_NEW_RESULTS_ROOT = os.path.join(_REPO_ROOT, "results_llava-hf", "llava-1.5-7b-hf")
_LOGIT_LENS_RAW_OUT_DIR = os.path.join(_NEW_RESULTS_ROOT, "logit_lens_raw")


def _find_level_dirs(base_dir: str) -> list[str]:
    if not os.path.isdir(base_dir):
        return []
    levels = [d for d in os.listdir(base_dir) if d.startswith("level_")]
    return sorted(levels)


def _load_records(summary_path: str) -> list[dict]:
    if not os.path.exists(summary_path):
        return []
    records = []
    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def _mean_curves(curves: list[list[float]]) -> list[float]:
    if not curves:
        return []
    min_len = min(len(c) for c in curves)
    if min_len == 0:
        return []
    out = []
    for i in range(min_len):
        out.append(sum(c[i] for c in curves) / len(curves))
    return out


def _plot_curves(
    correct: list[float],
    wrong: list[float],
    out_path: str,
    title: str,
    ylabel: str,
) -> None:
    if not correct and not wrong:
        return
    plt.figure(figsize=(6, 4))
    if correct:
        plt.plot(list(range(len(correct))), correct, label="correct")
    if wrong:
        plt.plot(list(range(len(wrong))), wrong, label="wrong")
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_four_curves(curves: dict, out_path: str, title: str, ylabel: str) -> None:
    if not curves:
        return
    plt.figure(figsize=(6, 4))
    for label, data in curves.items():
        if data:
            plt.plot(list(range(len(data))), data, label=label)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_multi_curves(curves: dict, out_path: str, title: str, ylabel: str) -> None:
    if not curves:
        return
    plt.figure(figsize=(6, 4))
    for label, data in curves.items():
        if data:
            plt.plot(list(range(len(data))), data, label=label)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _mean_by_layer(curves: list[list[float]]) -> list[float]:
    if not curves:
        return []
    min_len = min(len(c) for c in curves)
    if min_len == 0:
        return []
    out = []
    for i in range(min_len):
        out.append(sum(c[i] for c in curves) / len(curves))
    return out


def _dataset_label(base_dir: str) -> str:
    b = (base_dir or "").lower()
    if "visual_logit_lens" in b or "visual" in b:
        return "image"
    if "text_only_objective" in b or "text" in b:
        return "text"
    if "aug_logit_lens" in b or "aug":
        return "augmented"
    return "dataset"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Aggregate logit-lens raw logits (correct vs wrong)."
    )
    ap.add_argument(
        "--base_dir",
        type=str,
        default=os.path.join(_NEW_RESULTS_ROOT, "visual_logit_lens"),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=_LOGIT_LENS_RAW_OUT_DIR,
    )
    ap.add_argument(
        "--levels",
        nargs="*",
        default=None,
        help="Optional list of levels to include (e.g., level_2).",
    )
    return ap.parse_args()


def _extract_curve(lens: list[dict], key: str) -> list[float]:
    curve = []
    for d in lens:
        val = d.get(key)
        if val is None:
            return []
        curve.append(float(val))
    return curve


def _extract_full_softmax_curves(lens: list[dict]) -> tuple[list[float], list[float]]:
    yes = []
    no = []
    for d in lens:
        logit_yes = d.get("logit_yes")
        logit_no = d.get("logit_no")
        logit_total = d.get("logit_total")
        if logit_yes is None or logit_no is None or logit_total is None:
            return [], []
        yes.append(float(math.exp(float(logit_yes) - float(logit_total))))
        no.append(float(math.exp(float(logit_no) - float(logit_total))))
    return yes, no


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    levels = _find_level_dirs(args.base_dir)
    if args.levels:
        levels = [lvl for lvl in levels if lvl in args.levels]
    if not levels:
        raise SystemExit(f"No level_* directories found in: {args.base_dir}")

    dataset = _dataset_label(args.base_dir)
    if len(levels) == 1:
        scope_label = levels[0]
    else:
        scope_label = "all levels"
    title_suffix = f"{dataset}, {scope_label}"

    all_correct_yes = []
    all_wrong_yes = []
    all_correct_no = []
    all_wrong_no = []
    all_correct_yes_full = []
    all_wrong_yes_full = []
    all_correct_no_full = []
    all_wrong_no_full = []
    outcome_logit_yes = {"TP": [], "TN": [], "FP": [], "FN": []}
    outcome_logit_no = {"TP": [], "TN": [], "FP": [], "FN": []}
    outcome_yes_full = {"TP": [], "TN": [], "FP": [], "FN": []}
    outcome_no_full = {"TP": [], "TN": [], "FP": [], "FN": []}

    for lvl in levels:
        summary_path = os.path.join(args.base_dir, lvl, "summary.jsonl")
        records = _load_records(summary_path)
        lvl_correct_yes = []
        lvl_wrong_yes = []
        lvl_correct_no = []
        lvl_wrong_no = []
        for r in records:
            if not isinstance(r.get("is_correct"), bool):
                continue
            lens = r.get("logit_lens")
            if not isinstance(lens, list) or not lens:
                continue
            gt = (r.get("answer") or r.get("gt") or "").strip().lower()
            if gt not in ("yes", "no"):
                continue
            pred_final = (r.get("prediction") or "").strip().lower()
            if pred_final not in ("yes", "no"):
                continue

            logit_yes = _extract_curve(lens, "logit_yes")
            logit_no = _extract_curve(lens, "logit_no")
            yes_full, no_full = _extract_full_softmax_curves(lens)
            if not logit_yes or not logit_no or not yes_full or not no_full:
                continue

            if r["is_correct"]:
                lvl_correct_yes.append(logit_yes)
                lvl_correct_no.append(logit_no)
                all_correct_yes.append(logit_yes)
                all_correct_no.append(logit_no)
                all_correct_yes_full.append(yes_full)
                all_correct_no_full.append(no_full)
            else:
                lvl_wrong_yes.append(logit_yes)
                lvl_wrong_no.append(logit_no)
                all_wrong_yes.append(logit_yes)
                all_wrong_no.append(logit_no)
                all_wrong_yes_full.append(yes_full)
                all_wrong_no_full.append(no_full)

            if pred_final == "yes" and gt == "yes":
                outcome = "TP"
            elif pred_final == "no" and gt == "no":
                outcome = "TN"
            elif pred_final == "yes" and gt == "no":
                outcome = "FP"
            else:
                outcome = "FN"

            outcome_logit_yes[outcome].append(logit_yes)
            outcome_logit_no[outcome].append(logit_no)
            outcome_yes_full[outcome].append(yes_full)
            outcome_no_full[outcome].append(no_full)

        _plot_curves(
            _mean_curves(lvl_correct_yes),
            _mean_curves(lvl_wrong_yes),
            os.path.join(args.out_dir, f"logit_lens_raw_yes_{lvl}.png"),
            f"Logit lens logit(yes) by layer ({title_suffix}, {lvl})",
            "logit(yes)",
        )
        _plot_curves(
            _mean_curves(lvl_correct_no),
            _mean_curves(lvl_wrong_no),
            os.path.join(args.out_dir, f"logit_lens_raw_no_{lvl}.png"),
            f"Logit lens logit(no) by layer ({title_suffix}, {lvl})",
            "logit(no)",
        )

    _plot_curves(
        _mean_curves(all_correct_yes),
        _mean_curves(all_wrong_yes),
        os.path.join(args.out_dir, "logit_lens_raw_yes_overall.png"),
        f"Logit lens logit(yes) by layer ({title_suffix})",
        "logit(yes)",
    )
    _plot_curves(
        _mean_curves(all_correct_no),
        _mean_curves(all_wrong_no),
        os.path.join(args.out_dir, "logit_lens_raw_no_overall.png"),
        f"Logit lens logit(no) by layer ({title_suffix})",
        "logit(no)",
    )

    all_yes = all_correct_yes + all_wrong_yes
    all_no = all_correct_no + all_wrong_no
    _plot_multi_curves(
        {
            "logit_yes": _mean_curves(all_yes),
            "logit_no": _mean_curves(all_no),
        },
        os.path.join(args.out_dir, "logit_lens_raw_yes_no_overall.png"),
        f"Logit lens logit(yes/no) overall ({title_suffix})",
        "logit",
    )
    _plot_multi_curves(
        {
            "p_yes": _mean_curves(all_correct_yes_full + all_wrong_yes_full),
            "p_no": _mean_curves(all_correct_no_full + all_wrong_no_full),
        },
        os.path.join(args.out_dir, "logit_lens_softmax_yes_no_overall.png"),
        f"Logit lens softmax p(yes/no) overall ({title_suffix})",
        "P",
    )
    _plot_multi_curves(
        {
            "correct_yes": _mean_curves(all_correct_yes),
            "correct_no": _mean_curves(all_correct_no),
            "wrong_yes": _mean_curves(all_wrong_yes),
            "wrong_no": _mean_curves(all_wrong_no),
        },
        os.path.join(args.out_dir, "logit_lens_raw_yes_no_correct_wrong.png"),
        f"Logit lens logit(yes/no) by correctness ({title_suffix})",
        "logit",
    )
    _plot_multi_curves(
        {
            "correct_yes": _mean_curves(all_correct_yes_full),
            "correct_no": _mean_curves(all_correct_no_full),
            "wrong_yes": _mean_curves(all_wrong_yes_full),
            "wrong_no": _mean_curves(all_wrong_no_full),
        },
        os.path.join(args.out_dir, "logit_lens_softmax_yes_no_correct_wrong.png"),
        f"Logit lens softmax p(yes/no) by correctness ({title_suffix})",
        "P",
    )

    logit_yes_curves = {k: _mean_by_layer(v) for k, v in outcome_logit_yes.items()}
    logit_no_curves = {k: _mean_by_layer(v) for k, v in outcome_logit_no.items()}
    _plot_four_curves(
        logit_yes_curves,
        os.path.join(args.out_dir, "logit_lens_raw_yes_by_outcome.png"),
        f"Logit lens logit(yes) by outcome ({title_suffix})",
        "logit(yes)",
    )
    _plot_four_curves(
        logit_no_curves,
        os.path.join(args.out_dir, "logit_lens_raw_no_by_outcome.png"),
        f"Logit lens logit(no) by outcome ({title_suffix})",
        "logit(no)",
    )
    _plot_multi_curves(
        {
            "TP_yes": logit_yes_curves.get("TP", []),
            "TP_no": logit_no_curves.get("TP", []),
            "TN_yes": logit_yes_curves.get("TN", []),
            "TN_no": logit_no_curves.get("TN", []),
            "FP_yes": logit_yes_curves.get("FP", []),
            "FP_no": logit_no_curves.get("FP", []),
            "FN_yes": logit_yes_curves.get("FN", []),
            "FN_no": logit_no_curves.get("FN", []),
        },
        os.path.join(args.out_dir, "logit_lens_raw_yes_no_by_outcome.png"),
        f"Logit lens logit(yes/no) by outcome ({title_suffix})",
        "logit",
    )
    yes_full_curves = {k: _mean_by_layer(v) for k, v in outcome_yes_full.items()}
    no_full_curves = {k: _mean_by_layer(v) for k, v in outcome_no_full.items()}
    _plot_multi_curves(
        {
            "TP_yes": yes_full_curves.get("TP", []),
            "TP_no": no_full_curves.get("TP", []),
            "TN_yes": yes_full_curves.get("TN", []),
            "TN_no": no_full_curves.get("TN", []),
            "FP_yes": yes_full_curves.get("FP", []),
            "FP_no": no_full_curves.get("FP", []),
            "FN_yes": yes_full_curves.get("FN", []),
            "FN_no": no_full_curves.get("FN", []),
        },
        os.path.join(args.out_dir, "logit_lens_softmax_yes_no_by_outcome.png"),
        f"Logit lens softmax p(yes/no) by outcome ({title_suffix})",
        "P",
    )

    print(f"Saved raw logit-lens plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
