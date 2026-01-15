import argparse
import json
import os

import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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
    outcome_logit_yes = {"TP": [], "TN": [], "FP": [], "FN": []}
    outcome_logit_no = {"TP": [], "TN": [], "FP": [], "FN": []}

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
            if not logit_yes or not logit_no:
                continue

            if r["is_correct"]:
                lvl_correct_yes.append(logit_yes)
                lvl_correct_no.append(logit_no)
                all_correct_yes.append(logit_yes)
                all_correct_no.append(logit_no)
            else:
                lvl_wrong_yes.append(logit_yes)
                lvl_wrong_no.append(logit_no)
                all_wrong_yes.append(logit_yes)
                all_wrong_no.append(logit_no)

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

    print(f"Saved raw logit-lens plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
