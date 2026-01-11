import argparse
import json
import os

import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_NEW_RESULTS_ROOT = os.path.join(_REPO_ROOT, "results_llava-hf", "llava-1.5-7b-hf")
_LOGIT_LENS_OUT_DIR = os.path.join(_NEW_RESULTS_ROOT, "logit_lens")


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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate logit-lens curves (correct vs wrong).")
    ap.add_argument(
        "--base_dir",
        type=str,
        default=os.path.join(_NEW_RESULTS_ROOT, "text_only_objective"),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=_LOGIT_LENS_OUT_DIR,
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    levels = _find_level_dirs(args.base_dir)
    if not levels:
        raise SystemExit(f"No level_* directories found in: {args.base_dir}")

    all_correct = []
    all_wrong = []
    all_correct_no = []
    all_wrong_no = []
    all_layers_counts = {}
    outcome_p_yes = {"TP": [], "TN": [], "FP": [], "FN": []}
    outcome_p_no = {"TP": [], "TN": [], "FP": [], "FN": []}

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
            p_yes = [float(d.get("p_yes", 0.0)) for d in lens]
            p_no = [float(d.get("p_no", 0.0)) for d in lens]
            pred_by_layer = ["yes" if y >= n else "no" for y, n in zip(p_yes, p_no)]
            if r["is_correct"]:
                lvl_correct_yes.append(p_yes)
                lvl_correct_no.append(p_no)
                all_correct.append(p_yes)
                all_correct_no.append(p_no)
            else:
                lvl_wrong_yes.append(p_yes)
                lvl_wrong_no.append(p_no)
                all_wrong.append(p_yes)
                all_wrong_no.append(p_no)

            if pred_final == "yes" and gt == "yes":
                outcome = "TP"
            elif pred_final == "no" and gt == "no":
                outcome = "TN"
            elif pred_final == "yes" and gt == "no":
                outcome = "FP"
            else:
                outcome = "FN"

            outcome_p_yes[outcome].append(p_yes)
            outcome_p_no[outcome].append(p_no)

            for li, pred in enumerate(pred_by_layer):
                counts = all_layers_counts.setdefault(li, {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
                if pred == "yes" and gt == "yes":
                    counts["tp"] += 1
                elif pred == "no" and gt == "no":
                    counts["tn"] += 1
                elif pred == "yes" and gt == "no":
                    counts["fp"] += 1
                elif pred == "no" and gt == "yes":
                    counts["fn"] += 1

        corr_curve_yes = _mean_curves(lvl_correct_yes)
        wrong_curve_yes = _mean_curves(lvl_wrong_yes)
        corr_curve_no = _mean_curves(lvl_correct_no)
        wrong_curve_no = _mean_curves(lvl_wrong_no)
        _plot_curves(
            corr_curve_yes,
            wrong_curve_yes,
            os.path.join(args.out_dir, f"logit_lens_{lvl}.png"),
            f"Logit lens p(yes) by layer ({lvl})",
            "P(yes)",
        )
        _plot_curves(
            corr_curve_no,
            wrong_curve_no,
            os.path.join(args.out_dir, f"logit_lens_pno_{lvl}.png"),
            f"Logit lens p(no) by layer ({lvl})",
            "P(no)",
        )

    _plot_curves(
        _mean_curves(all_correct),
        _mean_curves(all_wrong),
        os.path.join(args.out_dir, "logit_lens_overall.png"),
        "Logit lens p(yes) by layer (overall)",
        "P(yes)",
    )
    _plot_curves(
        _mean_curves(all_correct_no),
        _mean_curves(all_wrong_no),
        os.path.join(args.out_dir, "logit_lens_pno_overall.png"),
        "Logit lens p(no) by layer (overall)",
        "P(no)",
    )

    p_yes_curves = {k: _mean_by_layer(v) for k, v in outcome_p_yes.items()}
    p_no_curves = {k: _mean_by_layer(v) for k, v in outcome_p_no.items()}
    _plot_four_curves(
        p_yes_curves,
        os.path.join(args.out_dir, "logit_lens_pyes_by_outcome.png"),
        "Logit lens p(yes) by outcome",
        "P(yes)",
    )
    _plot_four_curves(
        p_no_curves,
        os.path.join(args.out_dir, "logit_lens_pno_by_outcome.png"),
        "Logit lens p(no) by outcome",
        "P(no)",
    )

    if all_layers_counts:
        max_layer = max(all_layers_counts.keys())
        series = {"TP": [], "TN": [], "FP": [], "FN": []}
        for li in range(max_layer + 1):
            c = all_layers_counts.get(li)
            if not c:
                continue
            mat = [
                [c["tp"], c["fn"]],
                [c["fp"], c["tn"]],
            ]
            plt.figure(figsize=(4, 4))
            plt.imshow(mat, interpolation="nearest")
            plt.colorbar(label="Count")
            plt.xticks([0, 1], ["pred_yes", "pred_no"])
            plt.yticks([0, 1], ["gt_yes", "gt_no"])
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, str(mat[i][j]), ha="center", va="center")
            plt.title(f"Logit-lens confusion (layer {li})")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"logit_lens_confusion_layer_{li:02d}.png"), dpi=160)
            plt.close()

            series["TP"].append(c["tp"])
            series["TN"].append(c["tn"])
            series["FP"].append(c["fp"])
            series["FN"].append(c["fn"])

        _plot_four_curves(
            series,
            os.path.join(args.out_dir, "logit_lens_confusion_counts_by_layer.png"),
            "Logit-lens confusion counts by layer",
            "Count",
        )

        series_norm = {}
        for i in range(len(series["TP"])):
            total = series["TP"][i] + series["TN"][i] + series["FP"][i] + series["FN"][i]
            if total <= 0:
                series_norm.setdefault("TP", []).append(0.0)
                series_norm.setdefault("TN", []).append(0.0)
                series_norm.setdefault("FP", []).append(0.0)
                series_norm.setdefault("FN", []).append(0.0)
            else:
                series_norm.setdefault("TP", []).append(series["TP"][i] / total)
                series_norm.setdefault("TN", []).append(series["TN"][i] / total)
                series_norm.setdefault("FP", []).append(series["FP"][i] / total)
                series_norm.setdefault("FN", []).append(series["FN"][i] / total)

        _plot_four_curves(
            series_norm,
            os.path.join(args.out_dir, "logit_lens_confusion_rates_by_layer.png"),
            "Logit-lens confusion rates by layer",
            "Rate",
        )

    print(f"Saved logit-lens plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
