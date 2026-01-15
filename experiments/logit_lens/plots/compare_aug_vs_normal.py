#!/usr/bin/env python3
import argparse
import csv
import json
import os

import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RESULTS_ROOT = os.path.join(_REPO_ROOT, "results_llava-hf", "llava-1.5-7b-hf")


def _find_level_dirs(base_dir: str) -> list[str]:
    if not os.path.isdir(base_dir):
        return []
    return sorted([d for d in os.listdir(base_dir) if d.startswith("level_")])


def _load_records(level_dir: str) -> list[dict]:
    path = os.path.join(level_dir, "summary.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _confusion_counts(records: list[dict]) -> dict:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0, "correct": 0}
    for r in records:
        gt = (r.get("answer") or r.get("gt") or "").strip().lower()
        pred = (r.get("prediction") or "").strip().lower()
        if gt not in ("yes", "no") or pred not in ("yes", "no"):
            continue
        counts["total"] += 1
        if gt == pred:
            counts["correct"] += 1
        if pred == "yes" and gt == "yes":
            counts["tp"] += 1
        elif pred == "no" and gt == "no":
            counts["tn"] += 1
        elif pred == "yes" and gt == "no":
            counts["fp"] += 1
        elif pred == "no" and gt == "yes":
            counts["fn"] += 1
    return counts


def _yes_rate(records: list[dict]) -> float:
    total = 0
    yes = 0
    for r in records:
        pred = (r.get("prediction") or "").strip().lower()
        if pred not in ("yes", "no"):
            continue
        total += 1
        if pred == "yes":
            yes += 1
    return (yes / total) if total > 0 else 0.0


def _mean_confidence(records: list[dict]) -> float:
    vals = []
    for r in records:
        p_yes = r.get("p_yes")
        p_no = r.get("p_no")
        if isinstance(p_yes, (int, float)) and isinstance(p_no, (int, float)):
            vals.append(max(float(p_yes), float(p_no)))
    return (sum(vals) / len(vals)) if vals else 0.0


def _summarize_level(records: list[dict]) -> dict:
    conf = _confusion_counts(records)
    total = conf["total"]
    acc = (conf["correct"] / total) if total > 0 else 0.0
    return {
        "num_total": total,
        "num_correct": conf["correct"],
        "accuracy": acc,
        "yes_rate": _yes_rate(records),
        "mean_confidence": _mean_confidence(records),
        "confusion": conf,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare augmented vs normal visual logit-lens accuracy by level.")
    ap.add_argument(
        "--normal_dir",
        type=str,
        default=os.path.join(_RESULTS_ROOT, "visual_logit_lens"),
        help="Directory containing normal level_*/summary.jsonl files.",
    )
    ap.add_argument(
        "--aug_dir",
        type=str,
        default=os.path.join(_RESULTS_ROOT, "aug_logit_lens"),
        help="Directory containing augmented level_*/summary.jsonl files.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_RESULTS_ROOT, "analysis_aug_vs_normal"),
    )
    return ap.parse_args()


def _plot_accuracy(levels: list[str], normal_acc: list[float], aug_acc: list[float], out_path: str) -> None:
    x = list(range(len(levels)))
    width = 0.38
    plt.figure(figsize=(max(6, 0.7 * len(levels)), 4))
    plt.bar([i - width / 2 for i in x], normal_acc, width=width, label="normal")
    plt.bar([i + width / 2 for i in x], aug_acc, width=width, label="augmented")
    plt.xticks(x, levels)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by level (normal vs augmented)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_delta(levels: list[str], deltas: list[float], out_path: str) -> None:
    x = list(range(len(levels)))
    plt.figure(figsize=(max(6, 0.7 * len(levels)), 4))
    plt.bar(x, deltas)
    plt.axhline(0.0, color="gray", linewidth=1)
    plt.xticks(x, levels)
    plt.ylabel("Accuracy (aug - normal)")
    plt.title("Accuracy delta by level")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    levels = sorted(set(_find_level_dirs(args.normal_dir)) & set(_find_level_dirs(args.aug_dir)))
    if not levels:
        raise SystemExit("No overlapping level_* directories between normal and augmented results.")

    rows = []
    summary = {"levels": levels, "normal": {}, "aug": {}, "delta_accuracy": {}}
    normal_acc = []
    aug_acc = []
    deltas = []

    for lvl in levels:
        normal_records = _load_records(os.path.join(args.normal_dir, lvl))
        aug_records = _load_records(os.path.join(args.aug_dir, lvl))

        normal_stats = _summarize_level(normal_records)
        aug_stats = _summarize_level(aug_records)

        summary["normal"][lvl] = normal_stats
        summary["aug"][lvl] = aug_stats
        summary["delta_accuracy"][lvl] = aug_stats["accuracy"] - normal_stats["accuracy"]
        normal_acc.append(normal_stats["accuracy"])
        aug_acc.append(aug_stats["accuracy"])
        deltas.append(summary["delta_accuracy"][lvl])

        rows.append(
            {
                "level": lvl,
                "normal_accuracy": normal_stats["accuracy"],
                "aug_accuracy": aug_stats["accuracy"],
                "delta_accuracy": summary["delta_accuracy"][lvl],
                "normal_yes_rate": normal_stats["yes_rate"],
                "aug_yes_rate": aug_stats["yes_rate"],
                "normal_mean_confidence": normal_stats["mean_confidence"],
                "aug_mean_confidence": aug_stats["mean_confidence"],
                "normal_total": normal_stats["num_total"],
                "aug_total": aug_stats["num_total"],
            }
        )

    with open(os.path.join(args.out_dir, "comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(args.out_dir, "comparison_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _plot_accuracy(
        levels,
        normal_acc,
        aug_acc,
        os.path.join(args.out_dir, "accuracy_by_level_aug_vs_normal.png"),
    )
    _plot_delta(
        levels,
        deltas,
        os.path.join(args.out_dir, "accuracy_delta_by_level.png"),
    )

    print(f"Wrote summary to: {os.path.join(args.out_dir, 'comparison_summary.json')}")
    print(f"Wrote CSV to: {csv_path}")
    print(f"Wrote plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
