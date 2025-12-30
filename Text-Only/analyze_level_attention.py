import argparse
import json
import os

import matplotlib.pyplot as plt


ATTN_KEYS = [
    "caption_mass_mean",
    "question_mass_mean",
    "other_mass_mean",
]


def _find_level_dirs(base_dir: str) -> list[str]:
    if not os.path.isdir(base_dir):
        return []
    entries = [d for d in os.listdir(base_dir) if d.startswith("level_")]
    return sorted(entries)


def _load_records(level_dir: str) -> list[dict]:
    path = os.path.join(level_dir, "summary.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def _as_float(v):
    return float(v) if isinstance(v, (int, float)) else None


def _aggregate_level(records: list[dict]) -> dict:
    agg = {
        "total": 0,
        "correct": 0,
        "wrong": 0,
        "means": {k: {"correct": None, "wrong": None} for k in ATTN_KEYS},
        "counts": {k: {"correct": 0, "wrong": 0} for k in ATTN_KEYS},
        "values": {k: {"correct": [], "wrong": []} for k in ATTN_KEYS},
    }

    for r in records:
        is_correct = r.get("is_correct")
        if not isinstance(is_correct, bool):
            continue

        agg["total"] += 1
        if is_correct:
            agg["correct"] += 1
        else:
            agg["wrong"] += 1

        attn = r.get("attention_summary") or {}
        for k in ATTN_KEYS:
            v = _as_float(attn.get(k))
            if v is None:
                continue
            label = "correct" if is_correct else "wrong"
            agg["values"][k][label].append(v)
            agg["counts"][k][label] += 1

    for k in ATTN_KEYS:
        for label in ("correct", "wrong"):
            vals = agg["values"][k][label]
            if vals:
                agg["means"][k][label] = sum(vals) / len(vals)

    return agg


def _plot_accuracy(levels: list[str], aggs: list[dict], out_dir: str) -> None:
    accs = []
    for agg in aggs:
        if agg["total"] > 0:
            accs.append(agg["correct"] / agg["total"])
        else:
            accs.append(0.0)
    plt.figure(figsize=(max(6, 0.7 * len(levels)), 4))
    plt.bar(levels, accs)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Level")
    for i, a in enumerate(accs):
        plt.text(i, min(0.98, a + 0.02), f"{a:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_by_level.png"), dpi=160)
    plt.close()


def _plot_attn_means(levels: list[str], aggs: list[dict], out_dir: str, key: str, title: str) -> None:
    correct_means = [agg["means"][key]["correct"] or 0.0 for agg in aggs]
    wrong_means = [agg["means"][key]["wrong"] or 0.0 for agg in aggs]

    x = list(range(len(levels)))
    width = 0.38

    plt.figure(figsize=(max(6, 0.7 * len(levels)), 4))
    plt.bar([i - width / 2 for i in x], correct_means, width=width, label="correct")
    plt.bar([i + width / 2 for i in x], wrong_means, width=width, label="wrong")
    plt.xticks(x, levels)
    plt.ylabel("Mean attention mass")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{key}_by_level.png"), dpi=160)
    plt.close()


def _plot_attn_gaps(levels: list[str], aggs: list[dict], out_dir: str) -> None:
    plt.figure(figsize=(max(6, 0.7 * len(levels)), 4))
    x = list(range(len(levels)))
    for key, label in [
        ("caption_mass_mean", "caption"),
        ("question_mass_mean", "question"),
        ("other_mass_mean", "other"),
    ]:
        gaps = []
        for agg in aggs:
            c = agg["means"][key]["correct"]
            w = agg["means"][key]["wrong"]
            if c is None or w is None:
                gaps.append(0.0)
            else:
                gaps.append(c - w)
        plt.plot(x, gaps, marker="o", label=label)
    plt.axhline(0.0, color="gray", linewidth=1)
    plt.xticks(x, levels)
    plt.ylabel("Correct - Wrong attention mean")
    plt.title("Attention gap by level")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "attn_gaps_by_level.png"), dpi=160)
    plt.close()


def _plot_accuracy_vs_attention(levels: list[str], aggs: list[dict], out_dir: str) -> None:
    accs = []
    cap = []
    qst = []
    for agg in aggs:
        if agg["total"] > 0:
            accs.append(agg["correct"] / agg["total"])
        else:
            accs.append(0.0)
        cap.append(agg["means"]["caption_mass_mean"]["correct"] or 0.0)
        qst.append(agg["means"]["question_mass_mean"]["correct"] or 0.0)

    plt.figure(figsize=(5, 5))
    plt.scatter(cap, accs, s=30)
    for i, lvl in enumerate(levels):
        plt.text(cap[i], accs[i], lvl, fontsize=7, ha="left", va="bottom")
    plt.xlabel("Caption attention mean (correct)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Caption Attention")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_vs_caption_attention.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(qst, accs, s=30)
    for i, lvl in enumerate(levels):
        plt.text(qst[i], accs[i], lvl, fontsize=7, ha="left", va="bottom")
    plt.xlabel("Question self-attention mean (correct)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Question Self-Attention")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_vs_question_attention.png"), dpi=160)
    plt.close()


def _plot_attention_shares(levels: list[str], aggs: list[dict], out_dir: str) -> None:
    shares = {"caption": [], "question": [], "other": []}
    for agg in aggs:
        c = agg["means"]["caption_mass_mean"]["correct"] or 0.0
        q = agg["means"]["question_mass_mean"]["correct"] or 0.0
        o = agg["means"]["other_mass_mean"]["correct"] or 0.0
        total = c + q + o
        if total <= 0:
            shares["caption"].append(0.0)
            shares["question"].append(0.0)
            shares["other"].append(0.0)
        else:
            shares["caption"].append(c / total)
            shares["question"].append(q / total)
            shares["other"].append(o / total)

    x = list(range(len(levels)))
    plt.figure(figsize=(max(6, 0.7 * len(levels)), 4))
    plt.bar(x, shares["caption"], label="caption")
    plt.bar(x, shares["question"], bottom=shares["caption"], label="question")
    bottom = [shares["caption"][i] + shares["question"][i] for i in range(len(levels))]
    plt.bar(x, shares["other"], bottom=bottom, label="other")
    plt.xticks(x, levels)
    plt.ylim(0, 1)
    plt.ylabel("Share of attention mass (correct)")
    plt.title("Attention shares for correct answers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "attention_shares_correct.png"), dpi=160)
    plt.close()


def _plot_heatmap(levels: list[str], aggs: list[dict], out_dir: str) -> None:
    cols = []
    for key in ATTN_KEYS:
        cols.append(f"{key}|correct")
        cols.append(f"{key}|wrong")
    data = []
    for agg in aggs:
        row = []
        for key in ATTN_KEYS:
            row.append(agg["means"][key]["correct"] or 0.0)
            row.append(agg["means"][key]["wrong"] or 0.0)
        data.append(row)

    plt.figure(figsize=(max(6, 0.7 * len(cols)), max(4, 0.5 * len(levels))))
    plt.imshow(data, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Mean attention mass")
    plt.yticks(range(len(levels)), levels)
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.title("Attention means by level and correctness")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "attention_mean_heatmap.png"), dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze attention/correctness across levels.")
    ap.add_argument(
        "--base_dir",
        type=str,
        default=os.path.join(os.path.abspath(os.path.dirname(__file__)), "vis_results_caption", "interactive"),
        help="Directory containing level_* outputs with summary.jsonl.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.abspath(os.path.dirname(__file__)), "vis_results_caption", "analysis"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    levels = _find_level_dirs(args.base_dir)
    if not levels:
        raise SystemExit(f"No level_* directories found in: {args.base_dir}")

    aggs = []
    for lvl in levels:
        records = _load_records(os.path.join(args.base_dir, lvl))
        aggs.append(_aggregate_level(records))

    summary = {lvl: agg for lvl, agg in zip(levels, aggs)}
    with open(os.path.join(args.out_dir, "level_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_accuracy(levels, aggs, args.out_dir)
    _plot_attn_means(levels, aggs, args.out_dir, "caption_mass_mean", "Caption attention vs correctness")
    _plot_attn_means(levels, aggs, args.out_dir, "question_mass_mean", "Question self-attention vs correctness")
    _plot_attn_means(levels, aggs, args.out_dir, "other_mass_mean", "Other-token attention vs correctness")
    _plot_attn_gaps(levels, aggs, args.out_dir)
    _plot_accuracy_vs_attention(levels, aggs, args.out_dir)
    _plot_attention_shares(levels, aggs, args.out_dir)
    _plot_heatmap(levels, aggs, args.out_dir)

    print(f"Wrote summary to: {os.path.join(args.out_dir, 'level_summary.json')}")
    print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
