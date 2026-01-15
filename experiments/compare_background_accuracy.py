import argparse
import json
import os

import matplotlib.pyplot as plt


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


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


def _normalize_ann_path(path: str) -> str:
    if not path:
        return ""
    if os.path.exists(path):
        return path
    marker = "DeepLearningProject/"
    idx = path.find(marker)
    if idx >= 0:
        local = os.path.join(_REPO_ROOT, path[idx + len(marker) :])
        if os.path.exists(local):
            return local
    return path


def _load_background(ann_path: str, cache: dict[str, str]) -> str:
    if not ann_path:
        return ""
    ann_path = _normalize_ann_path(ann_path)
    if ann_path in cache:
        return cache[ann_path]
    if not os.path.exists(ann_path):
        cache[ann_path] = ""
        return ""
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        bg = (data.get("background") or "").strip().lower()
    except Exception:
        bg = ""
    cache[ann_path] = bg
    return bg


def _bg_label(bg: str) -> str:
    if bg == "b":
        return "black"
    if bg == "w":
        return "white"
    return "unknown"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare accuracy by background color for each level."
    )
    ap.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(
            _REPO_ROOT, "results_llava-hf", "llava-1.5-7b-hf", "visual_logit_lens"
        ),
        help="Path containing level_* directories with summary.jsonl files.",
    )
    ap.add_argument(
        "--levels",
        nargs="*",
        default=None,
        help="Optional list of levels to include (e.g., level_2).",
    )
    ap.add_argument(
        "--aggregate",
        choices=["image", "question"],
        default="image",
        help="Aggregate accuracy per image or per question.",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Save a bar plot comparing background accuracy by level.",
    )
    ap.add_argument(
        "--plot_path",
        type=str,
        default=None,
        help="Optional output path for the plot PNG.",
    )
    return ap.parse_args()


def _acc_fmt(n_correct: int, n_total: int) -> str:
    if n_total <= 0:
        return "n/a"
    return f"{(n_correct / n_total) * 100:.2f}%"


def main() -> None:
    args = parse_args()
    levels = _find_level_dirs(args.results_dir)
    if args.levels:
        levels = [lvl for lvl in levels if lvl in args.levels]
    if not levels:
        raise SystemExit(f"No level_* directories found in: {args.results_dir}")

    ann_cache: dict[str, str] = {}
    rows = []

    for lvl in levels:
        summary_path = os.path.join(args.results_dir, lvl, "summary.jsonl")
        records = _load_records(summary_path)
        if not records:
            continue

        if args.aggregate == "question":
            stats = {"b": [0, 0], "w": [0, 0], "": [0, 0]}
            for r in records:
                if not isinstance(r.get("is_correct"), bool):
                    continue
                bg = _load_background(r.get("ann_path", ""), ann_cache)
                key = bg if bg in ("b", "w") else ""
                stats[key][1] += 1
                if r["is_correct"]:
                    stats[key][0] += 1
        else:
            per_image: dict[str, dict[str, int]] = {}
            bg_by_image: dict[str, str] = {}
            for r in records:
                if not isinstance(r.get("is_correct"), bool):
                    continue
                image_id = str(r.get("image_id") or "")
                if not image_id:
                    continue
                bg = _load_background(r.get("ann_path", ""), ann_cache)
                if image_id not in per_image:
                    per_image[image_id] = {"correct": 0, "total": 0}
                    bg_by_image[image_id] = bg if bg in ("b", "w") else ""
                per_image[image_id]["total"] += 1
                if r["is_correct"]:
                    per_image[image_id]["correct"] += 1

            stats = {"b": [0.0, 0], "w": [0.0, 0], "": [0.0, 0]}
            for image_id, counts in per_image.items():
                bg = bg_by_image.get(image_id, "")
                key = bg if bg in ("b", "w") else ""
                stats[key][1] += 1
                if counts["total"] > 0:
                    stats[key][0] += counts["correct"] / counts["total"]

        for bg_key in ("b", "w"):
            correct, total = stats[bg_key]
            if args.aggregate == "question":
                accuracy = _acc_fmt(int(correct), int(total))
            else:
                accuracy = _acc_fmt(correct, total)
            acc_value = (correct / total) if total else None
            rows.append(
                {
                    "level": lvl,
                    "background": _bg_label(bg_key),
                    "correct": correct,
                    "total": total,
                    "accuracy": accuracy,
                    "accuracy_value": acc_value,
                }
            )

    if not rows:
        raise SystemExit("No records found to compute accuracy.")

    label = "per-image" if args.aggregate == "image" else "per-question"
    print(f"Background accuracy ({label}):")
    if args.aggregate == "question":
        print("level\tbackground\tcorrect\ttotal\taccuracy")
        for r in rows:
            print(
                f"{r['level']}\t{r['background']}\t{int(r['correct'])}\t{int(r['total'])}\t{r['accuracy']}"
            )
    else:
        print("level\tbackground\tn_images\tmean_image_acc")
        for r in rows:
            print(
                f"{r['level']}\t{r['background']}\t{int(r['total'])}\t{r['accuracy']}"
            )

    if args.plot:
        levels_sorted = sorted({r["level"] for r in rows})
        black = []
        white = []
        for lvl in levels_sorted:
            black_row = next((r for r in rows if r["level"] == lvl and r["background"] == "black"), None)
            white_row = next((r for r in rows if r["level"] == lvl and r["background"] == "white"), None)
            black.append(black_row["accuracy_value"] if black_row else None)
            white.append(white_row["accuracy_value"] if white_row else None)

        x = list(range(len(levels_sorted)))
        width = 0.36
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([i - width / 2 for i in x], black, width, label="black", color="black", alpha=0.7)
        ax.bar([i + width / 2 for i in x], white, width, label="white", color="lightgray", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(levels_sorted, rotation=0)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Level")
        label = "per-image" if args.aggregate == "image" else "per-question"
        ax.set_title(f"Background accuracy by level ({label})")
        ax.legend()
        fig.tight_layout()
        if not args.plot_path:
            plot_name = f"background_accuracy_{args.aggregate}.png"
            args.plot_path = os.path.join(args.results_dir, plot_name)
        fig.savefig(args.plot_path, dpi=160)
        plt.close(fig)
        print(f"Saved plot to: {args.plot_path}")


if __name__ == "__main__":
    main()
