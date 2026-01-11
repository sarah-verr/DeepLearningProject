import argparse
import json
import os

import matplotlib.pyplot as plt

_TEXT_ONLY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_TEXT_ONLY_DIR, "Results")
_REPO_ROOT = os.path.dirname(_TEXT_ONLY_DIR)


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


def _infer_rel_phrase(question: str) -> str:
    q = f" {question.lower()} "
    for phrase in REL_PHRASE_CANDIDATES:
        if f" {phrase} " in q:
            return phrase
    return "unknown"


def _find_level_dirs(base_dir: str) -> list[str]:
    if not os.path.isdir(base_dir):
        return []
    return sorted([d for d in os.listdir(base_dir) if d.startswith("level_")])


def _load_text_records(level_dir: str) -> list[dict]:
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


def _load_image_records(level_dir: str) -> list[dict]:
    out = []
    for root, _, files in os.walk(level_dir):
        if "results.json" in files:
            path = os.path.join(root, "results.json")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for r in data.get("results", []):
                    out.append(r)
            except Exception:
                continue
    return out


def _confusion_counts(records: list[dict], gt_key: str, pred_key: str) -> dict:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0, "correct": 0}
    for r in records:
        gt = (r.get(gt_key) or "").strip().lower()
        pred = (r.get(pred_key) or "").strip().lower()
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


def _plot_accuracy_by_level(levels: list[str], text_acc: list[float], img_acc: list[float], out_dir: str) -> None:
    x = list(range(len(levels)))
    width = 0.38
    plt.figure(figsize=(max(6, 0.7 * len(levels)), 4))
    plt.bar([i - width / 2 for i in x], text_acc, width=width, label="text")
    plt.bar([i + width / 2 for i in x], img_acc, width=width, label="image")
    plt.xticks(x, levels)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by level (text vs image)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_by_level_text_vs_image.png"), dpi=160)
    plt.close()


def _plot_confusion_by_level(levels: list[str], confs: list[dict], title: str, out_path: str) -> None:
    x = list(range(len(levels)))
    width = 0.2
    tp = [c["tp"] for c in confs]
    tn = [c["tn"] for c in confs]
    fp = [c["fp"] for c in confs]
    fn = [c["fn"] for c in confs]
    plt.figure(figsize=(max(6, 0.8 * len(levels)), 5))
    plt.bar([i - 1.5 * width for i in x], tp, width=width, label="TP")
    plt.bar([i - 0.5 * width for i in x], tn, width=width, label="TN")
    plt.bar([i + 0.5 * width for i in x], fp, width=width, label="FP")
    plt.bar([i + 1.5 * width for i in x], fn, width=width, label="FN")
    plt.xticks(x, levels)
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_accuracy_by_type(levels: list[str], type_stats: list[dict], title: str, out_path: str) -> None:
    entity_acc = []
    relation_acc = []
    for stats in type_stats:
        ent = stats["entity"]
        rel = stats["relation"]
        entity_acc.append((ent["correct"] / ent["total"]) if ent["total"] > 0 else 0.0)
        relation_acc.append((rel["correct"] / rel["total"]) if rel["total"] > 0 else 0.0)

    x = list(range(len(levels)))
    width = 0.38
    plt.figure(figsize=(max(6, 0.7 * len(levels)), 4))
    plt.bar([i - width / 2 for i in x], entity_acc, width=width, label="entity-only")
    plt.bar([i + width / 2 for i in x], relation_acc, width=width, label="relation")
    plt.xticks(x, levels)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_relation_accuracy(rel_stats: dict, title: str, out_path: str) -> None:
    labels = sorted(rel_stats.keys())
    accs = []
    for k in labels:
        total = rel_stats[k]["total"]
        correct = rel_stats[k]["correct"]
        accs.append((correct / total) if total > 0 else 0.0)
    plt.figure(figsize=(max(6, 0.7 * len(labels)), 4))
    plt.bar(labels, accs)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_confidence_hist(correct: list[float], wrong: list[float], title: str, out_path: str) -> None:
    if not correct and not wrong:
        return
    plt.figure(figsize=(6, 4))
    if correct:
        plt.hist(correct, bins=20, alpha=0.6, label="correct")
    if wrong:
        plt.hist(wrong, bins=20, alpha=0.6, label="wrong")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare text-only vs image results.")
    ap.add_argument(
        "--text_dir",
        type=str,
        default=os.path.join(_RESULTS_DIR, "objective_vlm_prompt"),
    )
    ap.add_argument(
        "--image_dir",
        type=str,
        default=os.path.join(_REPO_ROOT, "vis_results"),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_RESULTS_DIR, "analysis_compare_text_image"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    text_levels = _find_level_dirs(args.text_dir)
    img_levels = _find_level_dirs(args.image_dir)
    levels = sorted(set(text_levels) & set(img_levels))
    if not levels:
        raise SystemExit("No overlapping level_* dirs between text and image outputs.")

    text_acc = []
    img_acc = []
    text_confs = []
    img_confs = []
    text_type_stats = []
    img_type_stats = []
    text_rel_stats = {}
    img_rel_stats = {}

    text_conf_correct = []
    text_conf_wrong = []
    img_conf_correct = []
    img_conf_wrong = []

    for lvl in levels:
        text_records = _load_text_records(os.path.join(args.text_dir, lvl))
        img_records = _load_image_records(os.path.join(args.image_dir, lvl))

        text_conf = _confusion_counts(text_records, "answer", "prediction")
        img_conf = _confusion_counts(img_records, "gt", "prediction")
        text_confs.append(text_conf)
        img_confs.append(img_conf)

        text_acc.append((text_conf["correct"] / text_conf["total"]) if text_conf["total"] > 0 else 0.0)
        img_acc.append((img_conf["correct"] / img_conf["total"]) if img_conf["total"] > 0 else 0.0)

        text_type = {"entity": {"total": 0, "correct": 0}, "relation": {"total": 0, "correct": 0}}
        img_type = {"entity": {"total": 0, "correct": 0}, "relation": {"total": 0, "correct": 0}}

        for r in text_records:
            q = (r.get("question") or "").strip()
            rel = _infer_rel_phrase(q)
            q_type = "relation" if rel != "unknown" else "entity"
            text_type[q_type]["total"] += 1
            if r.get("is_correct") is True:
                text_type[q_type]["correct"] += 1

            text_rel_stats.setdefault(rel, {"total": 0, "correct": 0})
            text_rel_stats[rel]["total"] += 1
            text_rel_stats[rel]["correct"] += int(r.get("is_correct") is True)

            conf = r.get("p_yes")
            conf_no = r.get("p_no")
            if isinstance(conf, (int, float)) and isinstance(conf_no, (int, float)):
                c = max(float(conf), float(conf_no))
                if r.get("is_correct") is True:
                    text_conf_correct.append(c)
                elif r.get("is_correct") is False:
                    text_conf_wrong.append(c)

        for r in img_records:
            q = (r.get("question") or "").strip()
            rel = _infer_rel_phrase(q)
            q_type = "relation" if rel != "unknown" else "entity"
            img_type[q_type]["total"] += 1
            if r.get("is_correct") is True:
                img_type[q_type]["correct"] += 1

            img_rel_stats.setdefault(rel, {"total": 0, "correct": 0})
            img_rel_stats[rel]["total"] += 1
            img_rel_stats[rel]["correct"] += int(r.get("is_correct") is True)

            conf = r.get("confidence")
            if isinstance(conf, (int, float)):
                if r.get("is_correct") is True:
                    img_conf_correct.append(float(conf))
                elif r.get("is_correct") is False:
                    img_conf_wrong.append(float(conf))

        text_type_stats.append(text_type)
        img_type_stats.append(img_type)

    _plot_accuracy_by_level(levels, text_acc, img_acc, args.out_dir)
    _plot_confusion_by_level(levels, text_confs, "Text confusion by level", os.path.join(args.out_dir, "confusion_text_by_level.png"))
    _plot_confusion_by_level(levels, img_confs, "Image confusion by level", os.path.join(args.out_dir, "confusion_image_by_level.png"))
    _plot_accuracy_by_type(levels, text_type_stats, "Text accuracy by question type", os.path.join(args.out_dir, "accuracy_text_by_type.png"))
    _plot_accuracy_by_type(levels, img_type_stats, "Image accuracy by question type", os.path.join(args.out_dir, "accuracy_image_by_type.png"))
    _plot_relation_accuracy(text_rel_stats, "Text accuracy by relation phrase", os.path.join(args.out_dir, "accuracy_text_by_relation.png"))
    _plot_relation_accuracy(img_rel_stats, "Image accuracy by relation phrase", os.path.join(args.out_dir, "accuracy_image_by_relation.png"))
    _plot_confidence_hist(text_conf_correct, text_conf_wrong, "Text confidence by correctness", os.path.join(args.out_dir, "confidence_text.png"))
    _plot_confidence_hist(img_conf_correct, img_conf_wrong, "Image confidence by correctness", os.path.join(args.out_dir, "confidence_image.png"))

    summary = {
        "levels": levels,
        "text_accuracy": text_acc,
        "image_accuracy": img_acc,
        "text_confusion": text_confs,
        "image_confusion": img_confs,
    }
    with open(os.path.join(args.out_dir, "comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved comparison plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
