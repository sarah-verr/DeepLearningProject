import argparse
import json
import os

import matplotlib.pyplot as plt


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


def _infer_rel_phrase(question: str) -> str:
    q = f" {question.lower()} "
    for phrase in REL_PHRASE_CANDIDATES:
        if f" {phrase} " in q:
            return phrase
    return "unknown"


def _mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def _mean_layers(seqs: list[list[float]]) -> list[float]:
    if not seqs:
        return []
    min_len = min(len(s) for s in seqs)
    if min_len == 0:
        return []
    out = []
    for i in range(min_len):
        out.append(sum(s[i] for s in seqs) / len(seqs))
    return out


def _plot_accuracy(levels: list[str], accs: list[float], out_dir: str) -> None:
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


def _plot_attn_means(levels: list[str], means: list[dict], key: str, title: str, out_dir: str) -> None:
    correct_means = [m[key]["correct"] or 0.0 for m in means]
    wrong_means = [m[key]["wrong"] or 0.0 for m in means]
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


def _plot_attn_gaps(levels: list[str], means: list[dict], out_dir: str) -> None:
    plt.figure(figsize=(max(6, 0.7 * len(levels)), 4))
    x = list(range(len(levels)))
    for key, label in [
        ("caption_mass_mean", "caption"),
        ("question_mass_mean", "question"),
        ("other_mass_mean", "other"),
    ]:
        gaps = []
        for m in means:
            c = m[key]["correct"]
            w = m[key]["wrong"]
            gaps.append((c - w) if c is not None and w is not None else 0.0)
        plt.plot(x, gaps, marker="o", label=label)
    plt.axhline(0.0, color="gray", linewidth=1)
    plt.xticks(x, levels)
    plt.ylabel("Correct - Wrong attention mean")
    plt.title("Attention gap by level")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "attn_gaps_by_level.png"), dpi=160)
    plt.close()


def _plot_per_layer(overall: dict, out_dir: str) -> None:
    for key, label in [
        ("caption_mass_per_layer", "caption"),
        ("question_mass_per_layer", "question"),
        ("other_mass_per_layer", "other"),
    ]:
        corr = overall[key]["correct"]
        wrong = overall[key]["wrong"]
        if not corr and not wrong:
            continue
        plt.figure(figsize=(6, 4))
        if corr:
            plt.plot(list(range(len(corr))), corr, label="correct")
        if wrong:
            plt.plot(list(range(len(wrong))), wrong, label="wrong")
        plt.xlabel("Layer")
        plt.ylabel("Attention mass")
        plt.title(f"Per-layer attention ({label})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"per_layer_{label}_overall.png"), dpi=160)
        plt.close()


def _plot_level_per_layer(level: str, per_layer: dict, out_dir: str) -> None:
    lvl_dir = os.path.join(out_dir, "levels", level)
    os.makedirs(lvl_dir, exist_ok=True)
    for key, label in [
        ("caption_mass_per_layer", "caption"),
        ("question_mass_per_layer", "question"),
        ("other_mass_per_layer", "other"),
    ]:
        corr = per_layer[key]["correct"]
        wrong = per_layer[key]["wrong"]
        if not corr and not wrong:
            continue
        plt.figure(figsize=(6, 4))
        if corr:
            plt.plot(list(range(len(corr))), corr, label="correct")
        if wrong:
            plt.plot(list(range(len(wrong))), wrong, label="wrong")
        plt.xlabel("Layer")
        plt.ylabel("Attention mass")
        plt.title(f"{level}: per-layer attention ({label})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(lvl_dir, f"per_layer_{label}.png"), dpi=160)
        plt.close()


def _plot_accuracy_vs_attention(levels: list[str], accs: list[float], means: list[dict], out_dir: str) -> None:
    for key, label in [
        ("caption_mass_mean", "caption"),
        ("question_mass_mean", "question"),
        ("other_mass_mean", "other"),
    ]:
        xs = [m[key]["correct"] or 0.0 for m in means]
        plt.figure(figsize=(5, 5))
        plt.scatter(xs, accs, s=30)
        for i, lvl in enumerate(levels):
            plt.text(xs[i], accs[i], lvl, fontsize=7, ha="left", va="bottom")
        plt.xlabel(f"{label} attention mean (correct)")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs {label} attention")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"acc_vs_{label}_attention.png"), dpi=160)
        plt.close()


def _plot_confidence_scatter(records: list[dict], out_dir: str) -> None:
    xs = []
    ys = []
    cs = []
    for r in records:
        attn = r.get("attention_summary") or {}
        cap = attn.get("caption_mass_mean")
        p_yes = r.get("p_yes")
        p_no = r.get("p_no")
        if not isinstance(cap, (int, float)):
            continue
        if not isinstance(p_yes, (int, float)) or not isinstance(p_no, (int, float)):
            continue
        conf = max(float(p_yes), float(p_no))
        xs.append(float(cap))
        ys.append(conf)
        cs.append("tab:blue" if r.get("is_correct") else "tab:red")

    if not xs:
        return
    plt.figure(figsize=(5, 5))
    plt.scatter(xs, ys, c=cs, alpha=0.6, s=12)
    plt.xlabel("Caption attention mean")
    plt.ylabel("Confidence (max p_yes/p_no)")
    plt.title("Confidence vs caption attention")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confidence_vs_caption_attention.png"), dpi=160)
    plt.close()


def _plot_relation_accuracy(rel_stats: dict, out_dir: str) -> None:
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
    plt.title("Accuracy by relation phrase")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_by_relation.png"), dpi=160)
    plt.close()


def _plot_question_type_accuracy(levels: list[str], type_stats: list[dict], out_dir: str) -> None:
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
    plt.title("Accuracy by question type (per level)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_by_question_type_levels.png"), dpi=160)
    plt.close()


def _plot_question_type_overall(stats: dict, out_dir: str) -> None:
    labels = ["entity-only", "relation"]
    ent = stats["entity"]
    rel = stats["relation"]
    accs = [
        (ent["correct"] / ent["total"]) if ent["total"] > 0 else 0.0,
        (rel["correct"] / rel["total"]) if rel["total"] > 0 else 0.0,
    ]
    plt.figure(figsize=(5, 4))
    plt.bar(labels, accs)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by question type (overall)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_by_question_type_overall.png"), dpi=160)
    plt.close()


def _plot_grounding_boxplots(rows: list[dict], out_dir: str) -> None:
    def _vals(key: str, correct: bool) -> list[float]:
        vals = []
        for r in rows:
            if r.get("is_correct") != correct:
                continue
            v = r.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return vals

    for key, title, fname in [
        ("entity_mass_mean", "Entity grounding vs correctness", "entity_grounding_box.png"),
        ("rel_entity_mass_mean", "Relation→entity grounding vs correctness", "rel_entity_grounding_box.png"),
    ]:
        correct_vals = _vals(key, True)
        wrong_vals = _vals(key, False)
        if not correct_vals and not wrong_vals:
            continue
        plt.figure(figsize=(6, 4))
        plt.boxplot([correct_vals, wrong_vals], labels=["correct", "wrong"])
        plt.ylabel("Mean attention mass")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze objective attention + accuracy across levels.")
    ap.add_argument(
        "--base_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Text-Only", "vis_results_caption", "objective_vlm_prompt"),
        help="Directory containing level_*/summary.jsonl files.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Text-Only", "vis_results_caption", "analysis_objective_attention"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    levels = _find_level_dirs(args.base_dir)
    if not levels:
        raise SystemExit(f"No level_* directories found in: {args.base_dir}")

    accs = []
    means_by_level = []
    per_layer_by_level = []
    rel_stats = {}
    all_records = []
    type_stats_by_level = []
    overall_type_stats = {
        "entity": {"total": 0, "correct": 0},
        "relation": {"total": 0, "correct": 0},
    }
    grounding_rows = []

    overall_per_layer = {
        "caption_mass_per_layer": {"correct": [], "wrong": []},
        "question_mass_per_layer": {"correct": [], "wrong": []},
        "other_mass_per_layer": {"correct": [], "wrong": []},
    }

    for lvl in levels:
        summary_path = os.path.join(args.base_dir, lvl, "summary.jsonl")
        records = _load_records(summary_path)
        all_records.extend(records)

        total = 0
        correct = 0
        means = {
            "caption_mass_mean": {"correct": [], "wrong": []},
            "question_mass_mean": {"correct": [], "wrong": []},
            "other_mass_mean": {"correct": [], "wrong": []},
        }
        per_layer = {
            "caption_mass_per_layer": {"correct": [], "wrong": []},
            "question_mass_per_layer": {"correct": [], "wrong": []},
            "other_mass_per_layer": {"correct": [], "wrong": []},
        }

        for r in records:
            if not isinstance(r.get("is_correct"), bool):
                continue
            is_correct = bool(r["is_correct"])
            total += 1
            if is_correct:
                correct += 1

            question = (r.get("question") or "").strip()
            rel = _infer_rel_phrase(question)
            rel_stats.setdefault(rel, {"total": 0, "correct": 0})
            rel_stats[rel]["total"] += 1
            rel_stats[rel]["correct"] += int(is_correct)

            q_type = "relation" if rel != "unknown" else "entity"
            overall_type_stats[q_type]["total"] += 1
            overall_type_stats[q_type]["correct"] += int(is_correct)

            attn = r.get("attention_summary") or {}
            for key in ("caption_mass_mean", "question_mass_mean", "other_mass_mean"):
                v = attn.get(key)
                if isinstance(v, (int, float)):
                    means[key]["correct" if is_correct else "wrong"].append(float(v))

            for key in ("caption_mass_per_layer", "question_mass_per_layer", "other_mass_per_layer"):
                seq = attn.get(key)
                if isinstance(seq, list) and seq:
                    per_layer[key]["correct" if is_correct else "wrong"].append([float(x) for x in seq])
                    overall_per_layer[key]["correct" if is_correct else "wrong"].append([float(x) for x in seq])

            grounding_rows.append(
                {
                    "is_correct": is_correct,
                    "entity_mass_mean": attn.get("entity_mass_mean"),
                    "rel_entity_mass_mean": attn.get("rel_entity_mass_mean"),
                    "q_type": q_type,
                }
            )

        accs.append((correct / total) if total > 0 else 0.0)

        means_by_level.append(
            {
                "caption_mass_mean": {
                    "correct": _mean(means["caption_mass_mean"]["correct"]),
                    "wrong": _mean(means["caption_mass_mean"]["wrong"]),
                },
                "question_mass_mean": {
                    "correct": _mean(means["question_mass_mean"]["correct"]),
                    "wrong": _mean(means["question_mass_mean"]["wrong"]),
                },
                "other_mass_mean": {
                    "correct": _mean(means["other_mass_mean"]["correct"]),
                    "wrong": _mean(means["other_mass_mean"]["wrong"]),
                },
            }
        )

        per_layer_by_level.append(
            {
                "caption_mass_per_layer": {
                    "correct": _mean_layers(per_layer["caption_mass_per_layer"]["correct"]),
                    "wrong": _mean_layers(per_layer["caption_mass_per_layer"]["wrong"]),
                },
                "question_mass_per_layer": {
                    "correct": _mean_layers(per_layer["question_mass_per_layer"]["correct"]),
                    "wrong": _mean_layers(per_layer["question_mass_per_layer"]["wrong"]),
                },
                "other_mass_per_layer": {
                    "correct": _mean_layers(per_layer["other_mass_per_layer"]["correct"]),
                    "wrong": _mean_layers(per_layer["other_mass_per_layer"]["wrong"]),
                },
            }
        )

        type_stats_by_level.append(
            {
                "entity": {
                    "total": sum(1 for r in records if _infer_rel_phrase((r.get("question") or "").strip()) == "unknown"),
                    "correct": sum(
                        1
                        for r in records
                        if _infer_rel_phrase((r.get("question") or "").strip()) == "unknown"
                        and r.get("is_correct") is True
                    ),
                },
                "relation": {
                    "total": sum(1 for r in records if _infer_rel_phrase((r.get("question") or "").strip()) != "unknown"),
                    "correct": sum(
                        1
                        for r in records
                        if _infer_rel_phrase((r.get("question") or "").strip()) != "unknown"
                        and r.get("is_correct") is True
                    ),
                },
            }
        )

    summary = {
        "levels": levels,
        "accuracy": accs,
        "means_by_level": means_by_level,
        "relation_stats": rel_stats,
        "question_type_stats": {
            "overall": overall_type_stats,
            "by_level": type_stats_by_level,
        },
    }
    with open(os.path.join(args.out_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_accuracy(levels, accs, args.out_dir)
    _plot_attn_means(levels, means_by_level, "caption_mass_mean", "Caption attention vs correctness", args.out_dir)
    _plot_attn_means(levels, means_by_level, "question_mass_mean", "Question self-attention vs correctness", args.out_dir)
    _plot_attn_means(levels, means_by_level, "other_mass_mean", "Other-token attention vs correctness", args.out_dir)
    _plot_attn_gaps(levels, means_by_level, args.out_dir)
    _plot_accuracy_vs_attention(levels, accs, means_by_level, args.out_dir)
    _plot_confidence_scatter(all_records, args.out_dir)
    _plot_relation_accuracy(rel_stats, args.out_dir)
    _plot_question_type_accuracy(levels, type_stats_by_level, args.out_dir)
    _plot_question_type_overall(overall_type_stats, args.out_dir)
    _plot_grounding_boxplots(grounding_rows, args.out_dir)

    overall = {
        "caption_mass_per_layer": {
            "correct": _mean_layers(overall_per_layer["caption_mass_per_layer"]["correct"]),
            "wrong": _mean_layers(overall_per_layer["caption_mass_per_layer"]["wrong"]),
        },
        "question_mass_per_layer": {
            "correct": _mean_layers(overall_per_layer["question_mass_per_layer"]["correct"]),
            "wrong": _mean_layers(overall_per_layer["question_mass_per_layer"]["wrong"]),
        },
        "other_mass_per_layer": {
            "correct": _mean_layers(overall_per_layer["other_mass_per_layer"]["correct"]),
            "wrong": _mean_layers(overall_per_layer["other_mass_per_layer"]["wrong"]),
        },
    }
    _plot_per_layer(overall, args.out_dir)

    for lvl, per_layer in zip(levels, per_layer_by_level):
        _plot_level_per_layer(lvl, per_layer, args.out_dir)

    print(f"Saved analysis plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
