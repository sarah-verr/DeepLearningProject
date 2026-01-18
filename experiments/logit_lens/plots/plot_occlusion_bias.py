import argparse
import json
import os

import math
import matplotlib.pyplot as plt


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_RESULTS_ROOT = os.path.join(_REPO_ROOT, "results_llava-hf", "llava-1.5-7b-hf")


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


def _mean_curve_list(curves: list[list[float]]) -> list[float]:
    if not curves:
        return []
    min_len = min(len(c) for c in curves)
    if min_len == 0:
        return []
    out = []
    for i in range(min_len):
        out.append(sum(c[i] for c in curves) / len(curves))
    return out


def _load_ann(path: str) -> dict | None:
    if not path:
        return None
    if not os.path.exists(path):
        marker = "DeepLearningProject/"
        idx = path.find(marker)
        if idx >= 0:
            local = os.path.join(_REPO_ROOT, path[idx + len(marker) :])
            if os.path.exists(local):
                path = local
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _match_qa(ann: dict, question: str) -> dict | None:
    q = (question or "").strip()
    for qa in ann.get("qa", []) or []:
        if (qa.get("question") or "").strip() == q:
            return qa
    return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot yes-rate and p(yes) by removal role (subject/object)."
    )
    ap.add_argument(
        "--base_dir",
        type=str,
        default=os.path.join(_RESULTS_ROOT, "visual_logit_lens_occluded"),
        help="Directory containing level_*/summary.jsonl files.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_RESULTS_ROOT, "analysis_occlusion_bias"),
    )
    ap.add_argument(
        "--levels",
        nargs="*",
        default=None,
        help="Optional list of levels to include (e.g., level_1 level_2).",
    )
    ap.add_argument(
        "--split_by_level",
        action="store_true",
        help="Also write per-level plots to subdirectories.",
    )
    ap.add_argument(
        "--combine_low_levels",
        action="store_true",
        help="Combine levels 0-2 into a single group for level-comparison plots.",
    )
    ap.add_argument(
        "--fp_filter_dir",
        type=str,
        default=None,
        help="Optional dir with summary.jsonl files to build FP filter (gt=no, pred=yes).",
    )
    ap.add_argument(
        "--filter_original_gt_no",
        action="store_true",
        help="Filter occluded records to those with gt=no in the original results.",
    )
    ap.add_argument(
        "--filter_original_fp",
        action="store_true",
        help="Filter occluded records to those that are FP (gt=no, pred=yes) in the original results.",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Print a quick summary table for the FP per-level comparison and exit.",
    )
    return ap.parse_args()


def _plot_bars(
    title: str,
    ylabel: str,
    groups: list[str],
    values: list[float],
    out_path: str,
) -> None:
    x = list(range(len(groups)))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, values, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_lines(
    title: str,
    ylabel: str,
    curves: dict[str, list[float]],
    out_path: str,
) -> None:
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


def _confusion_from_records(
    records: list[dict],
    *,
    original_map: dict[tuple[str, str], tuple[str, str]] | None = None,
) -> list[list[int]]:
    # Rows: gt yes/no, Cols: pred yes/no
    mat = [[0, 0], [0, 0]]
    for r in records:
        if original_map is not None:
            image_id = _base_image_id(r.get("image_id") or "")
            question = (r.get("question") or "").strip()
            key = (image_id, question)
            if key not in original_map:
                continue
            gt = original_map[key][0]
        else:
            gt = (r.get("answer") or "").strip().lower()
        pred = (r.get("prediction") or "").strip().lower()
        if gt not in ("yes", "no") or pred not in ("yes", "no"):
            continue
        gi = 0 if gt == "yes" else 1
        pi = 0 if pred == "yes" else 1
        mat[gi][pi] += 1
    return mat


def _plot_confusion(mat: list[list[int]], out_path: str, title: str, normalize: bool) -> None:
    import numpy as np

    arr = np.array(mat, dtype=float)
    if normalize:
        row_sums = arr.sum(axis=1, keepdims=True)
        arr = np.divide(arr, row_sums, out=np.zeros_like(arr), where=row_sums != 0)
    plt.figure(figsize=(4, 4))
    plt.imshow(arr, interpolation="nearest", cmap="Blues", vmin=0.0 if normalize else None, vmax=1.0 if normalize else None)
    plt.title(title)
    plt.colorbar(label="Rate" if normalize else "Count")
    plt.xticks([0, 1], ["pred_yes", "pred_no"])
    plt.yticks([0, 1], ["gt_yes", "gt_no"])
    for i in range(2):
        for j in range(2):
            val = arr[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            plt.text(j, i, text, ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _base_image_id(image_id: str) -> str:
    if not image_id:
        return ""
    if "_rm" in image_id:
        return image_id.split("_rm", 1)[0]
    return image_id


def _build_original_map(fp_dir: str) -> dict[tuple[str, str], tuple[str, str]]:
    levels = _find_level_dirs(fp_dir)
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for lvl in levels:
        summary_path = os.path.join(fp_dir, lvl, "summary.jsonl")
        records = _load_records(summary_path)
        for r in records:
            gt = (r.get("answer") or "").strip().lower()
            pred = (r.get("prediction") or "").strip().lower()
            image_id = _base_image_id(r.get("image_id") or "")
            question = (r.get("question") or "").strip()
            if image_id and question and gt in ("yes", "no") and pred in ("yes", "no"):
                out[(image_id, question)] = (gt, pred)
    return out


def _run_plots(records: list[dict], out_dir: str, *, title_suffix: str) -> None:
    stats = {
        "subject_removed": {"count": 0, "p_yes": 0.0, "p_no": 0.0, "pred_yes": 0, "pred_no": 0},
        "object_removed": {"count": 0, "p_yes": 0.0, "p_no": 0.0, "pred_yes": 0, "pred_no": 0},
    }
    curve_yes = {"subject_removed": [], "object_removed": []}
    curve_no = {"subject_removed": [], "object_removed": []}

    for r in records:
        ann = _load_ann(r.get("ann_path"))
        if not ann:
            continue
        removed_id = ann.get("removed_object_id")
        if removed_id is None:
            continue
        qa = _match_qa(ann, r.get("question"))
        if not qa:
            continue
        subj_id = qa.get("subject_id")
        obj_id = qa.get("object_id")
        if removed_id == subj_id:
            role = "subject_removed"
        elif removed_id == obj_id:
            role = "object_removed"
        else:
            continue

        p_yes = r.get("p_yes")
        p_no = r.get("p_no")
        if p_yes is None:
            continue
        pred = (r.get("prediction") or "").strip().lower()
        bucket = stats[role]
        bucket["count"] += 1
        bucket["p_yes"] += float(p_yes)
        if p_no is not None:
            bucket["p_no"] += float(p_no)
        if pred == "yes":
            bucket["pred_yes"] += 1
        elif pred == "no":
            bucket["pred_no"] += 1

        lens = r.get("logit_lens")
        if isinstance(lens, list) and lens:
            yes_curve, no_curve = _extract_full_softmax_curves(lens)
            if yes_curve and no_curve:
                curve_yes[role].append(yes_curve)
                curve_no[role].append(no_curve)

    groups = ["subject_removed", "object_removed"]
    mean_p_yes = []
    mean_p_no = []
    yes_rate = []
    no_rate = []
    for g in groups:
        bucket = stats[g]
        count = bucket["count"]
        mean_p_yes.append(bucket["p_yes"] / count if count else 0.0)
        mean_p_no.append(bucket["p_no"] / count if count else 0.0)
        yes_rate.append(bucket["pred_yes"] / count if count else 0.0)
        no_rate.append(bucket["pred_no"] / count if count else 0.0)

    _plot_bars(
        f"Mean p(yes) by removal role{title_suffix}",
        "Mean p(yes)",
        groups,
        mean_p_yes,
        os.path.join(out_dir, "occlusion_mean_pyes_by_role.png"),
    )
    _plot_bars(
        f"Mean p(no) by removal role{title_suffix}",
        "Mean p(no)",
        groups,
        mean_p_no,
        os.path.join(out_dir, "occlusion_mean_pno_by_role.png"),
    )
    _plot_bars(
        f"Yes prediction rate by removal role{title_suffix}",
        "Yes rate",
        groups,
        yes_rate,
        os.path.join(out_dir, "occlusion_yes_rate_by_role.png"),
    )
    _plot_bars(
        f"No prediction rate by removal role{title_suffix}",
        "No rate",
        groups,
        no_rate,
        os.path.join(out_dir, "occlusion_no_rate_by_role.png"),
    )

    curve_yes_mean = {
        "subject_removed": _mean_curve_list(curve_yes["subject_removed"]),
        "object_removed": _mean_curve_list(curve_yes["object_removed"]),
    }
    curve_no_mean = {
        "subject_removed": _mean_curve_list(curve_no["subject_removed"]),
        "object_removed": _mean_curve_list(curve_no["object_removed"]),
    }
    _plot_lines(
        f"Softmax p(yes) by layer{title_suffix}",
        "P(yes)",
        curve_yes_mean,
        os.path.join(out_dir, "occlusion_softmax_pyes_by_layer.png"),
    )
    _plot_lines(
        f"Softmax p(no) by layer{title_suffix}",
        "P(no)",
        curve_no_mean,
        os.path.join(out_dir, "occlusion_softmax_pno_by_layer.png"),
    )

    print("Summary (count, mean_p_yes, mean_p_no, yes_rate, no_rate):")
    for g in groups:
        b = stats[g]
        c = b["count"]
        mean_p_y = b["p_yes"] / c if c else 0.0
        mean_p_n = b["p_no"] / c if c else 0.0
        y_rate = b["pred_yes"] / c if c else 0.0
        n_rate = b["pred_no"] / c if c else 0.0
        print(f"{g}\tcount={c}\tmean_p_yes={mean_p_y:.4f}\tmean_p_no={mean_p_n:.4f}\tyes_rate={y_rate:.4f}\tno_rate={n_rate:.4f}")
    return {
        "subject_removed": {"count": stats["subject_removed"]["count"], "mean_p_yes": mean_p_yes[0], "yes_rate": yes_rate[0]},
        "object_removed": {"count": stats["object_removed"]["count"], "mean_p_yes": mean_p_yes[1], "yes_rate": yes_rate[1]},
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    levels = _find_level_dirs(args.base_dir)
    if args.levels:
        levels = [lvl for lvl in levels if lvl in args.levels]
    if not levels:
        raise SystemExit(f"No level_* directories found in: {args.base_dir}")

    all_records = []
    for lvl in levels:
        summary_path = os.path.join(args.base_dir, lvl, "summary.jsonl")
        all_records.extend(_load_records(summary_path))

    original_map = _build_original_map(args.fp_filter_dir) if args.fp_filter_dir else None
    if original_map is not None:
        filtered = []
        for r in all_records:
            image_id = _base_image_id(r.get("image_id") or "")
            question = (r.get("question") or "").strip()
            key = (image_id, question)
            if key not in original_map:
                continue
            gt, pred = original_map[key]
            if args.filter_original_fp:
                if gt == "no" and pred == "yes":
                    filtered.append(r)
            elif args.filter_original_gt_no:
                if gt == "no":
                    filtered.append(r)
            else:
                filtered.append(r)
        all_records = filtered

    title_suffix = ""
    title_suffix = ""
    title_suffix_conf = ""
    if args.filter_original_fp:
        title_suffix = " (filtered to FP)"
        title_suffix_conf = "\n(filtered to FP)"
    elif args.filter_original_gt_no:
        title_suffix = " (filtered to gt=no)"
        title_suffix_conf = "\n(filtered to gt=no)"

    occlusion_stats = _run_plots(all_records, args.out_dir, title_suffix=title_suffix)

    if args.quick:
        if not (args.fp_filter_dir and args.filter_original_fp):
            raise SystemExit("--quick requires --fp_filter_dir and --filter_original_fp")

        # Quick summary for the per-level FP comparison.
        level_groups = []
        level_map = {}
        if args.combine_low_levels:
            low = [lvl for lvl in levels if lvl in ("level_0", "level_1", "level_2")]
            rest = [lvl for lvl in levels if lvl not in ("level_0", "level_1", "level_2")]
            if low:
                level_groups.append("levels_0_2")
                level_map["levels_0_2"] = low
            for lvl in rest:
                level_groups.append(lvl)
                level_map[lvl] = [lvl]
        else:
            for lvl in levels:
                level_groups.append(lvl)
                level_map[lvl] = [lvl]

        orig_level_map = {}
        for lvl in _find_level_dirs(args.fp_filter_dir):
            summary_path = os.path.join(args.fp_filter_dir, lvl, "summary.jsonl")
            orig_level_map[lvl] = _load_records(summary_path)

        rows = []
        for group in level_groups:
            orig_group = []
            for lvl in level_map[group]:
                orig_group.extend(orig_level_map.get(lvl, []))
            orig_fp = [
                r
                for r in orig_group
                if (r.get("answer") or "").strip().lower() == "no"
                and (r.get("prediction") or "").strip().lower() == "yes"
            ]
            orig_p_yes_vals = [float(r.get("p_yes")) for r in orig_fp if r.get("p_yes") is not None]
            original_mean = sum(orig_p_yes_vals) / len(orig_p_yes_vals) if orig_p_yes_vals else 0.0

            group_records = []
            for lvl in level_map[group]:
                summary_path = os.path.join(args.base_dir, lvl, "summary.jsonl")
                group_records.extend(_load_records(summary_path))
            if original_map is not None:
                filtered = []
                for r in group_records:
                    image_id = _base_image_id(r.get("image_id") or "")
                    question = (r.get("question") or "").strip()
                    key = (image_id, question)
                    if key not in original_map:
                        continue
                    gt, pred = original_map[key]
                    if gt == "no" and pred == "yes":
                        filtered.append(r)
                group_records = filtered

            stats = {
                "subject_removed": {"count": 0, "p_yes": 0.0},
                "object_removed": {"count": 0, "p_yes": 0.0},
            }
            for r in group_records:
                ann = _load_ann(r.get("ann_path"))
                if not ann:
                    continue
                removed_id = ann.get("removed_object_id")
                if removed_id is None:
                    continue
                qa = _match_qa(ann, r.get("question"))
                if not qa:
                    continue
                subj_id = qa.get("subject_id")
                obj_id = qa.get("object_id")
                if removed_id == subj_id:
                    role = "subject_removed"
                elif removed_id == obj_id:
                    role = "object_removed"
                else:
                    continue
                p_yes = r.get("p_yes")
                if p_yes is None:
                    continue
                bucket = stats[role]
                bucket["count"] += 1
                bucket["p_yes"] += float(p_yes)

            subj_mean = stats["subject_removed"]["p_yes"] / stats["subject_removed"]["count"] if stats["subject_removed"]["count"] else 0.0
            obj_mean = stats["object_removed"]["p_yes"] / stats["object_removed"]["count"] if stats["object_removed"]["count"] else 0.0
            rows.append((group, original_mean, subj_mean, obj_mean))

        print("level\toriginal_fp_mean_pyes\tsubject_removed_mean_pyes\tobject_removed_mean_pyes")
        for group, orig_mean, subj_mean, obj_mean in rows:
            print(f"{group}\t{orig_mean:.4f}\t{subj_mean:.4f}\t{obj_mean:.4f}")
        return

    conf = _confusion_from_records(all_records, original_map=original_map)
    _plot_confusion(
        conf,
        os.path.join(args.out_dir, "occlusion_confusion_counts.png"),
        f"Occlusion confusion matrix (counts){title_suffix_conf}",
        normalize=False,
    )
    _plot_confusion(
        conf,
        os.path.join(args.out_dir, "occlusion_confusion_norm.png"),
        f"Occlusion confusion matrix (normalized){title_suffix_conf}",
        normalize=True,
    )

    if args.split_by_level:
        for lvl in levels:
            summary_path = os.path.join(args.base_dir, lvl, "summary.jsonl")
            records = _load_records(summary_path)
            if original_map is not None:
                filtered = []
                for r in records:
                    image_id = _base_image_id(r.get("image_id") or "")
                    question = (r.get("question") or "").strip()
                    key = (image_id, question)
                    if key not in original_map:
                        continue
                    gt, pred = original_map[key]
                    if args.filter_original_fp:
                        if gt == "no" and pred == "yes":
                            filtered.append(r)
                    elif args.filter_original_gt_no:
                        if gt == "no":
                            filtered.append(r)
                    else:
                        filtered.append(r)
                records = filtered
            level_out = os.path.join(args.out_dir, lvl)
            os.makedirs(level_out, exist_ok=True)
            _run_plots(records, level_out, title_suffix=title_suffix)
            conf = _confusion_from_records(records, original_map=original_map)
            _plot_confusion(
                conf,
                os.path.join(level_out, "occlusion_confusion_counts.png"),
                f"Occlusion confusion matrix (counts){title_suffix_conf}",
                normalize=False,
            )
            _plot_confusion(
                conf,
                os.path.join(level_out, "occlusion_confusion_norm.png"),
                f"Occlusion confusion matrix (normalized){title_suffix_conf}",
                normalize=True,
            )

    level_groups = []
    level_map = {}
    if args.combine_low_levels:
        low = [lvl for lvl in levels if lvl in ("level_0", "level_1", "level_2")]
        rest = [lvl for lvl in levels if lvl not in ("level_0", "level_1", "level_2")]
        if low:
            level_groups.append("levels_0_2")
            level_map["levels_0_2"] = low
        for lvl in rest:
            level_groups.append(lvl)
            level_map[lvl] = [lvl]
    else:
        for lvl in levels:
            level_groups.append(lvl)
            level_map[lvl] = [lvl]

    group_stats = {}
    for group in level_groups:
        group_records = []
        for lvl in level_map[group]:
            summary_path = os.path.join(args.base_dir, lvl, "summary.jsonl")
            group_records.extend(_load_records(summary_path))
        stats = {
            "subject_removed": {"count": 0, "p_yes": 0.0, "pred_yes": 0},
            "object_removed": {"count": 0, "p_yes": 0.0, "pred_yes": 0},
        }
        for r in group_records:
            ann = _load_ann(r.get("ann_path"))
            if not ann:
                continue
            removed_id = ann.get("removed_object_id")
            if removed_id is None:
                continue
            qa = _match_qa(ann, r.get("question"))
            if not qa:
                continue
            subj_id = qa.get("subject_id")
            obj_id = qa.get("object_id")
            if removed_id == subj_id:
                role = "subject_removed"
            elif removed_id == obj_id:
                role = "object_removed"
            else:
                continue
            p_yes = r.get("p_yes")
            if p_yes is None:
                continue
            pred = (r.get("prediction") or "").strip().lower()
            bucket = stats[role]
            bucket["count"] += 1
            bucket["p_yes"] += float(p_yes)
            if pred == "yes":
                bucket["pred_yes"] += 1
        group_stats[group] = stats

    def _extract_group_metric(metric: str) -> dict[str, list[float]]:
        out = {"subject_removed": [], "object_removed": []}
        for group in level_groups:
            stats = group_stats[group]
            for role in ("subject_removed", "object_removed"):
                bucket = stats[role]
                count = bucket["count"]
                if count <= 0:
                    out[role].append(0.0)
                    continue
                if metric == "p_yes":
                    out[role].append(bucket["p_yes"] / count)
                else:
                    out[role].append(bucket["pred_yes"] / count)
        return out

    for metric, title, out_name, ylabel in [
        ("p_yes", f"Mean p(yes) by level{title_suffix}", "occlusion_mean_pyes_by_level.png", "Mean p(yes)"),
        ("yes_rate", f"Yes rate by level{title_suffix}", "occlusion_yes_rate_by_level.png", "Yes rate"),
    ]:
        vals = _extract_group_metric("p_yes" if metric == "p_yes" else "yes_rate")
        x = list(range(len(level_groups)))
        width = 0.35
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([i - width / 2 for i in x], vals["subject_removed"], width, label="subject_removed")
        ax.bar([i + width / 2 for i in x], vals["object_removed"], width, label="object_removed")
        ax.set_xticks(x)
        ax.set_xticklabels(level_groups, rotation=0)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, out_name), dpi=160)
        plt.close(fig)

    if args.fp_filter_dir and args.filter_original_fp:
        orig_records = []
        for lvl in _find_level_dirs(args.fp_filter_dir):
            summary_path = os.path.join(args.fp_filter_dir, lvl, "summary.jsonl")
            orig_records.extend(_load_records(summary_path))
        orig_fp = [r for r in orig_records if (r.get("answer") or "").strip().lower() == "no" and (r.get("prediction") or "").strip().lower() == "yes"]
        orig_p_yes_vals = [float(r.get("p_yes")) for r in orig_fp if r.get("p_yes") is not None]
        orig_mean_p_yes = sum(orig_p_yes_vals) / len(orig_p_yes_vals) if orig_p_yes_vals else 0.0
        orig_yes_rate = 1.0 if orig_fp else 0.0

        groups = ["original_fp", "subject_removed", "object_removed"]
        yes_rates = [
            orig_yes_rate,
            occlusion_stats["subject_removed"]["yes_rate"],
            occlusion_stats["object_removed"]["yes_rate"],
        ]
        _plot_bars(
            "Yes rate: original FP vs occluded FP",
            "Yes rate",
            groups,
            yes_rates,
            os.path.join(args.out_dir, "occlusion_fp_yes_rate_comparison.png"),
        )

        mean_p_yes_vals = [
            orig_mean_p_yes,
            occlusion_stats["subject_removed"]["mean_p_yes"],
            occlusion_stats["object_removed"]["mean_p_yes"],
        ]
        _plot_bars(
            "Mean p(yes): original FP vs occluded FP",
            "Mean p(yes)",
            groups,
            mean_p_yes_vals,
            os.path.join(args.out_dir, "occlusion_fp_pyes_comparison.png"),
        )

        # Per-level mean p(yes) comparison: original FP vs occluded (subject/object removed)
        level_groups = []
        level_map = {}
        if args.combine_low_levels:
            low = [lvl for lvl in levels if lvl in ("level_0", "level_1", "level_2")]
            rest = [lvl for lvl in levels if lvl not in ("level_0", "level_1", "level_2")]
            if low:
                level_groups.append("levels_0_2")
                level_map["levels_0_2"] = low
            for lvl in rest:
                level_groups.append(lvl)
                level_map[lvl] = [lvl]
        else:
            for lvl in levels:
                level_groups.append(lvl)
                level_map[lvl] = [lvl]

        orig_level_map = {}
        for lvl in _find_level_dirs(args.fp_filter_dir):
            summary_path = os.path.join(args.fp_filter_dir, lvl, "summary.jsonl")
            orig_level_map[lvl] = _load_records(summary_path)

        original_vals = []
        subj_vals = []
        obj_vals = []
        for group in level_groups:
            orig_group = []
            for lvl in level_map[group]:
                orig_group.extend(orig_level_map.get(lvl, []))
            orig_fp = [r for r in orig_group if (r.get("answer") or "").strip().lower() == "no" and (r.get("prediction") or "").strip().lower() == "yes"]
            orig_p_yes_vals = [float(r.get("p_yes")) for r in orig_fp if r.get("p_yes") is not None]
            original_vals.append(sum(orig_p_yes_vals) / len(orig_p_yes_vals) if orig_p_yes_vals else 0.0)

            # Occluded stats for this group
            group_records = []
            for lvl in level_map[group]:
                summary_path = os.path.join(args.base_dir, lvl, "summary.jsonl")
                group_records.extend(_load_records(summary_path))
            if original_map is not None:
                filtered = []
                for r in group_records:
                    image_id = _base_image_id(r.get("image_id") or "")
                    question = (r.get("question") or "").strip()
                    key = (image_id, question)
                    if key not in original_map:
                        continue
                    gt, pred = original_map[key]
                    if gt == "no" and pred == "yes":
                        filtered.append(r)
                group_records = filtered
            stats = {
                "subject_removed": {"count": 0, "p_yes": 0.0},
                "object_removed": {"count": 0, "p_yes": 0.0},
            }
            for r in group_records:
                ann = _load_ann(r.get("ann_path"))
                if not ann:
                    continue
                removed_id = ann.get("removed_object_id")
                if removed_id is None:
                    continue
                qa = _match_qa(ann, r.get("question"))
                if not qa:
                    continue
                subj_id = qa.get("subject_id")
                obj_id = qa.get("object_id")
                if removed_id == subj_id:
                    role = "subject_removed"
                elif removed_id == obj_id:
                    role = "object_removed"
                else:
                    continue
                p_yes = r.get("p_yes")
                if p_yes is None:
                    continue
                bucket = stats[role]
                bucket["count"] += 1
                bucket["p_yes"] += float(p_yes)
            subj_vals.append(stats["subject_removed"]["p_yes"] / stats["subject_removed"]["count"] if stats["subject_removed"]["count"] else 0.0)
            obj_vals.append(stats["object_removed"]["p_yes"] / stats["object_removed"]["count"] if stats["object_removed"]["count"] else 0.0)

        x = list(range(len(level_groups)))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([i - width for i in x], original_vals, width, label="original_fp")
        ax.bar(x, subj_vals, width, label="subject_removed")
        ax.bar([i + width for i in x], obj_vals, width, label="object_removed")
        ax.set_xticks(x)
        ax.set_xticklabels(level_groups, rotation=0)
        ax.set_ylabel("Mean p(yes)")
        ax.set_title("Mean p(yes) by level (original FP vs occluded FP)")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "occlusion_fp_pyes_by_level.png"), dpi=160)
        plt.close(fig)

    print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
