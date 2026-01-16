import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_NEW_RESULTS_ROOT = os.path.join(_REPO_ROOT, "results_llava-hf", "llava-1.5-7b-hf")
_OUT_DIR = os.path.join(_NEW_RESULTS_ROOT, "logit_lens_attribute")

SHAPES = ["square", "circle", "triangle", "star"]
COLORS = ["red", "blue", "green", "yellow", "purple", "cyan", "orange", "pink", "lime"]


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


def _plot_multi_curves(
    curves: dict,
    out_path: str,
    title: str,
    ylabel: str,
    *,
    color_map: dict[str, str] | None = None,
) -> None:
    if not curves:
        return
    plt.figure(figsize=(6, 4))
    for label, data in curves.items():
        if data:
            color = color_map.get(label) if color_map else None
            if color:
                plt.plot(list(range(len(data))), data, label=label, color=color)
            else:
                plt.plot(list(range(len(data))), data, label=label)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    out_path: str,
    title: str,
    cmap: str = "Blues",
) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar(label="Count")
    tick_marks = list(range(len(labels)))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center")
    plt.ylabel("Ground truth")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_confusion_matrix_norm(
    matrix: np.ndarray,
    labels: list[str],
    out_path: str,
    title: str,
    cmap: str = "Blues",
) -> None:
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
    plt.figure(figsize=(6, 5))
    plt.imshow(norm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.colorbar(label="Rate")
    tick_marks = list(range(len(labels)))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center")
    plt.ylabel("Ground truth")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _extract_p_correct(lens: list[dict], answer: str) -> list[float]:
    curve = []
    for layer in lens:
        p_by_choice = layer.get("p_by_choice")
        if not isinstance(p_by_choice, dict):
            return []
        val = p_by_choice.get(answer)
        if val is None:
            return []
        curve.append(float(val))
    return curve


def _extract_logit_curve(lens: list[dict], option: str) -> list[float]:
    curve = []
    for layer in lens:
        logit_by_choice = layer.get("logit_by_choice")
        if not isinstance(logit_by_choice, dict):
            return []
        val = logit_by_choice.get(option)
        if val is None:
            return []
        curve.append(float(val))
    return curve


def _extract_softmax_curve(lens: list[dict], option: str) -> list[float]:
    curve = []
    for layer in lens:
        logit_by_choice = layer.get("logit_by_choice")
        logit_total = layer.get("logit_total")
        if not isinstance(logit_by_choice, dict) or logit_total is None:
            return []
        val = logit_by_choice.get(option)
        if val is None:
            return []
        curve.append(float(np.exp(float(val) - float(logit_total))))
    return curve


def _extract_normalized_curve(lens: list[dict], option: str) -> list[float]:
    curve = []
    for layer in lens:
        p_by_choice = layer.get("p_by_choice")
        if not isinstance(p_by_choice, dict):
            return []
        val = p_by_choice.get(option)
        if val is None:
            return []
        curve.append(float(val))
    return curve


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Aggregate logit-lens attribute curves (p(correct) by layer)."
    )
    ap.add_argument(
        "--base_dir",
        type=str,
        default=os.path.join(_NEW_RESULTS_ROOT, "visual_logit_lens_attribute"),
    )
    ap.add_argument("--out_dir", type=str, default=_OUT_DIR)
    ap.add_argument(
        "--levels",
        nargs="*",
        default=None,
        help="Optional list of levels to include (e.g., level_2).",
    )
    ap.add_argument(
        "--split_by_level",
        action="store_true",
        help="Also write per-level plots to subdirectories.",
    )
    return ap.parse_args()


def _run_plots(records: list[dict], out_dir: str) -> None:
    all_correct = []
    all_wrong = []
    by_type = {"shape": {"correct": [], "wrong": []}, "color": {"correct": [], "wrong": []}}
    by_type_answer_logit: dict[str, dict[str, dict[str, list[list[float]]]]] = {
        "shape": {opt: {o: [] for o in SHAPES} for opt in SHAPES},
        "color": {opt: {o: [] for o in COLORS} for opt in COLORS},
    }
    by_type_answer_softmax = {
        "shape": {opt: {o: [] for o in SHAPES} for opt in SHAPES},
        "color": {opt: {o: [] for o in COLORS} for opt in COLORS},
    }
    by_type_answer_norm = {
        "shape": {opt: {o: [] for o in SHAPES} for opt in SHAPES},
        "color": {opt: {o: [] for o in COLORS} for opt in COLORS},
    }
    confusion = {
        "shape": np.zeros((len(SHAPES), len(SHAPES)), dtype=int),
        "color": np.zeros((len(COLORS), len(COLORS)), dtype=int),
    }

    for r in records:
        if not isinstance(r.get("is_correct"), bool):
            continue
        lens = r.get("logit_lens")
        if not isinstance(lens, list) or not lens:
            continue
        answer = (r.get("answer") or "").strip().lower()
        qtype = (r.get("question_type") or "").strip().lower()
        if qtype not in ("shape", "color") or not answer:
            continue
        curve = _extract_p_correct(lens, answer)
        if not curve:
            continue
        if r["is_correct"]:
            all_correct.append(curve)
            by_type[qtype]["correct"].append(curve)
        else:
            all_wrong.append(curve)
            by_type[qtype]["wrong"].append(curve)

        options = SHAPES if qtype == "shape" else COLORS
        if answer in options:
            for opt in options:
                opt_curve = _extract_logit_curve(lens, opt)
                if opt_curve:
                    by_type_answer_logit[qtype][answer][opt].append(opt_curve)
                opt_curve = _extract_softmax_curve(lens, opt)
                if opt_curve:
                    by_type_answer_softmax[qtype][answer][opt].append(opt_curve)
                opt_curve = _extract_normalized_curve(lens, opt)
                if opt_curve:
                    by_type_answer_norm[qtype][answer][opt].append(opt_curve)

        p_by_choice = r.get("p_by_choice")
        if isinstance(p_by_choice, dict):
            pred = None
            best = None
            for opt in options:
                val = p_by_choice.get(opt)
                if val is None:
                    continue
                if best is None or val > best:
                    best = val
                    pred = opt
            if pred in options and answer in options:
                gt_idx = options.index(answer)
                pred_idx = options.index(pred)
                confusion[qtype][gt_idx, pred_idx] += 1

    _plot_curves(
        _mean_curves(all_correct),
        _mean_curves(all_wrong),
        os.path.join(out_dir, "logit_lens_attribute_pcorrect_overall.png"),
        "Attribute logit-lens p(correct) by layer (overall)",
        "P(correct choice)",
    )

    for qtype in ("shape", "color"):
        _plot_curves(
            _mean_curves(by_type[qtype]["correct"]),
            _mean_curves(by_type[qtype]["wrong"]),
            os.path.join(out_dir, f"logit_lens_attribute_pcorrect_{qtype}.png"),
            f"Attribute logit-lens p(correct) by layer ({qtype})",
            "P(correct choice)",
        )
        options = SHAPES if qtype == "shape" else COLORS
        color_map = None
        if qtype == "color":
            color_map = {
                "red": "red",
                "blue": "blue",
                "green": "green",
                "yellow": "gold",
                "purple": "purple",
                "cyan": "cyan",
                "orange": "orange",
                "pink": "hotpink",
                "lime": "limegreen",
            }
        for answer in options:
            curves = {}
            for opt in options:
                curves[opt] = _mean_curves(by_type_answer_logit[qtype][answer][opt])
            _plot_multi_curves(
                curves,
                os.path.join(out_dir, f"logit_lens_attribute_logits_{qtype}_answer_{answer}.png"),
                f"Attribute logit-lens logits by layer ({qtype}, answer={answer})",
                "logit",
                color_map=color_map,
            )
            curves = {}
            for opt in options:
                curves[opt] = _mean_curves(by_type_answer_softmax[qtype][answer][opt])
            _plot_multi_curves(
                curves,
                os.path.join(out_dir, f"logit_lens_attribute_softmax_{qtype}_answer_{answer}.png"),
                f"Attribute logit-lens softmax p by layer ({qtype}, answer={answer})",
                "P",
                color_map=color_map,
            )
            curves = {}
            for opt in options:
                curves[opt] = _mean_curves(by_type_answer_norm[qtype][answer][opt])
            _plot_multi_curves(
                curves,
                os.path.join(out_dir, f"logit_lens_attribute_norm_{qtype}_answer_{answer}.png"),
                f"Attribute logit-lens normalized p by layer ({qtype}, answer={answer})",
                "P",
                color_map=color_map,
            )

        _plot_confusion_matrix(
            confusion[qtype],
            options,
            os.path.join(out_dir, f"logit_lens_attribute_confusion_{qtype}.png"),
            f"Attribute confusion matrix ({qtype}, counts)",
        )
        _plot_confusion_matrix_norm(
            confusion[qtype],
            options,
            os.path.join(out_dir, f"logit_lens_attribute_confusion_{qtype}_norm.png"),
            f"Attribute confusion matrix ({qtype}, normalized)",
        )


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

    _run_plots(all_records, args.out_dir)

    if args.split_by_level:
        for lvl in levels:
            summary_path = os.path.join(args.base_dir, lvl, "summary.jsonl")
            records = _load_records(summary_path)
            level_out = os.path.join(args.out_dir, lvl)
            os.makedirs(level_out, exist_ok=True)
            _run_plots(records, level_out)

    print(f"Saved attribute logit-lens plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
