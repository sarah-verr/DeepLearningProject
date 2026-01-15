#!/usr/bin/env python3
import argparse
import json
import os

import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_RESULTS_ROOT = os.path.join(_REPO_ROOT, "results_llava-hf", "llava-1.5-7b-hf")


def _find_level_dirs(base_dir: str) -> list[str]:
    if not os.path.isdir(base_dir):
        return []
    levels = [d for d in os.listdir(base_dir) if d.startswith("level_")]
    return sorted(levels)


def _load_records(base_dir: str) -> list[dict]:
    records = []
    for lvl in _find_level_dirs(base_dir):
        path = os.path.join(base_dir, lvl, "summary.jsonl")
        if not os.path.exists(path):
            continue
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


def _split_curves(records: list[dict]) -> tuple[list[list[float]], list[list[float]]]:
    pred_yes = []
    pred_no = []
    for r in records:
        lens = r.get("logit_lens")
        if not isinstance(lens, list) or not lens:
            continue
        pred = (r.get("prediction") or "").strip().lower()
        if pred not in ("yes", "no"):
            continue
        if pred == "yes":
            pred_yes.append([float(d.get("p_yes", 0.0)) for d in lens])
        else:
            pred_no.append([float(d.get("p_no", 0.0)) for d in lens])
    return pred_yes, pred_no


def _plot_overlay(
    text_curves: tuple[list[float], list[float]],
    image_curves: tuple[list[float], list[float]],
    out_path: str,
) -> None:
    text_pred_yes, text_pred_no = text_curves
    img_pred_yes, img_pred_no = image_curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax_yes, ax_no = axes

    if text_pred_yes:
        ax_yes.plot(list(range(len(text_pred_yes))), text_pred_yes, label="text pred_yes (p_yes)")
    if img_pred_yes:
        ax_yes.plot(list(range(len(img_pred_yes))), img_pred_yes, label="image pred_yes (p_yes)")
    ax_yes.set_xlabel("Layer")
    ax_yes.set_ylabel("P(yes)")
    ax_yes.set_title("Predicted yes")
    ax_yes.legend()

    if text_pred_no:
        ax_no.plot(list(range(len(text_pred_no))), text_pred_no, label="text pred_no (p_no)")
    if img_pred_no:
        ax_no.plot(list(range(len(img_pred_no))), img_pred_no, label="image pred_no (p_no)")
    ax_no.set_xlabel("Layer")
    ax_no.set_ylabel("P(no)")
    ax_no.set_title("Predicted no")
    ax_no.legend()

    fig.suptitle("Logit-lens by prediction: text vs image (overall)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot cross-modal logit-lens comparison (text vs image).")
    ap.add_argument(
        "--text_dir",
        type=str,
        default=os.path.join(_RESULTS_ROOT, "text_only_objective"),
    )
    ap.add_argument(
        "--image_dir",
        type=str,
        default=os.path.join(_RESULTS_ROOT, "visual_logit_lens"),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_RESULTS_ROOT, "logit_lens"),
    )
    ap.add_argument(
        "--out_name",
        type=str,
        default="logit_lens_text_vs_image_overall.png",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    text_records = _load_records(args.text_dir)
    image_records = _load_records(args.image_dir)
    if not text_records:
        raise SystemExit(f"No records found under: {args.text_dir}")
    if not image_records:
        raise SystemExit(f"No records found under: {args.image_dir}")

    text_pred_yes, text_pred_no = _split_curves(text_records)
    img_pred_yes, img_pred_no = _split_curves(image_records)
    text_mean = (_mean_curves(text_pred_yes), _mean_curves(text_pred_no))
    img_mean = (_mean_curves(img_pred_yes), _mean_curves(img_pred_no))

    out_path = os.path.join(args.out_dir, args.out_name)
    _plot_overlay(text_mean, img_mean, out_path)
    print(f"Saved cross-modal logit-lens plot to: {out_path}")


if __name__ == "__main__":
    main()
