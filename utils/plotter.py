from pathlib import Path
from typing import Any
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from scipy.ndimage import zoom

from utils.prompt_llava import MODEL_ID
from collections import defaultdict


class Plotter:
    """Central helper for results and plots.

    Manages a base results directory under results_{MODEL_ID} and allows
    callers to save different kinds of plots using simple filenames.
    """

    def __init__(self, experiment_name: str | None = None) -> None:
        project_root = Path(__file__).resolve().parents[1]
        base = project_root / f"results_{MODEL_ID}"
        if experiment_name:
            self._results_dir = base / experiment_name
        else:
            self._results_dir = base
        self._results_dir.mkdir(parents=True, exist_ok=True)

    @property
    def results_dir(self) -> Path:
        """Public accessor for the base results directory."""
        return self._results_dir

    def save_json(self, data: Any, filename: str, subdir: str | None = None) -> Path:
        """Save arbitrary data as pretty-printed JSON under the results dir."""
        out_dir = self._subdir(subdir)
        out_path = out_dir / filename
        with out_path.open("w") as f:
            json.dump(data, f, indent=2)
        return out_path

    def save_jsonl(self, data: list, filename: str, subdir: str | None = None) -> Path:
        """Save a list of dicts as line-delimited JSON (JSONL) under the results dir."""
        out_dir = self._subdir(subdir)
        out_path = out_dir / filename
        with out_path.open("w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return out_path

    def save_numpy(self, data: np.ndarray, filename: str, subdir: str | None = None) -> Path:
        """Save numpy array under the results dir."""
        out_dir = self._subdir(subdir)
        out_path = out_dir / filename
        np.save(out_path, data)
        return out_path

    def _subdir(self, name: str | None) -> Path:
        if name:
            d = self._results_dir / name
        else:
            d = self._results_dir
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------- Attention aggregation helpers -------

    def aggregate_attention_metrics(self, metrics_list: list[dict]) -> dict:
        """Aggregate a list of attention_metrics dicts into a single averaged one.

        Assumes all entries have the same layer/head structure.
        """
        if not metrics_list:
            return {"per_layer": []}

        first = metrics_list[0]
        per_layer_agg: list[dict] = []

        for layer_idx, base_layer in enumerate(first["per_layer"]):
            num_heads = len(base_layer.get("per_head", []))
            per_head_agg: list[dict] = []

            for h in range(num_heads):
                text_vals = [
                    m["per_layer"][layer_idx]["per_head"][h]["text_fraction"]
                    for m in metrics_list
                ]
                img_vals = [
                    m["per_layer"][layer_idx]["per_head"][h]["image_fraction"]
                    for m in metrics_list
                ]
                mean_text = float(np.mean(text_vals)) if text_vals else 0.0
                mean_img = float(np.mean(img_vals)) if img_vals else 0.0
                per_head_agg.append(
                    {
                        "head_idx": h,
                        "text_fraction": mean_text,
                        "image_fraction": mean_img,
                    }
                )

            if per_head_agg:
                layer_text_mean = float(
                    np.mean([h["text_fraction"] for h in per_head_agg])
                )
                layer_img_mean = float(
                    np.mean([h["image_fraction"] for h in per_head_agg])
                )
            else:
                layer_text_mean = 0.0
                layer_img_mean = 0.0

            per_layer_agg.append(
                {
                    "layer_idx": base_layer["layer_idx"],
                    "per_head": per_head_agg,
                    "mean_text_fraction": layer_text_mean,
                    "mean_image_fraction": layer_img_mean,
                }
            )

        return {"per_layer": per_layer_agg}

    # ------- Attention plots -------

    def plot_layer_text_vs_image(self, attn_metrics: dict) -> None:
        """Plot mean attention with transparent std bands for text vs image per layer."""
        per_layer = attn_metrics.get("per_layer", [])
        if not per_layer:
            return
        
        filename="layer_text_vs_image.png"
        subdir="plots"

        out_dir = self._subdir(subdir)
        out_path = out_dir / filename

        layers = np.array([pl["layer_idx"] for pl in per_layer])
        text_means = np.array([pl["mean_text_fraction"] for pl in per_layer])
        img_means = np.array([pl["mean_image_fraction"] for pl in per_layer])

        # Compute stddev across heads for each layer
        text_stds = []
        img_stds = []
        for pl in per_layer:
            heads = pl.get("per_head", [])
            text_vals = [h["text_fraction"] for h in heads] if heads else [0.0]
            img_vals = [h["image_fraction"] for h in heads] if heads else [0.0]
            text_stds.append(float(np.std(text_vals)))
            img_stds.append(float(np.std(img_vals)))

        text_stds = np.array(text_stds)
        img_stds = np.array(img_stds)

        plt.figure(figsize=(6, 4))

        # Text curve + shaded std band
        plt.plot(layers, text_means, label="Text", color="tab:blue", marker="o")
        plt.fill_between(
            layers,
            text_means - text_stds,
            text_means + text_stds,
            color="tab:blue",
            alpha=0.2,
        )

        # Image curve + shaded std band
        plt.plot(layers, img_means, label="Image", color="tab:orange", marker="o")
        plt.fill_between(
            layers,
            img_means - img_stds,
            img_means + img_stds,
            color="tab:orange",
            alpha=0.2,
        )
        plt.xlabel("Layer")
        plt.ylabel("Fraction of attention from last token")
        plt.title("Layer-wise attention: last token to text vs image")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    def plot_attention_by_relation(self, attn_results):
        """Plot head×layer grids per relation type and a global aggregate.

        Expects attn_results to be a list of dicts with keys:
            - "relation_type" (e.g., "above", "left_of")
            - "attention_metrics" (with "per_layer" as used elsewhere)

        Also prints heads/layers whose aggregated image-attention fraction
        exceeds a fixed threshold so you can see image-centric heads.
        """

        if not attn_results:
                return

        IMAGE_THRESHOLD = 0.5

        # Group metrics by relation type
        rel_groups = defaultdict(list)
        all_metrics = []
        for result in attn_results:
            rel_type = result.get("relation_type") or result.get("attention_metrics", {}).get("relation_type", "global")
            metrics = result["attention_metrics"]
            rel_groups[rel_type].append(metrics)
            all_metrics.append(metrics)

        # Plot for each relation_type (aggregated over all its QAs)
        for rel_type, metrics_list in rel_groups.items():
            if not metrics_list:
                continue
            agg = self.aggregate_attention_metrics(metrics_list)
            self.plot_head_layer_image_fraction(
                agg,
                filename=f"head_layer_image_fraction_{rel_type}.png",
                subdir="plots",
            )

            # Print image-centric heads for this relation type
            for layer in agg.get("per_layer", []):
                layer_idx = layer.get("layer_idx")
                for h in layer.get("per_head", []):
                    img_frac = float(h.get("image_fraction", 0.0))
                    if img_frac >= IMAGE_THRESHOLD:
                        head_idx = h.get("head_idx")
                        print(
                            f"[relation={rel_type}] layer={layer_idx}, head={head_idx} "
                            f"image_fraction={img_frac:.3f} (>= {IMAGE_THRESHOLD})"
                        )

        # Global aggregate across all relation types
        if all_metrics:
            agg_global = self.aggregate_attention_metrics(all_metrics)
            self.plot_head_layer_image_fraction(
                agg_global,
                filename="head_layer_image_fraction_global.png",
                subdir="plots",
            )

            # Print globally image-centric heads
            for layer in agg_global.get("per_layer", []):
                layer_idx = layer.get("layer_idx")
                for h in layer.get("per_head", []):
                    img_frac = float(h.get("image_fraction", 0.0))
                    if img_frac >= IMAGE_THRESHOLD:
                        head_idx = h.get("head_idx")
                        print(
                            f"[global] layer={layer_idx}, head={head_idx} "
                            f"image_fraction={img_frac:.3f} (>= {IMAGE_THRESHOLD})"
                        )

    def plot_head_layer_image_fraction(
        self,
        attn_metrics: dict,
        filename: str = "head_layer_image_fraction.png",
        subdir: str = "plots",
    ) -> None:
        """Build a head-layer grid with image-attention fraction for each head."""
        per_layer = attn_metrics.get("per_layer", [])
        if not per_layer:
            return

        out_dir = self._subdir(subdir)
        out_path = out_dir / filename

        num_layers = len(per_layer)
        max_heads = max(len(pl.get("per_head", [])) for pl in per_layer)
        if max_heads == 0:
            return

        grid = np.full((max_heads, num_layers), np.nan, dtype=float)
        for li, pl in enumerate(per_layer):
            for h_info in pl.get("per_head", []):
                h_idx = int(h_info.get("head_idx", -1))
                if 0 <= h_idx < max_heads:
                    grid[h_idx, li] = float(h_info.get("image_fraction", 0.0))

        plt.figure(figsize=(6, 6))
        plt.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        plt.colorbar(label="Image attention fraction")
        plt.xlabel("Layer index")
        plt.ylabel("Head index")
        plt.title("Head per Layer image-attention fractions")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    # ------- Evaluation plots (accuracy / confusion matrix) -------

    def plot_confusion_matrix(
        self,
        cm: Any,
        filename: str = "confusion_matrix.png",
        subdir: str = "plots",
        title: str = "Confusion Matrix (yes/no)",
    ) -> None:
        """Plot a 2x2 confusion matrix DataFrame as a heatmap."""
        out_dir = self._subdir(subdir)
        out_path = out_dir / filename

        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(title)
        plt.ylabel("Ground truth")
        plt.xlabel("Prediction")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    def plot_accuracy_bars(
        self,
        accuracy_by_level,
        filename: str = "accuracy_by_level.png",
        subdir: str = "plots",
        title: str = "Accuracy by level",
    ) -> None:
        """Plot a simple bar chart of accuracy per level."""
        out_dir = self._subdir(subdir)
        out_path = out_dir / filename

        # Convert to index + values
        if hasattr(accuracy_by_level, "index") and hasattr(accuracy_by_level, "values"):
            labels = list(accuracy_by_level.index)
            values = list(accuracy_by_level.values)
        else:
            labels = list(accuracy_by_level.keys())
            values = list(accuracy_by_level.values())

        plt.figure(figsize=(6, 4))
        plt.bar(labels, values)
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.xlabel("Level")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    def plot_attention_on_image(
        self,
        attention_vector: np.ndarray,
        layer_id: str,
        head_id: str,
        level_id: str,
        image_id: str,
        filename: str = "attention_overlay.png",
        subdir: str = "plots",
        alpha: float = 0.5,
        cmap: str = "viridis",
    ) -> None:
        """Plot attention weights overlaid on the image patches.
        
        Assumes attention_vector is 1D array of size num_patches,
        patches are in row-major order, and form a square grid.
        """
        
        # Load image
        project_root = Path(__file__).resolve().parents[1]
        img_path = project_root / "data" / "vlm_levels" / level_id / "images" / f"{image_id}.png"
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            return
        
        img = mpimg.imread(img_path)
        
        # Assume square grid
        num_patches = len(attention_vector)
        side = int(np.sqrt(num_patches))
        if side * side != num_patches:
            print(f"Cannot reshape {num_patches} patches to square grid")
            return
        
        # Reshape attention to grid
        attention_grid = attention_vector.reshape(side, side)
        
        # Normalize attention
        attention_norm = (attention_grid - attention_grid.min()) / (attention_grid.max() - attention_grid.min() + 1e-8)
        attention_norm = attention_norm.astype(np.float32)  # Ensure float32 for zoom
        
        # Resize attention to image size
        img_height, img_width = img.shape[:2]
        attention_resized = np.kron(attention_norm, np.ones((img_height // side, img_width // side)))

        
        out_dir = self._subdir(subdir)
        out_path = out_dir / filename
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.imshow(attention_resized, cmap=cmap, alpha=alpha, origin='upper')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        plt.title(f"Attention on Image at Layer {layer_id} and Head {head_id}")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()