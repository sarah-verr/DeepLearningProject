import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image

from metrics import attention_center_of_mass

# Heatmap visualization tuning
HEATMAP_CMAP = "viridis"  # perceptually-uniform, colorblind-friendly
HEATMAP_TRANSPARENCY_THRESHOLD = 0.05  # normalized attention in [0,1] below this is transparent
HEATMAP_NONZERO_ALPHA = 0.55  # opacity for any non-zero attention

# Patch marker drawing (target highlight rectangles)
# If False, overlay heatmaps will not draw patch rectangles (cleaner overlays).
DRAW_PATCH_MARKERS_ON_OVERLAY = False


def _draw_com_marker(ax, heatmap_data: np.ndarray, patch_size: int) -> None:
    """Draw a Center-of-Mass marker for a single head heatmap.

    The CoM is computed in patch coordinates using attention_center_of_mass,
    then mapped to pixel space using patch_size.
    """

    if heatmap_data.ndim != 2:
        return

    grid_dim = heatmap_data.shape[0]
    try:
        row_com, col_com = attention_center_of_mass(
            heatmap_data.reshape(-1), grid_dim
        )
    except Exception:
        return

    if row_com is None or col_com is None:
        return

    # Map patch coordinates (row, col) to pixel center inside the image
    x = (col_com + 0.5) * patch_size
    y = (row_com + 0.5) * patch_size

    ax.scatter(
        [x],
        [y],
        marker="x",
        s=40,
        linewidths=1.5,
        color="white",
    )


def draw_target_highlights(ax, target_groups_meta, patch_size: int) -> None:
    if not target_groups_meta:
        return
    for group in target_groups_meta:
        color = group["color"]
        for patch in group["patches"]:
            row, col = patch["row"], patch["col"]
            x = col * patch_size
            y = row * patch_size
            rect = patches.Rectangle(
                (x, y),
                patch_size,
                patch_size,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)


def add_heatmap_colorbar(fig, axes, *, cmap: str = HEATMAP_CMAP, label: str = "Normalized attention (0=low, 1=high)"):
    """Add a single colorbar that explains the heatmap colors.

    Note: heatmaps are normalized per-map in overlay_heatmap, so the legend is 0..1.
    """
    sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    return cbar


def overlay_heatmap(ax, base_image, heatmap_data, title, target_groups_meta=None, patch_size: int = 14):
    if heatmap_data.max() > heatmap_data.min():
        norm_map = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    else:
        norm_map = heatmap_data

    attn_image = Image.fromarray((norm_map * 255).astype("uint8"))
    attn_image = attn_image.resize(base_image.size, resample=Image.NEAREST)

    ax.imshow(base_image)

    # Binary opacity: exactly-zero attention is transparent; any non-zero attention
    # uses a constant opacity so the underlying shapes remain visible.
    attn_resized = (np.asarray(attn_image).astype(np.float32) / 255.0)
    attn_resized = np.clip(attn_resized, 0.0, 1.0)
    cmap = plt.get_cmap(HEATMAP_CMAP)
    rgba = cmap(attn_resized)
    mask_resized = attn_resized > HEATMAP_TRANSPARENCY_THRESHOLD
    rgba[..., 3] = np.where(mask_resized, HEATMAP_NONZERO_ALPHA, 0.0)
    im = ax.imshow(rgba, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    if target_groups_meta and DRAW_PATCH_MARKERS_ON_OVERLAY:
        draw_target_highlights(ax, target_groups_meta, patch_size=patch_size)

    return im


def create_layer_grid_plot(
    image,
    heads_data,
    avg_data,
    layer_idx,
    target_groups_meta,
    patch_size: int = 14,
    source_token_text=None,
):
    num_heads = heads_data.shape[0]
    total_plots = num_heads + 2
    cols = 6
    rows = (total_plots // cols) + (1 if total_plots % cols != 0 else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows), constrained_layout=True)
    if source_token_text is not None:
        title = f"Layer {layer_idx} Attention | source={repr(source_token_text)}"
    else:
        title = f"Layer {layer_idx} Attention"
    fig.suptitle(title, fontsize=16, weight="bold")
    axes_flat = axes.flatten()

    axes_flat[0].imshow(image)
    axes_flat[0].set_title("Original Image", fontsize=10, weight="bold")
    axes_flat[0].axis("off")

    if target_groups_meta:
        draw_target_highlights(axes_flat[0], target_groups_meta, patch_size=patch_size)
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color=g["color"], lw=2, label=f"Obj{g['object_id']}: {g['color']} {g['shape']}")
            for g in target_groups_meta
        ]
        axes_flat[0].legend(handles=legend_elements, loc="upper right", fontsize="small")

    overlay_heatmap(axes_flat[1], image, avg_data, "AVERAGE (All Heads)", target_groups_meta, patch_size=patch_size)

    for spine in axes_flat[1].spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(2)

    for i in range(num_heads):
        if i + 2 < len(axes_flat):
            overlay_heatmap(
                axes_flat[i + 2],
                image,
                heads_data[i],
                f"Head {i}",
                target_groups_meta,
                patch_size=patch_size,
            )
            _draw_com_marker(axes_flat[i + 2], heads_data[i], patch_size)

    for i in range(num_heads + 2, len(axes_flat)):
        axes_flat[i].axis("off")

    add_heatmap_colorbar(fig, axes_flat[1:total_plots])
    return fig


def plot_attention_trends(group_scores_history, group_metadata, output_dir, question_idx, question_text, rel_group=None, rel_type=None):
    plt.figure(figsize=(12, 7))
    for group_idx, scores in group_scores_history.items():
        layers = range(len(scores))

        # Use metadata for color and label
        if group_idx < len(group_metadata):
            meta = group_metadata[group_idx]
            color = meta["color"]  # Real color from annotation
            label = f"Obj {meta['object_id']}: {meta['color']} {meta['shape']}"
        else:
            color = "gray"  # Fallback
            label = f"Group {group_idx}"

        plt.plot(layers, scores, marker="o", color=color, linewidth=2, label=label)

    # Question as title with wrapping
    wrapped_question = "\n".join([question_text[i : i + 80] for i in range(0, len(question_text), 80)])
    suffix = ""
    if rel_group:
        suffix += f" | group={rel_group}"
    if rel_type:
        suffix += f" | rel={rel_type}"
    plt.title(f"Q{question_idx}: {wrapped_question}{suffix}", fontsize=11, pad=20)
    plt.xlabel("Layer Index")
    plt.ylabel("Total Attention")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "attention_trend_analysis.png"), bbox_inches="tight", dpi=100)
    plt.close()


def plot_subject_object_attention(subject_scores, object_scores, output_dir, question_idx, question_text, subject_id, object_id, rel_group=None, rel_type=None):
    """Per-question plot of attention on subject vs object patches over layers."""
    if not subject_scores and not object_scores:
        return

    layers = range(len(subject_scores))
    plt.figure(figsize=(10, 5))
    plt.plot(layers, subject_scores, marker="o", linewidth=2, label=f"subject_id={subject_id}")
    plt.plot(layers, object_scores, marker="o", linewidth=2, label=f"object_id={object_id}")

    wrapped_question = "\n".join([question_text[i : i + 80] for i in range(0, len(question_text), 80)])
    suffix = ""
    if rel_group:
        suffix += f" | group={rel_group}"
    if rel_type:
        suffix += f" | rel={rel_type}"
    plt.title(f"Q{question_idx}: {wrapped_question}{suffix}", fontsize=10)
    plt.xlabel("Layer Index")
    plt.ylabel("Total Attention (avg over heads)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "subject_object_attention.png"), bbox_inches="tight", dpi=120)
    plt.close()


def plot_correct_vs_incorrect_trends(correct_runs, incorrect_runs, output_dir):
    """Plot mean attention trend for correct vs incorrect answers."""
    if not correct_runs and not incorrect_runs:
        print("No data to compare.")
        return

    plt.figure(figsize=(12, 6))

    layers = range(len(correct_runs[0])) if correct_runs else range(len(incorrect_runs[0]))

    def get_stats(runs_matrix):
        if not runs_matrix:
            return None, None
        arr = np.array(runs_matrix)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        return mean, std

    mean_corr, std_corr = get_stats(correct_runs)
    if mean_corr is not None:
        plt.plot(layers, mean_corr, color="green", linewidth=3, label=f"Correct (n={len(correct_runs)})")
        plt.fill_between(layers, mean_corr - std_corr, mean_corr + std_corr, color="green", alpha=0.2)

    mean_inc, std_inc = get_stats(incorrect_runs)
    if mean_inc is not None:
        plt.plot(layers, mean_inc, color="red", linewidth=3, label=f"Incorrect (n={len(incorrect_runs)})")
        plt.fill_between(layers, mean_inc - std_inc, mean_inc + std_inc, color="red", alpha=0.2)

    plt.title("Attention on All Targets: Correct vs Incorrect Answers")
    plt.xlabel("Layer Index (0=Shallow, 32=Deep)")
    plt.ylabel("Avg Total Attention on Targets")
    plt.legend()
    plt.grid(True, alpha=0.4)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "comparison_correct_vs_incorrect.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Comparison plot saved to: {save_path}")
    plt.close()


def plot_evaluation_results(results, output_dir, title_id):
    questions = [f"Q{r['id']}" for r in results]
    confidences = [r["confidence"] for r in results]
    colors = ["green" if r["is_correct"] else "red" for r in results]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(questions, confidences, color=colors, alpha=0.7)

    plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    plt.ylim(0, 1.1)
    plt.ylabel("Model Confidence")
    plt.title(
        f"Evaluation Results: {title_id}\nTotal Accuracy: {sum([r['is_correct'] for r in results])/len(results):.1%}"
    )

    for bar, result in zip(bars, results):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{result['prediction']}\n({result['gt']})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    from matplotlib.lines import Line2D

    custom_lines = [Line2D([0], [0], color="green", lw=4), Line2D([0], [0], color="red", lw=4)]
    plt.legend(custom_lines, ["Correct", "Incorrect"], loc="upper right")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "evaluation_summary.png"))
    plt.close()


def create_phrase_thirds_plot(image, maps_3, rel_phrase: str, patch_size: int = 14):
    """Phrase→image attention aggregated over early/mid/late layers."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    fig.suptitle(f'Phrase→Image Attention (Layer Thirds) | "{rel_phrase}"', fontsize=14, weight="bold")

    overlay_heatmap(axes[0], image, maps_3["early"], "Early (0–33%)", target_groups_meta=None, patch_size=patch_size)
    overlay_heatmap(axes[1], image, maps_3["mid"], "Mid (33–66%)", target_groups_meta=None, patch_size=patch_size)
    overlay_heatmap(axes[2], image, maps_3["late"], "Late (66–100%)", target_groups_meta=None, patch_size=patch_size)

    add_heatmap_colorbar(fig, axes)
    return fig


def create_decision_thirds_plot(image, maps_3, patch_size: int = 14, source_token_text=None):
    """Aggregated decision-token→image attention over early/mid/late layers."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    if source_token_text is not None:
        title = f"Decision→Image Attention (Layer Thirds) | source={repr(source_token_text)}"
    else:
        title = "Decision→Image Attention (Layer Thirds)"
    fig.suptitle(title, fontsize=14, weight="bold")

    overlay_heatmap(axes[0], image, maps_3["early"], "Early (0–33%)", target_groups_meta=None, patch_size=patch_size)
    overlay_heatmap(axes[1], image, maps_3["mid"], "Mid (33–66%)", target_groups_meta=None, patch_size=patch_size)
    overlay_heatmap(axes[2], image, maps_3["late"], "Late (66–100%)", target_groups_meta=None, patch_size=patch_size)

    add_heatmap_colorbar(fig, axes)
    return fig


def create_phrase_layer_grid_plot(image, heads_data, avg_data, layer_idx, rel_phrase: str, patch_size: int = 14):
    """Per-layer phrase→image attention maps (heads + average)."""
    num_heads = heads_data.shape[0]
    total_plots = num_heads + 2
    cols = 6
    rows = (total_plots // cols) + (1 if total_plots % cols != 0 else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows), constrained_layout=True)
    fig.suptitle(f'Layer {layer_idx} | Phrase→Image Attention | "{rel_phrase}"', fontsize=14, weight="bold")
    axes_flat = axes.flatten()

    axes_flat[0].imshow(image)
    axes_flat[0].set_title("Original Image", fontsize=10, weight="bold")
    axes_flat[0].axis("off")

    overlay_heatmap(axes_flat[1], image, avg_data, "AVERAGE (All Heads)", target_groups_meta=None, patch_size=patch_size)
    for spine in axes_flat[1].spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(2)

    for i in range(num_heads):
        if i + 2 < len(axes_flat):
            overlay_heatmap(
                axes_flat[i + 2],
                image,
                heads_data[i],
                f"Head {i}",
                target_groups_meta=None,
                patch_size=patch_size,
            )
            _draw_com_marker(axes_flat[i + 2], heads_data[i], patch_size)

    for i in range(num_heads + 2, len(axes_flat)):
        axes_flat[i].axis("off")

    add_heatmap_colorbar(fig, axes_flat[1:total_plots])
    return fig


def plot_head_layer_fraction_heatmaps(
    subject_grid: np.ndarray | None,
    object_grid: np.ndarray | None,
    output_dir: str,
    question_idx: int,
    question_text: str,
    subject_id=None,
    object_id=None,
    subject_shape: str | None = None,
    object_shape: str | None = None,
):
    """Plot #heads x #layers heatmaps of bbox attention fractions.

    Each cell (head, layer) encodes the fraction of attention mass
    that falls inside the subject/object bounding boxes.
    """

    def _has_signal(grid: np.ndarray | None) -> bool:
        return grid is not None and grid.size > 0 and not np.all(np.isnan(grid))

    has_subj = _has_signal(subject_grid)
    has_obj = _has_signal(object_grid)
    if not has_subj and not has_obj:
        return

    wrapped_question = "\n".join([question_text[i : i + 80] for i in range(0, len(question_text), 80)])
    title_suffix = []
    if subject_shape is not None:
        title_suffix.append(f"subject={subject_shape}")
    elif subject_id is not None:
        title_suffix.append(f"subject_id={subject_id}")
    if object_shape is not None:
        title_suffix.append(f"object={object_shape}")
    elif object_id is not None:
        title_suffix.append(f"object_id={object_id}")

    os.makedirs(output_dir, exist_ok=True)

    for grid, name, slug in (
        (subject_grid, "Subject", "subject"),
        (object_grid, "Object", "object"),
    ):
        if grid is None or not _has_signal(grid):
            continue

		# Wider figure to give more room for all layer tick labels
        fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
        fig.suptitle(
            f"Q{question_idx}: Fraction of attention placed on {name}, compared to the whole image, across layers\n"
            + wrapped_question
            + (" | " + ", ".join(title_suffix) if title_suffix else ""),
            fontsize=12,
            weight="bold",
        )

        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Head Index")
        ax.set_title(f"{name} bbox fraction (0–1)")

        n_layers = grid.shape[1]
        n_heads = grid.shape[0]

        ax.set_yticks(range(n_heads))
        ax.set_xticks(range(n_layers))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Fraction of attention in box", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        plt.savefig(
            os.path.join(output_dir, f"head_layer_bbox_fraction_heatmap_{slug}.png"),
            bbox_inches="tight",
            dpi=140,
        )
        plt.close(fig)
