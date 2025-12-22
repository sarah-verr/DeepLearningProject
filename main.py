import torch
import sys
import importlib.util
import os
import time
import json
import argparse
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
from types import SimpleNamespace

# --- Configuration ---
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
BASE_OUTPUT_DIR = "vis_results"
BASE_DATA_PATH = f"/home/{os.environ['USER']}/deep-learning/DeepLearningProject/Synthetic-Data/vlm_levels"

# ---------------------- Config Loader (YAML/JSON) ----------------------
def _load_config_file(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in {".yaml", ".yml"}:
        try:
            import yaml  # requires: pip install pyyaml
        except Exception as e:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Install with: pip install pyyaml"
            ) from e
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {}

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}

    raise ValueError(f"Unsupported config extension '{ext}'. Use .yaml/.yml or .json")

def _normalize_config(cfg: dict) -> dict:
    """
    Fills defaults and validates required fields.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping/object at the top level.")

    # required
    if "level" not in cfg or "id" not in cfg:
        raise ValueError("Config must include required keys: level, id")

    out = dict(cfg)

    # defaults (match your existing CLI defaults)
    out.setdefault("num_questions", None)
    out.setdefault("plot_attention", False)
    out.setdefault("plot_trends", False)
    out.setdefault("plot_relational_phrase_attention", False)

    # optional: if you later want simple vs detailed phrase plots
    out.setdefault("relational_phrase_attention_mode", "detailed")  # "simple"|"detailed"
    if out["relational_phrase_attention_mode"] not in {"simple", "detailed"}:
        raise ValueError("relational_phrase_attention_mode must be one of: simple, detailed")

    # optional overrides for paths/model
    out.setdefault("model_id", None)
    out.setdefault("base_output_dir", None)
    out.setdefault("base_data_path", None)

    # normalize types
    out["level"] = str(out["level"])
    out["id"] = str(out["id"])
    if out["num_questions"] is not None:
        out["num_questions"] = int(out["num_questions"])

    out["plot_attention"] = bool(out["plot_attention"])
    out["plot_trends"] = bool(out["plot_trends"])
    out["plot_relational_phrase_attention"] = bool(out["plot_relational_phrase_attention"])

    return out

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLaVA inference (config-driven)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/run_configs.yaml",
        help="Path to YAML/JSON config (default: configs/run_configs.yaml)",
    )
    return parser.parse_args()

# --- DYNAMIC TARGET GROUP EXTRACTION ---
def extract_target_groups_from_annotation(annotation_data, grid_dim: int):
    """Extract per-object patch indices directly from the JSON.

    Expected JSON schema (newer): objects[*].patch_indices (flat row-major indices).
    Returns:
      - target_groups: List[List[int]] in stable (sorted object_id) order
      - group_metadata: List[dict] aligned with target_groups
      - obj_id_to_patch_indices: Dict[int, List[int]]
    """
    objects = annotation_data.get('objects', [])

    obj_id_to_patch_indices = {}
    obj_id_to_meta = {}

    max_idx = grid_dim * grid_dim - 1
    for obj in objects:
        if 'id' not in obj:
            continue
        obj_id = int(obj['id'])
        color = obj.get('color', 'gray')
        shape = obj.get('shape', 'unknown')

        patch_indices = obj.get('patch_indices', [])
        if not isinstance(patch_indices, list) or len(patch_indices) == 0:
            continue

        # sanitize
        cleaned = sorted({int(i) for i in patch_indices if 0 <= int(i) <= max_idx})
        if not cleaned:
            continue

        obj_id_to_patch_indices[obj_id] = cleaned
        obj_id_to_meta[obj_id] = {
            'object_id': obj_id,
            'shape': shape,
            'color': color,
            'patch_count': len(cleaned)
        }

    target_groups = []
    group_metadata = []
    for obj_id in sorted(obj_id_to_patch_indices.keys()):
        target_groups.append(obj_id_to_patch_indices[obj_id])
        group_metadata.append(obj_id_to_meta[obj_id])

    return target_groups, group_metadata, obj_id_to_patch_indices

# --- HELPER: Visualization ---
def draw_target_highlights(ax, target_groups_meta, patch_size: int):
    if not target_groups_meta: return
    for group in target_groups_meta:
        color = group['color']
        for patch in group['patches']:
            row, col = patch['row'], patch['col']
            x = col * patch_size
            y = row * patch_size
            rect = patches.Rectangle(
                (x, y), patch_size, patch_size,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)

def overlay_heatmap(ax, base_image, heatmap_data, title, target_groups_meta=None, patch_size: int = 14):
    if heatmap_data.max() > heatmap_data.min():
        norm_map = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    else:
        norm_map = heatmap_data

    attn_image = Image.fromarray((norm_map * 255).astype('uint8'))
    attn_image = attn_image.resize(base_image.size, resample=Image.BICUBIC)

    ax.imshow(base_image)
    ax.imshow(attn_image, cmap='jet', alpha=0.6)
    ax.set_title(title, fontsize=8)
    ax.axis('off')
    if target_groups_meta:
        draw_target_highlights(ax, target_groups_meta, patch_size=patch_size)

def create_layer_grid_plot(image, heads_data, avg_data, layer_idx, target_groups_meta, patch_size: int = 14):
    num_heads = heads_data.shape[0]
    total_plots = num_heads + 2
    cols = 6
    rows = (total_plots // cols) + (1 if total_plots % cols != 0 else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    fig.suptitle(f"Layer {layer_idx} Attention", fontsize=16, weight='bold')
    axes_flat = axes.flatten()

    axes_flat[0].imshow(image)
    axes_flat[0].set_title("Original Image", fontsize=10, weight='bold')
    axes_flat[0].axis('off')

    if target_groups_meta:
        draw_target_highlights(axes_flat[0], target_groups_meta, patch_size=patch_size)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=g['color'], lw=2, label=f"Obj{g['object_id']}: {g['color']} {g['shape']}")
            for i, g in enumerate(target_groups_meta)
        ]
        axes_flat[0].legend(handles=legend_elements, loc='upper right', fontsize='small')

    overlay_heatmap(axes_flat[1], image, avg_data, "AVERAGE (All Heads)", target_groups_meta, patch_size=patch_size)

    for spine in axes_flat[1].spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(2)

    for i in range(num_heads):
        if i + 2 < len(axes_flat):
            overlay_heatmap(axes_flat[i+2], image, heads_data[i], f"Head {i}", target_groups_meta, patch_size=patch_size)

    for i in range(num_heads + 2, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig

# --- PLOT 1: PER-QUESTION TREND ---
def plot_attention_trends(group_scores_history, group_metadata, output_dir, question_idx, question_text, rel_group=None, rel_type=None):
    plt.figure(figsize=(12, 7))
    for group_idx, scores in group_scores_history.items():
        layers = range(len(scores))
        
        # Use metadata for color and label
        if group_idx < len(group_metadata):
            meta = group_metadata[group_idx]
            color = meta['color']  # Real color from annotation
            label = f"Obj {meta['object_id']}: {meta['color']} {meta['shape']}"
        else:
            color = 'gray'  # Fallback
            label = f"Group {group_idx}"
        
        plt.plot(layers, scores, marker='o', color=color, linewidth=2, label=label)
    
    # Question as title with wrapping
    wrapped_question = "\n".join([question_text[i:i+80] for i in range(0, len(question_text), 80)])
    suffix = ""
    if rel_group:
        suffix += f" | group={rel_group}"
    if rel_type:
        suffix += f" | rel={rel_type}"
    plt.title(f"Q{question_idx}: {wrapped_question}{suffix}", fontsize=11, pad=20)
    plt.xlabel("Layer Index")
    plt.ylabel("Total Attention")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attention_trend_analysis.png"), bbox_inches='tight', dpi=100)
    plt.close()

def plot_subject_object_attention(subject_scores, object_scores, output_dir, question_idx, question_text, subject_id, object_id, rel_group=None, rel_type=None):
    """Per-question plot of attention on subject vs object patches over layers."""
    if not subject_scores and not object_scores:
        return

    layers = range(len(subject_scores))
    plt.figure(figsize=(10, 5))
    plt.plot(layers, subject_scores, marker='o', linewidth=2, label=f"subject_id={subject_id}")
    plt.plot(layers, object_scores, marker='o', linewidth=2, label=f"object_id={object_id}")

    wrapped_question = "\n".join([question_text[i:i+80] for i in range(0, len(question_text), 80)])
    suffix = ""
    if rel_group:
        suffix += f" | group={rel_group}"
    if rel_type:
        suffix += f" | rel={rel_type}"
    plt.title(f"Q{question_idx}: {wrapped_question}{suffix}", fontsize=10)
    plt.xlabel("Layer Index")
    plt.ylabel("Total Attention (avg over heads)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "subject_object_attention.png"), bbox_inches='tight', dpi=120)
    plt.close()

# --- NEW PLOT: CORRECT VS INCORRECT COMPARISON ---
def plot_correct_vs_incorrect_trends(correct_runs, incorrect_runs, output_dir):
    """
    Plots the mean attention trend (sum of all target groups) for correct vs incorrect answers.
    Includes shaded error bands for standard deviation.
    """
    if not correct_runs and not incorrect_runs:
        print("No data to compare.")
        return

    plt.figure(figsize=(12, 6))
    
    # Determine x-axis based on available data
    layers = range(len(correct_runs[0])) if correct_runs else range(len(incorrect_runs[0]))

    # Helper to calculate mean and std deviation
    def get_stats(runs_matrix):
        if not runs_matrix: return None, None
        arr = np.array(runs_matrix) # Shape: [num_samples, num_layers]
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        return mean, std

    # Plot Correct Curve (Green)
    mean_corr, std_corr = get_stats(correct_runs)
    if mean_corr is not None:
        plt.plot(layers, mean_corr, color='green', linewidth=3, label=f"Correct (n={len(correct_runs)})")
        plt.fill_between(layers, mean_corr - std_corr, mean_corr + std_corr, color='green', alpha=0.2)

    # Plot Incorrect Curve (Red)
    mean_inc, std_inc = get_stats(incorrect_runs)
    if mean_inc is not None:
        plt.plot(layers, mean_inc, color='red', linewidth=3, label=f"Incorrect (n={len(incorrect_runs)})")
        plt.fill_between(layers, mean_inc - std_inc, mean_inc + std_inc, color='red', alpha=0.2)

    plt.title("Attention on All Targets: Correct vs Incorrect Answers")
    plt.xlabel("Layer Index (0=Shallow, 32=Deep)")
    plt.ylabel("Avg Total Attention on Targets")
    plt.legend()
    plt.grid(True, alpha=0.4)
    
    save_path = os.path.join(output_dir, "comparison_correct_vs_incorrect.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    plt.close()

# --- HELPER: Probability Extraction ---
def get_yes_no_probability(outputs, tokenizer, prompt_context="ASSISTANT:"):
    first_token_logits = outputs.scores[0][0]
    probs = torch.softmax(first_token_logits, dim=-1)

    yes_tokens = [
        tokenizer.encode(f"{prompt_context} Yes", add_special_tokens=False)[-1],
        tokenizer.encode(f"{prompt_context} yes", add_special_tokens=False)[-1]
    ]
    no_tokens = [
        tokenizer.encode(f"{prompt_context} No", add_special_tokens=False)[-1],
        tokenizer.encode(f"{prompt_context} no", add_special_tokens=False)[-1]
    ]

    prob_yes = sum([probs[t_id].item() for t_id in yes_tokens if t_id < len(probs)])
    prob_no = sum([probs[t_id].item() for t_id in no_tokens if t_id < len(probs)])

    total = prob_yes + prob_no + 1e-9
    norm_yes = prob_yes / total
    norm_no = prob_no / total

    prediction = "yes" if norm_yes > norm_no else "no"
    confidence = norm_yes if prediction == "yes" else norm_no
    
    return prediction, confidence, norm_yes, norm_no

def plot_evaluation_results(results, output_dir, title_id):
    questions = [f"Q{r['id']}" for r in results]
    confidences = [r['confidence'] for r in results]
    colors = ['green' if r['is_correct'] else 'red' for r in results]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(questions, confidences, color=colors, alpha=0.7)
    
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.ylim(0, 1.1)
    plt.ylabel("Model Confidence")
    plt.title(f"Evaluation Results: {title_id}\nTotal Accuracy: {sum([r['is_correct'] for r in results])/len(results):.1%}")
    
    for bar, result in zip(bars, results):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f"{result['prediction']}\n({result['gt']})",
                ha='center', va='bottom', fontsize=9)

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='green', lw=4),
                    Line2D([0], [0], color='red', lw=4)]
    plt.legend(custom_lines, ['Correct', 'Incorrect'], loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_summary.png"))
    plt.close()


def _find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    """Return start index of needle in haystack, or None."""
    if not needle or not haystack or len(needle) > len(haystack):
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None

def _locate_rel_phrase_token_positions(tokenizer, full_input_ids_1d, question_text: str, rel_phrase: str) -> list[int]:
    """
    Returns absolute token positions (indices into full_input_ids_1d) corresponding to rel_phrase tokens.
    Strategy:
      1) find question token span inside full prompt tokens
      2) find rel_phrase token span inside question tokens
      3) convert to absolute positions
    """
    if not question_text or not rel_phrase:
        return []

    full_ids = full_input_ids_1d.tolist() if hasattr(full_input_ids_1d, "tolist") else list(full_input_ids_1d)

    q_ids = tokenizer(question_text, add_special_tokens=False).input_ids
    q_start = _find_subsequence(full_ids, q_ids)

    # Tokenize phrase in two ways to handle whitespace-sensitive tokenization
    phrase_ids_a = tokenizer(rel_phrase, add_special_tokens=False).input_ids
    phrase_ids_b = tokenizer(" " + rel_phrase, add_special_tokens=False).input_ids

    if q_start is not None:
        # find phrase within question tokens
        p_start = _find_subsequence(q_ids, phrase_ids_a)
        p_len = len(phrase_ids_a)
        if p_start is None:
            p_start = _find_subsequence(q_ids, phrase_ids_b)
            p_len = len(phrase_ids_b)

        if p_start is None:
            return []

        abs_start = q_start + p_start
        return list(range(abs_start, abs_start + p_len))

    # Fallback: search in the full prompt directly
    p_start = _find_subsequence(full_ids, phrase_ids_a)
    p_len = len(phrase_ids_a)
    if p_start is None:
        p_start = _find_subsequence(full_ids, phrase_ids_b)
        p_len = len(phrase_ids_b)
    if p_start is None:
        return []
    return list(range(p_start, p_start + p_len))

def create_phrase_thirds_plot(image, maps_3, rel_phrase: str, patch_size: int = 14):
    """
    maps_3: dict with keys {"early","mid","late"} each a [grid_dim, grid_dim] numpy array
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Phrase→Image Attention (Layer Thirds) | "{rel_phrase}"', fontsize=14, weight="bold")

    overlay_heatmap(axes[0], image, maps_3["early"], "Early (0–33%)", target_groups_meta=None, patch_size=patch_size)
    overlay_heatmap(axes[1], image, maps_3["mid"],   "Mid (33–66%)", target_groups_meta=None, patch_size=patch_size)
    overlay_heatmap(axes[2], image, maps_3["late"],  "Late (66–100%)", target_groups_meta=None, patch_size=patch_size)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    return fig

def create_phrase_layer_grid_plot(image, heads_data, avg_data, layer_idx, rel_phrase: str, patch_size: int = 14):
    """
    Same layout as create_layer_grid_plot, but for phrase->image attention maps.
    heads_data: [num_heads, grid_dim, grid_dim]
    avg_data:   [grid_dim, grid_dim]
    """
    num_heads = heads_data.shape[0]
    total_plots = num_heads + 2
    cols = 6
    rows = (total_plots // cols) + (1 if total_plots % cols != 0 else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
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
            overlay_heatmap(axes_flat[i + 2], image, heads_data[i], f"Head {i}", target_groups_meta=None, patch_size=patch_size)

    for i in range(num_heads + 2, len(axes_flat)):
        axes_flat[i].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def main():
    cli = parse_arguments()

    cfg_raw = _load_config_file(cli.config)
    cfg = _normalize_config(cfg_raw)
    args = SimpleNamespace(**cfg)  # preserves your existing args.* usage

    # Allow config to override globals (optional)
    global MODEL_ID, BASE_OUTPUT_DIR, BASE_DATA_PATH
    if args.model_id:
        MODEL_ID = args.model_id
    if args.base_output_dir:
        BASE_OUTPUT_DIR = args.base_output_dir
    if args.base_data_path:
        BASE_DATA_PATH = args.base_data_path

    # Path Construction
    level_dir = f"level_{args.level}"
    image_path = os.path.join(BASE_DATA_PATH, level_dir, "images", f"{args.id}.png")
    json_path = os.path.join(BASE_DATA_PATH, level_dir, "ann", f"{args.id}.json")

    if not os.path.exists(image_path) or not os.path.exists(json_path):
        print(f"Error: Files not found at constructed paths.")
        print(f"  image: {image_path}")
        print(f"  json : {json_path}")
        return

    # Output Setup - Organized structure with timestamp for multiple trials
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    level_output = os.path.join(BASE_OUTPUT_DIR, f"level_{args.level}")
    image_output = os.path.join(level_output, f"{args.id}_{timestamp}")

    # Create subdirectories
    overview_dir = os.path.join(image_output, "overview")
    trends_dir = os.path.join(image_output, "trend_analysis")
    attention_dir = os.path.join(image_output, "full_attention")
    phrase_attn_dir = os.path.join(image_output, "phrase_attention")

    os.makedirs(overview_dir, exist_ok=True)
    if args.plot_trends:
        os.makedirs(trends_dir, exist_ok=True)
    if args.plot_attention:
        os.makedirs(attention_dir, exist_ok=True)
    if args.plot_relational_phrase_attention:
        os.makedirs(phrase_attn_dir, exist_ok=True)

    # LOGIC: We need attention from model if ANY heavy/light attention plot is requested
    REQUIRES_ATTENTION = args.plot_attention or args.plot_trends or args.plot_relational_phrase_attention
    if REQUIRES_ATTENTION:
        print("NOTE: Attention extraction is ENABLED.")
        if args.plot_attention:
            print("      -> Generating full heatmaps (Expect slower performance)")
        if args.plot_trends:
            print("      -> Generating trend graphs")
        if args.plot_relational_phrase_attention:
            print(f"      -> Phrase→image attention enabled (mode={args.relational_phrase_attention_mode})")

    # Load Model
    print(f"Loading model: {MODEL_ID}...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto", attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Data Prep
    image = Image.open(image_path).convert('RGB')

    with open(json_path, 'r') as f:
        annotation_data = json.load(f)
        full_qa_list = annotation_data.get('qa', [])

    meta = annotation_data.get('meta', {}) if isinstance(annotation_data.get('meta', {}), dict) else {}
    img_size = int(meta.get('img_size', 336))
    patch_size = int(meta.get('patch', 14))
    grid_dim = int(meta.get('grid_dim', 24))
    num_patches = grid_dim * grid_dim

    # Extract per-object patch indices directly from JSON
    TARGET_GROUPS, group_metadata, obj_id_to_patch_indices = extract_target_groups_from_annotation(annotation_data, grid_dim=grid_dim)
    
    print(f"\nDynamically extracted {len(TARGET_GROUPS)} target groups:")
    for i, meta in enumerate(group_metadata):
        print(f"  Group {i}: Object {meta['object_id']} - {meta['color']} {meta['shape']} ({meta['patch_count']} patches)")
    
    # --- PREPARE GROUP METADATA FOR VISUALIZATION ---
    target_groups_meta = []
    if args.plot_attention:
        vis_image = image.resize((img_size, img_size), resample=Image.BICUBIC)
        for i, (group_indices, meta) in enumerate(zip(TARGET_GROUPS, group_metadata)):
            # Use the actual object color from annotation
            object_color = meta['color']
            group_patches = []
            for idx in group_indices:
                row = idx // grid_dim
                col = idx % grid_dim
                group_patches.append({'row': int(row), 'col': int(col), 'idx': idx})
            
            target_groups_meta.append({
                'group_id': i,
                'color': object_color,
                'patches': group_patches,
                'shape': meta['shape'],
                'object_id': meta['object_id']
            })

    # Question sampling
    if args.num_questions is not None and args.num_questions > 0:
        if args.num_questions < len(full_qa_list):
            print(f"Randomly sampling {args.num_questions} questions from {len(full_qa_list)} total available.")
            qa_list = random.sample(full_qa_list, args.num_questions)
        else:
            qa_list = full_qa_list
    else:
        qa_list = full_qa_list

    results_summary = []
    
    # --- NEW: Accumulators for Correct vs Incorrect Analysis ---
    correct_runs_layerwise = []   # Will store lists of [sum_layer_0, sum_layer_1, ...]
    incorrect_runs_layerwise = [] 

    print("\nStarting Inference...")
    print(f"{'Idx':<4} | {'Correct':<7} | {'Pred':<5} | {'GT':<5} | {'Conf':<6} | Question")
    print("-" * 80)

    for i, qa_item in tqdm(enumerate(qa_list), total=len(qa_list), desc="Total Progress"):
        question_text = qa_item["question"]
        ground_truth = qa_item["answer"].lower()
        subject_id = qa_item.get("subject_id", None)
        object_id = qa_item.get("object_id", None)
        rel_type = qa_item.get("rel_type", None)
        rel_group = qa_item.get("rel_group", None)
        rel_phrase = qa_item.get("rel_phrase", None)  # NEW

        prompt_text = f"USER: <image>\n{question_text}\nASSISTANT:"
        inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                return_dict_in_generate=True,
                output_scores=True,
                output_attentions=REQUIRES_ATTENTION,
            )

        # Eval
        prediction, confidence, p_yes, p_no = get_yes_no_probability(outputs, processor.tokenizer)
        is_correct = (prediction == ground_truth)
        status_icon = "✅" if is_correct else "❌"
        
        tqdm.write(f"{i:<4} | {status_icon:<7} | {prediction:<5} | {ground_truth:<5} | {confidence:.2f}   | {question_text}")

        results_summary.append({
            "id": i,
            "question": question_text,
            "gt": ground_truth,
            "prediction": prediction,
            "confidence": float(confidence),
            "is_correct": is_correct,
            "subject_id": subject_id,
            "object_id": object_id,
            "rel_type": rel_type,
            "rel_group": rel_group,
        })

        # --- ATTENTION PROCESSING ---
        if REQUIRES_ATTENTION:
            # Choose where to store per-question artifacts
            if args.plot_attention:
                q_dir = os.path.join(attention_dir, f"Q{i}")
            elif args.plot_trends:
                q_dir = os.path.join(trends_dir, f"Q{i}")
            else:
                q_dir = os.path.join(overview_dir, f"Q{i}")
            os.makedirs(q_dir, exist_ok=True)

            # --- VRAM OPTIMIZATION START ---
            raw_layers_gpu = outputs.attentions[0]  # first generation step attentions
            all_layers_data = [layer_tensor.detach().cpu() for layer_tensor in raw_layers_gpu]
            del outputs
            del raw_layers_gpu
            torch.cuda.empty_cache()
            # --- VRAM OPTIMIZATION END ---

            group_scores_layerwise = {g_idx: [] for g_idx in range(len(TARGET_GROUPS))}
            
            # This list will hold the sum of attention on ALL targets for each layer
            current_question_layer_totals = []

            # Per-question subject/object curves (dynamic targets from JSON)
            subject_layer_scores = []
            object_layer_scores = []
            subj_patch_indices = obj_id_to_patch_indices.get(int(subject_id), []) if subject_id is not None else []
            obj_patch_indices = obj_id_to_patch_indices.get(int(object_id), []) if object_id is not None else []

            start_idx = (inputs.input_ids[0] == model.config.image_token_index).nonzero(as_tuple=True)[0][0].item()
            end_idx = start_idx + num_patches

            if args.plot_relational_phrase_attention and rel_phrase:
                phrase_q_dir = os.path.join(phrase_attn_dir, f"Q{i}")
                os.makedirs(phrase_q_dir, exist_ok=True)

                phrase_positions = _locate_rel_phrase_token_positions(
                    processor.tokenizer,
                    inputs.input_ids[0],
                    question_text=question_text,
                    rel_phrase=rel_phrase,
                )
                if not phrase_positions:
                    tqdm.write(
                        f'  [Warning] Could not locate rel_phrase tokens for Q{i}: "{rel_phrase}". Skipping phrase plot.'
                    )
                else:
                    vis_image_phrase = image.resize((img_size, img_size), resample=Image.BICUBIC)

                    # Collect per-layer avg maps for the "simple" thirds aggregation
                    per_layer_avg_maps: list[np.ndarray] = []

                    for layer_idx, layer_attn_tensor in enumerate(all_layers_data):
                        try:
                            layer = layer_attn_tensor[0]  # [heads, tgt_len, src_len]

                            tgt_len = layer.shape[1]
                            src_len = layer.shape[2]
                            phrase_pos_in_range = [p for p in phrase_positions if 0 <= p < tgt_len]
                            if not phrase_pos_in_range:
                                continue

                            # Slice keys to image tokens
                            if end_idx > src_len:
                                img_key_slice = slice(src_len - num_patches, src_len)
                            else:
                                img_key_slice = slice(start_idx, end_idx)

                            # phrase_to_img: [heads, phrase_len, num_patches]
                            phrase_to_img = layer[:, phrase_pos_in_range, img_key_slice]

                            # heads_flat: [heads, num_patches] (avg over phrase tokens)
                            heads_flat = phrase_to_img.mean(dim=1).float().numpy()

                            # heads_map_2d: [heads, grid_dim, grid_dim]
                            heads_map_2d = heads_flat.reshape(heads_flat.shape[0], grid_dim, grid_dim)

                            # avg_map_2d: [grid_dim, grid_dim] (avg over heads)
                            avg_map_2d = heads_map_2d.mean(axis=0)

                            # store for thirds plot
                            per_layer_avg_maps.append(avg_map_2d)

                            # DETAILED mode: save per-layer grids
                            if args.relational_phrase_attention_mode == "detailed":
                                fig = create_phrase_layer_grid_plot(
                                    vis_image_phrase,
                                    heads_map_2d,
                                    avg_map_2d,
                                    layer_idx=layer_idx,
                                    rel_phrase=rel_phrase,
                                    patch_size=patch_size,
                                )
                                plt.savefig(
                                    os.path.join(phrase_q_dir, f"layer_{layer_idx:02d}.png"),
                                    bbox_inches="tight",
                                )
                                plt.close(fig)

                        except Exception as e:
                            tqdm.write(f"  [Warning] Phrase-attn error layer {layer_idx}: {e}")

                    # SIMPLE mode: one plot aggregated into early/mid/late thirds
                    if args.relational_phrase_attention_mode == "simple":
                        if not per_layer_avg_maps:
                            tqdm.write(f"  [Warning] No per-layer phrase maps collected for Q{i}.")
                        else:
                            L = len(per_layer_avg_maps)
                            a = L // 3
                            b = (2 * L) // 3

                            early_maps = per_layer_avg_maps[:a] if a > 0 else per_layer_avg_maps[:1]
                            mid_maps = per_layer_avg_maps[a:b] if b > a else per_layer_avg_maps[a : a + 1]
                            late_maps = per_layer_avg_maps[b:] if b < L else per_layer_avg_maps[-1:]

                            maps_3 = {
                                "early": np.mean(np.stack(early_maps, axis=0), axis=0),
                                "mid": np.mean(np.stack(mid_maps, axis=0), axis=0),
                                "late": np.mean(np.stack(late_maps, axis=0), axis=0),
                            }

                            fig3 = create_phrase_thirds_plot(
                                vis_image_phrase,
                                maps_3,
                                rel_phrase=rel_phrase,
                                patch_size=patch_size,
                            )
                            plt.savefig(
                                os.path.join(phrase_q_dir, "phrase_attention_thirds.png"),
                                bbox_inches="tight",
                                dpi=140,
                            )
                            plt.close(fig3)

            # Iterate layers
            iterator = tqdm(enumerate(all_layers_data), total=len(all_layers_data), desc=f"  > Processing Layers", leave=False) if args.plot_attention else enumerate(all_layers_data)

            for layer_idx, layer_attn_tensor in iterator:
                try:
                    heads_raw = layer_attn_tensor[0, :, -1, :] 
                    
                    if end_idx > heads_raw.shape[1]:
                        image_heads_flat = heads_raw[:, -num_patches:]
                    else:
                        image_heads_flat = heads_raw[:, start_idx : end_idx]
                    
                    # 1. ALWAYS Calculate Trends
                    avg_attention_flat = image_heads_flat.mean(dim=0).numpy()
                    
                    layer_total_target_attn = 0.0

                    for g_idx, g_indices in enumerate(TARGET_GROUPS):
                        group_sum = 0.0
                        for pid in g_indices:
                            val = avg_attention_flat[pid] if pid < len(avg_attention_flat) else 0.0
                            group_sum += float(val)
                        
                        group_scores_layerwise[g_idx].append(group_sum)
                        layer_total_target_attn += group_sum
                    
                    # Store the sum of ALL groups for this layer
                    current_question_layer_totals.append(layer_total_target_attn)

                    # Per-question: subject vs object attention
                    subj_sum = 0.0
                    for pid in subj_patch_indices:
                        val = avg_attention_flat[pid] if pid < len(avg_attention_flat) else 0.0
                        subj_sum += float(val)

                    obj_sum = 0.0
                    for pid in obj_patch_indices:
                        val = avg_attention_flat[pid] if pid < len(avg_attention_flat) else 0.0
                        obj_sum += float(val)

                    subject_layer_scores.append(subj_sum)
                    object_layer_scores.append(obj_sum)

                    # 2. ONLY Plot Heavy Grids if asked
                    if args.plot_attention:
                        current_num_heads = image_heads_flat.shape[0]
                        heads_map_2d = image_heads_flat.view(current_num_heads, grid_dim, grid_dim).float().numpy()
                        avg_map_2d = heads_map_2d.mean(axis=0)

                        fig = create_layer_grid_plot(vis_image, heads_map_2d, avg_map_2d, layer_idx, target_groups_meta, patch_size=patch_size)
                        plt.savefig(os.path.join(q_dir, f"layer_{layer_idx:02d}.png"), bbox_inches='tight')
                        plt.close(fig)

                except Exception as e:
                    tqdm.write(f"  [Warning] Error layer {layer_idx}: {e}")
            
            # --- ACCUMULATE DATA FOR COMPARISON PLOT ---
            if is_correct:
                correct_runs_layerwise.append(current_question_layer_totals)
            else:
                incorrect_runs_layerwise.append(current_question_layer_totals)

            # 3. SAVE INDIVIDUAL TREND PLOT (If requested)
            if args.plot_trends:
                trend_q_dir = os.path.join(trends_dir, f"Q{i}")
                os.makedirs(trend_q_dir, exist_ok=True)
                plot_attention_trends(group_scores_layerwise, group_metadata, trend_q_dir, i, question_text, rel_group=rel_group, rel_type=rel_type)

            # 4. SAVE per-question subject vs object plot (always when attention is computed)
            plot_subject_object_attention(
                subject_layer_scores,
                object_layer_scores,
                q_dir,
                i,
                question_text,
                subject_id,
                object_id,
                rel_group=rel_group,
                rel_type=rel_type,
            )
            

    # --- SAVE SUMMARY ---
    final_metadata = {
        "timestamp": timestamp,
        "level": args.level,
        "id": args.id,
        "num_objects": len(TARGET_GROUPS),
        "objects": group_metadata,
        "results": results_summary,
        "accuracy": sum(r['is_correct'] for r in results_summary) / len(results_summary) if results_summary else 0
    }
    
    with open(os.path.join(overview_dir, "results.json"), "w") as f:
        json.dump(final_metadata, f, indent=4)

    if results_summary:
        plot_evaluation_results(results_summary, overview_dir, f"{args.id} (Level {args.level})")
    
    # --- GENERATE THE COMPARISON PLOT ---
    if args.plot_trends:
        plot_correct_vs_incorrect_trends(correct_runs_layerwise, incorrect_runs_layerwise, overview_dir)
        
    print(f"\nCompleted. Results saved to: {os.path.abspath(image_output)}")

if __name__ == "__main__":
    main()