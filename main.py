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

# --- Dependency Check ---
if importlib.util.find_spec("bitsandbytes") is None:
    print("Error: 'bitsandbytes' library is missing.")
    print("Please install it by running: pip install bitsandbytes accelerate")
    sys.exit(1)

# --- Configuration ---
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
BASE_OUTPUT_DIR = "vis_results"
BASE_DATA_PATH = f"/home/{os.environ['USER']}/DeepLearningProject/Synthetic-Data/vlm_levels"

# --- DYNAMIC TARGET GROUP EXTRACTION ---
def extract_target_groups_from_annotation(annotation_data, image_width=336, image_height=336):
    """
    Converts object bounding boxes to patch indices for a 24x24 grid.
    Returns target groups organized by object ID.
    """
    objects = annotation_data.get('objects', [])
    
    target_groups = []
    group_metadata = []
    
    patch_width = image_width / 24
    patch_height = image_height / 24
    
    for obj in objects:
        obj_id = obj['id']
        bbox = obj['bbox']  # [x_min, y_min, x_max, y_max]
        color = obj['color']
        shape = obj['shape']
        
        # Calculate which patches this object overlaps
        x_min, y_min, x_max, y_max = bbox
        
        # Convert to patch coordinates
        col_min = int(x_min / patch_width)
        col_max = int(x_max / patch_width)
        row_min = int(y_min / patch_height)
        row_max = int(y_max / patch_height)
        
        # Clamp to valid range
        col_min = max(0, min(23, col_min))
        col_max = max(0, min(23, col_max))
        row_min = max(0, min(23, row_min))
        row_max = max(0, min(23, row_max))
        
        # Collect all patch indices in this bounding box
        patch_indices = []
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                patch_idx = row * 24 + col
                patch_indices.append(patch_idx)
        
        if patch_indices:
            target_groups.append(patch_indices)
            group_metadata.append({
                'object_id': obj_id,
                'shape': shape,
                'color': color,
                'patch_count': len(patch_indices)
            })
    
    return target_groups, group_metadata

# --- HELPER: Visualization ---
def draw_target_highlights(ax, target_groups_meta):
    if not target_groups_meta: return
    PATCH_SIZE = 14
    for group in target_groups_meta:
        color = group['color']
        for patch in group['patches']:
            row, col = patch['row'], patch['col']
            x = col * PATCH_SIZE
            y = row * PATCH_SIZE
            rect = patches.Rectangle(
                (x, y), PATCH_SIZE, PATCH_SIZE, 
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)

def overlay_heatmap(ax, base_image, heatmap_data, title, target_groups_meta=None):
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
        draw_target_highlights(ax, target_groups_meta)

def create_layer_grid_plot(image, heads_data, avg_data, layer_idx, target_groups_meta):
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
        draw_target_highlights(axes_flat[0], target_groups_meta)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=g['color'], lw=2, label=f"Obj{g['object_id']}: {g['color']} {g['shape']}")
            for i, g in enumerate(target_groups_meta)
        ]
        axes_flat[0].legend(handles=legend_elements, loc='upper right', fontsize='small')

    overlay_heatmap(axes_flat[1], image, avg_data, "AVERAGE (All Heads)", target_groups_meta)
    
    for spine in axes_flat[1].spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(2)

    for i in range(num_heads):
        if i + 2 < len(axes_flat):
            overlay_heatmap(axes_flat[i+2], image, heads_data[i], f"Head {i}", target_groups_meta)

    for i in range(num_heads + 2, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig

# --- PLOT 1: PER-QUESTION TREND ---
def plot_attention_trends(group_scores_history, group_metadata, output_dir, question_idx, question_text):
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
    plt.title(f"Q{question_idx}: {wrapped_question}", fontsize=11, pad=20)
    plt.xlabel("Layer Index")
    plt.ylabel("Total Attention")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attention_trend_analysis.png"), bbox_inches='tight', dpi=100)
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Batch LLaVA Inference")
    parser.add_argument("--level", type=str, required=True, help="Level number (e.g., '1')")
    parser.add_argument("--id", type=str, required=True, help="File ID (e.g., '00000_b')")
    parser.add_argument("--num_questions", type=int, default=None, help="Number of questions to sample randomly (default: all)")
    
    # SPLIT ARGUMENTS
    parser.add_argument("--plot_attention", action="store_true", help="Save HEAVY layer-wise heatmaps (Slow)")
    parser.add_argument("--plot_trends", action="store_true", help="Save LIGHTWEIGHT line plots of group attention (Fast)")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Path Construction
    level_dir = f"level_{args.level}"
    image_path = os.path.join(BASE_DATA_PATH, level_dir, "images", f"{args.id}.png")
    json_path = os.path.join(BASE_DATA_PATH, level_dir, "ann", f"{args.id}.json")

    if not os.path.exists(image_path) or not os.path.exists(json_path):
        print(f"Error: Files not found at constructed paths.")
        return

    # Output Setup - Organized structure with timestamp for multiple trials
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    level_output = os.path.join(BASE_OUTPUT_DIR, f"level_{args.level}")
    image_output = os.path.join(level_output, f"{args.id}_{timestamp}")
    
    # Create subdirectories
    overview_dir = os.path.join(image_output, "overview")
    trends_dir = os.path.join(image_output, "trend_analysis")
    attention_dir = os.path.join(image_output, "full_attention")
    
    os.makedirs(overview_dir, exist_ok=True)
    if args.plot_trends:
        os.makedirs(trends_dir, exist_ok=True)
    if args.plot_attention:
        os.makedirs(attention_dir, exist_ok=True)
    
    print(f"Processing Level: {args.level} | ID: {args.id}")
    print(f"Output: {image_output}")
    
    # LOGIC: We need attention from model if EITHER flag is true
    REQUIRES_ATTENTION = args.plot_attention or args.plot_trends
    if REQUIRES_ATTENTION:
        print("NOTE: Attention extraction is ENABLED.")
        
        if args.plot_attention:
            print("      -> Generating full heatmaps (Expect slower performance)")
        if args.plot_trends:
            print("      -> Generating trend graphs")

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
    # Extract dynamic target groups from annotation
    TARGET_GROUPS, group_metadata = extract_target_groups_from_annotation(annotation_data)
    
    print(f"\nDynamically extracted {len(TARGET_GROUPS)} target groups:")
    for i, meta in enumerate(group_metadata):
        print(f"  Group {i}: Object {meta['object_id']} - {meta['color']} {meta['shape']} ({meta['patch_count']} patches)")
    
    # --- PREPARE GROUP METADATA FOR VISUALIZATION ---
    target_groups_meta = []
    if args.plot_attention:
        vis_image = image.resize((336, 336), resample=Image.BICUBIC)
        for i, (group_indices, meta) in enumerate(zip(TARGET_GROUPS, group_metadata)):
            # Use the actual object color from annotation
            object_color = meta['color']
            group_patches = []
            for idx in group_indices:
                row = idx // 24
                col = idx % 24
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
        question_text = qa_item['question']
        ground_truth = qa_item['answer'].lower() 
        prompt_text = f"USER: <image>\n{question_text}\nASSISTANT:"
        
        inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=50, 
                return_dict_in_generate=True,
                output_scores=True,
                output_attentions=REQUIRES_ATTENTION # Pass True if either flag is set
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
            "is_correct": is_correct
        })

        # --- ATTENTION PROCESSING ---
        if REQUIRES_ATTENTION:
            # Create question directory in appropriate location
            if args.plot_attention:
                q_dir = os.path.join(attention_dir, f"Q{i}")
                os.makedirs(q_dir, exist_ok=True)
            elif args.plot_trends:
                q_dir = os.path.join(trends_dir, f"Q{i}")
                os.makedirs(q_dir, exist_ok=True)

            # --- VRAM OPTIMIZATION START ---
            raw_layers_gpu = outputs.attentions[0]
            # Move to CPU immediately
            all_layers_data = [layer_tensor.detach().cpu() for layer_tensor in raw_layers_gpu]
            # Delete GPU tensors
            del outputs
            del raw_layers_gpu
            torch.cuda.empty_cache()
            # --- VRAM OPTIMIZATION END ---

            group_scores_layerwise = {g_idx: [] for g_idx in range(len(TARGET_GROUPS))}
            
            # This list will hold the sum of attention on ALL targets for each layer
            current_question_layer_totals = []

            start_idx = (inputs.input_ids[0] == model.config.image_token_index).nonzero(as_tuple=True)[0][0].item()
            end_idx = start_idx + 576

            # Iterate layers
            iterator = tqdm(enumerate(all_layers_data), total=len(all_layers_data), desc=f"  > Processing Layers", leave=False) if args.plot_attention else enumerate(all_layers_data)

            for layer_idx, layer_attn_tensor in iterator:
                try:
                    heads_raw = layer_attn_tensor[0, :, -1, :] 
                    
                    if end_idx > heads_raw.shape[1]:
                        image_heads_flat = heads_raw[:, -576:]
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

                    # 2. ONLY Plot Heavy Grids if asked
                    if args.plot_attention:
                        current_num_heads = image_heads_flat.shape[0]
                        heads_map_2d = image_heads_flat.view(current_num_heads, 24, 24).float().numpy()
                        avg_map_2d = heads_map_2d.mean(axis=0)

                        fig = create_layer_grid_plot(vis_image, heads_map_2d, avg_map_2d, layer_idx, target_groups_meta)
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
                plot_attention_trends(group_scores_layerwise, group_metadata, trend_q_dir, i, question_text)
            

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