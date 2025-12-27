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
from probing import (
    compute_head_probing_scores,
    compute_layer_probing_scores,
    plot_head_probing_heatmap,
    plot_layer_probing_scores
)

# --- Dependency Check ---
if importlib.util.find_spec("bitsandbytes") is None:
    print("Error: 'bitsandbytes' library is missing.")
    print("Please install it by running: pip install bitsandbytes accelerate")
    sys.exit(1)

# --- Configuration ---
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
BASE_OUTPUT_DIR = "vis_results"
BASE_DATA_PATH = "/home/tenkhtuvshin/DeepLearningProject/Synthetic-Data/vlm_levels"

# --- TARGET GROUP CONFIGURATION ---
TARGET_GROUPS = [
    [475, 476, 477, 499, 500, 501, 523, 524, 525],  # Group 0: cyan star
    [33, 34, 35, 57, 58, 59, 81, 82, 83],  # Group 1: green square
    [248, 249, 250, 272, 273, 274, 296, 297, 298],  # Group 2: purple star
    [509, 510, 511, 533, 534, 535, 557, 558, 559],  # Group 3: pink triangle
]
GROUP_COLORS = ['cyan', 'magenta', 'yellow', 'lime']

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
            Line2D([0], [0], color=g['color'], lw=2, label=f"Group {i}") 
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
def plot_attention_trends(group_scores_history, output_dir, question_idx):
    plt.figure(figsize=(10, 6))
    for group_idx, scores in group_scores_history.items():
        layers = range(len(scores))
        color = GROUP_COLORS[group_idx % len(GROUP_COLORS)]
        plt.plot(layers, scores, marker='o', color=color, linewidth=2, label=f"Group {group_idx}")
    plt.title(f"Q{question_idx}: Attention Flow on Target Groups")
    plt.xlabel("Layer Index")
    plt.ylabel("Total Attention")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "attention_trend_analysis.png"), bbox_inches='tight')
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
    
    # PROBING ARGUMENTS
    parser.add_argument("--probe_heads", action="store_true", help="Compute head-level probing scores (mechanistic analysis)")
    parser.add_argument("--probe_layers", action="store_true", help="Compute layer-level probing scores (mechanistic analysis)")
    
    # DEBUG PROBING ARGUMENTS (for testing single head/layer)
    parser.add_argument("--probe_single_head", type=int, nargs=2, metavar=('LAYER', 'HEAD'), 
                        help="Probe a single head only (e.g., --probe_single_head 0 5 for layer 0, head 5)")
    parser.add_argument("--probe_single_layer", type=int, metavar='LAYER',
                        help="Probe a single layer only (all heads in that layer)")
    
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

    # Output Setup
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_folder_name = f"{timestamp}_{args.id}"
    current_output_dir = os.path.join(BASE_OUTPUT_DIR, output_folder_name)
    os.makedirs(current_output_dir, exist_ok=True)
    
    print(f"Processing Level: {args.level} | ID: {args.id}")
    print(f"Output: {current_output_dir}")
    
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
    
    # --- PREPARE GROUP METADATA ---
    target_groups_meta = []
    image_attention_data = []  

    # Only need to prepare this if we are plotting visual grids
    if args.plot_attention:
        vis_image = image.resize((336, 336), resample=Image.BICUBIC)
        for i, group_indices in enumerate(TARGET_GROUPS):
            group_color = GROUP_COLORS[i % len(GROUP_COLORS)]
            group_patches = []
            for idx in group_indices:
                row = idx // 24
                col = idx % 24
                group_patches.append({'row': int(row), 'col': int(col), 'idx': idx})
            
            target_groups_meta.append({
                'group_id': i,
                'color': group_color,
                'patches': group_patches
            })

    with open(json_path, 'r') as f:
        full_qa_list = json.load(f).get('qa', [])

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
            q_safe_text = "".join(x for x in question_text[:20] if x.isalnum() or x in " _-").strip()
            q_dir = os.path.join(current_output_dir, f"Q{i}_{q_safe_text}")
            os.makedirs(q_dir, exist_ok=True)

            # Initialize variables before try block
            current_question_image_mass = []
            current_question_layer_totals = []
            group_scores_layerwise = {g_idx: [] for g_idx in range(len(TARGET_GROUPS))}

            try:
                # Check if attentions are available
                if outputs.attentions is None or len(outputs.attentions) == 0:
                    tqdm.write(f"  [Warning] No attentions returned for question {i}. output_attentions may not be working.")
                    image_attention_data.append({
                        "question_idx": i,
                        "question_text": question_text,
                        "layers": []
                    })
                else:
                    # --- VRAM OPTIMIZATION START ---
                    raw_layers_gpu = outputs.attentions[0]
                    # Move to CPU immediately
                    all_layers_data = [layer_tensor.detach().cpu() for layer_tensor in raw_layers_gpu]
                    # Delete GPU tensors
                    del outputs
                    del raw_layers_gpu
                    torch.cuda.empty_cache()
                    # --- VRAM OPTIMIZATION END ---

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

                            image_attention_mass = float(avg_attention_flat.sum())
                            total_seq_attention = float(heads_raw.mean(dim=0).sum())
                            image_attention_percentage = (image_attention_mass / total_seq_attention * 100) if total_seq_attention > 0 else 0.0

                            current_question_image_mass.append({
                                "layer": layer_idx,
                                "image_mass": image_attention_mass,
                                "image_percentage": image_attention_percentage,
                                "total_seq_mass": total_seq_attention
                            })

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
                            tqdm.write(f"  [Warning] Error processing layer {layer_idx}: {e}")
                    
                    # --- ACCUMULATE DATA FOR COMPARISON PLOT ---
                    if is_correct:
                        correct_runs_layerwise.append(current_question_layer_totals)
                    else:
                        incorrect_runs_layerwise.append(current_question_layer_totals)

                    # 3. SAVE INDIVIDUAL TREND PLOT (If requested)
                    if args.plot_trends:
                        plot_attention_trends(group_scores_layerwise, q_dir, i)
                    
                    image_attention_data.append({
                        "question_idx": i,
                        "question_text": question_text,
                        "layers": current_question_image_mass
                    })
            except Exception as e:
                tqdm.write(f"  [Error] Failed to extract attention for question {i}: {e}")
                import traceback
                tqdm.write(f"  [Error] Traceback: {traceback.format_exc()}")
                # Still append empty data so we know this question was processed
                image_attention_data.append({
                    "question_idx": i,
                    "question_text": question_text,
                    "layers": []
                })
            

    # --- SAVE SUMMARY ---
    final_metadata = {
        "timestamp": timestamp,
        "level": args.level,
        "id": args.id,
        "results": results_summary,
        "accuracy": sum(r['is_correct'] for r in results_summary) / len(results_summary) if results_summary else 0,
        "image_attention_mass": image_attention_data  

    }
    
    with open(os.path.join(current_output_dir, "batch_results.json"), "w") as f:
        json.dump(final_metadata, f, indent=4)

    if results_summary:
        plot_evaluation_results(results_summary, current_output_dir, f"{args.id} (Level {args.level})")
    
    # --- GENERATE THE COMPARISON PLOT ---
    if args.plot_trends:
        plot_correct_vs_incorrect_trends(correct_runs_layerwise, incorrect_runs_layerwise, current_output_dir)
    
    # --- PROBING ANALYSIS ---
    if args.probe_layers:
        print("\n" + "="*60)
        print("Starting Layer-Level Probing Analysis...")
        print("="*60)
        print("Probing residual stream (4096 dims) at generation-step-1 position")
        layer_probing_scores = compute_layer_probing_scores(
            model, processor, image, qa_list, results_summary, debug=True
        )
        plot_layer_probing_scores(
            layer_probing_scores, current_output_dir, f"{args.id} (Level {args.level})"
        )
        # Save probing scores to metadata
        final_metadata["layer_probing_scores"] = layer_probing_scores
        # Re-save JSON with probing scores
        with open(os.path.join(current_output_dir, "batch_results.json"), "w") as f:
            json.dump(final_metadata, f, indent=4)
        print("Layer-level probing completed.")
    
    if args.probe_heads or args.probe_single_head is not None or args.probe_single_layer is not None:
        print("\n" + "="*60)
        print("Starting Head-Level Probing Analysis...")
        print("="*60)
        
        # Determine if we're probing a single head/layer
        single_layer = None
        single_head = None
        if args.probe_single_head is not None:
            single_layer, single_head = args.probe_single_head
            print(f"Probing single head: Layer {single_layer}, Head {single_head}")
        elif args.probe_single_layer is not None:
            single_layer = args.probe_single_layer
            print(f"Probing single layer: Layer {single_layer} (all heads)")
        else:
            print("NOTE: This may take a while as it probes each head individually.")
        
        # Enable debug mode for single head probing
        enable_debug = (single_layer is not None and single_head is not None)
        head_probing_scores = compute_head_probing_scores(
            model, processor, image, qa_list, results_summary, 
            debug=enable_debug,
            single_layer=single_layer,
            single_head=single_head
        )
        
        # Only plot heatmap if we probed multiple heads
        if single_layer is None or single_head is None:
            plot_head_probing_heatmap(
                head_probing_scores, current_output_dir, f"{args.id} (Level {args.level})"
            )
        
        # Save probing scores to metadata
        final_metadata["head_probing_scores"] = {f"L{k[0]}_H{k[1]}": v for k, v in head_probing_scores.items()}
        # Re-save JSON with probing scores
        with open(os.path.join(current_output_dir, "batch_results.json"), "w") as f:
            json.dump(final_metadata, f, indent=4)
        print("Head-level probing completed.")
        
    print(f"\nCompleted. Results saved to: {os.path.abspath(current_output_dir)}")

if __name__ == "__main__":
    main()