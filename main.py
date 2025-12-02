import torch
import sys
import importlib.util
import os
import shutil
import time
import json  # <--- NEW IMPORT
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# --- Dependency Check ---
if importlib.util.find_spec("bitsandbytes") is None:
    print("Error: 'bitsandbytes' library is missing.")
    print("Please install it by running: pip install bitsandbytes accelerate")
    sys.exit(1)

from transformers import AutoProcessor, LlavaForConditionalGeneration

# --- Configuration ---
MODEL_ID = "llava-hf/llava-1.5-13b-hf"
IMAGE_PATH = "sample.png"

# Hardcoded target indices
TARGET_PATCHES = [203, 204] 

QUESTION = "Is the blue triangle to the right of the red triangle?"
PROMPT_TEXT = f"USER: <image>\n{QUESTION}\nASSISTANT:"
BASE_OUTPUT_DIR = "vis_results"

# --- HELPER: Plotting ---
def draw_target_highlights(ax, target_list):
    if not target_list: return
    PATCH_SIZE = 14
    for t in target_list:
        row, col = t['row'], t['col']
        x = col * PATCH_SIZE
        y = row * PATCH_SIZE
        rect = patches.Rectangle((x, y), PATCH_SIZE, PATCH_SIZE, linewidth=2, edgecolor='cyan', facecolor='none')
        ax.add_patch(rect)

def overlay_heatmap(ax, base_image, heatmap_data, title, target_indices=None):
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
    if target_indices:
        draw_target_highlights(ax, target_indices)

def create_layer_grid_plot(image, heads_data, avg_data, layer_idx, target_indices):
    # Dynamic grid calculation
    num_heads = heads_data.shape[0]
    total_plots = num_heads + 2 
    cols = 6
    rows = (total_plots // cols) + (1 if total_plots % cols != 0 else 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    fig.suptitle(f"Layer {layer_idx} Attention (Squares = Hardcoded Targets {TARGET_PATCHES})", fontsize=16, weight='bold')
    axes_flat = axes.flatten()

    axes_flat[0].imshow(image)
    axes_flat[0].set_title("Original Image", fontsize=10, weight='bold')
    axes_flat[0].axis('off')
    draw_target_highlights(axes_flat[0], target_indices)

    overlay_heatmap(axes_flat[1], image, avg_data, "AVERAGE (All Heads)", target_indices)
    for spine in axes_flat[1].spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(2)

    for i in range(num_heads):
        if i + 2 < len(axes_flat):
            overlay_heatmap(axes_flat[i+2], image, heads_data[i], f"Head {i}", target_indices)

    for i in range(num_heads + 2, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig

def plot_attention_trends(attention_data, target_patches, output_dir):
    plt.figure(figsize=(12, 6))
    for patch_idx in target_patches:
        scores = attention_data[patch_idx]
        layers = range(len(scores))
        plt.plot(layers, scores, marker='o', label=f"Patch {patch_idx}")
    plt.title("Attention on Target Patches Across Layers")
    plt.xlabel("Layer Index")
    plt.ylabel("Raw Attention Score")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    filename = os.path.join(output_dir, "attention_trend_analysis.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def main():
    # --- 1. SETUP UNIQUE TIMESTAMP DIRECTORY ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    current_output_dir = os.path.join(BASE_OUTPUT_DIR, timestamp)
    os.makedirs(current_output_dir, exist_ok=True)
    print(f"Created unique output directory: {current_output_dir}/")
    
    # Initialize Metadata Dictionary
    metadata = {
        "timestamp": timestamp,
        "model_id": MODEL_ID,
        "image_path": IMAGE_PATH,
        "question": QUESTION,
        "full_prompt": PROMPT_TEXT,
        "target_patches": TARGET_PATCHES,
        "token_probabilities": [] # Will be filled later
    }

    print(f"Loading model: {MODEL_ID}...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        dtype=torch.float16, 
        device_map="auto",
        attn_implementation="eager"
    )
    model.config.output_attentions = True 
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    try:
        image = Image.open(IMAGE_PATH).convert('RGB')
        image = image.resize((336, 336), resample=Image.BICUBIC) 
    except FileNotFoundError:
        print(f"Error: Could not find {IMAGE_PATH}.")
        return

    targets = []
    for idx in TARGET_PATCHES:
        row = idx // 24
        col = idx % 24
        targets.append({'row': int(row), 'col': int(col), 'idx': idx})

    inputs = processor(text=PROMPT_TEXT, images=image, return_tensors="pt").to(model.device)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50,
            output_attentions=True,
            return_dict_in_generate=True,
            output_scores=True 
        )

    generated_text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    print(f"\nResponse: {generated_text}\n")
    
    # Update Metadata with Response
    metadata["response_text"] = generated_text

    # Probability Analysis
    print("="*40)
    print(" PROBABILITY ANALYSIS (First Token)")
    print("="*40)
    
    first_token_logits = outputs.scores[0][0] 
    probs = torch.softmax(first_token_logits, dim=-1)
    tokenizer = processor.tokenizer
    
    prompt_context = "ASSISTANT:"
    words_to_test = ["Yes", "No"]
    seen_ids = set()

    for word in words_to_test:
        text_with_space = prompt_context + " " + word
        id_with_space = tokenizer.encode(text_with_space, add_special_tokens=False)[-1]
        
        text_no_space = prompt_context + word
        id_no_space = tokenizer.encode(text_no_space, add_special_tokens=False)[-1]
        
        ids_to_check = [(id_with_space, f" {word}"), (id_no_space, f"{word}")]
        
        for token_id, label in ids_to_check:
            if token_id in seen_ids: continue
            seen_ids.add(token_id)
            
            token_prob = probs[token_id].item()
            is_chosen = (token_id == torch.argmax(first_token_logits).item())
            marker = "<<" if is_chosen else ""
            
            print(f"Token: '{label:<5}' | ID: {token_id:<6} | Probability: {token_prob:.4f} ({token_prob*100:.2f}%) {marker}")
            
            # Store in Metadata
            metadata["token_probabilities"].append({
                "token_label": label,
                "token_id": token_id,
                "probability": float(token_prob), # Ensure it's a python float, not tensor
                "is_chosen": bool(is_chosen)
            })
    
    print("="*40 + "\n")
    
    # --- SAVE METADATA JSON ---
    json_path = os.path.join(current_output_dir, "metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to: {json_path}")
    # --------------------------

    # Generate Images
    print(f"Generating images in {current_output_dir}...")
    all_layers_data = outputs.attentions[0] 
    patch_scores_history = {pid: [] for pid in TARGET_PATCHES}

    start_idx = (inputs.input_ids[0] == model.config.image_token_index).nonzero(as_tuple=True)[0][0].item()
    end_idx = start_idx + 576

    for layer_idx, layer_attn_tensor in enumerate(all_layers_data):
        try:
            heads_raw = layer_attn_tensor[0, :, -1, :] 
            
            if end_idx > heads_raw.shape[1]:
                image_heads_flat = heads_raw[:, -576:]
            else:
                image_heads_flat = heads_raw[:, start_idx : end_idx]
            
            current_num_heads = image_heads_flat.shape[0] 
            
            avg_attention_flat = image_heads_flat.mean(dim=0).cpu().numpy()
            
            for pid in TARGET_PATCHES:
                if pid < len(avg_attention_flat):
                    patch_scores_history[pid].append(float(avg_attention_flat[pid])) # Cast to float for safety
                else:
                    patch_scores_history[pid].append(0.0)

            heads_map_2d = image_heads_flat.view(current_num_heads, 24, 24).float().cpu().numpy()
            avg_map_2d = heads_map_2d.mean(axis=0)

            fig = create_layer_grid_plot(image, heads_map_2d, avg_map_2d, layer_idx, targets)
            
            filename = os.path.join(current_output_dir, f"layer_{layer_idx:02d}.png")
            plt.savefig(filename, bbox_inches='tight')
            plt.close(fig) 

            if layer_idx % 5 == 0:
                print(f"  - Saved Layer {layer_idx}")

        except Exception as e:
            print(f"Error processing layer {layer_idx}: {e}")

    plot_attention_trends(patch_scores_history, TARGET_PATCHES, current_output_dir)
    print(f"\nSuccess! All images saved to folder: {os.path.abspath(current_output_dir)}")

if __name__ == "__main__":
    main()