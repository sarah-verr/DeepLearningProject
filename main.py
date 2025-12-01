import torch
import sys
import importlib.util
import os
import shutil

# --- Dependency Check ---
if importlib.util.find_spec("bitsandbytes") is None:
    print("Error: 'bitsandbytes' library is missing.")
    print("Please install it by running: pip install bitsandbytes accelerate")
    sys.exit(1)

from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
IMAGE_PATH = "sample.png"
PROMPT_TEXT = "USER: <image>\nIs the yellow triangle right of the red triangle?\nASSISTANT:"
OUTPUT_DIR = "layer_vis_results"
OVERLAP_OUTPUT_FILE = "all_layers_overlap_attention.png" # New output file for overlap

def save_attention_map(image, attn_map, layer_idx, output_dir):
    """
    Helper function to plot and save a single layer's attention map.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original Image
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    # Heatmap Overlay
    # Resize attention map to match original image size
    attn_image = Image.fromarray((attn_map * 255).astype('uint8'))
    attn_image = attn_image.resize(image.size, resample=Image.BICUBIC)
    
    ax[1].imshow(image)
    ax[1].imshow(attn_image, cmap='jet', alpha=0.6)
    
    ax[1].set_title(f"Attention of 1st Gen Token\n(Layer {layer_idx})")
    ax[1].axis('off')
    
    # Save file with zero-padded index (e.g., layer_05.png)
    filename = os.path.join(output_dir, f"layer_{layer_idx:02d}.png")
    plt.savefig(filename, bbox_inches='tight')
    
    # CRITICAL: Close the figure to free memory
    plt.close(fig) 

def save_overlap_attention_map(image, aggregated_attn_map, output_file_path):
    """
    Helper function to plot and save the aggregated attention map.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original Image
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    # Heatmap Overlay
    # Normalize the aggregated map to [0, 1] for proper visualization
    normalized_aggregated_attn = (aggregated_attn_map - aggregated_attn_map.min()) / \
                                 (aggregated_attn_map.max() - aggregated_attn_map.min() + 1e-8) # Add epsilon for stability
    
    attn_image = Image.fromarray((normalized_aggregated_attn * 255).astype('uint8'))
    attn_image = attn_image.resize(image.size, resample=Image.BICUBIC)
    
    ax[1].imshow(image)
    ax[1].imshow(attn_image, cmap='jet', alpha=0.6)
    
    ax[1].set_title("Aggregated Attention Across All Layers")
    ax[1].axis('off')
    
    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close(fig)

def main():
    # 0. Output Setup
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}/")

    # 1. Setup Model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    print(f"Loading model: {MODEL_ID} (4-bit mode)...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        quantization_config=quantization_config,
        device_map="auto",
        attn_implementation="eager"
    )
    
    model.config.output_attentions = True 
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    num_layers = model.config.text_config.num_hidden_layers
    print(f"Model Loaded. Processing {num_layers} layers.")

    # 2. Process Input
    print(f"Loading image from {IMAGE_PATH}...")
    try:
        image = Image.open(IMAGE_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {IMAGE_PATH}. Please ensure the image exists.")
        return

    inputs = processor(text=PROMPT_TEXT, images=image, return_tensors="pt").to(model.device)

    # 3. Inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50,
            output_attentions=True,
            return_dict_in_generate=True
        )

    # Decode response
    generated_text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    print(f"\nResponse: {generated_text}\n")

    if outputs.attentions is None:
        print("Error: Model did not return attentions.")
        return

    # List to store all attention maps for aggregation
    all_attention_maps = []

    # 4. Attention Extraction Loop
    print(f"Generating visualizations for all {num_layers} layers...")
    
    all_layers_data = outputs.attentions[0]

    for layer_idx, layer_attn_tensor in enumerate(all_layers_data):
        try:
            avg_attn = layer_attn_tensor[0, :, -1, :].mean(dim=0)

            num_image_tokens = 576
            start_idx = 5 
            end_idx = start_idx + num_image_tokens
            
            if end_idx > len(avg_attn):
                image_attn = avg_attn[-num_image_tokens:] 
            else:
                image_attn = avg_attn[start_idx:end_idx]

            attn_map = image_attn.view(24, 24).float().cpu().numpy()
            
            # Store the attention map for aggregation later
            all_attention_maps.append(attn_map)

            # 6. Save Individual Visualization
            save_attention_map(image, attn_map, layer_idx, OUTPUT_DIR)
            
            if layer_idx % 5 == 0:
                print(f"Processed Layer {layer_idx}...")

        except Exception as e:
            print(f"Error processing layer {layer_idx}: {e}")

    print(f"\nIndividual layer images saved in: {os.path.abspath(OUTPUT_DIR)}")

    # 7. Generate Overlap Visualization
    if all_attention_maps:
        print(f"\nGenerating overlap visualization: {OVERLAP_OUTPUT_FILE}...")
        # Sum all collected attention maps
        aggregated_attn_map = np.sum(all_attention_maps, axis=0)
        
        # Save the aggregated overlap map
        save_overlap_attention_map(image, aggregated_attn_map, os.path.join(OUTPUT_DIR, OVERLAP_OUTPUT_FILE))
        print(f"Overlap visualization saved to: {os.path.join(OUTPUT_DIR, OVERLAP_OUTPUT_FILE)}")
    else:
        print("No attention maps were collected to create an overlap visualization.")

    print("\nAll tasks completed!")

if __name__ == "__main__":
    main()