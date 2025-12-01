import torch
import sys
import importlib.util
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np

# --- Dependency Check ---
if importlib.util.find_spec("bitsandbytes") is None:
    print("Error: 'bitsandbytes' library is missing.")
    print("Please install it by running: pip install bitsandbytes accelerate")
    sys.exit(1)

from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

# --- Configuration ---
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
IMAGE_PATH = "sample.png"
PROMPT_TEXT = "USER: <image>\nIs the orange triangle above the yellow triangle?\nASSISTANT: "
PDF_FILENAME = "llava_attention_analysis.pdf"

def create_plot(image, attn_map, title):
    """
    Creates a Matplotlib figure for a specific attention map.
    Returns the figure object (does not save it).
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    
    # Left: Original Image
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    # Right: Heatmap Overlay
    # Normalize map for visualization
    if attn_map.max() > attn_map.min():
        norm_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    else:
        norm_map = attn_map 

    attn_image = Image.fromarray((norm_map * 255).astype('uint8'))
    attn_image = attn_image.resize(image.size, resample=Image.BICUBIC)
    
    ax[1].imshow(image)
    ax[1].imshow(attn_image, cmap='jet', alpha=0.6)
    
    ax[1].set_title(title)
    ax[1].axis('off')
    
    return fig

def main():
    # 1. Setup Model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    print(f"Loading model: {MODEL_ID} (4-bit mode)...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        # quantization_config=quantization_config,
        dtype=torch.float16,
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

    # --- NEW: INSPECTION BLOCK ---
    print("\n" + "="*30)
    print(" INSPECTING PROCESSOR OUTPUTS")
    print("="*30)
    
    for key, value in inputs.items():
        if hasattr(value, 'shape'):
            print(f"Key: {key:15} | Shape: {value.shape} | Type: {value.dtype}")
    
    print("="*30 + "\n")
    # -----------------------------
    # 3. Inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50,
            output_attentions=True,
            return_dict_in_generate=True
        )

    generated_text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    print(f"\nResponse: {generated_text}\n")
    # print(f"Output Length (Prompt + Image + Generated Answer): {outputs.sequences.shape[1]}")

    if outputs.attentions is None:
        print("Error: Model did not return attentions.")
        return

    all_attention_maps = [] # To store numpy arrays for the final overlap

    # 4. Initialize PDF
    print(f"Initializing PDF: {PDF_FILENAME}...")
    
    with PdfPages(PDF_FILENAME) as pdf:
        
        # Loop through all layers
        print(f"Type of outputs.attentions = {type(outputs.attentions)} and is of shape {len(outputs.attentions)}")
        all_layers_data = outputs.attentions[0] # outputs.attentions gives us a tuple of tuples, and the outermost tuple has the attentions at each step of the generation. We do 0 because we want to visualise the attention that the model had when generating the first word
        print(f"Retrieved {len(all_layers_data)} attention matrices")
        print(f"Generating pages for {num_layers} layers...")
        for layer_idx, layer_attn_tensor in enumerate(all_layers_data):
            # print(f"Retrieved attention matrix of shape {layer_attn_tensor.shape}for {layer_idx}")
            try:
                # Extract Attention
                print(f"Shape before collapsing: {layer_attn_tensor.shape}")
                # 0 ensures you take the first batch only (we only have 1 batch so far)
                # : ensures you take all the attention heads
                # -1 here ensures you take the attention for the last token that is generated
                # ensures you take the attention placed by the generated token on all the tokens
                avg_attn = layer_attn_tensor[0, :, -1, :].mean(dim=0)
                print(f"Shape after collapsing along head dimension: {avg_attn.shape}")
                # Extract Image tokens (approx 576 tokens)
                num_image_tokens = 576
                start_idx = 5 
                end_idx = start_idx + num_image_tokens
                
                if end_idx > len(avg_attn):
                    image_attn = avg_attn[-num_image_tokens:] 
                else:
                    image_attn = avg_attn[start_idx:end_idx]

                attn_map = image_attn.view(24, 24).float().cpu().numpy()
                all_attention_maps.append(attn_map)

                # Create Figure
                fig = create_plot(image, attn_map, title=f"Layer {layer_idx} Attention\n(1st Generated Token)")
                
                # Save into PDF
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig) # Close figure to free memory

                if layer_idx % 5 == 0:
                    print(f"  - Saved Layer {layer_idx}")

            except Exception as e:
                print(f"Error processing layer {layer_idx}: {e}")

        # 5. Generate Overlap Page
        if all_attention_maps:
            print("Generating final overlap page...")
            aggregated_attn_map = np.sum(all_attention_maps, axis=0)
            
            fig_overlap = create_plot(image, aggregated_attn_map, title="Aggregated Attention (All Layers Summed)")
            
            pdf.savefig(fig_overlap, bbox_inches='tight')
            plt.close(fig_overlap)
        
        # Set Metadata
        d = pdf.infodict()
        d['Title'] = 'LLaVA Attention Map Analysis'
        d['Author'] = 'LLaVA Visualizer'

    print(f"\nSuccess! PDF saved to: {os.path.abspath(PDF_FILENAME)}")

if __name__ == "__main__":
    main()