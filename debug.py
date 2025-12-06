import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import argparse
import os
import json

# --- CONFIG ---
BASE_DATA_PATH = "/home/kkarthikeyan/deep-learning/DeepLearningProject/Synthetic-Data/vlm_levels"
LLAVA_SIZE = 336
PATCH_SIZE = 14  # 336 / 24
GRID_DIM = 24

def get_intersecting_patches(bbox, scale_x, scale_y):
    """
    Calculates which 14x14 patches intersect with a specific bounding box.
    bbox format: [x_min, y_min, x_max, y_max] (Original coords)
    """
    # 1. Scale bbox to 336x336 space
    x1 = bbox[0] * scale_x
    y1 = bbox[1] * scale_y
    x2 = bbox[2] * scale_x
    y2 = bbox[3] * scale_y
    
    intersecting_indices = []
    
    # 2. Iterate through all patches to check intersection
    for r in range(GRID_DIM):
        for c in range(GRID_DIM):
            # Patch coordinates in 336x336 space
            p_x1 = c * PATCH_SIZE
            p_y1 = r * PATCH_SIZE
            p_x2 = p_x1 + PATCH_SIZE
            p_y2 = p_y1 + PATCH_SIZE
            
            # Intersection logic:
            # overlap exists if max(lefts) < min(rights) AND max(tops) < min(bottoms)
            inter_x1 = max(x1, p_x1)
            inter_y1 = max(y1, p_y1)
            inter_x2 = min(x2, p_x2)
            inter_y2 = min(y2, p_y2)
            
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                flat_idx = (r * GRID_DIM) + c
                intersecting_indices.append(flat_idx)
                
    return sorted(intersecting_indices)

def analyze_json_targets(image_path, json_path):
    print(f"Loading Image: {image_path}")
    print(f"Loading Ann  : {json_path}")
    
    # Load Image
    try:
        img_orig = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    
    # Load JSON
    try:
        with open(json_path, 'r') as f:
            ann_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON not found at {json_path}")
        return None

    # Calculate Scaling Factors
    w_orig, h_orig = img_orig.size
    scale_x = LLAVA_SIZE / w_orig
    scale_y = LLAVA_SIZE / h_orig
    
    # Resize Image for visualization
    img_resized = img_orig.resize((LLAVA_SIZE, LLAVA_SIZE), resample=Image.BICUBIC)
    
    objects_data = []
    
    # Process Objects
    for obj in ann_data.get('objects', []):
        indices = get_intersecting_patches(obj['bbox'], scale_x, scale_y)
        objects_data.append({
            'id': obj['id'],
            'color': obj['color'],
            'shape': obj['shape'],
            'bbox_orig': obj['bbox'],
            'indices': indices
        })
        
    return img_resized, objects_data

def visualize_patches(img, objects_data):
    """Visualizes the specific patches associated with each object."""
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    # --- Plot 1: Standard Bounding Boxes ---
    ax[0].imshow(img)
    ax[0].set_title("Ground Truth Bounding Boxes (Scaled)")
    ax[0].axis('off')
    
    # Map typical JSON colors to matplotlib colors
    color_map = {'pink': 'magenta', 'green': 'lime', 'blue': 'cyan', 'red': 'red', 'yellow': 'yellow'}
    
    for obj in objects_data:
        # Re-scale bbox just for drawing
        # Note: We calculated indices earlier, here we just draw the box for visual confirmation
        # Since we don't have exact scale factors passed here, we infer from indices roughly
        # Or better: We reconstruct the rect from the patches for visualization
        
        # Highlight patches in background
        for idx in obj['indices']:
            r = idx // GRID_DIM
            c = idx % GRID_DIM
            rect = patches.Rectangle((c*PATCH_SIZE, r*PATCH_SIZE), PATCH_SIZE, PATCH_SIZE, 
                                   linewidth=0, facecolor=color_map.get(obj['color'], 'white'), alpha=0.3)
            ax[0].add_patch(rect)

    # --- Plot 2: Patch Indices Grid ---
    ax[1].imshow(img)
    ax[1].set_title("Patch Indices (for TARGET_GROUPS)")
    ax[1].axis('off')
    
    # Draw Grid
    for r in range(GRID_DIM):
        for c in range(GRID_DIM):
            x, y = c * PATCH_SIZE, r * PATCH_SIZE
            rect = patches.Rectangle((x, y), PATCH_SIZE, PATCH_SIZE, linewidth=0.5, edgecolor='gray', facecolor='none', alpha=0.2)
            ax[1].add_patch(rect)

    # Draw Object Patches
    for obj in objects_data:
        c_code = color_map.get(obj['color'], 'white')
        for idx in obj['indices']:
            r = idx // GRID_DIM
            c = idx % GRID_DIM
            x, y = c * PATCH_SIZE, r * PATCH_SIZE
            
            # Thick border for active patch
            rect = patches.Rectangle((x, y), PATCH_SIZE, PATCH_SIZE, linewidth=2, edgecolor=c_code, facecolor='none')
            ax[1].add_patch(rect)
            
            # Text Index
            ax[1].text(x + PATCH_SIZE/2, y + PATCH_SIZE/2, str(idx), 
                       color=c_code, fontsize=7, ha='center', va='center', weight='bold')

    plt.tight_layout()
    out_name = "debug_objects_grid.png"
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {out_name}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map JSON Bounding Boxes to LLaVA Patches")
    parser.add_argument("--level", type=str, required=True, help="Level number (e.g., '1')")
    parser.add_argument("--id", type=str, required=True, help="File ID (e.g., '00007_b')")
    args = parser.parse_args()

    # Construct Paths
    level_dir = f"level_{args.level}"
    img_path = os.path.join(BASE_DATA_PATH, level_dir, "images", f"{args.id}.png")
    json_path = os.path.join(BASE_DATA_PATH, level_dir, "ann", f"{args.id}.json")

    result = analyze_json_targets(img_path, json_path)

    if result:
        image, obj_data = result
        
        print("\n" + "="*40)
        print(f"TARGET CONFIGURATION for {args.id}")
        print("="*40)
        print("Copy the lines below into main.py:\n")
        
        print("TARGET_GROUPS = [")
        for obj in obj_data:
            print(f"    {obj['indices']},  # Group {obj['id']}: {obj['color']} {obj['shape']}")
        print("]")
        
        visualize_patches(image, obj_data)