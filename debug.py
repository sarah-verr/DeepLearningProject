import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# --- CONFIG ---
IMAGE_PATH = "sample.png" 

def analyze_pixels_grid(image_path, top_k=2):
    """
    Analyzes the image grid and returns pixel metrics.
    Returns:
        img: The resized PIL image
        heatmap_grid: 2D numpy array of scores
        patch_data: List of dicts with full details for every patch
        target_indices: List of integers (e.g., [152, 314]) representing the top_k active patches
    """
    print(f"Loading {image_path}...")
    # 1. Resize exactly like LLaVA
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found. Please ensure it exists.")
        return None, None, None, []
        
    img = img.resize((336, 336), resample=Image.BICUBIC)
    img_arr = np.array(img)

    # LLaVA Constants
    GRID_ROWS = 24
    GRID_COLS = 24
    PATCH_SIZE = 14 # 336 / 24 = 14
    
    # Grid to store the "Activity Score" of each patch
    heatmap_grid = np.zeros((GRID_ROWS, GRID_COLS))
    patch_data = []

    # 2. Iterate over every patch
    print("Scanning patches...")
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            # Calculate pixel boundaries
            y_start = r * PATCH_SIZE
            y_end = y_start + PATCH_SIZE
            x_start = c * PATCH_SIZE
            x_end = x_start + PATCH_SIZE
            
            # Slice the image array
            patch_pixels = img_arr[y_start:y_end, x_start:x_end]
            
            # 3. COMPUTE METRIC: Standard Deviation
            score = np.std(patch_pixels)
            
            heatmap_grid[r, c] = score
            
            # Compute flat index (0 to 575)
            flat_idx = (r * GRID_COLS) + c
            
            patch_data.append({
                'index': flat_idx,
                'row': r,
                'col': c,
                'score': score,
                'x': x_start,
                'y': y_start
            })

    # 4. Identify Top Targets internally
    # Sort by score descending
    sorted_patches = sorted(patch_data, key=lambda x: x['score'], reverse=True)
    
    # Take top K
    top_patches = sorted_patches[:top_k]
    
    # Sort them by index (so they appear in reading order 0->575)
    top_patches = sorted(top_patches, key=lambda x: x['index'])
    
    # Extract just the integers
    target_indices = [p['index'] for p in top_patches]

    return img, heatmap_grid, patch_data, target_indices

def plot_reference_grid(ax, image, patch_data):
    """Helper function to plot the full reference grid with all labels."""
    ax.imshow(image)
    ax.set_title("Full Reference Grid (Indices 0-575)")
    ax.axis('off')
    
    PATCH_SIZE = 14
    
    for t in patch_data:
        rect = patches.Rectangle((t['x'], t['y']), PATCH_SIZE, PATCH_SIZE, linewidth=0.5, edgecolor='white', facecolor='none', alpha=0.3)
        ax.add_patch(rect)
        
        # Calculate center coordinates for text
        text_x = t['x'] + PATCH_SIZE / 2
        text_y = t['y'] + PATCH_SIZE / 2
        
        ax.text(text_x, text_y, str(t['index']), color='cyan', fontsize=5, ha='center', va='center', alpha=0.8)


# --- RUN ANALYSIS ---
# Unpack the 4th return value (target_indices)
result = analyze_pixels_grid(IMAGE_PATH, top_k=2)

if result[0] is not None:
    image, heatmap, all_patch_data, target_indices = result

    print("\n" + "="*30)
    print("PIXEL ANALYSIS RESULTS")
    print("="*30)
    print(f"Top {len(target_indices)} token indices found: {target_indices}")
    print("Use this list in your LLaVA script!")

    # Filter data for visualization based on the returned indices
    top_2_targets = [p for p in all_patch_data if p['index'] in target_indices]

    # --- VISUALIZATION ---
    fig, ax = plt.subplots(1, 3, figsize=(22, 7))

    # Plot 1: The 'Activity' Heatmap
    im = ax[0].imshow(heatmap, cmap='magma')
    ax[0].set_title("Pixel Variation Heatmap (24x24)")
    plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    for t in top_2_targets:
        ax[0].add_patch(patches.Rectangle((t['col']-0.5, t['row']-0.5), 1, 1, fill=False, edgecolor='cyan', lw=2))
        ax[0].text(t['col'], t['row'], str(t['index']), color='cyan', ha='center', va='center', weight='bold')

    # Plot 2: Original Image with Boxes
    ax[1].imshow(image)
    ax[1].set_title(f"Original Image (Targets: {target_indices})")
    ax[1].axis('off')
    for t in top_2_targets:
        rect = patches.Rectangle((t['x'], t['y']), 14, 14, linewidth=2, edgecolor='cyan', facecolor='none')
        ax[1].add_patch(rect)
        ax[1].text(t['x'], t['y']-2, f"ID:{t['index']}", color='cyan', fontsize=9, weight='bold')

    # Plot 3: Full Reference Grid
    plot_reference_grid(ax[2], image, all_patch_data)

    plt.tight_layout()
    plt.savefig("patch_reference_grid.png", dpi=300, bbox_inches='tight')
    print("\nDetailed reference image saved to: patch_reference_grid.png")
    plt.show()