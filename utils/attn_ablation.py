"""
Attention Head Ablation for LLaVA model.
Zero out specific attention heads and measure impact on yes/no accuracy.
"""

import os
import json
import argparse
import time
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from utils.prompt_templates import build_visual_yesno_prompt
from utils.prompt_llava import _init_model, generate_output_for_model, score_yesno

# --- Configuration ---
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
BASE_DATA_PATH = f"/home/{os.environ['USER']}/DeepLearningProject/data/vlm_levels"


class AttentionHeadAblator:
    """
    Context manager to ablate (zero out) specific attention heads during inference.
    """
    def __init__(self, model, layers_heads: List[Tuple[int, int]]):
        """
        Args:
            model: LlavaForConditionalGeneration model
            layers_heads: List of (layer_idx, head_idx) tuples to ablate
        """
        self.model = model
        self.layers_heads = layers_heads
        self.hooks = []
        self.num_heads = model.language_model.config.num_attention_heads
        self.head_dim = model.language_model.config.hidden_size // self.num_heads
        
    def _create_ablation_hook(self, heads_to_ablate: List[int]):
        """Create a hook that zeros out specific heads in attention output."""
        def hook(module, input, output):
            # output[0] is attention output: [batch, seq_len, hidden_size]
            attn_output = output[0]
            batch_size, seq_len, hidden_size = attn_output.shape
            
            # Reshape to [batch, seq_len, num_heads, head_dim]
            attn_output_reshaped = attn_output.view(batch_size, seq_len, self.num_heads, self.head_dim)
            
            # Zero out specified heads
            for head_idx in heads_to_ablate:
                attn_output_reshaped[:, :, head_idx, :] = 0.0
            
            # Reshape back
            attn_output_modified = attn_output_reshaped.view(batch_size, seq_len, hidden_size)
            
            # Return modified output (preserve other elements)
            return (attn_output_modified,) + output[1:]
        
        return hook
    
    def __enter__(self):
        # Group heads by layer
        layer_to_heads = {}
        for layer_idx, head_idx in self.layers_heads:
            if layer_idx not in layer_to_heads:
                layer_to_heads[layer_idx] = []
            layer_to_heads[layer_idx].append(head_idx)
        
        # Register hooks for each layer
        for layer_idx, heads in layer_to_heads.items():
            layer = self.model.language_model.layers[layer_idx]
            hook = layer.self_attn.register_forward_hook(
                self._create_ablation_hook(heads)
            )
            self.hooks.append(hook)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def _pick_model_input_device(model) -> torch.device:
    """Get the device where model inputs should be placed."""
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_accuracy(model, processor, level: int, max_images: Optional[int] = None) -> Tuple[float, int, int]:
    """Evaluate yes/no accuracy on the data."""
    level_dir = os.path.join(BASE_DATA_PATH, f"level_{level}")
    ann_dir = os.path.join(level_dir, "ann")
    img_dir = os.path.join(level_dir, "images")
    
    if not os.path.exists(ann_dir):
        raise FileNotFoundError(f"Level {level} directory not found: {ann_dir}")
    
    json_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".json")])
    if max_images:
        json_files = json_files[:max_images]
    
    device = _pick_model_input_device(model)
    correct = 0
    total = 0
    
    for filename in tqdm(json_files, desc="Evaluating", leave=False):
        file_id = filename.replace(".json", "")
        ann_path = os.path.join(ann_dir, filename)
        
        with open(ann_path, "r") as f:
            annotation = json.load(f)
        
        # Try different image extensions
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = os.path.join(img_dir, file_id + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        
        if img_path is None:
            continue
        
        image = Image.open(img_path).convert("RGB")
        qa_pairs = annotation.get("qa", [])
        
        for qa in qa_pairs:
            question = qa["question"]
            gt_answer = qa["answer"].lower().strip()
            
            prompt = build_visual_yesno_prompt(question)
            conversation = processor.apply_chat_template(
                prompt, add_generation_prompt=True
            )
            
            inputs = processor(
                text=conversation,
                images=image,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = generate_output_for_model(model, inputs, max_new_tokens=10)
            
            pred, _, _ = score_yesno(outputs, processor.tokenizer)
            
            if pred and pred.lower() == gt_answer:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def run_ablation_study(
    model, processor, level: int,
    layers: List[int], num_heads: int = 32,
    max_images: Optional[int] = None
) -> Dict[str, Any]:
    """Run ablation study on specified layers/heads."""
    
    # Get baseline
    print("Computing baseline accuracy (no ablation)...")
    baseline_acc, correct, total = evaluate_accuracy(model, processor, level, max_images)
    print(f"Baseline: {baseline_acc:.4f} ({correct}/{total})")
    
    results = {
        'baseline_accuracy': baseline_acc,
        'baseline_correct': correct,
        'baseline_total': total,
        'ablation_results': [],
        'layers': layers,
        'num_heads': num_heads,
        'level': level,
    }
    
    # Ablate each head
    total_heads = len(layers) * num_heads
    pbar = tqdm(total=total_heads, desc="Ablating heads")
    
    for layer_idx in layers:
        for head_idx in range(num_heads):
            # Ablate single head
            with AttentionHeadAblator(model, [(layer_idx, head_idx)]):
                accuracy, _, _ = evaluate_accuracy(model, processor, level, max_images)
            
            results['ablation_results'].append({
                'layer': layer_idx,
                'head': head_idx,
                'accuracy': accuracy,
                'accuracy_drop': baseline_acc - accuracy,
            })
            
            pbar.set_postfix({
                'L': layer_idx, 'H': head_idx,
                'acc': f'{accuracy:.3f}',
                'drop': f'{baseline_acc - accuracy:+.3f}'
            })
            pbar.update(1)
    
    pbar.close()
    return results


def run_layer_ablation_study(
    model, processor, level: int,
    layers: Optional[List[int]] = None,
    max_images: Optional[int] = None
) -> Dict[str, Any]:
    """Run ablation study ablating entire layers (all heads at once)."""
    
    num_layers_total = len(model.language_model.layers)
    num_heads = model.language_model.config.num_attention_heads
    
    if layers is None:
        layers = list(range(num_layers_total))
    
    # Get baseline
    print("Computing baseline accuracy (no ablation)...")
    baseline_acc, correct, total = evaluate_accuracy(model, processor, level, max_images)
    print(f"Baseline: {baseline_acc:.4f} ({correct}/{total})")
    
    results = {
        'baseline_accuracy': baseline_acc,
        'baseline_correct': correct,
        'baseline_total': total,
        'layer_ablation_results': [],
        'layers': layers,
        'level': level,
    }
    
    # Ablate each layer (all heads at once)
    for layer_idx in tqdm(layers, desc="Ablating layers"):
        # Ablate all heads in this layer
        heads_to_ablate = [(layer_idx, h) for h in range(num_heads)]
        
        with AttentionHeadAblator(model, heads_to_ablate):
            accuracy, _, _ = evaluate_accuracy(model, processor, level, max_images)
        
        results['layer_ablation_results'].append({
            'layer': layer_idx,
            'accuracy': accuracy,
            'accuracy_drop': baseline_acc - accuracy,
        })
        
        print(f"  Layer {layer_idx}: acc={accuracy:.4f}, drop={baseline_acc - accuracy:+.4f}")
    
    return results


def plot_layer_ablation(results: Dict, output_dir: str):
    """Plot bar chart of accuracy drop when ablating each layer."""
    baseline = results['baseline_accuracy']
    level = results['level']
    
    layers = [r['layer'] for r in results['layer_ablation_results']]
    drops = [r['accuracy_drop'] for r in results['layer_ablation_results']]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ['red' if d > 0 else 'blue' for d in drops]
    bars = ax.bar(layers, drops, color=colors, alpha=0.7, edgecolor='black')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Accuracy Drop', fontsize=12)
    ax.set_title(f'Layer Ablation: Accuracy Drop per Layer\n'
                 f'(Level {level}, Baseline: {baseline:.3f})', fontsize=14)
    ax.set_xticks(layers)
    
    # Add value labels on bars
    for bar, drop in zip(bars, drops):
        height = bar.get_height()
        ax.annotate(f'{drop:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -10),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8, rotation=90)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'layer_ablation_chart.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to: {plot_path}")


def plot_ablation_heatmap(results: Dict, output_dir: str):
    """Plot heatmap of accuracy drop when ablating each head."""
    layers = results['layers']
    num_heads = results['num_heads']
    baseline = results['baseline_accuracy']
    level = results['level']
    
    # Create heatmap data
    heatmap = np.zeros((len(layers), num_heads))
    
    for entry in results['ablation_results']:
        layer_idx = layers.index(entry['layer'])
        head_idx = entry['head']
        heatmap[layer_idx, head_idx] = entry['accuracy_drop']
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, max(6, len(layers) * 0.4)))
    
    # Use diverging colormap: red = important (positive drop), blue = negative drop
    vmax = max(0.05, np.abs(heatmap).max())
    im = ax.imshow(heatmap, aspect='auto', cmap='RdBu_r', 
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    
    ax.set_xlabel('Attention Head', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title(f'Accuracy Drop from Single Head Ablation\n'
                 f'(Level {level}, Baseline: {baseline:.3f})', fontsize=14)
    
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xticks(range(0, num_heads, 4))
    ax.set_xticklabels(range(0, num_heads, 4))
    
    plt.colorbar(im, ax=ax, label='Accuracy Drop (red = important)')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'head_ablation_heatmap.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Attention Head Ablation Study")
    parser.add_argument("--level", type=int, default=2, help="Data level (0-4)")
    parser.add_argument("--max_images", type=int, default=None, 
                       help="Maximum images to process")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                       help="Layers to ablate (default: 13 16 19 24)")
    parser.add_argument("--single", type=int, nargs=2, default=None,
                       metavar=('LAYER', 'HEAD'),
                       help="Ablate single head and report accuracy")
    parser.add_argument("--layer_ablation", action="store_true",
                       help="Ablate entire layers instead of individual heads")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading {MODEL_ID}...")
    processor, model, device = _init_model(
        MODEL_ID,
        output_hidden_states=False,
        output_attentions=False,
    )
    
    num_layers = len(model.language_model.layers)
    num_heads = model.language_model.config.num_attention_heads
    print(f"Model has {num_layers} layers, {num_heads} heads per layer")
    
    # Default layers (based on typical interesting layers)
    if args.layers is None:
        args.layers = [13, 16, 19, 24]
    
    # Output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"vis_results/level_{args.level}/ablation_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Single head ablation mode
    if args.single:
        layer_idx, head_idx = args.single
        print(f"\nAblating single head: Layer {layer_idx}, Head {head_idx}")
        
        # Baseline
        print("Computing baseline...")
        baseline, correct, total = evaluate_accuracy(model, processor, args.level, args.max_images)
        print(f"Baseline accuracy: {baseline:.4f} ({correct}/{total})")
        
        # Ablated
        print(f"Computing ablated accuracy...")
        with AttentionHeadAblator(model, [(layer_idx, head_idx)]):
            ablated, _, _ = evaluate_accuracy(model, processor, args.level, args.max_images)
        print(f"Ablated accuracy: {ablated:.4f}")
        print(f"Accuracy drop: {baseline - ablated:+.4f}")
        return
    
    # Layer ablation mode
    if args.layer_ablation:
        # For layer ablation, default to all layers if not specified
        if args.layers is None:
            args.layers = list(range(num_layers))
        
        print(f"\nRunning layer ablation study on layers {args.layers}...")
        results = run_layer_ablation_study(
            model, processor, args.level,
            layers=args.layers, max_images=args.max_images
        )
        
        # Save results
        results_path = os.path.join(output_dir, 'layer_ablation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")
        
        # Plot
        plot_layer_ablation(results, output_dir)
        
        # Print most important layers
        sorted_results = sorted(results['layer_ablation_results'], 
                               key=lambda x: x['accuracy_drop'], reverse=True)
        print("\nMost important layers (largest accuracy drop):")
        for i, entry in enumerate(sorted_results[:10]):
            print(f"  {i+1}. Layer {entry['layer']}: drop = {entry['accuracy_drop']:+.4f}")
        return
    
    # Full head ablation study
    print(f"\nRunning head ablation study on layers {args.layers}...")
    results = run_ablation_study(
        model, processor, args.level,
        args.layers, num_heads, args.max_images
    )
    
    # Save results
    results_path = os.path.join(output_dir, 'ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # Plot heatmap
    plot_ablation_heatmap(results, output_dir)
    
    # Print top important heads
    sorted_results = sorted(results['ablation_results'], 
                           key=lambda x: x['accuracy_drop'], reverse=True)
    print("\nTop 10 most important heads (largest accuracy drop):")
    for i, entry in enumerate(sorted_results[:10]):
        print(f"  {i+1}. Layer {entry['layer']}, Head {entry['head']}: "
              f"drop = {entry['accuracy_drop']:+.4f}")


if __name__ == "__main__":
    main()

