"""
This file contains the code that uses the output from pure inference, and then runs the same input over the model again (without generation) to compute attentions that the model saw during generation
"""

import numpy as np
from utils.plotter import Plotter
from utils.prompt_llava import infer_model_with_attention, visualise_full_attention, _init_model, MODEL_ID
from utils.attention_utils import get_phrase_token_positions
import random
import os
import json
from pathlib import Path
import torch

def aggregate_attention_and_save_to_file(levels, filename="attention_results_detailed.json", use_masked_attention = False, always_mask = False):
    """
    This function computes that attention (aggregated as mean, per head, layer) and saves it to a json file: attention_results_all_levels.jsonl in the results_{model} folder
    """

    # These are the source -> target combinations of interest for us and hence we will store the mean for these
    key_pairs = [
                    ("object", "visual_object"),
                    ("subject", "visual_subject"),
                    ("object", "all_visual"),
                    ("subject", "all_visual"),
                    ("last", "visual_subject"),
                    ("last", "visual_object"),
                    ("relation", "visual_subject"),
                    ("relation", "visual_object"),
                    ("last", "all_text"),
                    ("last", "all_visual"),
                    ("relation", "all_text"),
                    ("relation", "all_visual")
    ]

    attn_results = infer_model_with_attention(levels, key_pairs, "visual", filename, use_attention_mask=use_masked_attention, always_mask=always_mask, num_questions_per_image=6)

    if not attn_results:
        return

    plotter = Plotter(experiment_name="attention_analysis")

    # Always save raw attention results to JSON via Plotter
    plotter.save_jsonl(attn_results, filename)

    # Once I write down the json, I will then analyse them on a notebook
    # # 1) Layer-wise text vs image fractions averaged over all QAs
    # all_metrics = [r["attention_metrics"] for r in attn_results]
    # agg_metrics = plotter.aggregate_attention_metrics(all_metrics)
    # plotter.plot_layer_text_vs_image(agg_metrics)

    # # 2) Per‑relation and global head×layer grids
    # #    (plot_attention_by_relation expects the full attn_results list)
    # plotter.plot_attention_by_relation(attn_results)

def full_detailed_attention_map_for_sample(levels, num_samples=1, layers_and_heads_to_plot=None, use_relational_phrase=True, use_masked_attention = False, always_mask = False):
    # Load model once
    processor, model, device = _init_model(MODEL_ID, output_attentions=True)

    print(f"Model loaded on {device}")
    
    project_root = Path(__file__).resolve().parents[1]
    for level_id in levels:
        print(f"Infering for level {level_id}")
        for _ in range(num_samples):
            # Sample image_id
            images_dir = project_root / "data" / "vlm_levels" / level_id / "images"
            image_files = [f for f in os.listdir(images_dir) if f.endswith(('_b.png', '_w.png'))]
            image_ids = [f.replace('.png', '') for f in image_files]
            image_id = random.choice(image_ids)
            
            # Load annotation to get available qa_ids
            ann_path = project_root / "data" / "vlm_levels" / level_id / "ann" / f"{image_id}.json"
            with open(ann_path, 'r') as f:
                ann = json.load(f)
            qa_ids = [q['id'] for q in ann['qa']]
            
            # Randomly sample one qa_id
            qa_id = random.choice(qa_ids)
            
            attentions, output, rel_positions = visualise_full_attention(level_id, image_id, qa_id, processor, model, device)

            # Convert to numpy array for processing
            attentions_array = np.array(attentions)

            # Save prompt as text
            plotter = Plotter(experiment_name="attention_analysis")
            output_dir = plotter.results_dir / f'{level_id}_{image_id}'
            output_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = output_dir / "output.txt"
            with open(prompt_path, "w") as f:
                f.write(output)
            print(f"Saved prompt to {prompt_path}")

            # pick the only batch item
            attentions = attentions_array[:, 0, :, :, :]
            print("Prompt + Ouput: ", output)

            # Pick specific combinations of layer and head of interest
            if layers_and_heads_to_plot is None:
                layers_and_heads_to_plot = [
                    (5, 15),
                    (12, 23),  # layer 12, head 23
                    (14, 6),
                    (16, 25),
                    (19, 6),
                    (29, 30), # layer 29, head 30
                    # Add more as needed
                ]

            for layer_idx, head_idx in layers_and_heads_to_plot:
                # Get attention for this layer and head: (seq_len, num_patches)
                attention = attentions[layer_idx, head_idx, :, :]
                if use_relational_phrase:
                    # Pick attention from the relational phrase tokens to the image patches
                    attention_to_patches = attention[rel_positions, :].mean(axis=0)
                    attention_type = "relational"
                else:
                    # Pick attention from the last token to the image patches
                    attention_to_patches = attention[-1, :]
                    attention_type = "last"
                # Plot
                filename = f"attention_overlay_layer_{layer_idx}_head_{head_idx}_{attention_type}.png"
                plotter.plot_attention_on_image(attention_to_patches, layer_idx, head_idx, level_id, image_id, filename=filename, subdir=f'{level_id}_{image_id}')
                print(f"Plotted {filename}")
            
            del attentions
            torch.cuda.empty_cache()
        
    # Clean up model after all samples
    del processor, model
    torch.cuda.empty_cache()


def average_attention_over_layer_groups(levels, layer_groups, num_samples=1, use_relational_phrase=True):
    """
    Similar to full_detailed_attention_map_for_sample, but averages attention over all heads and all layers in each specified group.
    
    Args:
        levels: List of level IDs to process
        layer_groups: List of lists, where each sublist contains layer indices to average over
        num_samples: Number of samples per level
        use_relational_phrase: Whether to use relational phrase tokens or last token
    """
    # Load model once
    processor, model, device = _init_model(MODEL_ID, output_attentions=True)

    print(f"Model loaded on {device}")
    
    project_root = Path(__file__).resolve().parents[1]
    for level_id in levels:
        print(f"Inferring for level {level_id}")
        for _ in range(num_samples):
            # Sample image_id
            images_dir = project_root / "data" / "vlm_levels" / level_id / "images"
            image_files = [f for f in os.listdir(images_dir) if f.endswith(('_b.png', '_w.png'))]
            image_ids = [f.replace('.png', '') for f in image_files]
            image_id = random.choice(image_ids)
            
            # Load annotation to get available qa_ids
            ann_path = project_root / "data" / "vlm_levels" / level_id / "ann" / f"{image_id}.json"
            with open(ann_path, 'r') as f:
                ann = json.load(f)
            qa_ids = [q['id'] for q in ann['qa']]
            
            # Randomly sample one qa_id
            qa_id = random.choice(qa_ids)
            
            attentions, output, rel_positions = visualise_full_attention(level_id, image_id, qa_id, processor, model, device)

            # Convert to numpy array for processing
            attentions_array = np.array(attentions)

            # Save prompt as text
            plotter = Plotter(experiment_name="attention_analysis_by_level_groups")
            output_dir = plotter.results_dir / f'{level_id}_{image_id}'
            output_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = output_dir / "output.txt"
            with open(prompt_path, "w") as f:
                f.write(output)
            print(f"Saved prompt to {prompt_path}")

            # pick the only batch item
            attentions = attentions_array[:, 0, :, :, :]  # (num_layers, num_heads, seq_len, num_patches)
            print("Prompt + Output: ", output)

            for group_idx, layer_indices in enumerate(layer_groups):
                # Average over layers and heads in the group
                group_attentions = attentions[layer_indices, :, :, :]  # (num_layers_in_group, num_heads, seq_len, num_patches)
                mean_attention = group_attentions.mean(axis=(0, 1))  # (seq_len, num_patches)
                
                if use_relational_phrase:
                    # Pick attention from the relational phrase tokens to the image patches
                    attention_to_patches = mean_attention[rel_positions, :].mean(axis=0)
                    attention_type = "relational"
                else:
                    # Pick attention from the last token to the image patches
                    attention_to_patches = mean_attention[-1, :]
                    attention_type = "last"
                
                # Plot
                filename = f"attention_overlay_group_{group_idx}_{attention_type}.png"
                # Using -1 for layer_idx and head_idx as placeholders for averaged
                plotter.plot_attention_on_image(attention_to_patches, layer_indices, -1, level_id, image_id, filename=filename, subdir=f'{level_id}_{image_id}')
                print(f"Plotted {filename}")
            
            del attentions
            torch.cuda.empty_cache()
        
    # Clean up model after all samples
    del processor, model
    torch.cuda.empty_cache()