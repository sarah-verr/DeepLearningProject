import os
import json
import argparse
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from transformers import (
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
)

from utils.prompt_templates import build_visual_yesno_prompt

# --- Configuration ---
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
BASE_DATA_PATH = "/home/tenkhtuvshin/DeepLearningProject/data/vlm_levels"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)


def _pick_model_input_device(model) -> torch.device:
    """Best-effort device for placing input tensors with device_map='auto'."""
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


def _move_to_device(batch: dict, device: torch.device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if hasattr(v, "to") else v
    return out


def extract_all_layer_hidden_states(model, processor, inputs, num_layers):
    """
    Extract hidden states from all layers at the decision point in a single generation pass.

    Args:
        model: The LLaVA model
        processor: The processor
        inputs: Input dictionary with input_ids, pixel_values, etc.

    Returns:
        dict mapping layer_idx -> numpy array of shape (hidden_dim,)
    """
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    # outputs.hidden_states is a tuple of tuples: (step_0, step_1, ...)
    # Each step contains hidden states for all layers
    if len(outputs.hidden_states) == 0:
        raise ValueError("No hidden states returned from generation")

    # Step 0 = input step
    input_step_hidden_states = outputs.hidden_states[0]

    # Extract hidden state at the last input token (decision point) for all layers
    all_layer_hidden_states = {}
    for layer_idx in range(num_layers):
        # layer_hidden shape: (batch, seq_len, hidden_dim)
        layer_hidden = input_step_hidden_states[layer_idx]
        decision_point_hidden = layer_hidden[0, input_length - 1, :]  # Shape: (hidden_dim,)
        all_layer_hidden_states[layer_idx] = decision_point_hidden.detach().cpu().numpy()

    return all_layer_hidden_states


def collect_probing_data(model, processor, level, max_images=None):
    """
    Collect hidden states and labels from all questions in a level.

    Returns:
        hidden_states_by_layer: dict mapping layer_idx -> list of hidden state arrays
        labels: list of binary labels (0=no, 1=yes)
        metadata: list of dicts with question/image info
    """
    level_dir = os.path.join(BASE_DATA_PATH, f"level_{level}")
    ann_dir = os.path.join(level_dir, "ann")
    # ann_dir = os.path.join(base_dir, "ann")
    img_dir = os.path.join(level_dir, "images")
    # img_dir = os.path.join(base_dir, "images")

    if not os.path.exists(ann_dir):
        raise FileNotFoundError(f"Level {level} directory not found: {ann_dir}")

    json_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".json")])
    if max_images:
        json_files = json_files[:max_images]

    # Determine number of layers from model
    num_layers = len(model.language_model.layers)
    hidden_states_by_layer = {i: [] for i in range(num_layers)}
    labels = []
    metadata = []

    device = _pick_model_input_device(model)

    print(f"Collecting data from {len(json_files)} images...")

    for filename in tqdm(json_files, desc="Processing images"):
        file_id = filename.replace(".json", "")
        ann_path = os.path.join(ann_dir, filename)

        with open(ann_path, "r") as f:
            data = json.load(f)

        image_path = os.path.join(img_dir, f"{file_id}.png")
        if not os.path.exists(image_path):
            continue

        image = Image.open(image_path).convert("RGB")

        qa_list = data.get("qa", [])

        for qa in qa_list:
            question = (qa.get("question") or "").strip()
            gt = (qa.get("answer") or "").strip().lower()

            if not question or gt not in {"yes", "no"}:
                continue

            # Prepare inputs
            conversation = build_visual_yesno_prompt(question)
            prompt_text = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = processor(text=prompt_text, images=image, return_tensors="pt")
            inputs = _move_to_device(inputs, device)

            # Extract hidden states from all layers in a single generation pass
            try:
                all_layer_states = extract_all_layer_hidden_states(
                    model, processor, inputs, num_layers
                )

                # Store hidden states for each layer
                for layer_idx, hidden_state in all_layer_states.items():
                    hidden_states_by_layer[layer_idx].append(hidden_state)

                # Store label (0=no, 1=yes)
                labels.append(1 if gt == "yes" else 0)
                metadata.append(
                    {
                        "image": file_id,
                        "question": question,
                        "ground_truth": gt,
                    }
                )
            except Exception as e:
                print(
                    f"Error processing {file_id}, question: {question[:50]}... - {e}"
                )

    print(f"Collected {len(labels)} examples")
    return hidden_states_by_layer, labels, metadata


def probe_layer(hidden_states, labels, cv_folds=5):
    """
    Train a linear probe on hidden states and return cross-validation accuracy.

    Args:
        hidden_states: list of numpy arrays, each of shape (hidden_dim,)
        labels: list of binary labels (0 or 1)
        cv_folds: number of cross-validation folds

    Returns:
        mean_accuracy: mean CV accuracy
        std_accuracy: std of CV accuracy
    """
    if len(hidden_states) != len(labels):
        raise ValueError(
            f"Mismatch: {len(hidden_states)} hidden states, {len(labels)} labels"
        )

    if len(set(labels)) < 2:
        # Need at least 2 classes
        return 0.0, 0.0

    # Convert to numpy arrays
    X = np.array(hidden_states)  # Shape: (n_samples, hidden_dim)
    y = np.array(labels)  # Shape: (n_samples,)

    # Train linear probe with cross-validation
    probe = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(probe, X, y, cv=cv_folds, scoring="accuracy")

    return cv_scores.mean(), cv_scores.std()


def plot_layer_accuracy(results, output_dir, level):
    """
    Plot layer accuracy and save to vis_results folder.

    Args:
        results: dict with 'layer_scores' containing list of dicts with 'layer', 'accuracy', 'std'
        output_dir: directory to save the plot
        level: data level for title
    """
    layer_scores = results.get("layer_scores", [])
    if not layer_scores:
        print("No layer scores to plot")
        return

    # Extract data
    layers = [s["layer"] for s in layer_scores]
    accuracies = [s["accuracy"] for s in layer_scores]
    num_examples = results.get("num_examples", 0)

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot line without error bars
    plt.plot(
        layers,
        accuracies,
        "o-",
        linewidth=2,
        markersize=8,
        label="Layer Accuracy",
    )

    # Add horizontal line at 0.5 (random chance)
    plt.axhline(y=0.5, linestyle="--", alpha=0.5, label="Random (0.5)")

    # Find and highlight best layer
    best_layer_idx = np.argmax(accuracies)
    best_layer = layers[best_layer_idx]
    best_acc = accuracies[best_layer_idx]
    plt.plot(
        best_layer,
        best_acc,
        "*",
        markersize=20,
        label=f"Best Layer {best_layer} ({best_acc:.3f})",
    )

    # Formatting
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Cross-Validation Accuracy", fontsize=12)
    plt.title(
        f"Layer Probing Accuracy (Level {level})\n{num_examples} examples, {len(layers)} layers",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.xlim(-0.5, max(layers) + 0.5)
    plt.ylim(0, 1.0)
    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "layer_probing_accuracy.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Probe model layers for yes/no decision encoding"
    )
    parser.add_argument(
        "--level", type=int, default=1, help="Data level to probe (0-4)"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process",
    )
    # parser.add_argument(
    #     "--output",
    #     type=str,
    #     default="layer_probing_results.json",
    #     help="Output JSON file",
    # )
    parser.add_argument(
        "--cv_folds", type=int, default=5, help="Number of CV folds"
    )
    args = parser.parse_args()

    print(f"Loading {MODEL_ID}...")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        quantization_config=bnb,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
    model.eval()

    num_layers = len(model.language_model.layers)
    print(f"Model has {num_layers} layers")

    # Collect data
    hidden_states_by_layer, labels, metadata = collect_probing_data(
        model, processor, args.level, max_images=args.max_images
    )

    if len(labels) == 0:
        print("No data collected. Exiting.")
        return

    print(f"\nProbing {num_layers} layers on {len(labels)} examples...")

    # Probe each layer
    results = {
        "level": args.level,
        "num_examples": len(labels),
        "num_layers": num_layers,
        "layer_scores": [],
    }

    for layer_idx in tqdm(range(num_layers), desc="Probing layers"):
        if layer_idx not in hidden_states_by_layer:
            continue

        hidden_states = hidden_states_by_layer[layer_idx]
        if len(hidden_states) != len(labels):
            print(
                f"Warning: Layer {layer_idx} has {len(hidden_states)} states "
                f"but {len(labels)} labels"
            )
            continue

        mean_acc, std_acc = probe_layer(
            hidden_states, labels, cv_folds=args.cv_folds
        )

        results["layer_scores"].append(
            {
                "layer": layer_idx,
                "accuracy": float(mean_acc),
                "std": float(std_acc),
            }
        )

        print(f"Layer {layer_idx:2d}: {mean_acc:.4f} ± {std_acc:.4f}")

    # # Save results
    # with open(args.output, "w") as f:
    #     json.dump(results, f, indent=2)

    # print(f"\nResults saved to {args.output}")
    best_layer_info = max(
        results["layer_scores"], key=lambda x: x["accuracy"]
    )
    print(f"\nBest layer: {best_layer_info}")

    # Create vis_results output directory and plot
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_output_dir = "vis_results"
    level_output = os.path.join(base_output_dir, f"level_{args.level}")
    probing_output = os.path.join(level_output, f"probing_{timestamp}")
    os.makedirs(probing_output, exist_ok=True)

    # Plot and save
    plot_layer_accuracy(results, probing_output, args.level)

    # Also save a copy of the results JSON in the vis_results folder
    results_copy_path = os.path.join(
        probing_output, "layer_probing_results.json"
    )
    with open(results_copy_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nVisualization results saved to: {os.path.abspath(probing_output)}")


if __name__ == "__main__":
    main()
