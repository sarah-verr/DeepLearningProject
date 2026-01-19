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
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
)

from utils.prompt_templates import build_visual_yesno_prompt
from utils.prompt_llava import _init_model, generate_output_for_model, score_yesno
from utils.attention_utils import get_entity_indices, get_phrase_token_positions

# --- Configuration ---
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
BASE_DATA_PATH = f"/home/{os.environ['USER']}/DeepLearningProject/data/vlm_levels"


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    
    Args:
        pred_mask: Predicted binary mask (2D numpy array)
        gt_mask: Ground truth binary mask (2D numpy array)
    
    Returns:
        IoU score (0.0 to 1.0)
    """
    intersection = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
    union = np.logical_or(pred_mask > 0, gt_mask > 0).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)

# bnb = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
# )


def _pick_model_input_device(model) -> torch.device:
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


def extract_all_layer_hidden_states(model, processor, inputs, num_layers, first_token=False):
    
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    if outputs.hidden_states is None or len(outputs.hidden_states) == 0:
        raise ValueError("No hidden states returned from generation")

    generated_sequence_length = outputs.sequences[0].shape[0] if hasattr(outputs, 'sequences') and len(outputs.sequences) > 0 else input_length
    token_was_generated = generated_sequence_length > input_length

    if first_token:
        if not token_was_generated:
            raise ValueError(f"No new token generated. Input length: {input_length}, Generated sequence length: {generated_sequence_length}")
        
        if len(outputs.hidden_states) < 2:
            generated_token_id = outputs.sequences[0][input_length:input_length+1]
            
            full_input_ids = torch.cat([inputs["input_ids"], generated_token_id.unsqueeze(0)], dim=1)
            
            if "attention_mask" in inputs:
                full_attention_mask = torch.cat([
                    inputs["attention_mask"], 
                    torch.ones((1, 1), device=inputs["attention_mask"].device, dtype=inputs["attention_mask"].dtype)
                ], dim=1)
            else:
                full_attention_mask = None
            
            model_kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
            
            with torch.no_grad():
                forward_outputs = model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    **model_kwargs
                )
            
            input_step_hidden_states = forward_outputs.hidden_states
        else:
            input_step_hidden_states = outputs.hidden_states[1]
    else:
        input_step_hidden_states = outputs.hidden_states[0]

    all_layer_hidden_states = {}
    for layer_idx in range(num_layers):
        layer_hidden = input_step_hidden_states[layer_idx]
        if first_token:
            seq_len = layer_hidden.shape[1]
            if seq_len <= input_length:
                raise ValueError(f"Expected sequence length > {input_length} for first token, got {seq_len}")
            decision_point_hidden = layer_hidden[0, input_length, :]  # First generated token position
        else:
            decision_point_hidden = layer_hidden[0, input_length - 1, :]  # Last input token position
        all_layer_hidden_states[layer_idx] = decision_point_hidden.detach().cpu().numpy()

    return all_layer_hidden_states


def collect_probing_data(model, processor, level, max_images=None, first_token=False):

    level_dir = os.path.join(BASE_DATA_PATH, f"level_{level}")
    ann_dir = os.path.join(level_dir, "ann")
    img_dir = os.path.join(level_dir, "images")

    if not os.path.exists(ann_dir):
        raise FileNotFoundError(f"Level {level} directory not found: {ann_dir}")

    json_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".json")])
    if max_images:
        json_files = json_files[:max_images]

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

            conversation = build_visual_yesno_prompt(question)
            prompt_text = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = processor(text=prompt_text, images=image, return_tensors="pt")
            inputs = _move_to_device(inputs, device)

            try:
                all_layer_states = extract_all_layer_hidden_states(
                    model, processor, inputs, num_layers, first_token
                )

                for layer_idx, hidden_state in all_layer_states.items():
                    hidden_states_by_layer[layer_idx].append(hidden_state)

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
        return 0.0, 0.0

    X = np.array(hidden_states)  # Shape: (n_samples, hidden_dim)
    y = np.array(labels)  # Shape: (n_samples,)

    probe = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(probe, X, y, cv=cv_folds, scoring="accuracy")

    return cv_scores.mean(), cv_scores.std()

def extract_all_head_attentions(model, processor, inputs, specific_layers=[13, 16, 19, 24], first_token=False):
    """
    Extract attention patterns from specific heads in specific layers.
    
    Args:
        model: The LLaVA model
        processor: The processor
        inputs: Model inputs
        specific_layers: List of layer indices to extract from
        first_token: Whether to use first generated token or last input token (currently only last input token supported)
    
    Returns:
        Dictionary: {layer_idx: {head_idx: attention_pattern}}
        where attention_pattern is attention from last token to image patches [num_image_tokens]
    """
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            output_attentions=True,
            return_dict_in_generate=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    
    if not outputs.attentions or len(outputs.attentions) == 0:
        raise ValueError("No attentions returned from generation")
    
    input_step_attentions = outputs.attentions[0]  # tuple of [num_layers] tensors
    
    image_token_id = model.config.image_token_index
    full_input_ids = inputs["input_ids"][0]  # [seq_len]
    image_token_positions = (full_input_ids == image_token_id).nonzero(as_tuple=True)[0].tolist()
    
    if not image_token_positions:
        raise ValueError("No image tokens found in sequence")
    
    last_token_idx = input_length - 1
    
    all_head_attentions = {}
    for layer_idx in specific_layers:
        if layer_idx >= len(input_step_attentions):
            continue
        
        layer_attn = input_step_attentions[layer_idx]  # [1, heads, seq, seq]
        layer_attn = layer_attn[0]  # Remove batch dim: [heads, seq, seq]
        num_heads = layer_attn.shape[0]
        all_head_attentions[layer_idx] = {}
        
        for head_idx in range(num_heads):
            head_attn = layer_attn[head_idx, last_token_idx, :]  # [seq]
            
            image_attn = head_attn[image_token_positions]  # [num_image_tokens]
            
            image_attn_sum = image_attn.sum()
            if image_attn_sum > 0:
                image_attn_probs = image_attn / image_attn_sum
            else:
                image_attn_probs = image_attn
            
            all_head_attentions[layer_idx][head_idx] = image_attn_probs.detach().cpu().numpy()
    
    return all_head_attentions

def collect_head_probing_data(model, processor, level, max_images=None, 
                              specific_layers=[13, 16, 19, 24], first_token=False):
    """
    Collect head-specific attention patterns for probing.
    Reuses the same structure as collect_probing_data but extracts attentions instead.
    """
    level_dir = os.path.join(BASE_DATA_PATH, f"level_{level}")
    ann_dir = os.path.join(level_dir, "ann")
    img_dir = os.path.join(level_dir, "images")
    
    if not os.path.exists(ann_dir):
        raise FileNotFoundError(f"Level {level} directory not found: {ann_dir}")
    
    json_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".json")])
    if max_images:
        json_files = json_files[:max_images]
    
    head_features_by_layer_head = {}
    for layer in specific_layers:
        head_features_by_layer_head[layer] = {head: [] for head in range(32)}
    
    labels = []
    metadata = []
    
    device = _pick_model_input_device(model)
    
    print(f"Collecting head probing data from {len(json_files)} images...")
    
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
            
            conversation = build_visual_yesno_prompt(question)
            prompt_text = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = processor(text=prompt_text, images=image, return_tensors="pt")
            inputs = _move_to_device(inputs, device)
            
            try:
                all_head_attentions = extract_all_head_attentions(
                    model, processor, inputs, specific_layers, first_token
                )
                
                for layer_idx, heads in all_head_attentions.items():
                    for head_idx, attention_pattern in heads.items():
                        head_features_by_layer_head[layer_idx][head_idx].append(attention_pattern)
                
                labels.append(1 if gt == "yes" else 0)
                metadata.append({
                    "image": file_id,
                    "question": question,
                    "ground_truth": gt,
                })
            except Exception as e:
                print(f"Error processing {file_id}, question: {question[:50]}... - {e}")
    
    print(f"Collected {len(labels)} examples")
    return head_features_by_layer_head, labels, metadata

def probe_head(head_features, labels, cv_folds=5):
    """
    Train a linear probe on head-specific features and return cross-validation accuracy.
    Same pattern as probe_layer.
    
    Args:
        head_features: list of numpy arrays, each representing features from a specific head
                      (e.g., attention pattern from that head)
        labels: list of binary labels (0 or 1)
        cv_folds: number of cross-validation folds
    
    Returns:
        mean_accuracy: mean CV accuracy
        std_accuracy: std of CV accuracy
    """
    if len(head_features) != len(labels):
        raise ValueError(
            f"Mismatch: {len(head_features)} head features, {len(labels)} labels"
        )
    
    if len(set(labels)) < 2:
        return 0.0, 0.0
    
    X = np.array(head_features)  # Shape: (n_samples, feature_dim)
    y = np.array(labels)
    
    probe = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(probe, X, y, cv=cv_folds, scoring="accuracy")
    
    return cv_scores.mean(), cv_scores.std()

def collect_head_com_data(model, processor, level, max_images=None, 
                          specific_layers=None, source_token_type="object"):
    """
    Collect center of mass data for attention heads across specified layers.
    
    Args:
        model: The LLaVA model
        processor: The processor
        level: Data level to process
        max_images: Maximum number of images to process
        specific_layers: List of layer indices to analyze (None = all layers)
        source_token_type: Which token to use as source for attention. Options: "subject", "relation", "object", "last"
    
    Returns:
        List of dictionaries containing CoM results for each image/QA pair
    """
    level_dir = os.path.join(BASE_DATA_PATH, f"level_{level}")
    ann_dir = os.path.join(level_dir, "ann")
    img_dir = os.path.join(level_dir, "images")
    
    if not os.path.exists(ann_dir):
        raise FileNotFoundError(f"Level {level} directory not found: {ann_dir}")
    
    json_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".json")])
    if max_images:
        json_files = json_files[:max_images]
    
    device = _pick_model_input_device(model)
    all_com_results = []
    
    print(f"Collecting head CoM data from {len(json_files)} images...")
    
    for filename in tqdm(json_files, desc="Processing images"):
        file_id = filename.replace(".json", "")
        ann_path = os.path.join(ann_dir, filename)
        
        with open(ann_path, "r") as f:
            annotation = json.load(f)
        
        image_path = os.path.join(img_dir, f"{file_id}.png")
        if not os.path.exists(image_path):
            continue
        
        image = Image.open(image_path).convert("RGB")
        qa_list = annotation.get("qa", [])
        
        for qa_pair in qa_list:
            question = (qa_pair.get("question") or "").strip()
            if not question:
                continue
            
            conversation = build_visual_yesno_prompt(question)
            prompt_text = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = processor(text=prompt_text, images=image, return_tensors="pt")
            inputs = _move_to_device(inputs, device)
            
            try:
                com_results, prediction, p_yes, p_no = calc_com(
                    model, processor, inputs, image, annotation, qa_pair,
                    specific_layers=specific_layers,
                    source_token_type=source_token_type
                )
                
                all_com_results.append({
                    "image_id": file_id,
                    "qa_id": qa_pair.get("id"),
                    "question": question,
                    "ground_truth": qa_pair.get("answer"),
                    "model_prediction": prediction,
                    "p_yes": float(p_yes) if p_yes is not None else None,
                    "p_no": float(p_no) if p_no is not None else None,
                    "com_results": com_results,
                    "source_token_type": source_token_type
                })
            except Exception as e:
                print(
                    f"Error processing {file_id}, question: {question[:50]}... - {e}"
                )
                continue
    
    return all_com_results

def calc_com(model, processor, inputs, image, annotation, qa_pair, 
             specific_layers=None, source_token_type="object"):
    """
    Calculate the center of mass distance between the head attention map and the ground truth centers
    for both subject and object.
    
    Uses attention from a specified source token (subject, relation, or object) instead of the last input token.
    
    Args:
        model: The LLaVA model
        processor: The processor
        inputs: Model inputs (from processor)
        image: PIL Image
        annotation: Annotation dict with objects and meta
        qa_pair: QA pair dict with subject_id/object_id
        specific_layers: List of layer indices to analyze (None = all layers)
        source_token_type: Which token to use as source for attention. Options: "subject", "relation", "object", "last"
    
    Returns:
        Dictionary with per-layer, per-head center of mass distances to subject and object centers
    """
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            output_attentions=True,
            output_scores=True,  # For Yes/No prediction
            return_dict_in_generate=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    
    if not outputs.attentions or len(outputs.attentions) == 0:
        raise ValueError("No attentions returned from generation")
    
    input_step_attentions = outputs.attentions[0]  # tuple of [num_layers] tensors
    
    if specific_layers is None:
        num_layers = len(input_step_attentions)
        specific_layers = list(range(num_layers))
    
    image_token_id = model.config.image_token_index
    full_input_ids = inputs["input_ids"][0]  # [seq_len]
    image_token_positions = (full_input_ids == image_token_id).nonzero(as_tuple=True)[0].tolist()
    
    if not image_token_positions:
        raise ValueError("No image tokens found in sequence")
    
    entity_indices = get_entity_indices(
        processor.tokenizer,
        full_input_ids,
        qa_pair,
        annotation
    )

    
    source_token_idx = None
    source_token_name = None
    
    if source_token_type == "subject":
        subject_token_indices = entity_indices.get("subject", [])
        if not subject_token_indices:
            raise ValueError(f"Subject tokens not found in sequence for subject_id={qa_pair.get('subject_id')}")
        source_token_idx = subject_token_indices[-1]  # Use last token of subject phrase
        source_token_name = "subject"
    
    elif source_token_type == "relation":
        rel_phrase = qa_pair.get("rel_phrase", "")
        if not rel_phrase:
            raise ValueError(f"rel_phrase not found in qa_pair")
        relation_indices = get_phrase_token_positions(
            processor.tokenizer,
            full_input_ids,
            rel_phrase
        ).get("relation", [])
        if not relation_indices:
            raise ValueError(f"Relation tokens not found for rel_phrase='{rel_phrase}'")
        source_token_idx = relation_indices[-1]  # Use last token of relation phrase
        source_token_name = "relation"
    
    elif source_token_type == "object":
        object_token_indices = entity_indices.get("object", [])
        if not object_token_indices:
            raise ValueError(f"Object tokens not found in sequence for object_id={qa_pair.get('object_id')}")
        source_token_idx = object_token_indices[-1]  # Use last token of object phrase
        source_token_name = "object"
    
    elif source_token_type == "last":
        source_token_idx = full_input_ids.shape[-1] - 1
        source_token_name = "last"
    
    else:
        raise ValueError(f"Invalid source_token_type: {source_token_type}. Must be 'subject', 'relation', 'object', or 'last'")
    
    grid_dim = annotation.get("meta", {}).get("grid_dim", 24)
    patch_size = annotation.get("meta", {}).get("patch", 14)
    
    objects = annotation.get("objects", [])
    subject_id = qa_pair.get("subject_id")
    object_id = qa_pair.get("object_id")
    
    gt_subject_patch_indices = None
    gt_object_patch_indices = None
    
    for obj in objects:
        if obj.get("id") == subject_id and "patch_indices" in obj:
            gt_subject_patch_indices = obj["patch_indices"]
        if obj.get("id") == object_id and "patch_indices" in obj:
            gt_object_patch_indices = obj["patch_indices"]
    
    if gt_subject_patch_indices is None or len(gt_subject_patch_indices) == 0:
        raise ValueError(f"Subject patch_indices not found for subject_id={subject_id}")
    if gt_object_patch_indices is None or len(gt_object_patch_indices) == 0:
        raise ValueError(f"Object patch_indices not found for object_id={object_id}")
    
    gt_subject_mask = np.zeros((grid_dim, grid_dim))
    gt_object_mask = np.zeros((grid_dim, grid_dim))
    
    for idx in gt_subject_patch_indices:
        row = idx // grid_dim
        col = idx % grid_dim
        if 0 <= row < grid_dim and 0 <= col < grid_dim:
            gt_subject_mask[row, col] = 1
    
    for idx in gt_object_patch_indices:
        row = idx // grid_dim
        col = idx % grid_dim
        if 0 <= row < grid_dim and 0 <= col < grid_dim:
            gt_object_mask[row, col] = 1
    
    rows, cols = np.meshgrid(np.arange(grid_dim), np.arange(grid_dim), indexing='ij')
    gt_subject_sum = gt_subject_mask.sum()
    gt_object_sum = gt_object_mask.sum()
    
    if gt_subject_sum > 0:
        gt_subject_com_row = (gt_subject_mask * rows).sum() / gt_subject_sum
        gt_subject_com_col = (gt_subject_mask * cols).sum() / gt_subject_sum
        gt_subject_grid = (gt_subject_com_row, gt_subject_com_col)
    else:
        raise ValueError(f"Subject mask is empty for subject_id={subject_id}")
    
    if gt_object_sum > 0:
        gt_object_com_row = (gt_object_mask * rows).sum() / gt_object_sum
        gt_object_com_col = (gt_object_mask * cols).sum() / gt_object_sum
        gt_object_grid = (gt_object_com_row, gt_object_com_col)
    else:
        raise ValueError(f"Object mask is empty for object_id={object_id}")
    
    results = {}
    
    low_image_attn_threshold = 0.01  # Less than 1% attention to image
    total_heads = 0
    low_attn_heads = 0
    
    for layer_idx in specific_layers:
        if layer_idx >= len(input_step_attentions):
            continue
        
        layer_attn = input_step_attentions[layer_idx]
        layer_attn = layer_attn[0]  # Remove batch dim: [heads, seq, seq]
        
        num_heads = layer_attn.shape[0]
        results[layer_idx] = {}
        
        for head_idx in range(num_heads):
            total_heads += 1
            
            head_attn = layer_attn[head_idx, source_token_idx, :]
            
            image_attn = head_attn[image_token_positions]
            
            image_attn_sum = image_attn.sum().item()
            
            has_low_image_attention = image_attn_sum < low_image_attn_threshold
            if has_low_image_attention:
                low_attn_heads += 1
            
            if image_attn_sum > 0:
                image_attn_probs = image_attn / image_attn_sum
            else:
                image_attn_probs = image_attn
            
            if len(image_token_positions) != grid_dim * grid_dim:
                attn_2d = np.zeros((grid_dim, grid_dim))
                for i, pos in enumerate(image_token_positions):
                    if i < grid_dim * grid_dim:
                        row = i // grid_dim
                        col = i % grid_dim
                        attn_2d[row, col] = image_attn_probs[i].item()
            else:
                attn_2d = image_attn_probs.reshape(grid_dim, grid_dim).cpu().numpy()
            
            rows, cols = np.meshgrid(np.arange(grid_dim), np.arange(grid_dim), indexing='ij')
            com_row = (attn_2d * rows).sum() / (attn_2d.sum() + 1e-9)
            com_col = (attn_2d * cols).sum() / (attn_2d.sum() + 1e-9)
            com_grid = (com_row, com_col)
            
            dist_to_subject = np.sqrt(
                (com_row - gt_subject_grid[0])**2 + 
                (com_col - gt_subject_grid[1])**2
            )
            
            dist_to_object = np.sqrt(
                (com_row - gt_object_grid[0])**2 + 
                (com_col - gt_object_grid[1])**2
            )
            
            
            subject_intersection = (attn_2d * gt_subject_mask).sum()
            subject_union = attn_2d.sum() + gt_subject_mask.sum() - subject_intersection
            iou_with_subject = float(subject_intersection / (subject_union + 1e-10))
            
            object_intersection = (attn_2d * gt_object_mask).sum()
            object_union = attn_2d.sum() + gt_object_mask.sum() - object_intersection
            iou_with_object = float(object_intersection / (object_union + 1e-10))
            
            results[layer_idx][head_idx] = {
                "com_grid": (float(com_row), float(com_col)),
                "gt_subject_grid": gt_subject_grid,
                "gt_object_grid": gt_object_grid,
                "distance_to_subject": float(dist_to_subject),
                "distance_to_object": float(dist_to_object),
                "iou_with_subject": float(iou_with_subject),
                "iou_with_object": float(iou_with_object),
                "attention_sum": float(image_attn_sum),
                "has_low_image_attention": has_low_image_attention,
                "source_token_type": source_token_name,
                "source_token_idx": int(source_token_idx),
            }
    
    prediction, p_yes, p_no = score_yesno(outputs, processor.tokenizer)
    
    return results, prediction, p_yes, p_no

def plot_com_heatmaps_from_json(json_path, output_dir=None):
    """
    Load CoM results from JSON file and plot heatmaps.
    
    Args:
        json_path: Path to the head_com_results.json file
        output_dir: Directory to save plots (default: same directory as JSON file)
    """
    with open(json_path, 'r') as f:
        com_results = json.load(f)
    
    if not com_results:
        print(f"No data found in {json_path}")
        return
    
    json_dir = os.path.dirname(json_path)
    json_basename = os.path.basename(json_dir)
    
    level = None
    for part in json_path.split(os.sep):
        if part.startswith('level_'):
            try:
                level = int(part.split('_')[1])
                break
            except (ValueError, IndexError):
                continue
    
    if level is None:
        if 'level_' in json_basename:
            try:
                level = int(json_basename.split('level_')[1].split('_')[0])
            except (ValueError, IndexError):
                pass
    
    if level is None:
        print("Warning: Could not extract level from path. Using level 1 as default.")
        level = 1
    
    source_token_type = com_results[0].get("source_token_type", "object")
    
    if output_dir is None:
        output_dir = json_dir
    
    plot_com_heatmaps(com_results, output_dir, level, source_token_type)
    plot_iou_heatmaps(com_results, output_dir, level, source_token_type)
    print_top_grounding_heads(com_results, output_dir, level, source_token_type)
    plot_com_distance_lines(com_results, output_dir, level, source_token_type)
    plot_iou_by_prediction(com_results, output_dir, level, source_token_type)
    plot_com_distance_by_prediction(com_results, output_dir, level, source_token_type)
    print(f"Heatmaps saved to: {os.path.abspath(output_dir)}")

def plot_com_heatmaps(com_results, output_dir, level, source_token_type):
    """
    Plot heatmaps showing distance to subject/object for each head in each layer.
    
    Args:
        com_results: List of dicts with 'com_results' containing per-layer, per-head data
        output_dir: Directory to save the plots
        level: Data level for title
        source_token_type: Source token type used for CoM calculation (from args)
    """
    if not com_results:
        print("No CoM results to plot")
        return
    
    aggregated = {}
    
    for result in com_results:
        com_data = result.get("com_results", {})
        
        all_distances_subject = []
        all_distances_object = []
        
        for layer_str, heads in com_data.items():
            for head_str, head_data in heads.items():
                dist_subj = head_data.get('distance_to_subject', 0)
                dist_obj = head_data.get('distance_to_object', 0)
                if dist_subj > 0:  # Only include non-zero distances
                    all_distances_subject.append(dist_subj)
                if dist_obj > 0:
                    all_distances_object.append(dist_obj)
        
        if all_distances_subject:
            min_subj = min(all_distances_subject)
            max_subj = max(all_distances_subject)
            range_subj = max_subj - min_subj if max_subj > min_subj else 1.0
        else:
            min_subj, max_subj, range_subj = 0, 1, 1
        
        if all_distances_object:
            min_obj = min(all_distances_object)
            max_obj = max(all_distances_object)
            range_obj = max_obj - min_obj if max_obj > min_obj else 1.0
        else:
            min_obj, max_obj, range_obj = 0, 1, 1
        
        for layer_str, heads in com_data.items():
            layer = int(layer_str)
            if layer not in aggregated:
                aggregated[layer] = {head: {'dist_subject': [], 'dist_object': []} 
                                     for head in range(32)}
            
            for head_str, head_data in heads.items():
                head = int(head_str)
                if head < 32:
                    dist_subj = head_data.get('distance_to_subject', 0)
                    dist_obj = head_data.get('distance_to_object', 0)
                    
                    if dist_subj > 0:
                        normalized_subj = (dist_subj - min_subj) / range_subj
                        aggregated[layer][head]['dist_subject'].append(normalized_subj)
                    
                    if dist_obj > 0:
                        normalized_obj = (dist_obj - min_obj) / range_obj
                        aggregated[layer][head]['dist_object'].append(normalized_obj)
    
    if not aggregated:
        print("No aggregated data to plot")
        return
    
    layers = sorted(aggregated.keys())
    num_layers = len(layers)
    num_heads = 32
    
    dist_subject_heatmap = np.zeros((num_layers, num_heads))
    dist_object_heatmap = np.zeros((num_layers, num_heads))
    
    for i, layer in enumerate(layers):
        for head in range(num_heads):
            if aggregated[layer][head]['dist_subject']:
                dist_subject_heatmap[i, head] = np.mean(aggregated[layer][head]['dist_subject'])
            if aggregated[layer][head]['dist_object']:
                dist_object_heatmap[i, head] = np.mean(aggregated[layer][head]['dist_object'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    im1 = ax1.imshow(dist_subject_heatmap, aspect='auto', cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1)
    ax1.set_xlabel('Attention Head', fontsize=12)
    ax1.set_ylabel('Layer', fontsize=12)
    ax1.set_title(f'Average Distance to Subject CoM\n(Level {level}, Source: {source_token_type})', fontsize=14)
    ax1.set_yticks(range(num_layers))
    ax1.set_yticklabels(layers)
    ax1.set_xticks(range(0, num_heads, 4))
    ax1.set_xticklabels(range(0, num_heads, 4))
    plt.colorbar(im1, ax=ax1, label='Normalized Distance (0-1, per-image)')
    
    im2 = ax2.imshow(dist_object_heatmap, aspect='auto', cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1)
    ax2.set_xlabel('Attention Head', fontsize=12)
    ax2.set_ylabel('Layer', fontsize=12)
    ax2.set_title(f'Average Distance to Object CoM\n(Level {level}, Source: {source_token_type})', fontsize=14)
    ax2.set_yticks(range(num_layers))
    ax2.set_yticklabels(layers)
    ax2.set_xticks(range(0, num_heads, 4))
    ax2.set_xticklabels(range(0, num_heads, 4))
    plt.colorbar(im2, ax=ax2, label='Normalized Distance (0-1, per-image)')
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "com_distance_heatmaps.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"CoM heatmaps saved to: {save_path}")


def plot_iou_heatmaps(com_results, output_dir, level, source_token_type):
    """
    Plot heatmaps showing IoU with subject/object for each head in each layer.
    
    Args:
        com_results: List of dicts with 'com_results' containing per-layer, per-head data
        output_dir: Directory to save the plots
        level: Data level for title
        source_token_type: Source token type used for CoM calculation (from args)
    """
    if not com_results:
        print("No CoM results to plot for IoU")
        return
    
    aggregated = {}
    
    for result in com_results:
        com_data = result.get("com_results", {})
        
        for layer_str, heads in com_data.items():
            layer = int(layer_str)
            if layer not in aggregated:
                aggregated[layer] = {head: {'iou_subject': [], 'iou_object': []} 
                                     for head in range(32)}
            
            for head_str, head_data in heads.items():
                head = int(head_str)
                if head < 32:
                    iou_subj = head_data.get('iou_with_subject', 0)
                    iou_obj = head_data.get('iou_with_object', 0)
                    
                    aggregated[layer][head]['iou_subject'].append(iou_subj)
                    aggregated[layer][head]['iou_object'].append(iou_obj)
    
    if not aggregated:
        print("No aggregated IoU data to plot")
        return
    
    layers = sorted(aggregated.keys())
    num_layers = len(layers)
    num_heads = 32
    
    iou_subject_heatmap = np.zeros((num_layers, num_heads))
    iou_object_heatmap = np.zeros((num_layers, num_heads))
    
    for i, layer in enumerate(layers):
        for head in range(num_heads):
            if aggregated[layer][head]['iou_subject']:
                iou_subject_heatmap[i, head] = np.mean(aggregated[layer][head]['iou_subject'])
            if aggregated[layer][head]['iou_object']:
                iou_object_heatmap[i, head] = np.mean(aggregated[layer][head]['iou_object'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    im1 = ax1.imshow(iou_subject_heatmap, aspect='auto', cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    ax1.set_xlabel('Attention Head', fontsize=12)
    ax1.set_ylabel('Layer', fontsize=12)
    ax1.set_title(f'Average Soft IoU with Subject\n(Level {level}, Source: {source_token_type})', fontsize=14)
    ax1.set_yticks(range(num_layers))
    ax1.set_yticklabels(layers)
    ax1.set_xticks(range(0, num_heads, 4))
    ax1.set_xticklabels(range(0, num_heads, 4))
    plt.colorbar(im1, ax=ax1, label='Soft IoU (0-1)')
    
    im2 = ax2.imshow(iou_object_heatmap, aspect='auto', cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    ax2.set_xlabel('Attention Head', fontsize=12)
    ax2.set_ylabel('Layer', fontsize=12)
    ax2.set_title(f'Average Soft IoU with Object\n(Level {level}, Source: {source_token_type})', fontsize=14)
    ax2.set_yticks(range(num_layers))
    ax2.set_yticklabels(layers)
    ax2.set_xticks(range(0, num_heads, 4))
    ax2.set_xticklabels(range(0, num_heads, 4))
    plt.colorbar(im2, ax=ax2, label='Soft IoU (0-1)')
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "iou_heatmaps.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Soft IoU heatmaps saved to: {save_path}")


def print_top_grounding_heads(com_results, output_dir, level, source_token_type, top_k=20):
    """
    Rank attention heads by average Soft IoU with object to identify grounding heads.
    
    Args:
        com_results: List of dicts with 'com_results'
        output_dir: Directory to save results
        level: Data level
        source_token_type: Source token type used
        top_k: Number of top heads to display
    """
    if not com_results:
        print("No CoM results for head ranking")
        return
    
    head_scores = {}  # (layer, head) -> list of IoU values
    
    for result in com_results:
        com_data = result.get("com_results", {})
        
        for layer_str, heads in com_data.items():
            layer = int(layer_str)
            for head_str, head_data in heads.items():
                head = int(head_str)
                key = (layer, head)
                
                iou = head_data.get("iou_with_object", 0)
                
                if key not in head_scores:
                    head_scores[key] = {"iou": []}
                
                head_scores[key]["iou"].append(iou)
    
    if not head_scores:
        print("No head scores to rank")
        return
    
    ranked = []
    for (layer, head), scores in head_scores.items():
        mean_iou = np.mean(scores["iou"])
        ranked.append((layer, head, mean_iou))
    
    ranked.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n{'='*60}")
    print(f"TOP {top_k} GROUNDING HEADS BY SOFT IoU")
    print(f"(Level {level}, Source: {source_token_type})")
    print(f"{'='*60}")
    print(f"{'Rank':>4} | {'Layer':>5} | {'Head':>4} | {'Soft IoU':>10}")
    print(f"{'-'*4}-+-{'-'*5}-+-{'-'*4}-+-{'-'*10}")
    
    for i, (layer, head, iou) in enumerate(ranked[:top_k], 1):
        print(f"{i:>4} | {layer:>5} | {head:>4} | {iou:>10.4f}")
    
    print(f"{'='*60}")
    
    print(f"\nBOTTOM 5 HEADS (least grounding):")
    print(f"{'Rank':>4} | {'Layer':>5} | {'Head':>4} | {'Soft IoU':>10}")
    print(f"{'-'*4}-+-{'-'*5}-+-{'-'*4}-+-{'-'*10}")
    for i, (layer, head, iou) in enumerate(ranked[-5:], len(ranked)-4):
        print(f"{i:>4} | {layer:>5} | {head:>4} | {iou:>10.4f}")
    
    print(f"{'='*70}\n")
    
    save_path = os.path.join(output_dir, "top_grounding_heads.txt")
    with open(save_path, 'w') as f:
        f.write(f"Top Grounding Heads by Soft IoU\n")
        f.write(f"Level {level}, Source: {source_token_type}\n")
        f.write(f"{'='*70}\n")
        f.write(f"{'Rank':>4} | {'Layer':>5} | {'Head':>4} | {'Soft IoU':>10}\n")
        for i, (layer, head, iou) in enumerate(ranked, 1):
            f.write(f"{i:>4} | {layer:>5} | {head:>4} | {iou:>10.4f}\n")
    
    print(f"Full ranking saved to: {save_path}")


def plot_com_distance_lines(com_results, output_dir, level, source_token_type):
    """
    Plot line graphs showing average CoM distance to subject/object per layer.
    Uses per-image min-max normalization to reduce noise.
    
    Args:
        com_results: List of dicts with 'com_results' containing per-layer, per-head data
        output_dir: Directory to save the plots
        level: Data level for title
        source_token_type: Source token type used for CoM calculation
    """
    if not com_results:
        print("No CoM results to plot for line graph")
        return
    
    aggregated = {}
    
    for result in com_results:
        com_data = result.get("com_results", {})
        
        all_distances_subject = []
        all_distances_object = []
        
        for layer_str, heads in com_data.items():
            for head_str, head_data in heads.items():
                dist_subj = head_data.get('distance_to_subject', 0)
                dist_obj = head_data.get('distance_to_object', 0)
                if dist_subj > 0:
                    all_distances_subject.append(dist_subj)
                if dist_obj > 0:
                    all_distances_object.append(dist_obj)
        
        if all_distances_subject:
            min_subj = min(all_distances_subject)
            max_subj = max(all_distances_subject)
            range_subj = max_subj - min_subj if max_subj > min_subj else 1.0
        else:
            min_subj, range_subj = 0, 1
        
        if all_distances_object:
            min_obj = min(all_distances_object)
            max_obj = max(all_distances_object)
            range_obj = max_obj - min_obj if max_obj > min_obj else 1.0
        else:
            min_obj, range_obj = 0, 1
        
        for layer_str, heads in com_data.items():
            layer = int(layer_str)
            if layer not in aggregated:
                aggregated[layer] = {'dist_subject': [], 'dist_object': []}
            
            for head_str, head_data in heads.items():
                dist_subj = head_data.get('distance_to_subject', 0)
                dist_obj = head_data.get('distance_to_object', 0)
                
                if dist_subj > 0:
                    normalized_subj = (dist_subj - min_subj) / range_subj
                    aggregated[layer]['dist_subject'].append(normalized_subj)
                
                if dist_obj > 0:
                    normalized_obj = (dist_obj - min_obj) / range_obj
                    aggregated[layer]['dist_object'].append(normalized_obj)
    
    if not aggregated:
        print("No aggregated distance data for line plot")
        return
    
    layers = sorted(aggregated.keys())
    
    mean_subject = [np.mean(aggregated[l]['dist_subject']) if aggregated[l]['dist_subject'] else 0 for l in layers]
    std_subject = [np.std(aggregated[l]['dist_subject']) if aggregated[l]['dist_subject'] else 0 for l in layers]
    mean_object = [np.mean(aggregated[l]['dist_object']) if aggregated[l]['dist_object'] else 0 for l in layers]
    std_object = [np.std(aggregated[l]['dist_object']) if aggregated[l]['dist_object'] else 0 for l in layers]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(layers, mean_subject, 'b-o', label='Distance to Subject', linewidth=2, markersize=4)
    plt.fill_between(layers, 
                     np.array(mean_subject) - np.array(std_subject),
                     np.array(mean_subject) + np.array(std_subject),
                     alpha=0.2, color='blue')
    
    plt.plot(layers, mean_object, 'r-o', label='Distance to Object', linewidth=2, markersize=4)
    plt.fill_between(layers,
                     np.array(mean_object) - np.array(std_object),
                     np.array(mean_object) + np.array(std_object),
                     alpha=0.2, color='red')
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Normalized Distance (0-1, per-image)', fontsize=12)
    plt.ylim(0, 1)
    plt.title(f'CoM Distance by Layer\n(Level {level}, Source: {source_token_type})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "com_distance_by_layer.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"CoM distance line plot saved to: {save_path}")


def plot_iou_by_prediction(com_results, output_dir, level, source_token_type):
    """
    Compare IoU distributions for TP, TN, FP, FN predictions.
    
    TP: GT=Yes, Pred=Yes | TN: GT=No, Pred=No
    FP: GT=No, Pred=Yes  | FN: GT=Yes, Pred=No
    
    Args:
        com_results: List of dicts with 'com_results', 'model_prediction', 'ground_truth'
        output_dir: Directory to save the plot
        level: Data level for title
        source_token_type: Source token type used
    """
    if not com_results:
        print("No CoM results for IoU by prediction analysis")
        return
    
    has_predictions = any(result.get("model_prediction") for result in com_results)
    if not has_predictions:
        print("No model predictions found. Skipping IoU by prediction analysis.")
        return
    
    tp_ious = []  # GT=Yes, Pred=Yes
    tn_ious = []  # GT=No, Pred=No
    fp_ious = []  # GT=No, Pred=Yes
    fn_ious = []  # GT=Yes, Pred=No
    
    for result in com_results:
        model_prediction = (result.get("model_prediction") or "").lower()
        ground_truth = (result.get("ground_truth") or "").lower()
        com_data = result.get("com_results", {})
        
        if model_prediction not in ["yes", "no"] or ground_truth not in ["yes", "no"]:
            continue
        
        iou_values = []
        for layer_str, heads in com_data.items():
            for head_str, head_data in heads.items():
                iou = head_data.get("iou_with_object", 0)
                iou_values.append(iou)
        
        if not iou_values:
            continue
        
        avg_iou = np.mean(iou_values)
        
        if ground_truth == "yes" and model_prediction == "yes":
            tp_ious.append(avg_iou)
        elif ground_truth == "no" and model_prediction == "no":
            tn_ious.append(avg_iou)
        elif ground_truth == "no" and model_prediction == "yes":
            fp_ious.append(avg_iou)
        elif ground_truth == "yes" and model_prediction == "no":
            fn_ious.append(avg_iou)
    
    if not any([tp_ious, tn_ious, fp_ious, fn_ious]):
        print("No valid predictions for IoU analysis")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    data_to_plot = []
    labels = []
    colors = []
    
    if tp_ious:
        data_to_plot.append(tp_ious)
        labels.append(f'TP\n(n={len(tp_ious)})')
        colors.append('green')
    if tn_ious:
        data_to_plot.append(tn_ious)
        labels.append(f'TN\n(n={len(tn_ious)})')
        colors.append('blue')
    if fp_ious:
        data_to_plot.append(fp_ious)
        labels.append(f'FP\n(n={len(fp_ious)})')
        colors.append('orange')
    if fn_ious:
        data_to_plot.append(fn_ious)
        labels.append(f'FN\n(n={len(fn_ious)})')
        colors.append('red')
    
    bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax1.set_ylabel('Soft IoU with Object (avg across heads)')
    ax1.set_title(f'Soft IoU Distribution by Outcome\n(Level {level}, Source: {source_token_type})')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(0.3, ax1.get_ylim()[1]))  # Ensure we see low values
    
    for i, (data, color) in enumerate(zip(data_to_plot, colors), 1):
        mean_val = np.mean(data)
        ax1.hlines(mean_val, i - 0.3, i + 0.3, colors=color, linewidth=2, linestyles='--')
    
    all_ious = tp_ious + tn_ious + fp_ious + fn_ious
    max_iou = max(all_ious) if all_ious else 0.3
    bins = np.linspace(0, max(0.3, max_iou), 20)
    
    if tp_ious:
        ax2.hist(tp_ious, bins=bins, alpha=0.5, label=f'TP (n={len(tp_ious)})', color='green')
    if tn_ious:
        ax2.hist(tn_ious, bins=bins, alpha=0.5, label=f'TN (n={len(tn_ious)})', color='blue')
    if fp_ious:
        ax2.hist(fp_ious, bins=bins, alpha=0.5, label=f'FP (n={len(fp_ious)})', color='orange')
    if fn_ious:
        ax2.hist(fn_ious, bins=bins, alpha=0.5, label=f'FN (n={len(fn_ious)})', color='red')
    
    ax2.set_xlabel('Soft IoU with Object')
    ax2.set_ylabel('Count')
    ax2.set_title('Soft IoU Histogram by Outcome')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "iou_by_prediction.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()



def plot_com_distance_by_prediction(com_results, output_dir, level, source_token_type):
    """
    Compare CoM Distance distributions for TP, TN, FP, FN predictions.
    Lower distance = attention center is closer to object.
    
    TP: GT=Yes, Pred=Yes | TN: GT=No, Pred=No
    FP: GT=No, Pred=Yes  | FN: GT=Yes, Pred=No
    
    Args:
        com_results: List of dicts with 'com_results', 'model_prediction', 'ground_truth'
        output_dir: Directory to save the plot
        level: Data level for title
        source_token_type: Source token type used
    """
    if not com_results:
        print("No CoM results for CoM distance by prediction analysis")
        return
    
    has_predictions = any(result.get("model_prediction") for result in com_results)
    if not has_predictions:
        print("No model predictions found. Skipping CoM distance by prediction analysis.")
        return
    
    tp_dist = []  # GT=Yes, Pred=Yes
    tn_dist = []  # GT=No, Pred=No
    fp_dist = []  # GT=No, Pred=Yes
    fn_dist = []  # GT=Yes, Pred=No
    
    for result in com_results:
        model_prediction = (result.get("model_prediction") or "").lower()
        ground_truth = (result.get("ground_truth") or "").lower()
        com_data = result.get("com_results", {})
        
        if model_prediction not in ["yes", "no"] or ground_truth not in ["yes", "no"]:
            continue
        
        dist_values = []
        for layer_str, heads in com_data.items():
            for head_str, head_data in heads.items():
                dist = head_data.get("distance_to_object", 0)
                dist_values.append(dist)
        
        if not dist_values:
            continue
        
        avg_dist = np.mean(dist_values)
        
        if ground_truth == "yes" and model_prediction == "yes":
            tp_dist.append(avg_dist)
        elif ground_truth == "no" and model_prediction == "no":
            tn_dist.append(avg_dist)
        elif ground_truth == "no" and model_prediction == "yes":
            fp_dist.append(avg_dist)
        elif ground_truth == "yes" and model_prediction == "no":
            fn_dist.append(avg_dist)
    
    if not any([tp_dist, tn_dist, fp_dist, fn_dist]):
        print("No valid predictions for CoM distance analysis")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    data_to_plot = []
    labels = []
    colors = []
    
    if tp_dist:
        data_to_plot.append(tp_dist)
        labels.append(f'TP\n(n={len(tp_dist)})')
        colors.append('green')
    if tn_dist:
        data_to_plot.append(tn_dist)
        labels.append(f'TN\n(n={len(tn_dist)})')
        colors.append('blue')
    if fp_dist:
        data_to_plot.append(fp_dist)
        labels.append(f'FP\n(n={len(fp_dist)})')
        colors.append('orange')
    if fn_dist:
        data_to_plot.append(fn_dist)
        labels.append(f'FN\n(n={len(fn_dist)})')
        colors.append('red')
    
    bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax1.set_ylabel('CoM Distance to Object (patches) - avg across heads')
    ax1.set_title(f'CoM Distance Distribution by Outcome\n(Level {level}, Source: {source_token_type})\n(Lower = closer to object)')
    ax1.grid(True, alpha=0.3)
    
    for i, (data, color) in enumerate(zip(data_to_plot, colors), 1):
        mean_val = np.mean(data)
        ax1.hlines(mean_val, i - 0.3, i + 0.3, colors=color, linewidth=2, linestyles='--')
    
    all_dist = tp_dist + tn_dist + fp_dist + fn_dist
    max_dist = max(all_dist) if all_dist else 20
    bins = np.linspace(0, max_dist * 1.1, 20)
    
    if tp_dist:
        ax2.hist(tp_dist, bins=bins, alpha=0.5, label=f'TP (n={len(tp_dist)})', color='green')
    if tn_dist:
        ax2.hist(tn_dist, bins=bins, alpha=0.5, label=f'TN (n={len(tn_dist)})', color='blue')
    if fp_dist:
        ax2.hist(fp_dist, bins=bins, alpha=0.5, label=f'FP (n={len(fp_dist)})', color='orange')
    if fn_dist:
        ax2.hist(fn_dist, bins=bins, alpha=0.5, label=f'FN (n={len(fn_dist)})', color='red')
    
    ax2.set_xlabel('CoM Distance to Object (patches)')
    ax2.set_ylabel('Count')
    ax2.set_title('CoM Distance Histogram by Outcome')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "com_distance_by_prediction.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()




def plot_head_accuracy(results, output_dir, level):
    """
    Plot head probing accuracy heatmap.
    
    Args:
        results: dict with 'head_scores' containing list of dicts with 'layer', 'head', 'accuracy', 'std'
        output_dir: directory to save the plot
        level: data level for title
    """
    head_scores = results.get("head_scores", [])
    if not head_scores:
        print("No head scores to plot")
        return
    
    layers = sorted(set(s["layer"] for s in head_scores))
    num_layers = len(layers)
    num_heads = 32
    
    accuracy_heatmap = np.zeros((num_layers, num_heads))
    
    for score in head_scores:
        layer_idx = layers.index(score["layer"])
        head_idx = score["head"]
        if head_idx < num_heads:
            accuracy_heatmap[layer_idx, head_idx] = score["accuracy"]
    
    plt.figure(figsize=(14, 6))
    
    im = plt.imshow(accuracy_heatmap, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.xlabel('Attention Head', fontsize=12)
    plt.ylabel('Layer', fontsize=12)
    plt.title(f'Head Probing Accuracy (Level {level})\n{results.get("num_examples", 0)} examples', fontsize=14)
    plt.yticks(range(num_layers), layers)
    plt.xticks(range(0, num_heads, 4), range(0, num_heads, 4))
    plt.colorbar(im, label='Cross-Validation Accuracy')
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "head_probing_accuracy.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Head probing plot saved to: {save_path}")

def plot_accuracy_vs_distance(head_probing_results_path, com_results_path, output_dir, level):
    """
    Plot head probing accuracy vs CoM distance to subject/object.
    
    Args:
        head_probing_results_path: Path to head_probing_results.json
        com_results_path: Path to head_com_results.json
        output_dir: Directory to save the plot
        level: Data level for title
    """
    with open(head_probing_results_path, "r") as f:
        head_results = json.load(f)
    
    with open(com_results_path, "r") as f:
        com_results = json.load(f)
    
    com_aggregated = {}
    
    for result in com_results:
        com_data = result.get("com_results", {})
        for layer_str, heads in com_data.items():
            layer = int(layer_str)
            if layer not in com_aggregated:
                com_aggregated[layer] = {head: {'dist_subject': [], 'dist_object': []} 
                                        for head in range(32)}
            
            for head_str, head_data in heads.items():
                head = int(head_str)
                if head < 32:
                    com_aggregated[layer][head]['dist_subject'].append(
                        head_data.get('distance_to_subject', 0)
                    )
                    com_aggregated[layer][head]['dist_object'].append(
                        head_data.get('distance_to_object', 0)
                    )
    
    accuracies = []
    dists_subject = []
    dists_object = []
    dists_avg = []
    layers = []
    heads = []
    
    for score in head_results.get("head_scores", []):
        layer = score["layer"]
        head = score["head"]
        accuracy = score["accuracy"]
        
        if layer in com_aggregated and head in com_aggregated[layer]:
            dist_subj_list = com_aggregated[layer][head]['dist_subject']
            dist_obj_list = com_aggregated[layer][head]['dist_object']
            
            if dist_subj_list and dist_obj_list:
                avg_dist_subject = np.mean(dist_subj_list)
                avg_dist_object = np.mean(dist_obj_list)
                avg_dist = (avg_dist_subject + avg_dist_object) / 2
                
                accuracies.append(accuracy)
                dists_subject.append(avg_dist_subject)
                dists_object.append(avg_dist_object)
                dists_avg.append(avg_dist)
                layers.append(layer)
                heads.append(head)
    
    if not accuracies:
        print("No matching data found between head probing and CoM results")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].scatter(dists_subject, accuracies, alpha=0.6, s=50)
    axes[0].set_xlabel('Average Distance to Subject CoM (grid units)', fontsize=11)
    axes[0].set_ylabel('Head Probing Accuracy', fontsize=11)
    axes[0].set_title('Accuracy vs Distance to Subject', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.5, linestyle="--", alpha=0.5, color='red', label='Random (0.5)')
    axes[0].legend()
    
    if len(dists_subject) > 1:
        corr_subj = np.corrcoef(dists_subject, accuracies)[0, 1]
        axes[0].text(0.05, 0.95, f'r = {corr_subj:.3f}', transform=axes[0].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[1].scatter(dists_object, accuracies, alpha=0.6, s=50, color='orange')
    axes[1].set_xlabel('Average Distance to Object CoM (grid units)', fontsize=11)
    axes[1].set_ylabel('Head Probing Accuracy', fontsize=11)
    axes[1].set_title('Accuracy vs Distance to Object', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.5, linestyle="--", alpha=0.5, color='red', label='Random (0.5)')
    axes[1].legend()
    
    if len(dists_object) > 1:
        corr_obj = np.corrcoef(dists_object, accuracies)[0, 1]
        axes[1].text(0.05, 0.95, f'r = {corr_obj:.3f}', transform=axes[1].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[2].scatter(dists_avg, accuracies, alpha=0.6, s=50, color='green')
    axes[2].set_xlabel('Average Distance to Subject+Object CoM (grid units)', fontsize=11)
    axes[2].set_ylabel('Head Probing Accuracy', fontsize=11)
    axes[2].set_title('Accuracy vs Average Distance', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0.5, linestyle="--", alpha=0.5, color='red', label='Random (0.5)')
    axes[2].legend()
    
    if len(dists_avg) > 1:
        corr_avg = np.corrcoef(dists_avg, accuracies)[0, 1]
        axes[2].text(0.05, 0.95, f'r = {corr_avg:.3f}', transform=axes[2].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Head Probing Accuracy vs CoM Distance (Level {level})', fontsize=14, y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "accuracy_vs_distance.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Accuracy vs distance plot saved to: {save_path}")

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

    layers = [s["layer"] for s in layer_scores]
    accuracies = [s["accuracy"] for s in layer_scores]
    num_examples = results.get("num_examples", 0)

    plt.figure(figsize=(12, 6))

    plt.plot(
        layers,
        accuracies,
        "o-",
        linewidth=2,
        markersize=8,
        label="Layer Accuracy",
    )

    plt.axhline(y=0.5, linestyle="--", alpha=0.5, label="Random (0.5)")

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
    parser.add_argument(
        "--cv_folds", type=int, default=5, help="Number of CV folds"
    )
    parser.add_argument("--first_gen_token", action="store_true")
    parser.add_argument("--calc_com", action="store_true", 
                       help="Calculate center of mass for attention heads")
    parser.add_argument("--com_layers", type=int, nargs="+", 
                       default=None,
                       help="Layers to analyze for CoM (default: all layers)")
    parser.add_argument("--com_source_token", type=str, default="object",
                       choices=["subject", "relation", "object", "last"],
                       help="Source token type for CoM calculation (default: object)")
    parser.add_argument("--plot_json", type=str, default=None,
                       help="Plot heatmaps from existing JSON file (path to head_com_results.json)")
    parser.add_argument("--skip_probing", action="store_true",
                       help="Skip layer probing and only run CoM calculation")
    parser.add_argument("--probe_heads", action="store_true",
                       help="Probe individual attention heads (requires attentions)")
    parser.add_argument("--head_layers", type=int, nargs="+", 
                       default=[13, 16, 19, 24],
                       help="Layers to probe for head probing (default: 13 16 19 24)")
    parser.add_argument("--skip_layer_probing", action="store_true",
                       help="Skip layer probing and only run head probing")
    args = parser.parse_args()

    if args.plot_json:
        plot_com_heatmaps_from_json(args.plot_json)
        return

    if args.probe_heads and args.skip_layer_probing:
        print(f"Loading {MODEL_ID} for head probing only...")
        processor, model, device = _init_model(
            MODEL_ID,
            output_hidden_states=False,
            output_attentions=True,  # Need attentions for head probing
        )
        
        head_features_by_layer_head, labels, metadata = collect_head_probing_data(
            model, processor, args.level,
            max_images=args.max_images,
            specific_layers=args.head_layers,
            first_token=args.first_gen_token
        )
        
        if len(labels) == 0:
            print("No data collected. Exiting.")
            return
        
        print(f"\nProbing heads in layers {args.head_layers} on {len(labels)} examples...")
        
        results = {
            "level": args.level,
            "num_examples": len(labels),
            "layers": args.head_layers,
            "head_scores": [],
        }
        
        total_heads = sum(len(heads) for heads in head_features_by_layer_head.values())
        with tqdm(total=total_heads, desc="Probing heads") as pbar:
            for layer_idx in sorted(head_features_by_layer_head.keys()):
                heads = head_features_by_layer_head[layer_idx]
                for head_idx in sorted(heads.keys()):
                    head_features = heads[head_idx]
                    if len(head_features) != len(labels):
                        pbar.update(1)
                        continue
                    
                    mean_acc, std_acc = probe_head(head_features, labels, cv_folds=args.cv_folds)
                    
                    results["head_scores"].append({
                        "layer": layer_idx,
                        "head": head_idx,
                        "accuracy": float(mean_acc),
                        "std": float(std_acc),
                    })
                    pbar.update(1)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base_output_dir = "vis_results"
        level_output = os.path.join(base_output_dir, f"level_{args.level}")
        token_suffix = "_first_token" if args.first_gen_token else ""
        head_output = os.path.join(level_output, f"head_probing{token_suffix}_{timestamp}")
        os.makedirs(head_output, exist_ok=True)
        
        plot_head_accuracy(results, head_output, args.level)
        
        results_path = os.path.join(head_output, "head_probing_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nHead probing results saved to: {os.path.abspath(head_output)}")
        
        com_dirs = [d for d in os.listdir(level_output) if d.startswith("com_results_")]
        if com_dirs:
            latest_com_dir = sorted(com_dirs)[-1]
            com_results_path = os.path.join(level_output, latest_com_dir, "head_com_results.json")
            if os.path.exists(com_results_path):
                print(f"\nCreating accuracy vs distance comparison plot...")
                plot_accuracy_vs_distance(results_path, com_results_path, head_output, args.level)
        
        return  # Exit early, skip layer probing
    
    if args.calc_com and args.skip_probing:
        print(f"Loading {MODEL_ID} for CoM calculation only...")
        processor, model, device = _init_model(
            MODEL_ID,
            output_hidden_states=True,
            output_attentions=True,  # Need attentions for CoM
        )
        
        com_results = collect_head_com_data(
            model, processor, args.level, 
            max_images=args.max_images,
            specific_layers=args.com_layers,
            source_token_type=args.com_source_token
        )
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base_output_dir = "vis_results"
        level_output = os.path.join(base_output_dir, f"level_{args.level}")
        com_output = os.path.join(level_output, f"com_results_{timestamp}")
        os.makedirs(com_output, exist_ok=True)
        
        com_output_path = os.path.join(com_output, "head_com_results.json")
        with open(com_output_path, "w") as f:
            json.dump(com_results, f, indent=2)
        
        plot_com_heatmaps(com_results, com_output, args.level, source_token_type=args.com_source_token)
        plot_iou_heatmaps(com_results, com_output, args.level, source_token_type=args.com_source_token)
        print_top_grounding_heads(com_results, com_output, args.level, source_token_type=args.com_source_token)
        plot_com_distance_lines(com_results, com_output, args.level, source_token_type=args.com_source_token)
        plot_iou_by_prediction(com_results, com_output, args.level, source_token_type=args.com_source_token)
        plot_com_distance_by_prediction(com_results, com_output, args.level, source_token_type=args.com_source_token)
        
        print(f"CoM results saved to: {os.path.abspath(com_output_path)}")
        
        head_dirs = [d for d in os.listdir(level_output) if d.startswith("head_probing")]
        if head_dirs:
            latest_head_dir = sorted(head_dirs)[-1]
            head_results_path = os.path.join(level_output, latest_head_dir, "head_probing_results.json")
            if os.path.exists(head_results_path):
                print(f"\nCreating accuracy vs distance comparison plot...")
                plot_accuracy_vs_distance(head_results_path, com_output_path, com_output, args.level)
        
        return  # Exit early, skip all probing

    print(f"Loading {MODEL_ID}...")
    processor, model, device = _init_model(
        MODEL_ID,
        output_hidden_states=True,
        output_attentions=False,
    )

    num_layers = len(model.language_model.layers)
    print(f"Model has {num_layers} layers")

    hidden_states_by_layer, labels, metadata = collect_probing_data(
        model, processor, args.level, max_images=args.max_images, first_token=args.first_gen_token
    )

    if len(labels) == 0:
        print("No data collected. Exiting.")
        return

    print(f"\nProbing {num_layers} layers on {len(labels)} examples...")

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


    best_layer_info = max(
        results["layer_scores"], key=lambda x: x["accuracy"]
    )
    print(f"\nBest layer: {best_layer_info}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_output_dir = "vis_results"
    level_output = os.path.join(base_output_dir, f"level_{args.level}")
    token_suffix = "_first_token" if args.first_gen_token else ""
    probing_output = os.path.join(level_output, f"probing{token_suffix}_{timestamp}")
    os.makedirs(probing_output, exist_ok=True)

    plot_layer_accuracy(results, probing_output, args.level)

    results_copy_path = os.path.join(
        probing_output, "layer_probing_results.json"
    )
    with open(results_copy_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nVisualization results saved to: {os.path.abspath(probing_output)}")
    
    if args.probe_heads:
        print(f"\nProbing attention heads...")
        processor_heads, model_heads, device_heads = _init_model(
            MODEL_ID,
            output_hidden_states=False,
            output_attentions=True,  # Need attentions for head probing
        )
        
        head_features_by_layer_head, head_labels, head_metadata = collect_head_probing_data(
            model_heads, processor_heads, args.level,
            max_images=args.max_images,
            specific_layers=args.head_layers,
            first_token=args.first_gen_token
        )
        
        if len(head_labels) > 0:
            print(f"\nProbing heads in layers {args.head_layers} on {len(head_labels)} examples...")
            
            head_results = {
                "level": args.level,
                "num_examples": len(head_labels),
                "layers": args.head_layers,
                "head_scores": [],
            }
            
            total_heads = sum(len(heads) for heads in head_features_by_layer_head.values())
            for layer_idx in sorted(head_features_by_layer_head.keys()):
                heads = head_features_by_layer_head[layer_idx]
                for head_idx in sorted(heads.keys()):
                    head_features = heads[head_idx]
                    if len(head_features) != len(head_labels):
                        continue
                    
                    mean_acc, std_acc = probe_head(head_features, head_labels, cv_folds=args.cv_folds)
                    
                    head_results["head_scores"].append({
                        "layer": layer_idx,
                        "head": head_idx,
                        "accuracy": float(mean_acc),
                        "std": float(std_acc),
                    })
            
            plot_head_accuracy(head_results, probing_output, args.level)
            
            head_results_path = os.path.join(probing_output, "head_probing_results.json")
            with open(head_results_path, "w") as f:
                json.dump(head_results, f, indent=2)
            
            print(f"Head probing results saved to: {os.path.abspath(head_results_path)}")
            
            com_output = os.path.join(level_output, f"com_results_{timestamp}")
            com_results_path = os.path.join(com_output, "head_com_results.json")
            if os.path.exists(com_results_path):
                print(f"\nCreating accuracy vs distance comparison plot...")
                plot_accuracy_vs_distance(head_results_path, com_results_path, probing_output, args.level)
    
    if args.calc_com:
        print(f"\nCalculating center of mass for attention heads...")
        processor_com, model_com, device_com = _init_model(
            MODEL_ID,
            output_hidden_states=True,
            output_attentions=True,  # Need attentions for CoM
        )
        
        com_results = collect_head_com_data(
            model_com, processor_com, args.level, 
            max_images=args.max_images,
            specific_layers=args.com_layers,
            source_token_type=args.com_source_token
        )
        
        com_output = os.path.join(level_output, f"com_results_{timestamp}")
        os.makedirs(com_output, exist_ok=True)
        
        com_output_path = os.path.join(com_output, "head_com_results.json")
        with open(com_output_path, "w") as f:
            json.dump(com_results, f, indent=2)
        
        plot_com_heatmaps(com_results, com_output, args.level, source_token_type=args.com_source_token)
        plot_iou_heatmaps(com_results, com_output, args.level, source_token_type=args.com_source_token)
        print_top_grounding_heads(com_results, com_output, args.level, source_token_type=args.com_source_token)
        plot_com_distance_lines(com_results, com_output, args.level, source_token_type=args.com_source_token)
        plot_iou_by_prediction(com_results, com_output, args.level, source_token_type=args.com_source_token)
        plot_com_distance_by_prediction(com_results, com_output, args.level, source_token_type=args.com_source_token)
        
        print(f"CoM results saved to: {os.path.abspath(com_output_path)}")
        
        head_dirs = [d for d in os.listdir(level_output) if d.startswith("head_probing")]
        if head_dirs:
            latest_head_dir = sorted(head_dirs)[-1]
            head_results_path = os.path.join(level_output, latest_head_dir, "head_probing_results.json")
            if os.path.exists(head_results_path):
                print(f"\nCreating accuracy vs distance comparison plot...")
                plot_accuracy_vs_distance(head_results_path, com_output_path, com_output, args.level)


if __name__ == "__main__":
    main()