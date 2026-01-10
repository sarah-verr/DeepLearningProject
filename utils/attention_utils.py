from typing import Optional, Dict, Any, List, Tuple
import torch

def aggregate_attention_from_source_to_target(
    attentions: Tuple[torch.Tensor, ...],
    source_token_indices: List[int],
    target_token_indices: List[int],
) -> Dict[str, Any]:
    """
    For each layer/head, compute how much attention the source tokens pay to the target tokens.

    attentions: tuple[num_layers] of tensors [1, heads, seq, seq]
    source_token_indices: list of indices in the sequence to use as source
    target_token_indices: list of indices in the sequence to use as target

    Returns:
        {
            "source_token_indices": [...],
            "target_token_indices": [...],
            "per_layer": [
                {
                    "layer_idx": int,
                    "per_head_fraction": List[float],  # fraction for each head
                    "mean_fraction": float,
                },
                ...
            ]
        }
    """
    if not source_token_indices or not target_token_indices:
        return {
            "source_token_indices": source_token_indices,
            "target_token_indices": target_token_indices,
            "per_layer": [],
        }

    per_layer: List[Dict[str, Any]] = []
    for layer_idx, layer_att in enumerate(attentions):
        # layer_att: [1, heads, seq, seq]
        layer_all = layer_att[0].float()  # [heads, seq, seq]
        num_heads = layer_all.shape[0]

        head_fracs = []
        for h in range(num_heads):
            mat = layer_all[h]  # [seq, seq]
            # sum attention from all source tokens to all tokens
            src_to_all = mat[source_token_indices, :].sum().item()
            # sum attention from all source tokens to all target tokens
            src_to_tgt = mat[source_token_indices][:, target_token_indices].sum().item()
            frac = src_to_tgt / (src_to_all + 1e-9) if src_to_all > 0 else 0.0
            head_fracs.append(float(frac))

        mean_frac = float(sum(head_fracs) / len(head_fracs)) if head_fracs else 0.0
        per_layer.append(
            {
                "layer_idx": int(layer_idx),
                "per_head_fraction": head_fracs,
                "mean_fraction": mean_frac,
            }
        )

    return {
        "per_layer": per_layer,
    }

def aggregate_attention_between_groups(
    attentions: Tuple[torch.Tensor, ...],
    source_groups: Dict[str, List[int]],
    target_groups: Dict[str, List[int]],
    key_pairs: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Compute attention for all combinations of source groups -> target groups.
    
    For each source group and target group combination, computes per-layer, per-head
    attention fractions (how much attention flows from source group to target group).
    
    Args:
        attentions: tuple[num_layers] of tensors [1, heads, seq, seq]
        source_groups: Dict mapping group names to token indices (e.g., {"last": [123], "relation": [45, 46]})
        target_groups: Dict mapping group names to token indices (e.g., {"all_visual": [0, 1, 2], "all_text": [100, 101]})
    
    Returns:
        {
            "group_pairs": [
                {
                    "source_group": str,
                    "target_group": str,
                    "per_layer": [
                        {
                            "layer_idx": int,
                            "per_head_fraction": List[float],
                            "mean_fraction": float,
                        },
                        ...
                    ]
                },
                ...
            ]
        }
    """
    group_pairs = []
    
    # If key_pairs specified, only compute those
    if key_pairs:
        pairs_to_compute = key_pairs
    else:
        # Compute all combinations
        pairs_to_compute = [
            (src_name, tgt_name)
            for src_name in source_groups.keys()
            for tgt_name in target_groups.keys()
        ]
    
    for src_name, tgt_name in pairs_to_compute:
        src_indices = source_groups.get(src_name, [])
        tgt_indices = target_groups.get(tgt_name, [])
        
        if not src_indices or not tgt_indices:
            continue
            
        result = aggregate_attention_from_source_to_target(
            attentions, src_indices, tgt_indices
        )
        
        group_pairs.append({
            "source_group": src_name,
            "target_group": tgt_name,
            "per_layer": result["per_layer"],
        })
    
    return {
        "group_pairs": group_pairs,
    }

def get_phrase_token_positions(tokenizer, full_input_ids_1d, relation_phrase: str) -> dict[str, list[int]]:
    """
    Returns a dictionary with 'relation' key containing indices in full_input_ids_1d 
    that correspond to the relation_phrase.
    Tries both normal and leading-space tokenization, and returns all matches.
    """
    if not relation_phrase or not isinstance(relation_phrase, str):
        return {"relation": []}

    full_ids = full_input_ids_1d.tolist()
    # Try both with and without leading space
    phrase_variants = [relation_phrase, " " + relation_phrase]
    matches = []

    for variant in phrase_variants:
        phrase_ids = tokenizer(variant, add_special_tokens=False).input_ids
        if not phrase_ids:
            continue
        # Find all occurrences
        for i in range(len(full_ids) - len(phrase_ids) + 1):
            if full_ids[i:i+len(phrase_ids)] == phrase_ids:
                matches.extend(range(i, i+len(phrase_ids)))
    # Remove duplicates and sort
    return {"relation": sorted(set(matches))}

def get_last_token_index(full_input_ids_1d: torch.Tensor) -> dict[str, list[int]]:
    """
    Returns a dictionary with 'last' key containing the index of the last token in the sequence.
    """
    return {"last": [full_input_ids_1d.shape[0] - 1]}

def get_text_token_indices(full_input_ids_1d: torch.Tensor, image_token_id: int | None) -> dict[str, list[int]]:
    """
    Returns a dictionary with 'all_text' key containing indices for all text tokens (i.e., not image tokens).
    """
    if image_token_id is None:
        return {"all_text": list(range(full_input_ids_1d.shape[0]))}
    return {"all_text": (full_input_ids_1d != image_token_id).nonzero(as_tuple=True)[0].tolist()}

def get_image_token_indices(full_input_ids_1d: torch.Tensor, image_token_id: int | None) -> dict[str, list[int]]:
    """
    Returns a dictionary with 'all_visual' key containing indices for all image tokens.
    """
    if image_token_id is None:
        return {"all_visual": []}
    return {"all_visual": (full_input_ids_1d == image_token_id).nonzero(as_tuple=True)[0].tolist()}

def get_entity_indices(
    tokenizer,
    full_input_ids_1d: torch.Tensor,
    qa_pair: dict,
    annotation: dict,
) -> dict[str, list[int]]:
    """
    Returns a dictionary with 'subject' and 'object' keys containing token indices for each entity in the prompt.
    Uses annotation["objects"] to map id to color/shape.
    """
    full_ids = full_input_ids_1d.tolist()
    result = {"subject": [], "object": []}
    
    # Process subject
    if qa_pair.get("subject_id") is not None:
        entity = next((obj for obj in annotation.get("objects", []) if obj.get("id") == qa_pair["subject_id"]), None)
        if entity:
            color = entity.get("color", "")
            shape = entity.get("shape", "")
            if color and shape:
                entity_phrase = f"{color} {shape}".strip()
                if entity_phrase:
                    indices = []
                    for variant in [entity_phrase, " " + entity_phrase]:
                        phrase_ids = tokenizer(variant, add_special_tokens=False).input_ids
                        for i in range(len(full_ids) - len(phrase_ids) + 1):
                            if full_ids[i:i+len(phrase_ids)] == phrase_ids:
                                indices.extend(range(i, i+len(phrase_ids)))
                                break
                    result["subject"] = sorted(set(indices))
    
    # Process object
    if qa_pair.get("object_id") is not None:
        entity = next((obj for obj in annotation.get("objects", []) if obj.get("id") == qa_pair["object_id"]), None)
        if entity:
            color = entity.get("color", "")
            shape = entity.get("shape", "")
            if color and shape:
                entity_phrase = f"{color} {shape}".strip()
                if entity_phrase:
                    indices = []
                    for variant in [entity_phrase, " " + entity_phrase]:
                        phrase_ids = tokenizer(variant, add_special_tokens=False).input_ids
                        for i in range(len(full_ids) - len(phrase_ids) + 1):
                            if full_ids[i:i+len(phrase_ids)] == phrase_ids:
                                indices.extend(range(i, i+len(phrase_ids)))
                                break
                    result["object"] = sorted(set(indices))
    
    return result

def get_image_entity_indices(
    full_input_ids_1d: torch.Tensor,
    image_token_id: int | None,
    qa_pair: dict,
    annotation: dict,
) -> dict[str, list[int]]:
    """
    Returns a dictionary with 'visual_subject' and 'visual_object' keys containing image token indices 
    corresponding to the patches occupied by each entity.
    
    Args:
        full_input_ids_1d: Full sequence of token IDs
        image_token_id: Token ID used for image patches
        qa_pair: QA pair containing subject_id/object_id
        annotation: Annotation dict with objects[*].patch_indices
    
    Returns:
        Dictionary with 'visual_subject' and 'visual_object' keys containing image token indices for each entity's patches
    """
    if image_token_id is None:
        return {"visual_subject": [], "visual_object": []}
    
    # Get all image token positions in the sequence
    image_token_positions = (full_input_ids_1d == image_token_id).nonzero(as_tuple=True)[0].tolist()
    
    if not image_token_positions:
        return {"visual_subject": [], "visual_object": []}
    
    result = {"visual_subject": [], "visual_object": []}
    objects = annotation.get('objects', [])
    
    # Process subject
    if qa_pair.get("subject_id") is not None:
        patch_indices = []
        for obj in objects:
            if obj.get('id') == qa_pair["subject_id"]:
                patch_indices = obj.get('patch_indices', [])
                break
        
        if patch_indices:
            subject_result = []
            for patch_idx in patch_indices:
                if 0 <= patch_idx < len(image_token_positions):
                    subject_result.append(image_token_positions[patch_idx])
            result["visual_subject"] = sorted(set(subject_result))
    
    # Process object
    if qa_pair.get("object_id") is not None:
        patch_indices = []
        for obj in objects:
            if obj.get('id') == qa_pair["object_id"]:
                patch_indices = obj.get('patch_indices', [])
                break
        
        if patch_indices:
            object_result = []
            for patch_idx in patch_indices:
                if 0 <= patch_idx < len(image_token_positions):
                    object_result.append(image_token_positions[patch_idx])
            result["visual_object"] = sorted(set(object_result))
    
    return result
