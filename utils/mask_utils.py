"""
Utilities for computing attention masks based on spatial relations.
"""

from typing import Optional, List, Tuple
import torch
import numpy as np
from .attention_utils import get_image_token_indices

# Standard grid dimensions for LLaVA (24x24 = 576 patches)
GRID_DIM = 24


def patch_index_to_row_col(patch_idx: int, grid_dim: int = GRID_DIM) -> Tuple[int, int]:
    """Convert flat patch index to (row, col) coordinates."""
    row = patch_idx // grid_dim
    col = patch_idx % grid_dim
    return row, col


def row_col_to_patch_index(row: int, col: int, grid_dim: int = GRID_DIM) -> int:
    """Convert (row, col) coordinates to flat patch index."""
    return row * grid_dim + col


def get_patches_to_mask_for_relation(
    object_b_patches: List[int],
    relation_type: str,
    grid_dim: int = GRID_DIM,
) -> List[int]:
    """
    Get patch indices that should be masked based on the relation type.
    
    For relation "above" (A above B), mask all patches BELOW B's row (row > B's max row).
    For relation "below" (A below B), mask all patches ABOVE B's row (row < B's min row).
    For relation "left_of" (A left of B), mask all patches RIGHT of B's column (col > B's max col).
    For relation "right_of" (A right of B), mask all patches LEFT of B's column (col < B's min col).
    
    Args:
        object_b_patches: List of patch indices for object B
        relation_type: Type of relation ("above", "below", "left_of", "right_of")
        grid_dim: Grid dimension (default 24)
    
    Returns:
        List of patch indices to mask
    """
    if not object_b_patches:
        return []
    
    # Convert patches to row/col coordinates
    b_rows = []
    b_cols = []
    for patch_idx in object_b_patches:
        row, col = patch_index_to_row_col(patch_idx, grid_dim)
        b_rows.append(row)
        b_cols.append(col)
    
    min_row = min(b_rows)
    max_row = max(b_rows)
    min_col = min(b_cols)
    max_col = max(b_cols)
    
    patches_to_mask = []
    
    if relation_type == "above":
        # Mask patches below B's row (rows > max_row)
        for row in range(max_row + 1, grid_dim):
            for col in range(grid_dim):
                patches_to_mask.append(row_col_to_patch_index(row, col, grid_dim))
    
    elif relation_type == "below":
        # Mask patches above B's row (rows < min_row)
        for row in range(min_row):
            for col in range(grid_dim):
                patches_to_mask.append(row_col_to_patch_index(row, col, grid_dim))
    
    elif relation_type == "left_of":
        # Mask patches right of B's column (cols > max_col)
        for row in range(grid_dim):
            for col in range(max_col + 1, grid_dim):
                patches_to_mask.append(row_col_to_patch_index(row, col, grid_dim))
    
    elif relation_type == "right_of":
        # Mask patches left of B's column (cols < min_col)
        for row in range(grid_dim):
            for col in range(min_col):
                patches_to_mask.append(row_col_to_patch_index(row, col, grid_dim))
    
    else:
        # Unknown relation type, don't mask anything
        return []
    
    return sorted(set(patches_to_mask))


def compute_attention_mask_for_qa(
    input_ids: torch.Tensor,
    image_token_id: int,
    annotation: dict,
    qa_pair: dict,
    mask_opposite_side: bool = True,
    grid_dim: int = GRID_DIM,
    always_mask: bool = False,
) -> Optional[torch.Tensor]:
    """
    Compute attention mask that masks out patches on the opposite side of a relation.
    
    Args:
        input_ids: Input token IDs [batch, seq_len]
        image_token_id: Token ID used for image patches
        annotation: Annotation dict containing objects with patch_indices
        qa_pair: QA pair dict containing rel_type and object_id
        mask_opposite_side: If True, mask patches on opposite side of relation
        grid_dim: Grid dimension (default 24)
        always_mask: If True, mask even when relation doesn't exist (always_mask=True ignores ground truth)
    
    Returns:
        Attention mask tensor [batch, seq_len] or None if masking not needed/possible
    """
    if not mask_opposite_side:
        return None
    
    # Get relation type from qa_pair
    relation_type = qa_pair.get("rel_type")
    object_id = qa_pair.get("object_id")
    
    if not relation_type or object_id is None:
        # Can't mask without relation info
        return None
    
    # If always_mask is False and this is a yes/no question, only mask when answer is "yes" (relation exists)
    if not always_mask:
        ground_truth = qa_pair.get("answer", "").lower().strip()
        if ground_truth in ["no", "false"]:
            # Relation doesn't exist, don't mask
            return None
    
    # Handle inverse relations: normalize "above" to "above", but check if we need to invert
    # For questions like "Is A above B?", relation_type might be "above" with object_id=B
    # For questions like "Where is A in relation to B?" with answer "above", relation_type is "above"
    
    # Find object B
    objects = annotation.get("objects", [])
    object_b = next((obj for obj in objects if obj.get("id") == object_id), None)
    
    if object_b is None:
        return None
    
    # Get object B's patch indices
    object_b_patches = object_b.get("patch_indices", [])
    if not object_b_patches:
        return None
    
    # Get patches to mask
    patches_to_mask_flat = get_patches_to_mask_for_relation(
        object_b_patches, relation_type, grid_dim
    )
    
    if not patches_to_mask_flat:
        # Nothing to mask
        return None
    
    # Get all image token positions in the sequence using existing utility function
    # input_ids is [batch, seq_len], so we get positions for the first (and only) batch item
    image_token_positions = get_image_token_indices(input_ids[0], image_token_id)["all_visual"]
    
    if len(image_token_positions) != grid_dim * grid_dim:
        # Unexpected number of image tokens, skip masking
        return None
    
    # Create attention mask (1 = attend, 0 = mask/ignore)
    # Start with all tokens attended to (1)
    attention_mask = torch.ones_like(input_ids)
    
    # Mask out the specified patches by setting their attention_mask to 0
    # This prevents any tokens from attending to these image patch positions
    for patch_idx in patches_to_mask_flat:
        if 0 <= patch_idx < len(image_token_positions):
            seq_pos = image_token_positions[patch_idx]  # Sequence position of this image patch
            attention_mask[:, seq_pos] = 0  # Set to 0 to mask out this patch for all batch items
    
    return attention_mask


def compute_attention_mask_objects_only(
    input_ids: torch.Tensor,
    image_token_id: int,
    annotation: dict,
    qa_pair: dict,
    grid_dim: int = GRID_DIM,
) -> Optional[torch.Tensor]:
    """
    Compute attention mask that masks all image patches EXCEPT those belonging to the subject and object.
    
    This allows attention only to the objects mentioned in the question, masking out all background
    and other objects in the scene.
    
    Args:
        input_ids: Input token IDs [batch, seq_len]
        image_token_id: Token ID used for image patches
        annotation: Annotation dict containing objects with patch_indices
        qa_pair: QA pair dict containing subject_id and object_id
        grid_dim: Grid dimension (default 24)
    
    Returns:
        Attention mask tensor [batch, seq_len] or None if masking not needed/possible
    """
    # Get subject and object IDs from qa_pair
    subject_id = qa_pair.get("subject_id")
    object_id = qa_pair.get("object_id")
    
    if subject_id is None and object_id is None:
        # Can't mask without object IDs
        return None
    
    # Find objects in annotation
    objects = annotation.get("objects", [])
    subject_obj = next((obj for obj in objects if obj.get("id") == subject_id), None) if subject_id is not None else None
    object_obj = next((obj for obj in objects if obj.get("id") == object_id), None) if object_id is not None else None
    
    # Collect patches to keep (subject and object)
    patches_to_keep = set()
    
    if subject_obj:
        subject_patches = subject_obj.get("patch_indices", [])
        if subject_patches:
            patches_to_keep.update(subject_patches)
    
    if object_obj:
        object_patches = object_obj.get("patch_indices", [])
        if object_patches:
            patches_to_keep.update(object_patches)
    
    if not patches_to_keep:
        # No patches to keep, can't mask meaningfully
        return None
    
    # Get all image token positions in the sequence
    image_token_positions = get_image_token_indices(input_ids[0], image_token_id)["all_visual"]
    
    if len(image_token_positions) != grid_dim * grid_dim:
        # Unexpected number of image tokens, skip masking
        raise ValueError(f"Unexpected number of image tokens: {len(image_token_positions)} != {grid_dim * grid_dim}")
    
    # Create attention mask (1 = attend, 0 = mask/ignore)
    # Start with all tokens masked (0), then unmask the object patches (set to 1)
    attention_mask = torch.zeros_like(input_ids)
    
    # Unmask (set to 1) only the patches belonging to subject and object
    for patch_idx in patches_to_keep:
        if 0 <= patch_idx < len(image_token_positions):
            seq_pos = image_token_positions[patch_idx]
            attention_mask[:, seq_pos] = 1  # Allow attention to this patch
        else:
            raise ValueError(f"Patch index {patch_idx} is out of range for image token positions: {image_token_positions}")
    
    # Also ensure all non-image tokens are attended to (text tokens, special tokens)
    # Find all non-image token positions
    # input_ids[0] is shape [seq_len], non_image_mask will be boolean tensor [seq_len]
    # Note: We process one question at a time, so batch_size=1 (input_ids shape is [1, seq_len])
    non_image_mask = input_ids[0] != image_token_id
    # Apply mask to all items in batch (though typically batch_size=1)
    for batch_idx in range(input_ids.shape[0]):
        attention_mask[batch_idx][non_image_mask] = 1  # Always attend to text tokens
    
    return attention_mask
