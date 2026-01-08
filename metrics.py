import numpy as np


def compute_bbox_attention_fraction(attention_flat, bbox_indices, eps: float = 1e-9) -> float:
	"""Compute attention mass inside a set of patches relative to all image patches.

	Args:
		attention_flat: 1D array-like of length = num visual patches (e.g., numpy array
			or torch tensor) containing attention from a source token to each patch.
		bbox_indices: Iterable of patch indices corresponding to a bounding box or
			group of patches.
		eps: Small constant to avoid division by zero.

	Returns:
		Fraction in [0, 1] equal to sum(attention[bbox_indices]) / sum(attention).
		Returns 0.0 if there is no valid attention mass or no valid indices.
	"""
    
	arr = np.asarray(attention_flat, dtype=float).reshape(-1)
	if arr.size == 0:
		return 0.0
	if not bbox_indices:
		return 0.0

	valid_indices = [int(i) for i in bbox_indices if 0 <= int(i) < arr.size]
	if not valid_indices:
		return 0.0

	bbox_sum = float(arr[valid_indices].sum())
	total_sum = float(arr.sum())
	if total_sum <= eps:
		return 0.0
	return bbox_sum / (total_sum + eps)

def attention_center_of_mass(attn_flat, grid_dim):
    """
    attn_flat: 1D array-like, length == grid_dim * grid_dim
    returns (row_com, col_com) in patch coordinates (0..grid_dim-1)
    """
    arr = np.asarray(attn_flat, dtype=float).reshape(-1)
    if arr.size != grid_dim * grid_dim:
        raise ValueError("attn_flat length must be grid_dim * grid_dim")

    total = arr.sum()
    if total <= 0:
        return None  # or (np.nan, np.nan)

    arr /= total  # normalize to a probability distribution

    idxs = np.arange(arr.size)
    rows = idxs // grid_dim
    cols = idxs % grid_dim

    row_com = float((arr * rows).sum())
    col_com = float((arr * cols).sum())
    return row_com, col_com