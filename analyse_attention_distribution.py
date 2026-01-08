import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

from prompt_templates import build_visual_yesno_prompt


# --- Model / Data Defaults (match main.py) ---
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
BASE_DATA_PATH = "Synthetic-Data/vlm_levels"


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


def _move_processor_inputs(inputs, model):
    dev = _pick_model_input_device(model)
    return inputs.to(dev)


@dataclass
class LevelAttentionStats:
    level: int
    total_image_incoming: float = 0.0
    total_text_incoming: float = 0.0
    num_sequences: int = 0
    head_image_fraction_map: Optional[Dict[str, List[float]]] = None
    head_text_fraction_map: Optional[Dict[str, List[float]]] = None
    total_image_tokens: int = 0
    total_text_tokens: int = 0

    @property
    def image_fraction(self) -> float:
        total = self.total_image_incoming + self.total_text_incoming
        return float(self.total_image_incoming / (total + 1e-9)) if total > 0 else 0.0

    @property
    def text_fraction(self) -> float:
        total = self.total_image_incoming + self.total_text_incoming
        return float(self.total_text_incoming / (total + 1e-9)) if total > 0 else 0.0

    @property
    def avg_image_tokens(self) -> float:
        return float(self.total_image_tokens / self.num_sequences) if self.num_sequences > 0 else 0.0

    @property
    def avg_text_tokens(self) -> float:
        return float(self.total_text_tokens / self.num_sequences) if self.num_sequences > 0 else 0.0


def _find_image_token_span(input_ids: torch.Tensor, num_patches: int, image_token_index: int) -> Tuple[int, int]:
    """Return (start_idx, end_idx) for the block of image patch tokens.

    This mirrors the logic used in main.py: locate the first occurrence of
    model.config.image_token_index and then assume a contiguous span of
    length num_patches.
    """
    indices = (input_ids == image_token_index).nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        # No explicit image token found; treat all tokens as text.
        return -1, -1
    start_idx = int(indices[0].item())
    end_idx = start_idx + int(num_patches)
    return start_idx, end_idx


def _array_to_layer_map(arr: np.ndarray) -> Dict[str, List[float]]:
    """Convert a 2D array [layers, heads] into {layer_k: [...]} mapping."""
    if arr.size == 0:
        return {}
    layer_map: Dict[str, List[float]] = {}
    for idx, row in enumerate(arr):
        layer_map[f"layer_{idx}"] = row.tolist()
    return layer_map


def compute_incoming_attention_split(
    attentions: List[torch.Tensor],
    start_idx: int,
    end_idx: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute per-head text/image fractions plus global fractions.

    Returns:
        - head_text_fractions: array [num_layers, num_heads]
        - head_image_fractions: array [num_layers, num_heads]
        - global_text_fraction: float averaged over all heads/layers
        - global_image_fraction: float averaged over all heads/layers
    """
    if not attentions:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            0.0,
            0.0,
        )

    def _split_text_image(incoming_np: np.ndarray) -> Tuple[float, float]:
        seq_len = incoming_np.shape[0]
        if start_idx < 0 or end_idx <= start_idx or start_idx >= seq_len:
            return float(incoming_np.sum()), 0.0
        s_idx = max(0, min(start_idx, seq_len))
        e_idx = max(s_idx, min(end_idx, seq_len))
        img_mask = np.zeros(seq_len, dtype=bool)
        img_mask[s_idx:e_idx] = True
        txt_mask = ~img_mask
        total_image = float(incoming_np[img_mask].sum())
        total_text = float(incoming_np[txt_mask].sum())
        return total_text, total_image

    with torch.no_grad():
        stacked = torch.stack(attentions, dim=0)[:, 0].float()  # [L, H, S, S]
        L, H, _, _ = stacked.shape
        head_image = np.zeros((L, H), dtype=np.float32)
        head_text = np.zeros((L, H), dtype=np.float32)

        for l in range(L):
            for h in range(H):
                incoming = stacked[l, h].sum(dim=0)
                incoming_np = incoming.detach().cpu().numpy().astype(np.float64)
                if not np.all(np.isfinite(incoming_np)):
                    incoming_np = np.nan_to_num(incoming_np, nan=0.0, posinf=0.0, neginf=0.0)
                total_text, total_image = _split_text_image(incoming_np)
                denom = total_text + total_image + 1e-9
                head_image[l, h] = float(total_image / denom)
                head_text[l, h] = float(total_text / denom)

        incoming_per_token = stacked.sum(dim=(0, 1, 2))
        incoming_np = incoming_per_token.detach().cpu().numpy().astype(np.float64)
        if not np.all(np.isfinite(incoming_np)):
            incoming_np = np.nan_to_num(incoming_np, nan=0.0, posinf=0.0, neginf=0.0)
        total_text, total_image = _split_text_image(incoming_np)
        denom = total_text + total_image + 1e-9
        global_image_fraction = float(total_image / denom)
        global_text_fraction = float(total_text / denom)

    return head_text, head_image, global_text_fraction, global_image_fraction


def analyze_levels(
    base_data_path: str,
    levels: List[int],
    max_questions_per_image: int | None = None,
) -> Tuple[Dict[int, LevelAttentionStats], Dict[str, object]]:
    """Run attention-spread analysis and aggregate per-head distributions."""

    print(f"Loading model: {MODEL_ID} ...")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        quantization_config=bnb,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)

    stats_by_level: Dict[int, LevelAttentionStats] = {
        lvl: LevelAttentionStats(level=lvl) for lvl in levels
    }
    level_head_image_sums = {lvl: None for lvl in levels}
    level_head_text_sums = {lvl: None for lvl in levels}
    level_sequence_counts = {lvl: 0 for lvl in levels}

    overall_head_image_sum: Optional[np.ndarray] = None
    overall_head_text_sum: Optional[np.ndarray] = None
    overall_sequence_count = 0
    overall_global_image_sum = 0.0
    overall_global_text_sum = 0.0
    overall_total_image_tokens = 0
    overall_total_text_tokens = 0

    for level in levels:
        level_dir = os.path.join(base_data_path, f"level_{level}")
        ann_dir = os.path.join(level_dir, "ann")
        img_dir = os.path.join(level_dir, "images")

        if not os.path.isdir(ann_dir) or not os.path.isdir(img_dir):
            print(f"[Warning] Missing ann/images for level {level} at {level_dir}, skipping.")
            continue

        ann_files = sorted(f for f in os.listdir(ann_dir) if f.endswith(".json"))
        print(f"Level {level}: found {len(ann_files)} annotation files.")

        for ann_file in tqdm(ann_files, desc=f"Level {level}"):
            ann_path = os.path.join(ann_dir, ann_file)
            img_id = os.path.splitext(ann_file)[0]
            img_path = os.path.join(img_dir, f"{img_id}.png")

            if not os.path.exists(img_path):
                print(f"  [Warning] Missing image for {ann_file} -> {img_path}, skipping.")
                continue

            with open(ann_path, "r", encoding="utf-8") as f:
                ann_data = json.load(f)

            qa_list = ann_data.get("qa", []) or []
            meta = ann_data.get("meta", {}) or {}
            grid_dim = int(meta.get("grid_dim", 24))
            num_patches = int(grid_dim * grid_dim)

            if max_questions_per_image is not None and max_questions_per_image > 0:
                qa_list = qa_list[: max_questions_per_image]

            if not qa_list:
                continue

            image = Image.open(img_path).convert("RGB")

            for qa_item in qa_list:
                question_text = qa_item.get("question", "")
                conversation = build_visual_yesno_prompt(question_text)
                prompt_text = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=False,
                )

                inputs = processor(text=prompt_text, images=image, return_tensors="pt")
                inputs = _move_processor_inputs(inputs, model)

                try:
                    with torch.no_grad():
                        outputs = model(
                            **inputs,
                            output_attentions=True,
                            use_cache=False,
                            return_dict=True,
                        )
                    attentions = [layer.detach().cpu() for layer in outputs.attentions]
                except Exception as e:  # pragma: no cover - runtime/VRAM issues
                    print(f"  [Warning] Could not compute attentions for {ann_file}: {e}")
                    continue

                input_ids = inputs["input_ids"][0]
                image_token_index = model.config.image_token_index
                start_idx, end_idx = _find_image_token_span(
                    input_ids, num_patches=num_patches, image_token_index=image_token_index
                )
                seq_len = int(input_ids.shape[-1])
                if start_idx >= 0 and end_idx > start_idx:
                    s_idx = max(0, min(start_idx, seq_len))
                    e_idx = max(s_idx, min(end_idx, seq_len))
                    image_tokens = e_idx - s_idx
                else:
                    image_tokens = 0
                text_tokens = max(0, seq_len - image_tokens)

                head_text_frac, head_image_frac, global_text_frac, global_image_frac = compute_incoming_attention_split(
                    attentions,
                    start_idx=start_idx,
                    end_idx=end_idx,
                )

                head_image_np = np.asarray(head_image_frac, dtype=np.float64)
                head_text_np = np.asarray(head_text_frac, dtype=np.float64)

                if level_head_image_sums[level] is None:
                    level_head_image_sums[level] = np.zeros_like(head_image_np)
                    level_head_text_sums[level] = np.zeros_like(head_text_np)

                level_head_image_sums[level] += head_image_np
                level_head_text_sums[level] += head_text_np
                level_sequence_counts[level] += 1

                if overall_head_image_sum is None:
                    overall_head_image_sum = np.zeros_like(head_image_np)
                    overall_head_text_sum = np.zeros_like(head_text_np)

                overall_head_image_sum += head_image_np
                overall_head_text_sum += head_text_np
                overall_sequence_count += 1
                overall_global_image_sum += global_image_frac
                overall_global_text_sum += global_text_frac

                stats = stats_by_level[level]
                stats.total_text_incoming += global_text_frac
                stats.total_image_incoming += global_image_frac
                stats.num_sequences += 1
                stats.total_image_tokens += int(image_tokens)
                stats.total_text_tokens += int(text_tokens)
                overall_total_image_tokens += int(image_tokens)
                overall_total_text_tokens += int(text_tokens)

                del outputs, attentions, inputs
                torch.cuda.empty_cache()

    for level in levels:
        count = level_sequence_counts[level]
        if count > 0:
            stats = stats_by_level[level]
            if level_head_image_sums[level] is not None:
                stats.head_image_fraction_map = _array_to_layer_map(level_head_image_sums[level] / count)
            else:
                stats.head_image_fraction_map = {}
            if level_head_text_sums[level] is not None:
                stats.head_text_fraction_map = _array_to_layer_map(level_head_text_sums[level] / count)
            else:
                stats.head_text_fraction_map = {}
        else:
            stats_by_level[level].head_image_fraction_map = {}
            stats_by_level[level].head_text_fraction_map = {}

    overall_summary: Dict[str, object]
    if overall_sequence_count > 0:
        overall_head_image_map = (
            _array_to_layer_map(overall_head_image_sum / overall_sequence_count)
            if overall_head_image_sum is not None
            else {}
        )
        overall_head_text_map = (
            _array_to_layer_map(overall_head_text_sum / overall_sequence_count)
            if overall_head_text_sum is not None
            else {}
        )
        overall_summary = {
            "head_image_fraction_map": overall_head_image_map,
            "head_text_fraction_map": overall_head_text_map,
            "image_fraction": float(overall_global_image_sum / overall_sequence_count),
            "text_fraction": float(overall_global_text_sum / overall_sequence_count),
            "num_sequences": overall_sequence_count,
            "total_image_tokens": overall_total_image_tokens,
            "total_text_tokens": overall_total_text_tokens,
            "avg_image_tokens": float(overall_total_image_tokens / overall_sequence_count),
            "avg_text_tokens": float(overall_total_text_tokens / overall_sequence_count),
        }
    else:
        overall_summary = {
            "head_image_fraction_map": {},
            "head_text_fraction_map": {},
            "image_fraction": 0.0,
            "text_fraction": 0.0,
            "num_sequences": 0,
            "total_image_tokens": 0,
            "total_text_tokens": 0,
            "avg_image_tokens": 0.0,
            "avg_text_tokens": 0.0,
        }

    return stats_by_level, overall_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze global attention spread between text and image tokens across levels."
    )
    parser.add_argument(
        "--base_data_path",
        type=str,
        default=BASE_DATA_PATH,
        help="Base path to vlm_levels (default: Synthetic-Data/vlm_levels)",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="*",
        default=[0, 1, 2, 3, 4],
        help="Levels to analyze (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--max_questions_per_image",
        type=int,
        default=1,
        help="Optional cap on QAs per image (for speed).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="attention_spread_summary.json",
        help="Output JSON file to write summary stats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stats_by_level, overall_summary = analyze_levels(
        base_data_path=args.base_data_path,
        levels=args.levels,
        max_questions_per_image=args.max_questions_per_image,
    )

    summary = {
        "model_id": MODEL_ID,
        "base_data_path": os.path.abspath(args.base_data_path),
        "levels": args.levels,
        "per_level": {
            int(level): {
                **asdict(stat),
                "image_fraction": stat.image_fraction,
                "text_fraction": stat.text_fraction,
                "avg_image_tokens": stat.avg_image_tokens,
                "avg_text_tokens": stat.avg_text_tokens,
            }
            for level, stat in stats_by_level.items()
        },
        "overall": overall_summary,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved attention spread summary to {os.path.abspath(args.output)}")


if __name__ == "__main__":  # pragma: no cover
    main()
