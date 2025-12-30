import torch
import sys
import importlib.util
import os
import time
import json
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from tqdm import tqdm
from types import SimpleNamespace
from transformers import BitsAndBytesConfig

from prompt_templates import build_visual_yesno_prompt
from metrics import compute_bbox_attention_fraction
from plot_utils import (
    draw_target_highlights,
    add_heatmap_colorbar,
    overlay_heatmap,
    create_layer_grid_plot,
    plot_attention_trends,
    plot_subject_object_attention,
    plot_correct_vs_incorrect_trends,
    plot_evaluation_results,
    create_phrase_thirds_plot,
    create_decision_thirds_plot,
    create_phrase_layer_grid_plot,
    plot_head_layer_fraction_heatmaps,
)

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

# --- Configuration ---
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
BASE_OUTPUT_DIR = "vis_results"
BASE_DATA_PATH = f"/home/{os.environ['USER']}/deep-learning/DeepLearningProject/Synthetic-Data/vlm_levels"

# ---------------------- Config Loader (YAML/JSON) ----------------------
def _load_config_file(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in {".yaml", ".yml"}:
        try:
            import yaml  # requires: pip install pyyaml
        except Exception as e:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Install with: pip install pyyaml"
            ) from e
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {}

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}

    raise ValueError(f"Unsupported config extension '{ext}'. Use .yaml/.yml or .json")

def _normalize_config(cfg: dict) -> dict:
    """
    Fills defaults and validates required fields.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping/object at the top level.")

    # required
    if "level" not in cfg or "id" not in cfg:
        raise ValueError("Config must include required keys: level, id")

    out = dict(cfg)

    # defaults (match your existing CLI defaults)
    out.setdefault("num_questions", None)
    out.setdefault("plot_trends", False)

    # attention_mode controls attention extraction detail level
    # None     -> no attention plots
    # simple   -> lightweight summaries (e.g., layer-thirds maps), no per-layer grids
    # detailed -> full per-layer grids (and optional summaries)
    out.setdefault("attention_mode", None)
    if out["attention_mode"] not in {None, "simple", "detailed"}:
        raise ValueError("attention_mode must be one of: None, simple, detailed")

    # optional overrides for paths/model
    out.setdefault("model_id", None)
    out.setdefault("base_output_dir", None)
    out.setdefault("base_data_path", None)

    # attention_source controls which token(s) act as the query for
    # image attention when building maps/trends.
    #   - "first_generated_token" (default): use first generated token
    #   - "rel_phrase": use relational phrase tokens from the prompt
    out.setdefault("attention_source", "first_generated_token")
    if out["attention_source"] not in {"first_generated_token", "rel_phrase"}:
        raise ValueError(
            "attention_source must be one of: first_generated_token, rel_phrase"
        )

    # normalize types
    out["level"] = str(out["level"])
    out["id"] = str(out["id"])
    if out["num_questions"] is not None:
        out["num_questions"] = int(out["num_questions"])

    out["plot_trends"] = bool(out["plot_trends"])

    return out


def _get_query_positions_for_layer(
    source_mode: str,
    *,
    prompt_len: int,
    seq_len: int,
    phrase_positions: list[int] | None = None,
) -> list[int]:
    """Return token indices in [0, seq_len) to use as query positions.

    - first_generated_token: single index at the first generated token
    - rel_phrase: all relational-phrase token indices that fall in range
    Falls back to first_generated_token if nothing is found.
    """

    if source_mode == "rel_phrase" and phrase_positions:
        in_range = [p for p in phrase_positions if 0 <= p < seq_len]
        if in_range:
            return in_range

    # Default / fallback: first generated token
    idx = prompt_len if prompt_len < seq_len else seq_len - 1
    return [idx]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLaVA inference (config-driven)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/run_configs.yaml",
        help="Path to YAML/JSON config (default: configs/run_configs.yaml)",
    )
    return parser.parse_args()

# --- DYNAMIC TARGET GROUP EXTRACTION ---
def extract_target_groups_from_annotation(annotation_data, grid_dim: int):
    """Extract per-object patch indices directly from the JSON.

    Expected JSON schema (newer): objects[*].patch_indices (flat row-major indices).
    Returns:
      - target_groups: List[List[int]] in stable (sorted object_id) order
      - group_metadata: List[dict] aligned with target_groups
      - obj_id_to_patch_indices: Dict[int, List[int]]
    """
    objects = annotation_data.get('objects', [])

    obj_id_to_patch_indices = {}
    obj_id_to_meta = {}

    max_idx = grid_dim * grid_dim - 1
    for obj in objects:
        if 'id' not in obj:
            continue
        obj_id = int(obj['id'])
        color = obj.get('color', 'gray')
        shape = obj.get('shape', 'unknown')

        patch_indices = obj.get('patch_indices', [])
        if not isinstance(patch_indices, list) or len(patch_indices) == 0:
            continue

        # sanitize
        cleaned = sorted({int(i) for i in patch_indices if 0 <= int(i) <= max_idx})
        if not cleaned:
            continue

        obj_id_to_patch_indices[obj_id] = cleaned
        obj_id_to_meta[obj_id] = {
            'object_id': obj_id,
            'shape': shape,
            'color': color,
            'patch_count': len(cleaned)
        }

    target_groups = []
    group_metadata = []
    for obj_id in sorted(obj_id_to_patch_indices.keys()):
        target_groups.append(obj_id_to_patch_indices[obj_id])
        group_metadata.append(obj_id_to_meta[obj_id])

    return target_groups, group_metadata, obj_id_to_patch_indices


# --- HELPER: Probability Extraction ---
def get_yes_no_probability(outputs, tokenizer):
    first_token_logits = outputs.scores[0][0]
    probs = torch.softmax(first_token_logits, dim=-1)

    # Do NOT depend on a particular chat template prefix like "ASSISTANT:".
    # Across LLaVA 1.6 variants (Vicuna/Mistral), the prompt formatting can differ,
    # but the first generated token is typically a variant of " yes"/" no".
    def _last_token_id(text: str) -> int | None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return ids[-1] if ids else None

    yes_variants = [" yes", " Yes", "yes", "Yes"]
    no_variants = [" no", " No", "no", "No"]
    yes_tokens = [tid for tid in (_last_token_id(v) for v in yes_variants) if tid is not None]
    no_tokens = [tid for tid in (_last_token_id(v) for v in no_variants) if tid is not None]

    prob_yes = sum([probs[t_id].item() for t_id in yes_tokens if t_id < len(probs)])
    prob_no = sum([probs[t_id].item() for t_id in no_tokens if t_id < len(probs)])

    total = prob_yes + prob_no + 1e-9
    norm_yes = prob_yes / total
    norm_no = prob_no / total

    prediction = "yes" if norm_yes > norm_no else "no"
    confidence = norm_yes if prediction == "yes" else norm_no
    
    return prediction, confidence, norm_yes, norm_no



def _find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    """Return start index of needle in haystack, or None."""
    if not needle or not haystack or len(needle) > len(haystack):
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None

def _locate_rel_phrase_token_positions(tokenizer, full_input_ids_1d, question_text: str, rel_phrase: str) -> list[int]:
    """
    Returns absolute token positions (indices into full_input_ids_1d) corresponding to rel_phrase tokens.
    Strategy:
      1) find question token span inside full prompt tokens
      2) find rel_phrase token span inside question tokens
      3) convert to absolute positions
    """
    if not question_text or not rel_phrase:
        return []

    full_ids = full_input_ids_1d.tolist() if hasattr(full_input_ids_1d, "tolist") else list(full_input_ids_1d)

    q_ids = tokenizer(question_text, add_special_tokens=False).input_ids
    q_start = _find_subsequence(full_ids, q_ids)

    # Tokenize phrase in two ways to handle whitespace-sensitive tokenization
    phrase_ids_a = tokenizer(rel_phrase, add_special_tokens=False).input_ids
    phrase_ids_b = tokenizer(" " + rel_phrase, add_special_tokens=False).input_ids

    if q_start is not None:
        # find phrase within question tokens
        p_start = _find_subsequence(q_ids, phrase_ids_a)
        p_len = len(phrase_ids_a)
        if p_start is None:
            p_start = _find_subsequence(q_ids, phrase_ids_b)
            p_len = len(phrase_ids_b)

        if p_start is None:
            return []

        abs_start = q_start + p_start
        return list(range(abs_start, abs_start + p_len))

    # Fallback: search in the full prompt directly
    p_start = _find_subsequence(full_ids, phrase_ids_a)
    p_len = len(phrase_ids_a)
    if p_start is None:
        p_start = _find_subsequence(full_ids, phrase_ids_b)
        p_len = len(phrase_ids_b)
    if p_start is None:
        return []
    return list(range(p_start, p_start + p_len))


def main():
    cli = parse_arguments()

    cfg_raw = _load_config_file(cli.config)
    cfg = _normalize_config(cfg_raw)
    args = SimpleNamespace(**cfg)  # preserves your existing args.* usage

    # Allow config to override globals (optional)
    global MODEL_ID, BASE_OUTPUT_DIR, BASE_DATA_PATH
    if args.model_id:
        MODEL_ID = args.model_id
    if args.base_output_dir:
        BASE_OUTPUT_DIR = args.base_output_dir
    if args.base_data_path:
        BASE_DATA_PATH = args.base_data_path

    # Path Construction
    level_dir = f"level_{args.level}"
    image_path = os.path.join(BASE_DATA_PATH, level_dir, "images", f"{args.id}.png")
    json_path = os.path.join(BASE_DATA_PATH, level_dir, "ann", f"{args.id}.json")

    if not os.path.exists(image_path) or not os.path.exists(json_path):
        print(f"Error: Files not found at constructed paths.")
        print(f"  image: {image_path}")
        print(f"  json : {json_path}")
        return

    # Output Setup - Organized structure with timestamp for multiple trials
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    level_output = os.path.join(BASE_OUTPUT_DIR, f"level_{args.level}")
    image_output = os.path.join(level_output, f"{args.id}_{timestamp}")

    # Create subdirectories
    overview_dir = os.path.join(image_output, "overview")
    trends_dir = os.path.join(image_output, "trend_analysis")
    attention_dir = os.path.join(image_output, "attention")
    phrase_attn_dir = os.path.join(image_output, "phrase_attention")

    os.makedirs(overview_dir, exist_ok=True)
    if args.plot_trends:
        os.makedirs(trends_dir, exist_ok=True)
    if args.attention_mode is not None:
        os.makedirs(attention_dir, exist_ok=True)
        # Phrase-specific outputs only when relational phrase is the source
        if args.attention_source == "rel_phrase":
            os.makedirs(phrase_attn_dir, exist_ok=True)

    # LOGIC: We need attention if attention_mode is set or trends are requested
    REQUIRES_ATTENTION = bool(args.attention_mode) or args.plot_trends
    if REQUIRES_ATTENTION:
        print("NOTE: Attention extraction is ENABLED.")
        if args.attention_mode:
            print(f"      -> attention_mode={args.attention_mode}")
        if args.plot_trends:
            print("      -> Generating trend graphs")

    # Load Model
    print(f"Loading model: {MODEL_ID}...")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        quantization_config=bnb,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)

    # Data Prep
    image = Image.open(image_path).convert('RGB')

    with open(json_path, 'r') as f:
        annotation_data = json.load(f)
        full_qa_list = annotation_data.get('qa', [])

    meta = annotation_data.get('meta', {}) if isinstance(annotation_data.get('meta', {}), dict) else {}
    img_size = int(meta.get('img_size', 336))
    patch_size = int(meta.get('patch', 14))
    grid_dim = int(meta.get('grid_dim', 24))
    num_patches = grid_dim * grid_dim

    # Extract per-object patch indices directly from JSON
    TARGET_GROUPS, group_metadata, obj_id_to_patch_indices = extract_target_groups_from_annotation(annotation_data, grid_dim=grid_dim)
    # Map from object_id -> shape name for nicer plot titles
    object_id_to_shape = {m["object_id"]: m.get("shape", "unknown") for m in group_metadata}
    
    print(f"\nDynamically extracted {len(TARGET_GROUPS)} target groups:")
    for i, meta in enumerate(group_metadata):
        print(f"  Group {i}: Object {meta['object_id']} - {meta['color']} {meta['shape']} ({meta['patch_count']} patches)")
    
    # --- PREPARE GROUP METADATA FOR VISUALIZATION ---
    target_groups_meta = []
    if args.attention_mode is not None:
        vis_image = image.resize((img_size, img_size), resample=Image.BICUBIC)
        for i, (group_indices, meta) in enumerate(zip(TARGET_GROUPS, group_metadata)):
            # Use the actual object color from annotation
            object_color = meta['color']
            group_patches = []
            for idx in group_indices:
                row = idx // grid_dim
                col = idx % grid_dim
                group_patches.append({'row': int(row), 'col': int(col), 'idx': idx})
            
            target_groups_meta.append({
                'group_id': i,
                'color': object_color,
                'patches': group_patches,
                'shape': meta['shape'],
                'object_id': meta['object_id']
            })

    # Question sampling
    if args.num_questions is not None and args.num_questions > 0:
        if args.num_questions < len(full_qa_list):
            print(f"Randomly sampling {args.num_questions} questions from {len(full_qa_list)} total available.")
            qa_list = random.sample(full_qa_list, args.num_questions)
        else:
            qa_list = full_qa_list
    else:
        qa_list = full_qa_list

    results_summary = []
    
    # --- NEW: Accumulators for Correct vs Incorrect Analysis ---
    correct_runs_layerwise = []   # Will store lists of [sum_layer_0, sum_layer_1, ...]
    incorrect_runs_layerwise = [] 

    print("\nStarting Inference...")
    print(f"{'Idx':<4} | {'Correct':<7} | {'Pred':<5} | {'GT':<5} | {'Conf':<6} | Question")
    print("-" * 80)

    for i, qa_item in tqdm(enumerate(qa_list), total=len(qa_list), desc="Total Progress"):
        question_text = qa_item["question"]
        ground_truth = qa_item["answer"].lower()
        subject_id = qa_item.get("subject_id", None)
        object_id = qa_item.get("object_id", None)
        rel_type = qa_item.get("rel_type", None)
        rel_group = qa_item.get("rel_group", None)
        rel_phrase = qa_item.get("rel_phrase", None)  # NEW

        conversation = build_visual_yesno_prompt(question_text)
        prompt_text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(text=prompt_text, images=image, return_tensors="pt")
        inputs = _move_processor_inputs(inputs, model)
        # Index of the first generated token in the full sequence (prompt + generation)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                output_attentions=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )

        # Decode the first generated token (used for yes/no decision & plot titles)
        first_gen_text = None
        try:
            seq_full = outputs.sequences[0]
            if seq_full.shape[0] > prompt_len:
                first_gen_id = int(seq_full[prompt_len].item())
                first_gen_text = processor.tokenizer.decode([first_gen_id], skip_special_tokens=False)
                if i < 10:
                    tqdm.write(
                        f"Q{i}: first generated token id={first_gen_id}, text={repr(first_gen_text)}"
                    )
        except Exception as e:
            tqdm.write(f"  [Debug] Could not decode first generated token for Q{i}: {e}")

        # Eval
        prediction, confidence, p_yes, p_no = get_yes_no_probability(outputs, processor.tokenizer)
        is_correct = (prediction == ground_truth)
        status_icon = "✅" if is_correct else "❌"
        
        tqdm.write(f"{i:<4} | {status_icon:<7} | {prediction:<5} | {ground_truth:<5} | {confidence:.2f}   | {question_text}")

        results_summary.append({
            "id": i,
            "question": question_text,
            "gt": ground_truth,
            "prediction": prediction,
            "confidence": float(confidence),
            "is_correct": is_correct,
            "subject_id": subject_id,
            "object_id": object_id,
            "rel_type": rel_type,
            "rel_group": rel_group,
        })

        # If attention plots are requested, compute attentions in a separate controlled forward pass.
        # This avoids the large VRAM spike from `generate(output_attentions=True)`.
        all_layers_data = None
        if REQUIRES_ATTENTION:
            try:
                seq = outputs.sequences.detach()
                attn_inputs = dict(inputs)
                attn_inputs["input_ids"] = seq
                attn_inputs["attention_mask"] = torch.ones_like(seq, dtype=torch.long, device=seq.device)

                with torch.no_grad():
                    fwd = model(
                        **attn_inputs,
                        output_attentions=True,
                        use_cache=False,
                        return_dict=True,
                    )
                all_layers_data = [layer_tensor.detach().cpu() for layer_tensor in fwd.attentions]
                del fwd
                torch.cuda.empty_cache()
            except Exception as e:
                tqdm.write(f"  [Warning] Could not compute attentions in forward pass: {e}")

        # Free generation outputs early (keep only derived scalars above).
        del outputs
        torch.cuda.empty_cache()

        # --- ATTENTION PROCESSING ---
        if REQUIRES_ATTENTION and all_layers_data is not None:
            # Choose where to store per-question artifacts
            if args.attention_mode:
                q_dir = os.path.join(attention_dir, f"Q{i}")
            elif args.plot_trends:
                q_dir = os.path.join(trends_dir, f"Q{i}")
            else:
                q_dir = os.path.join(overview_dir, f"Q{i}")
            os.makedirs(q_dir, exist_ok=True)

            # We use `all_layers_data` (CPU) which comes from a forward pass over
            # prompt + generated token. This includes:
            # - last token query (generated yes/no) -> image keys
            # - phrase token queries -> image keys
            prompt_attn_layers = all_layers_data
            phrase_positions: list[int] = []
            decision_per_layer_maps: list[np.ndarray] = []
            # Compute relational-phrase token positions if needed either by
            # attention_mode (phrase*) or by attention_source configuration.
            if rel_phrase and (
                (args.attention_mode and args.attention_mode.startswith("phrase"))
                or args.attention_source == "rel_phrase"
            ):
                phrase_positions = _locate_rel_phrase_token_positions(
                    processor.tokenizer,
                    inputs.input_ids[0],
                    question_text=question_text,
                    rel_phrase=rel_phrase,
                )

            group_scores_layerwise = {g_idx: [] for g_idx in range(len(TARGET_GROUPS))}
            
            # This list will hold the sum of attention on ALL targets for each layer
            current_question_layer_totals = []

            # Per-question subject/object curves (dynamic targets from JSON)
            subject_layer_scores = []
            object_layer_scores = []
            # Per-layer fractions of attention on subject/object bounding boxes (avg over heads)
            subject_fraction_layers = []
            object_fraction_layers = []
            # Detailed per-layer, per-head fractions (filled inside the loop)
            per_layer_head_fractions: list[dict] = []
            subj_patch_indices = obj_id_to_patch_indices.get(int(subject_id), []) if subject_id is not None else []
            obj_patch_indices = obj_id_to_patch_indices.get(int(object_id), []) if object_id is not None else []

            start_idx = (inputs.input_ids[0] == model.config.image_token_index).nonzero(as_tuple=True)[0][0].item()
            end_idx = start_idx + num_patches

            if args.attention_mode and args.attention_source == "rel_phrase" and rel_phrase:
                phrase_q_dir = os.path.join(phrase_attn_dir, f"Q{i}")
                os.makedirs(phrase_q_dir, exist_ok=True)
                if not phrase_positions:
                    tqdm.write(
                        f'  [Warning] Could not locate rel_phrase tokens for Q{i}: "{rel_phrase}". Skipping phrase plot.'
                    )
                elif prompt_attn_layers is None:
                    tqdm.write(
                        f"  [Warning] Prompt attentions unavailable for Q{i}; skipping phrase-attention plots."
                    )
                else:
                    vis_image_phrase = image.resize((img_size, img_size), resample=Image.BICUBIC)

                    # Collect per-layer avg maps for the "simple" thirds aggregation
                    per_layer_avg_maps: list[np.ndarray] = []

                    for layer_idx, layer_attn_tensor in enumerate(prompt_attn_layers):
                        try:
                            layer = layer_attn_tensor[0]  # [heads, seq_len, seq_len]

                            seq_len = layer.shape[1]
                            phrase_pos_in_range = [p for p in phrase_positions if 0 <= p < seq_len]
                            if not phrase_pos_in_range:
                                continue

                            # Slice keys to image tokens
                            if end_idx > seq_len:
                                img_key_slice = slice(seq_len - num_patches, seq_len)
                            else:
                                img_key_slice = slice(start_idx, end_idx)

                            # phrase_to_img: [heads, phrase_len, num_patches]
                            phrase_to_img = layer[:, phrase_pos_in_range, img_key_slice]

                            # heads_flat: [heads, num_patches] (avg over phrase tokens)
                            heads_flat = phrase_to_img.mean(dim=1).float().numpy()

                            # heads_map_2d: [heads, grid_dim, grid_dim]
                            heads_map_2d = heads_flat.reshape(heads_flat.shape[0], grid_dim, grid_dim)

                            # avg_map_2d: [grid_dim, grid_dim] (avg over heads)
                            avg_map_2d = heads_map_2d.mean(axis=0)

                            # store for thirds plot
                            per_layer_avg_maps.append(avg_map_2d)

                            # DETAILED mode: save per-layer grids
                            if args.attention_mode == "detailed":
                                fig = create_phrase_layer_grid_plot(
                                    vis_image_phrase,
                                    heads_map_2d,
                                    avg_map_2d,
                                    layer_idx=layer_idx,
                                    rel_phrase=rel_phrase,
                                    patch_size=patch_size,
                                )
                                plt.savefig(
                                    os.path.join(phrase_q_dir, f"layer_{layer_idx:02d}.png"),
                                    bbox_inches="tight",
                                )
                                plt.close(fig)

                        except Exception as e:
                            tqdm.write(f"  [Warning] Phrase-attn error layer {layer_idx}: {e}")

                    # SIMPLE mode: one plot aggregated into early/mid/late thirds
                    if args.attention_mode == "simple":
                        if not per_layer_avg_maps:
                            tqdm.write(f"  [Warning] No per-layer phrase maps collected for Q{i}.")
                        else:
                            L = len(per_layer_avg_maps)
                            a = L // 3
                            b = (2 * L) // 3

                            early_maps = per_layer_avg_maps[:a] if a > 0 else per_layer_avg_maps[:1]
                            mid_maps = per_layer_avg_maps[a:b] if b > a else per_layer_avg_maps[a : a + 1]
                            late_maps = per_layer_avg_maps[b:] if b < L else per_layer_avg_maps[-1:]

                            maps_3 = {
                                "early": np.mean(np.stack(early_maps, axis=0), axis=0),
                                "mid": np.mean(np.stack(mid_maps, axis=0), axis=0),
                                "late": np.mean(np.stack(late_maps, axis=0), axis=0),
                            }

                            fig3 = create_phrase_thirds_plot(
                                vis_image_phrase,
                                maps_3,
                                rel_phrase=rel_phrase,
                                patch_size=patch_size,
                            )
                            plt.savefig(
                                os.path.join(phrase_q_dir, "phrase_attention_thirds.png"),
                                bbox_inches="tight",
                                dpi=140,
                            )
                            plt.close(fig3)

            # Iterate layers
            iterator = tqdm(
                enumerate(all_layers_data),
                total=len(all_layers_data),
                desc=f"  > Processing Layers",
                leave=False,
            ) if args.attention_mode else enumerate(all_layers_data)

            for layer_idx, layer_attn_tensor in iterator:
                try:
                    seq_len_layer = layer_attn_tensor.shape[2]
                    # Determine which token positions act as the attention source
                    query_positions = _get_query_positions_for_layer(
                        args.attention_source,
                        prompt_len=prompt_len,
                        seq_len=seq_len_layer,
                        phrase_positions=phrase_positions,
                    )

                    # Slice keys to image tokens
                    if end_idx > seq_len_layer:
                        img_key_slice = slice(seq_len_layer - num_patches, seq_len_layer)
                    else:
                        img_key_slice = slice(start_idx, end_idx)

                    layer_all = layer_attn_tensor[0]  # [heads, seq_len, seq_len]

                    if len(query_positions) == 1:
                        # Single-token query (e.g., first generated token)
                        heads_raw = layer_all[:, query_positions[0], :]
                        image_heads_flat = heads_raw[:, img_key_slice]
                    else:
                        # Multi-token query (e.g., relational phrase): average over phrase tokens
                        phrase_to_img = layer_all[:, query_positions, img_key_slice]
                        image_heads_flat = phrase_to_img.mean(dim=1)
                    # 1. ALWAYS Calculate Trends
                    avg_attention_flat = image_heads_flat.mean(dim=0).numpy()

                    # For decision simple aggregation, keep per-layer avg map
                    if (
                        args.attention_source == "first_generated_token"
                        and args.attention_mode == "simple"
                    ):
                        try:
                            decision_per_layer_maps.append(avg_attention_flat.reshape(grid_dim, grid_dim))
                        except Exception:
                            pass
                    
                    layer_total_target_attn = 0.0

                    for g_idx, g_indices in enumerate(TARGET_GROUPS):
                        group_sum = 0.0
                        for pid in g_indices:
                            val = avg_attention_flat[pid] if pid < len(avg_attention_flat) else 0.0
                            group_sum += float(val)
                        
                        group_scores_layerwise[g_idx].append(group_sum)
                        layer_total_target_attn += group_sum
                    
                    # Store the sum of ALL groups for this layer
                    current_question_layer_totals.append(layer_total_target_attn)

                    # Per-question: subject vs object attention
                    subj_sum = 0.0
                    for pid in subj_patch_indices:
                        val = avg_attention_flat[pid] if pid < len(avg_attention_flat) else 0.0
                        subj_sum += float(val)

                    obj_sum = 0.0
                    for pid in obj_patch_indices:
                        val = avg_attention_flat[pid] if pid < len(avg_attention_flat) else 0.0
                        obj_sum += float(val)

                    subject_layer_scores.append(subj_sum)
                    object_layer_scores.append(obj_sum)

                    # Also compute fractional attention on subject/object boxes (avg over heads)
                    subj_frac = None
                    obj_frac = None
                    if subj_patch_indices:
                        subj_frac = compute_bbox_attention_fraction(avg_attention_flat, subj_patch_indices)
                        subject_fraction_layers.append(subj_frac)
                    if obj_patch_indices:
                        obj_frac = compute_bbox_attention_fraction(avg_attention_flat, obj_patch_indices)
                        object_fraction_layers.append(obj_frac)

                    # Per-layer, per-head fractions using the same definition
                    layer_entry = {
                        "layer_idx": int(layer_idx),
                        "avg_subject_fraction": float(subj_frac) if subj_frac is not None else None,
                        "avg_object_fraction": float(obj_frac) if obj_frac is not None else None,
                        "heads": [],
                    }
                    num_heads_layer = image_heads_flat.shape[0]
                    for h in range(num_heads_layer):
                        head_vec = image_heads_flat[h].detach().cpu().numpy()
                        h_subj = compute_bbox_attention_fraction(head_vec, subj_patch_indices) if subj_patch_indices else None
                        h_obj = compute_bbox_attention_fraction(head_vec, obj_patch_indices) if obj_patch_indices else None
                        layer_entry["heads"].append(
                            {
                                "head_idx": int(h),
                                "subject_fraction": float(h_subj) if h_subj is not None else None,
                                "object_fraction": float(h_obj) if h_obj is not None else None,
                            }
                        )
                    per_layer_head_fractions.append(layer_entry)

                    # 2. Plot per-layer grids depending on attention_mode
                    if args.attention_mode:
                        # Skip heavy per-layer grids in lightweight modes
                        if args.attention_mode == "simple":
                            continue

                        if (
                            args.attention_source == "rel_phrase"
                            and prompt_attn_layers is not None
                            and phrase_positions
                            and layer_idx < len(prompt_attn_layers)
                        ):
                            layer_prompt = prompt_attn_layers[layer_idx][0]  # [heads, seq_len, seq_len]
                            seq_len = layer_prompt.shape[1]
                            phrase_pos_in_range = [p for p in phrase_positions if 0 <= p < seq_len]
                            if end_idx > seq_len:
                                img_key_slice = slice(seq_len - num_patches, seq_len)
                            else:
                                img_key_slice = slice(start_idx, end_idx)

                            if phrase_pos_in_range:
                                phrase_to_img = layer_prompt[:, phrase_pos_in_range, img_key_slice]
                                plot_heads_flat = phrase_to_img.mean(dim=1).float().numpy()  # [heads, num_patches]
                                current_num_heads = plot_heads_flat.shape[0]
                                heads_map_2d = plot_heads_flat.reshape(current_num_heads, grid_dim, grid_dim)
                                avg_map_2d = heads_map_2d.mean(axis=0)
                            else:
                                current_num_heads = image_heads_flat.shape[0]
                                heads_map_2d = image_heads_flat.view(current_num_heads, grid_dim, grid_dim).float().numpy()
                                avg_map_2d = heads_map_2d.mean(axis=0)
                        else:
                            current_num_heads = image_heads_flat.shape[0]
                            heads_map_2d = image_heads_flat.view(current_num_heads, grid_dim, grid_dim).float().numpy()
                            avg_map_2d = heads_map_2d.mean(axis=0)

                        fig = create_layer_grid_plot(
                            vis_image,
                            heads_map_2d,
                            avg_map_2d,
                            layer_idx,
                            target_groups_meta,
                            patch_size=patch_size,
                            source_token_text=first_gen_text,
                        )
                        plt.savefig(os.path.join(q_dir, f"layer_{layer_idx:02d}.png"), bbox_inches='tight')
                        plt.close(fig)

                except Exception as e:
                    tqdm.write(f"  [Warning] Error layer {layer_idx}: {e}")
            
            # --- STORE PER-QUESTION ATTENTION METRICS ---
            try:
                # Aggregate attention on all target patches across layers
                total_targets_all_layers = float(sum(current_question_layer_totals)) if current_question_layer_totals else 0.0
                mean_targets_per_layer = float(np.mean(current_question_layer_totals)) if current_question_layer_totals else 0.0

                # Aggregate per-object attention across layers (absolute sums)
                per_object_metrics = []
                for g_idx, scores in group_scores_layerwise.items():
                    total_obj = float(sum(scores)) if scores else 0.0
                    mean_obj = float(np.mean(scores)) if scores else 0.0
                    meta = group_metadata[g_idx] if g_idx < len(group_metadata) else {"object_id": g_idx}
                    per_object_metrics.append(
                        {
                            "object_id": meta.get("object_id", g_idx),
                            "total_attention_all_layers": total_obj,
                            "mean_attention_per_layer": mean_obj,
                        }
                    )

                subject_total = float(sum(subject_layer_scores)) if subject_layer_scores else 0.0
                object_total = float(sum(object_layer_scores)) if object_layer_scores else 0.0

                # Fractional attention on subject/object boxes averaged over layers
                subject_fraction_mean = (
                    float(np.mean(subject_fraction_layers)) if subject_fraction_layers else 0.0
                )
                object_fraction_mean = (
                    float(np.mean(object_fraction_layers)) if object_fraction_layers else 0.0
                )

                # Attach metrics to the latest results_summary entry for this question
                if results_summary:
                    results_summary[-1]["attention_metrics"] = {
                        "source_mode": args.attention_source,
                        "total_targets_all_layers": total_targets_all_layers,
                        "mean_targets_per_layer": mean_targets_per_layer,
                        "per_object": per_object_metrics,
                        "subject_total_all_layers": subject_total,
                        "object_total_all_layers": object_total,
                        "subject_fraction_mean": subject_fraction_mean,
                        "object_fraction_mean": object_fraction_mean,
                        "per_layer_head_fractions": per_layer_head_fractions,
                    }

                # Also build head x layer grids (subject/object fractions) for visualization
                if per_layer_head_fractions:
                    num_layers_plots = len(per_layer_head_fractions)
                    max_heads = max(len(le["heads"]) for le in per_layer_head_fractions)

                    if max_heads > 0 and num_layers_plots > 0:
                        subj_grid = np.full((max_heads, num_layers_plots), np.nan, dtype=float)
                        obj_grid = np.full((max_heads, num_layers_plots), np.nan, dtype=float)

                        for li, layer_entry in enumerate(per_layer_head_fractions):
                            for head_info in layer_entry.get("heads", []):
                                h_idx = int(head_info.get("head_idx", -1))
                                if 0 <= h_idx < max_heads:
                                    sf = head_info.get("subject_fraction", None)
                                    of = head_info.get("object_fraction", None)
                                    if sf is not None:
                                        subj_grid[h_idx, li] = float(sf)
                                    if of is not None:
                                        obj_grid[h_idx, li] = float(of)

                        # Lookup human-readable shapes for subject/object (if available)
                        subj_shape = None
                        obj_shape = None
                        try:
                            if subject_id is not None:
                                subj_shape = object_id_to_shape.get(int(subject_id))
                            if object_id is not None:
                                obj_shape = object_id_to_shape.get(int(object_id))
                        except Exception:
                            pass

                        # Save per-question heatmaps into the same question directory
                        plot_head_layer_fraction_heatmaps(
                            subj_grid,
                            obj_grid,
                            q_dir,
                            i,
                            question_text,
                            subject_id=subject_id,
                            object_id=object_id,
                            subject_shape=subj_shape,
                            object_shape=obj_shape,
                        )
            except Exception as e:
                tqdm.write(f"  [Warning] Could not compute attention metrics for Q{i}: {e}")

            # --- ACCUMULATE DATA FOR COMPARISON PLOT ---
            if is_correct:
                correct_runs_layerwise.append(current_question_layer_totals)
            else:
                incorrect_runs_layerwise.append(current_question_layer_totals)

            # Decision simple: aggregate thirds and save
            if (
                args.attention_source == "first_generated_token"
                and args.attention_mode == "simple"
                and decision_per_layer_maps
            ):
                L = len(decision_per_layer_maps)
                a = L // 3
                b = (2 * L) // 3

                early_maps = decision_per_layer_maps[:a] if a > 0 else decision_per_layer_maps[:1]
                mid_maps = decision_per_layer_maps[a:b] if b > a else decision_per_layer_maps[a : a + 1]
                late_maps = decision_per_layer_maps[b:] if b < L else decision_per_layer_maps[-1:]

                maps_3 = {
                    "early": np.mean(np.stack(early_maps, axis=0), axis=0),
                    "mid": np.mean(np.stack(mid_maps, axis=0), axis=0),
                    "late": np.mean(np.stack(late_maps, axis=0), axis=0),
                }

                fig_dec = create_decision_thirds_plot(
                    vis_image,
                    maps_3,
                    patch_size=patch_size,
                    source_token_text=first_gen_text,
                )
                plt.savefig(os.path.join(q_dir, "decision_attention_thirds.png"), bbox_inches="tight", dpi=140)
                plt.close(fig_dec)

            # 3. SAVE INDIVIDUAL TREND PLOT (If requested)
            if args.plot_trends:
                trend_q_dir = os.path.join(trends_dir, f"Q{i}")
                os.makedirs(trend_q_dir, exist_ok=True)
                plot_attention_trends(group_scores_layerwise, group_metadata, trend_q_dir, i, question_text, rel_group=rel_group, rel_type=rel_type)

            # 4. SAVE per-question subject vs object plot (always when attention is computed)
            plot_subject_object_attention(
                subject_layer_scores,
                object_layer_scores,
                q_dir,
                i,
                question_text,
                subject_id,
                object_id,
                rel_group=rel_group,
                rel_type=rel_type,
            )
            

    # --- SAVE SUMMARY ---
    final_metadata = {
        "timestamp": timestamp,
        "level": args.level,
        "id": args.id,
        "num_objects": len(TARGET_GROUPS),
        "objects": group_metadata,
        "results": results_summary,
        "accuracy": sum(r['is_correct'] for r in results_summary) / len(results_summary) if results_summary else 0
    }
    
    with open(os.path.join(overview_dir, "results.json"), "w") as f:
        json.dump(final_metadata, f, indent=4)

    if results_summary:
        plot_evaluation_results(results_summary, overview_dir, f"{args.id} (Level {args.level})")
    
    # --- GENERATE THE COMPARISON PLOT ---
    if args.plot_trends:
        plot_correct_vs_incorrect_trends(correct_runs_layerwise, incorrect_runs_layerwise, overview_dir)
        
    print(f"\nCompleted. Results saved to: {os.path.abspath(image_output)}")

if __name__ == "__main__":
    main()