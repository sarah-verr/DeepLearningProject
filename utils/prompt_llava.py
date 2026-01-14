"""
Central prompting script for LLaVA inference.
"""

import os
import json
from typing import Literal, Optional, Dict, Any, List, Tuple
import torch
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from tqdm.auto import tqdm
import numpy as np

from .prompt_templates import (
    build_visual_yesno_prompt,
    build_caption_yesno_prompt,
    build_scene_yesno_prompt,
    build_caption_text_yesno_prompt,
)

from .attention_utils import (get_phrase_token_positions, get_image_token_indices, get_last_token_index, get_text_token_indices, get_entity_indices, get_image_entity_indices, aggregate_attention_between_groups)

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
VLM_LEVELS_DIR = "data/vlm_levels"
VLM_TEXT_DIR = "data/vlm_levels_objective"

# CORE FUNCTIONS: The below two functions ensure that model initialisation params and generation config are standardised across inferences
def _init_model(MODEL_ID: str,
    output_hidden_states: bool = False,
    output_attentions: bool = False,
):
    processor = LlavaProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        output_attentions=output_attentions,
        attn_implementation="eager",
        output_hidden_states=output_hidden_states,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available() and device != "cuda":
        device = "mps"
        print("Using mps")
    model.to(device)
    model.eval()
    return processor, model, device

def generate_output_for_model(model, inputs, *, max_new_tokens: int = 10):
    generated_out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        output_scores=True,
                        output_attentions=False,
                        return_dict_in_generate=True,)
    
    return generated_out


def build_prompt(
    processor,
    prompt_strategy: Literal["visual", "caption", "scene", "text_only"],
    question: str,
    *,
    annotation: Optional[Dict[str, Any]] = None,
    caption: Optional[str] = None,
    qa_item: Optional[Dict[str, Any]] = None,
) -> str:
    if prompt_strategy == "text_only":
        if not caption:
            raise ValueError("caption is required for text_only prompts.")
        convo = build_caption_text_yesno_prompt(caption, question)
    elif prompt_strategy == "visual":
        convo = build_visual_yesno_prompt(question)
    elif prompt_strategy == "caption":
        if annotation is None:
            raise ValueError("annotation is required for caption prompts.")
        if qa_item is None:
            qa_item = {"question": question}
        convo = build_caption_yesno_prompt(annotation, qa_item)
    elif prompt_strategy == "scene":
        if annotation is None:
            raise ValueError("annotation is required for scene prompts.")
        convo = build_scene_yesno_prompt(annotation, question)
    else:
        raise ValueError(f"Unknown prompt strategy: {prompt_strategy}")
    return processor.apply_chat_template(convo, add_generation_prompt=True)

def prepare_inputs(processor, prompt: str, *, image: Optional[Image.Image] = None, device: Optional[str] = None):
    if image is None:
        inputs = processor(text=prompt, return_tensors="pt")
    else:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
    if device:
        inputs = inputs.to(device)
    return inputs

def infer_model_for_levels(
    level_ids: List[str],
    prompt_strategy: Literal["visual", "caption", "scene"] = "visual",
    show_llm_output: bool = False,
    use_plain_images: bool = False,
) -> Dict[str, Any]:
    """
    Run inference on all QA pairs in for the images in a level.
    
    Args:
        image_id: Image identifier (e.g., "00000_b")
        level_id: Level identifier (e.g., "level_0")
        prompt_strategy: How to format prompts ("visual", "caption", or "scene")
        output_attentions: Whether to extract attention patterns
        attention_source_token: Which token to use as attention source
        
    Returns:
        Dictionary with results for all QA pairs
    """
    # Step 1: Load annotation and images for all levels
    annotations_all_levels = []
    images_by_level = []
    for level in level_ids:
        annotations_all_levels.append(_load_annotation(level))
        images_by_level.append(_get_images(level))

    # Step 2: Initialize model and processor
    processor, model, device = _init_model(MODEL_ID, output_hidden_states=False)

    results_list = []
    # Step 3: Infer model for all levels
    for level_idx, (level_id, annotation_for_level, images_for_level) in enumerate(tqdm(
        list(zip(level_ids, annotations_all_levels, images_by_level)), desc="Levels",)
    ):
        print(f"Running inference for level: {level_id}")
        for image, annotation in tqdm(
            list(zip(images_for_level, annotation_for_level)),
            desc=f"Images in {level_id}",
            leave=False,
        ):
            image_id = annotation['image_id']
            print(f"Processing image... {image_id}")
            
            if use_plain_images:
                if image_id.endswith('_b'):
                    image = Image.open("data/plain_black.png")
                elif image_id.endswith('_w'):
                    image = Image.open("data/plain_white.png")
                else:
                    raise ValueError(f"Image ID {image_id} does not end with '_b' or '_w' for plain images")
            # else: image is already loaded from _get_images
            
            qa_pairs = annotation["qa"]
            processed_prompts = [build_prompt(processor,prompt_strategy,qa["question"],annotation=annotation,qa_item=qa,)for qa in qa_pairs]
            
            results_for_level = {
                "image_id": image_id,
                "level_id": level_id,
                "results": [],
            }
            # Step 5: Run inference for each prompt
            for idx, (prompt, qa_pair) in enumerate(
                tqdm(
                    list(zip(processed_prompts, qa_pairs)),
                    desc=f"QAs for {image_id}",
                    leave=False,
                )
            ):
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

                with torch.no_grad():
                        output = generate_output_for_model(model, inputs)
                # Extract just the generated part (after the prompt)]
                full_output = processor.decode(output.sequences[0], skip_special_tokens=True)
                if show_llm_output:
                    if output == None:
                        print("No output generated.")

                    # Also decode full output to see prompt + answer
                    print(f"\n--- Model Response {level_id} image: {image_id} ---")
                    print(f"{full_output}")
                    print("---" * 20)

                prediction, confidence, _, _ = get_yes_no_probability(output, tokenizer=processor.tokenizer)

                # Count image and text tokens in the prompt
                input_ids = inputs["input_ids"][0]
                image_token_id = model.config.image_token_index
                if image_token_id is not None:
                    n_image_tokens = (input_ids == image_token_id).sum().item()
                else:
                    n_image_tokens = 0
                n_text_tokens = len(input_ids) - n_image_tokens

                results_list.append({
                    "level_id": level_id,
                    "image_id": image_id,
                    "qa_id": qa_pair["id"],
                    "question": qa_pair["question"],
                    "response": full_output,
                    "prediction": prediction,
                    "confidence": confidence,
                    "num_image_tokens": n_image_tokens,
                    "num_text_tokens": n_text_tokens,
                })


    return results_list


def infer_model_with_attention(level_ids: List[str], key_pairs, prompt_strategy: Literal["visual", "caption", "scene"] = "visual"):
    """
    Runs model but this time with attention values aggregated according to source and target in the params
    """
    # --- Load data for all levels (same as infer_model_for_levels) ---
    annotations_all_levels = []
    images_by_level = []
    for level in level_ids:
        annotations_all_levels.append(_load_annotation(level))
        images_by_level.append(_get_images(level))

    # --- Load model & processor once, with attentions enabled ---
    processor, model, device = _init_model(MODEL_ID, output_hidden_states=False)

    # Import Plotter here to avoid circular import issues
    from utils.plotter import Plotter
    plotter = Plotter()

    # --- Loop over levels / images / QAs with tqdm progress bars ---
    for level_id, ann_for_level, imgs_for_level in tqdm(
        list(zip(level_ids, annotations_all_levels, images_by_level)),
        desc="Levels",
    ):
        attn_results: List[Dict[str, Any]] = []  # Per-level results
        for image, annotation in tqdm(
            list(zip(imgs_for_level, ann_for_level)),
            desc=f"Images in {level_id}",
            leave=False,
        ):
            image_id = annotation["image_id"]
            qa_pairs = annotation["qa"]

            # Build chat-style prompts as in infer_model_for_levels
            chat_prompts = [build_prompt(processor,prompt_strategy,qa["question"],annotation=annotation,qa_item=qa,) for qa in qa_pairs]

            for qa, chat_prompt in tqdm(
                list(zip(qa_pairs, chat_prompts)),
                desc=f"QAs for {image_id}",
                leave=False,
            ):
                # --- Build inputs (same as infer_model_for_levels) ---
                inputs = processor(
                    text=chat_prompt,
                    images=image,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # --- 1) Generation to get prediction (pure inference) ---
                with torch.no_grad():
                    gen_out = generate_output_for_model(model, inputs)

                full_output = processor.decode(gen_out.sequences[0], skip_special_tokens=True)

                prediction, confidence, _, _ = get_yes_no_probability(
                    gen_out, tokenizer=processor.tokenizer
                )

                # --- 2) Re-run forward pass on *full* sequence (prompt + generated) with attentions ---
                full_inputs = inputs.copy()
                full_inputs["input_ids"] = gen_out.sequences  # shape: [batch, prompt+generated_len]
                # if needed, build a matching attention mask
                full_inputs["attention_mask"] = torch.ones_like(gen_out.sequences).to(device)

                with torch.no_grad():
                    fwd_out = model(
                        **full_inputs,
                        output_attentions=True,
                        output_hidden_states=False,
                        use_cache=False,
                        return_dict=True,
                    )

                # move attentions + ids to CPU to save GPU memory
                attentions_cpu = tuple(
                    layer.detach().cpu() for layer in fwd_out.attentions
                )
                full_ids_1d = full_inputs["input_ids"][0].detach().cpu()
                
                # free GPU tensors ASAP 
                del fwd_out
                torch.cuda.empty_cache()

                # --- 3) Build source and target token groups ---

                image_token_id = model.config.image_token_index

                # Build ALL possible source groups
                source_groups = {}
                source_groups.update(get_last_token_index(full_ids_1d))
                source_groups.update(get_phrase_token_positions(processor.tokenizer, full_ids_1d, qa["rel_phrase"]))
                source_groups.update(get_text_token_indices(full_ids_1d, image_token_id))
                source_groups.update(get_entity_indices(processor.tokenizer, full_ids_1d, qa, annotation))

                # Build ALL possible target groups
                target_groups = {}
                target_groups.update(get_image_token_indices(full_ids_1d, image_token_id))
                target_groups.update(get_text_token_indices(full_ids_1d, image_token_id))
                target_groups.update(get_image_entity_indices(full_ids_1d, image_token_id, qa, annotation))
                target_groups.update(get_entity_indices(processor.tokenizer, full_ids_1d, qa, annotation))

                # Compute ALL combinations
                attention_metrics = aggregate_attention_between_groups(
                    attentions_cpu,
                    source_groups,
                    target_groups,
                    key_pairs=key_pairs  # specifies the source -> target combinations of interest for which we need to aggregate attention values
                )

                attn_results.append(
                    {
                        "level_id": level_id,
                        "image_id": image_id,
                        "qa_id": qa["id"],
                        "question": qa["question"],
                        "response": full_output,
                        "prediction": prediction,
                        "confidence": confidence,
                        "relation_type": qa["rel_type"],
                        "attention_metrics": attention_metrics,
                    }
                )

                # clean up per‑QA intermediates
                del gen_out, inputs, full_inputs, attentions_cpu, full_ids_1d
                torch.cuda.empty_cache()

        # After finishing this level, append results to a single file
        output_filename = "attention_results_all_levels.jsonl"
        output_path = plotter.results_dir / output_filename
        with open(output_path, "a") as f:
            for item in attn_results:
                f.write(json.dumps(item) + "\n")

    # Optionally, return nothing or a summary
    return None


def  visualise_full_attention(level_id: str, image_id: str, qa_id: int, processor=None, model=None, device=None):
    """
    Reads the image and annotation for the specified level and image_id,
    and for the specific qa_id, computes attentions and returns them along with the prompt.
    """
    
    # Load annotation for the specific image
    ann_path = os.path.join(VLM_LEVELS_DIR, level_id, "ann", f"{image_id}.json")
    with open(ann_path, "r") as f:
        annotation = json.load(f)
    annotation["image_id"] = image_id
    
    # Find the specific QA pair
    qa = next((q for q in annotation["qa"] if q["id"] == qa_id), None)
    if qa is None:
        raise ValueError(f"QA with id {qa_id} not found in annotation for {image_id}")
    
    # Load image
    img_dir = os.path.join(VLM_LEVELS_DIR, level_id, "images")
    img_filename = None
    for filename in os.listdir(img_dir):
        if filename.startswith(image_id) and filename.endswith((".png", ".jpg", ".jpeg")):
            img_filename = filename
            break
    if img_filename is None:
        raise FileNotFoundError(f"Image file for {image_id} not found in {img_dir}")
    img_path = os.path.join(img_dir, img_filename)
    image = Image.open(img_path)
    
    # Initialize model with attentions enabled if not provided
    if processor is None or model is None or device is None:
        processor, model, device = _init_model(MODEL_ID, output_attentions=True)
    
    # Build prompt (assuming 'visual' strategy, adjust if needed)
    prompt = build_prompt(processor, "visual", qa["question"], annotation=annotation, qa_item=qa)
    
    # Prepare inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate output
    with torch.no_grad():
        gen_out = generate_output_for_model(model, inputs)
    
    # Prepare full inputs (prompt + generated)
    full_inputs = inputs.copy()
    full_inputs["input_ids"] = gen_out.sequences
    full_inputs["attention_mask"] = torch.ones_like(gen_out.sequences).to(device)
    
    # Forward pass with attentions
    with torch.no_grad():
        fwd_out = model(
            **full_inputs,
            output_attentions=True,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
    
    # Convert attentions to numpy (list of [num_layers] arrays, each [1, heads, seq_len, seq_len])
    attentions_np = [att.detach().cpu().numpy() for att in fwd_out.attentions]
    # Slice to only keep attention to image tokens (columns)
    image_token_id = model.config.image_token_index
    image_positions = torch.where(full_inputs["input_ids"][0] == image_token_id)[0].cpu().numpy()
    attentions_np = [att[:, :, :, image_positions] for att in attentions_np]
    
    full_ids_1d = full_inputs["input_ids"][0].detach().cpu()
    rel_positions = get_phrase_token_positions(processor.tokenizer, full_ids_1d, qa["rel_phrase"])["relation"]
    
    full_output = processor.decode(gen_out.sequences[0], skip_special_tokens=True)
    
    # Clean up
    del gen_out, inputs, full_inputs, fwd_out
    torch.cuda.empty_cache()

    return attentions_np, full_output, rel_positions

# HELPERS to load data, build prompts, get yes\no probabilities

def get_yes_no_probability(outputs, tokenizer) -> Tuple[str, float, float, float]:
    """Extract yes/no prediction and confidence from model outputs."""
    scores = getattr(outputs, "scores", None)
    if not scores:
        return None, None, None, None

    first_token_logits = scores[0][0]
    probs = torch.softmax(first_token_logits, dim=-1)

    yes_variants = [" yes", " Yes", "yes", "Yes"]
    no_variants = [" no", " No", "no", "No"]

    def _last_token_id(text: str) -> int | None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return ids[-1] if ids else None

    # TODO: Made it to SET, needs to be tested!!
    yes_tokens = set(
        tid for tid in (_last_token_id(v) for v in yes_variants)
        if tid is not None
    )
    no_tokens = set(
        tid for tid in (_last_token_id(v) for v in no_variants)
        if tid is not None
    )

    prob_yes = sum([probs[t_id].item() for t_id in yes_tokens if t_id < len(probs)] + [0])
    prob_no = sum([probs[t_id].item() for t_id in no_tokens if t_id < len(probs)] + [0])

    total = prob_yes + prob_no
    if total <= 0:
        return None, None, None, None

    norm_yes = prob_yes / total
    norm_no = prob_no / total

    prediction = "yes" if norm_yes > norm_no else "no"
    confidence = norm_yes if prediction == "yes" else norm_no

    return prediction, confidence, norm_yes, norm_no

def _load_annotation(level_id: str) -> List[Dict[str, Any]]:
    """Load all annotation JSONs for all images in a given level."""
    ann_dir = os.path.join(
        VLM_LEVELS_DIR,
        level_id,
        "ann",
    )
    annotations = []
    for filename in sorted(os.listdir(ann_dir)):
        if filename.endswith(".json"):
            ann_path = os.path.join(ann_dir, filename)
            with open(ann_path, "r") as f:
                annotations.append(json.load(f))
            # get image id from filename (e.g., "00000_b.json" -> "00000_b")
            image_id = os.path.splitext(filename)[0]
            # add image_id to annotation
            annotations[-1]["image_id"] = image_id
    return annotations

def _get_images(level_id: str) -> List[Image.Image]:
    """Load all images for a given level."""
    img_dir = os.path.join(
        VLM_LEVELS_DIR,
        level_id,
        "images",
    )
    images = []
    for filename in sorted(os.listdir(img_dir)):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(img_dir, filename)
            image = Image.open(img_path) # .convert("RGB")
            images.append(image)
    return images
