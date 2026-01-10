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

from .prompt_templates import (
    build_visual_yesno_prompt,
    build_caption_yesno_prompt,
    build_scene_yesno_prompt,
)

from .attention_utils import (get_phrase_token_positions, get_image_token_indices, get_last_token_index, get_text_token_indices, get_entity_indices, get_image_entity_indices, aggregate_attention_between_groups)

MODEL_ID = "llava-hf/llava-1.5-7b-hf"


# CORE FUNCTIONS: The below two functions ensure that model initialisation params and generation config are standardised across inferences
def _init_model(MODEL_ID: str,
    output_hidden_states: bool = False,
):
    processor = LlavaProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        output_attentions=False,
        attn_implementation = "eager",
        output_hidden_states=output_hidden_states,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, device

def generate_output_for_model(model, inputs):
    generated_out = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        output_scores=True,
                        output_attentions=False,
                        return_dict_in_generate=True,)
    
    return generated_out

def infer_model_for_levels(
    level_ids: List[str],
    prompt_strategy: Literal["visual", "caption", "scene"] = "visual",
    show_llm_output: bool = False,
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
            qa_pairs = annotation["qa"]
            prompts = _build_prompts(qa_pairs, annotation, prompt_strategy)
            processed_prompts = [processor.apply_chat_template(prompt, add_generation_prompt=True) for prompt in prompts]
            
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
                if show_llm_output:
                    if output == None:
                        print("No output generated.")

                    # Also decode full output to see prompt + answer
                    full_output = processor.decode(output.sequences[0], skip_special_tokens=True)
                    print(f"\n--- Model Response {level_id} image: {image_id} ---")
                    print(f"{full_output}")
                    print("---" * 20)

                prediction , confidence = get_yes_no_probability(output, tokenizer=processor.tokenizer)

                results_list.append({
                    "level_id": level_id,
                    "image_id": image_id,
                    "qa_id": qa_pair["id"],
                    "question": qa_pair["question"],
                    "ground_truth": qa_pair["answer"],
                    "prediction": prediction,
                    "confidence": confidence,
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
    attn_results: List[Dict[str, Any]] = []

    # --- Loop over levels / images / QAs with tqdm progress bars ---
    for level_id, ann_for_level, imgs_for_level in tqdm(
        list(zip(level_ids, annotations_all_levels, images_by_level)),
        desc="Levels",
    ):
        for image, annotation in tqdm(
            list(zip(imgs_for_level, ann_for_level)),
            desc=f"Images in {level_id}",
            leave=False,
        ):
            image_id = annotation["image_id"]
            qa_pairs = annotation["qa"]

            # Build chat-style prompts as in infer_model_for_levels
            prompts = _build_prompts(qa_pairs, annotation, prompt_strategy)
            chat_prompts = [
                processor.apply_chat_template(p, add_generation_prompt=True)
                for p in prompts
            ]

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

                prediction, confidence = get_yes_no_probability(
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
                        "ground_truth": qa["answer"],
                        "prediction": prediction,
                        "confidence": confidence,
                        "relation_type": qa["rel_type"],
                        "attention_metrics": attention_metrics,
                    }
                )

                # clean up per‑QA intermediates
                del gen_out, inputs, full_inputs, attentions_cpu, full_ids_1d
                torch.cuda.empty_cache()
                
    return attn_results

# HELPERS to load data, build prompts, get yes\no probabilities

def get_yes_no_probability(outputs, tokenizer) -> Tuple[str, float, float, float]:
    """Extract yes/no prediction and confidence from model outputs."""
    first_token_logits = outputs.scores[0][0]
    probs = torch.softmax(first_token_logits, dim=-1)

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
    
    return prediction, confidence

def _load_annotation(level_id: str) -> List[Dict[str, Any]]:
    """Load all annotation JSONs for all images in a given level."""
    ann_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data/vlm_levels",
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
        os.path.dirname(os.path.dirname(__file__)),
        "data/vlm_levels",
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

def _build_prompts(qa_pairs: List[Dict], annotation: Dict[str, Any], strategy: str) -> List:
    """Build conversation prompts for all QA pairs based on strategy."""
    prompts = []
    for qa in qa_pairs:
        question = qa["question"]
        if strategy == "visual":
            p = build_visual_yesno_prompt(question)
        elif strategy == "caption": # TODO
            p = build_caption_yesno_prompt(annotation, qa)
        elif strategy == "scene": # TODO
            # TODO: Implement scene-based prompting
            p = build_scene_yesno_prompt(annotation, question)
        else:
            p = build_visual_yesno_prompt(question)
        prompts.append(p)
    return prompts