"""
Central prompting script for LLaVA inference.
"""

import os
import json
from typing import Literal, Optional, Dict, Any, List, Tuple
import torch
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration

from .prompt_templates import (
    build_visual_yesno_prompt,
    build_caption_yesno_prompt,
    build_scene_yesno_prompt,
)

# Model configuration
MODEL_ID = "llava-hf/llava-1.5-7b-hf"


def run_prompt(
    image_id: str,
    level_id: str,
    prompt_strategy: Literal["visual", "caption", "scene"] = "visual",
    show_llm_output: bool = True,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
    attention_source_token: Optional[Literal["entity", "relation", "last"]] = "last",
) -> Dict[str, Any]:
    """
    Run batch inference on all QA pairs for a given image and level.
    
    Args:
        image_id: Image identifier (e.g., "00000_b")
        level_id: Level identifier (e.g., "level_0")
        prompt_strategy: How to format prompts ("visual", "caption", or "scene")
        output_attentions: Whether to extract attention patterns
        attention_source_token: Which token to use as attention source
        
    Returns:
        Dictionary with results for all QA pairs
    """
    # Step 1: Load annotation and extract QA pairs
    annotation = _load_annotation(level_id, image_id)
    qa_pairs = annotation["qa"]
    
    # Step 2: Load image
    image_path = _get_image_path(level_id, image_id, annotation.get("background"))
    image = Image.open(image_path)
    
    # Step 3: Initialize model and processor
    processor = LlavaProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Step 4: Build prompts for all QA pairs
    prompts = _build_prompts(qa_pairs, annotation, prompt_strategy)
    processed_prompts = [processor.apply_chat_template(prompt, add_generation_prompt=True) for prompt in prompts]
    
    results = {
        "image_id": image_id,
        "level_id": level_id,
        "results": [],
    }
    # Step 5: Run inference for each prompt
    for idx, (prompt, qa_pair) in enumerate(zip(processed_prompts, qa_pairs)):
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

        with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=10, output_scores=True, 
                return_dict_in_generate=True)

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

        result = {
            "qa_id": qa_pair["id"],
            "question": qa_pair["question"],
            "ground_truth": qa_pair["answer"],
            "prediction": prediction,
            "confidence": confidence,
        }

        results["results"].append(result)

    return results

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

def _load_annotation(level_id: str, image_id: str) -> Dict[str, Any]:
    """Load annotation JSON for a given image."""
    ann_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "vlm_levels",
        level_id,
        "ann",
        f"{image_id}.json"
    )
    with open(ann_path, "r") as f:
        return json.load(f)

def _get_image_path(level_id: str, image_id: str, background: str) -> str:
    """Construct path to image file."""
    # Adjust this based on your actual image storage structure
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "vlm_levels",
        level_id,
        "images",
    )
    return os.path.join(base_dir, f"{image_id}.png")  # Adjust extension as needed

def _build_prompts(qa_pairs: List[Dict], annotation: Dict[str, Any], strategy: str) -> List:
    """Build conversation prompts for all QA pairs based on strategy."""
    prompts = []
    for qa in qa_pairs:
        question = qa["question"]
        if strategy == "visual":
            p = build_visual_yesno_prompt(question)
        elif strategy == "caption": # TODO
            caption = annotation["captions"][0]  # Use first caption
            p = build_caption_yesno_prompt(caption, question)
        elif strategy == "scene": # TODO
            # TODO: Implement scene-based prompting
            p = build_visual_yesno_prompt(question)
        else:
            p = build_visual_yesno_prompt(question)
        prompts.append(p)
    return prompts