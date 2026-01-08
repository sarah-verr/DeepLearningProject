"""
Central prompting script for LLaVA inference.
"""

import os
import json
from typing import Literal, Optional, Dict, Any, List, Tuple
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from .prompt_templates import (
    build_visual_yesno_prompt,
    build_caption_yesno_prompt,
    build_scene_yesno_prompt,
)

# Model configuration
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

def scene_to_text(annotation_data: dict) -> str:
    """Convert annotation data to scene description text."""
    meta = annotation_data.get("meta", {}) or {}
    patch = int(meta.get("patch", 14))

    lines = []
    for obj in annotation_data.get("objects", []):
        oid = obj.get("id", None)
        color = obj.get("color", "unknown")
        shape = obj.get("shape", "unknown")

        center = obj.get("center", None)
        if isinstance(center, list) and len(center) == 2:
            cx, cy = center
            gx, gy = int(cx) // patch, int(cy) // patch
            pos = f"grid ({gx}, {gy})"
        else:
            pos = "grid (unknown, unknown)"

        lines.append(f"Object {oid}: {color} {shape} at {pos}")

    return "\n".join(lines)


def get_caption_for_qa(annotation_data: dict, qa_item: dict) -> str | None:
    """Fetch the caption text associated with this QA item, if available."""
    cap_id = qa_item.get("caption_id")
    if cap_id is None:
        # Fallback to first caption
        captions = annotation_data.get("captions", []) or []
        if captions and isinstance(captions[0], str):
            return captions[0].strip()
        return None

    captions_meta = annotation_data.get("captions_meta")
    if not isinstance(captions_meta, list) or not captions_meta:
        return None

    try:
        cap_id_int = int(cap_id)
    except Exception:
        return None

    # Direct indexing
    if 0 <= cap_id_int < len(captions_meta):
        c = captions_meta[cap_id_int]
        if isinstance(c, dict):
            cap_text = c.get("caption")
            return cap_text.strip() if isinstance(cap_text, str) and cap_text.strip() else None

    # Fallback scan by explicit id
    for c in captions_meta:
        if not isinstance(c, dict):
            continue
        if c.get("id") == cap_id_int:
            cap_text = c.get("caption")
            return cap_text.strip() if isinstance(cap_text, str) and cap_text.strip() else None

    return None


def run_prompt(
    image_id: str,
    level_id: str,
    prompt_strategy: Literal["visual", "caption", "scene"],
    attention_required: bool = False,
    question: Optional[str] = None,
    model: Optional[LlavaNextForConditionalGeneration] = None,
    processor: Optional[LlavaNextProcessor] = None,
    base_data_path: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a prompt with the specified strategy and return results.
    
    Args:
        image_id: Image identifier (e.g., "001")
        level_id: Level identifier (e.g., "1")
        prompt_strategy: One of "visual", "caption", or "scene"
        attention_required: Whether to compute and return attention data
        question: Specific question to ask (if None, uses first question from annotation)
        model: Pre-loaded model (if None, loads model)
        processor: Pre-loaded processor (if None, loads processor)
        base_data_path: Base path to data directory (defaults to standard path)
        model_id: Model identifier (defaults to MODEL_ID constant)
    
    Returns:
        Dictionary containing:
        - prediction: "yes" or "no"
        - confidence: Confidence score (0-1)
        - prob_yes: Probability of "yes"
        - prob_no: Probability of "no"
        - question: The question asked
        - ground_truth: Ground truth answer if available
        - attention_data: Attention data if attention_required=True (None otherwise)
        - metadata: Additional metadata (subject_id, object_id, rel_type, etc.)
    """
    # Set defaults
    if base_data_path is None:
        base_data_path = f"/home/{os.environ.get('USER', 'user')}/DeepLearningProject/Synthetic-Data/vlm_levels"
    if model_id is None:
        model_id = MODEL_ID
    
    # Construct paths
    level_dir = f"level_{level_id}"
    image_path = os.path.join(base_data_path, level_dir, "images", f"{image_id}.png")
    json_path = os.path.join(base_data_path, level_dir, "ann", f"{image_id}.json")
    
    # Validate paths
    if prompt_strategy == "visual" and not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Annotation not found: {json_path}")
    
    # Load annotation
    with open(json_path, 'r') as f:
        annotation_data = json.load(f)
    
    # Get question
    if question is None:
        qa_list = annotation_data.get('qa', [])
        if not qa_list:
            raise ValueError(f"No questions found in annotation: {json_path}")
        qa_item = qa_list[0]
        question = qa_item.get("question", "")
        ground_truth = qa_item.get("answer", "").lower()
        metadata = {
            "subject_id": qa_item.get("subject_id"),
            "object_id": qa_item.get("object_id"),
            "rel_type": qa_item.get("rel_type"),
            "rel_group": qa_item.get("rel_group"),
            "rel_phrase": qa_item.get("rel_phrase"),
        }
    else:
        # Find matching question in annotation
        qa_list = annotation_data.get('qa', [])
        qa_item = None
        for qa in qa_list:
            if qa.get("question", "").strip() == question.strip():
                qa_item = qa
                break
        
        if qa_item is None:
            # Use first question as fallback, but use provided question text
            qa_item = qa_list[0] if qa_list else {}
            ground_truth = qa_item.get("answer", "").lower() if qa_item else ""
            metadata = {
                "subject_id": qa_item.get("subject_id") if qa_item else None,
                "object_id": qa_item.get("object_id") if qa_item else None,
                "rel_type": qa_item.get("rel_type") if qa_item else None,
                "rel_group": qa_item.get("rel_group") if qa_item else None,
                "rel_phrase": qa_item.get("rel_phrase") if qa_item else None,
            }
        else:
            ground_truth = qa_item.get("answer", "").lower()
            metadata = {
                "subject_id": qa_item.get("subject_id"),
                "object_id": qa_item.get("object_id"),
                "rel_type": qa_item.get("rel_type"),
                "rel_group": qa_item.get("rel_group"),
                "rel_phrase": qa_item.get("rel_phrase"),
            }
    
    if not question:
        raise ValueError("No question provided or found in annotation")
    
    # Load model and processor if not provided
    model_loaded = model is not None
    if model is None:
        print(f"Loading model: {model_id}...")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        model.to("cuda:0")
    if processor is None:
        processor = LlavaNextProcessor.from_pretrained(model_id)
    
    # Build prompt based on strategy
    if prompt_strategy == "visual":
        image = Image.open(image_path).convert('RGB')
        conversation = build_visual_yesno_prompt(question)
        prompt_text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(text=prompt_text, images=image, return_tensors="pt")
        inputs = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
    elif prompt_strategy == "caption":
        caption = get_caption_for_qa(annotation_data, qa_item)
        if not caption:
            raise ValueError("No caption found in annotation for caption strategy")
        conversation = build_caption_yesno_prompt(caption, question)
        prompt_text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(text=prompt_text, return_tensors="pt")
        inputs = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
    elif prompt_strategy == "scene":
        scene_text = scene_to_text(annotation_data)
        conversation = build_scene_yesno_prompt(scene_text, question)
        prompt_text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(text=prompt_text, return_tensors="pt")
        inputs = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
    else:
        raise ValueError(f"Unknown prompt strategy: {prompt_strategy}")
    
    prompt_len = inputs["input_ids"].shape[1]
    
    # Run inference
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            output_attentions=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    # Extract prediction
    prediction, confidence, prob_yes, prob_no = get_yes_no_probability(outputs, processor.tokenizer)
    
    # Compute attention if required
    attention_data = None
    if attention_required:
        try:
            seq = outputs.sequences.detach()
            attn_inputs = dict(inputs)
            attn_inputs["input_ids"] = seq
            attn_inputs["attention_mask"] = torch.ones_like(seq, dtype=torch.long, device="cuda:0")
            
            with torch.no_grad():
                fwd = model(
                    **attn_inputs,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True,
                )
            attention_data = [layer_tensor.detach().cpu() for layer_tensor in fwd.attentions]
            del fwd
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Could not compute attentions: {e}")
            attention_data = None
    
    # Clean up
    del outputs
    if not model_loaded:
        # Only clear cache if we loaded the model (don't interfere with external model)
        torch.cuda.empty_cache()
    
    # Build result
    result = {
        "prediction": prediction,
        "confidence": float(confidence),
        "prob_yes": float(prob_yes),
        "prob_no": float(prob_no),
        "question": question,
        "ground_truth": ground_truth,
        "is_correct": prediction == ground_truth if ground_truth else None,
        "attention_data": attention_data,
        "metadata": metadata,
        "prompt_strategy": prompt_strategy,
        "image_id": image_id,
        "level_id": level_id,
    }
    
    return result


# Example usage
if __name__ == "__main__":
    """
    Example usage of the run_prompt function:
    
    # Basic usage with visual strategy
    result = run_prompt(
        image_id="001",
        level_id="1",
        prompt_strategy="visual",
        attention_required=False
    )
    print(f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.2f}")
    
    # With caption strategy
    result = run_prompt(
        image_id="001",
        level_id="1",
        prompt_strategy="caption",
        attention_required=False
    )
    
    # With scene strategy
    result = run_prompt(
        image_id="001",
        level_id="1",
        prompt_strategy="scene",
        attention_required=False
    )
    
    # With specific question and attention
    result = run_prompt(
        image_id="001",
        level_id="1",
        prompt_strategy="visual",
        attention_required=True,
        question="Is the green star below the pink circle?"
    )
    
    # Reusing a pre-loaded model (more efficient for multiple calls)
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf", 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    )
    model.to("cuda:0")
    
    for img_id in ["001", "002", "003"]:
        result = run_prompt(
            image_id=img_id,
            level_id="1",
            prompt_strategy="visual",
            attention_required=False,
            model=model,
            processor=processor
        )
        print(f"Image {img_id}: {result['prediction']} (confidence: {result['confidence']:.2f})")
    """
    pass

def get_yes_no_probability(outputs, tokenizer):
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
    
    return prediction, confidence, norm_yes, norm_no