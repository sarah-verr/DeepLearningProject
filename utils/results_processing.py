"""
Unified results processing for storing model predictions across datasets.
"""

import torch
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path


def extract_raw_answer_from_response(response: str) -> str:
    """Extract the raw answer from the full model response.
    
    This extracts only the model's generated answer, not normalized.
    """
    # The response format is: prompt + "QUESTION: {question}\n ASSISTANT: {answer}"
    # Try to find the ASSISTANT marker (most reliable)
    if "ASSISTANT:" in response:
        parts = response.split("ASSISTANT:", 1)
        if len(parts) > 1:
            generated = parts[-1].strip()
            # Remove any trailing special tokens or formatting
            generated = generated.replace("</s>", "").strip()
            # Remove trailing newlines and whitespace
            generated = generated.split("\n")[0].strip()
            # Remove trailing punctuation but keep the raw text
            generated = generated.rstrip(".,!?").strip()
            return generated
    
    # Fallback: try to find "QUESTION:" and extract what comes after ASSISTANT
    if "QUESTION:" in response and "ASSISTANT:" in response:
        assistant_idx = response.rfind("ASSISTANT:")
        if assistant_idx != -1:
            generated = response[assistant_idx + len("ASSISTANT:"):].strip()
            generated = generated.split("\n")[0].strip()
            generated = generated.rstrip(".,!?").strip()
            return generated
    
    # Last resort: return the full response
    return response.strip()


def extract_confidence_from_output(outputs, tokenizer, predicted_answer: str) -> Optional[float]:
    """Extract confidence (probability) for a predicted answer from model outputs.
    
    For yes/no questions, this matches the existing get_yes_no_probability logic.
    For other questions, it extracts the probability of the first generated token(s).
    """
    scores = getattr(outputs, "scores", None)
    if not scores or len(scores) == 0:
        return None
    
    # Get probabilities for the first generated token
    first_token_logits = scores[0][0]  # [vocab_size]
    probs = torch.softmax(first_token_logits, dim=-1)
    
    # For yes/no, use existing logic
    if predicted_answer and predicted_answer.lower() in ["yes", "no"]:
        yes_variants = [" yes", " Yes", "yes", "Yes"]
        no_variants = [" no", " No", "no", "No"]
        
        def _last_token_id(text: str) -> int | None:
            ids = tokenizer.encode(text, add_special_tokens=False)
            return ids[-1] if ids else None
        
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
            return None
        
        if predicted_answer.lower() == "yes":
            return float(prob_yes / total)
        else:
            return float(prob_no / total)
    
    # For other answers, get the probability of the first token(s) of the answer
    # Encode the predicted answer and get the probability of its first token
    answer_tokens = tokenizer.encode(predicted_answer, add_special_tokens=False)
    if not answer_tokens:
        # Fallback: use the maximum probability token
        return float(probs.max().item())
    
    # Use the first token of the answer
    first_answer_token_id = answer_tokens[0]
    if first_answer_token_id < len(probs):
        return float(probs[first_answer_token_id].item())
    
    # Fallback: use max probability
    return float(probs.max().item())


def create_unified_results_df(
    results_list: List[Dict[str, Any]],
    dataset_name: str,
    output_model_answer_raw: bool = True,
    extract_confidence_from_outputs: bool = False,
    outputs_list: Optional[List] = None,
    tokenizer = None,
) -> pd.DataFrame:
    """Create a unified results DataFrame from model inference results.
    
    Args:
        results_list: List of result dicts from infer_model_for_levels
        dataset_name: Name of the dataset (e.g., "vlm_levels", "vlm_levels_v2", "vlm_levels_v3")
        output_model_answer_raw: If True, extract and store raw model answer (not normalized)
        extract_confidence_from_outputs: If True, extract confidence from model outputs
        outputs_list: Optional list of model outputs (for confidence extraction)
        tokenizer: Optional tokenizer (for confidence extraction)
    
    Returns:
        DataFrame with unified structure:
        - dataset: dataset name
        - level_id, image_id, qa_id
        - question, question_type (if present)
        - ground_truth
        - model_answer_raw: raw extracted answer (not normalized)
        - model_confidence: confidence/probability of predicted answer
        - is_correct: boolean indicating if prediction matches ground truth
        - Additional metadata from results_list
    """
    df_list = []
    
    for idx, result in enumerate(results_list):
        row = {
            "dataset": dataset_name,
            "level_id": result.get("level_id"),
            "image_id": result.get("image_id"),
            "qa_id": result.get("qa_id"),
            "question": result.get("question"),
            "ground_truth": result.get("ground_truth"),
            "model_answer_raw": None,
            "model_confidence": None,
            "is_correct": None,
        }
        
        # Add question_type if present
        if "question_type" in result:
            row["question_type"] = result["question_type"]
        
        # Extract raw model answer from response
        if output_model_answer_raw and "response" in result:
            row["model_answer_raw"] = extract_raw_answer_from_response(result["response"])
        
        # Use existing prediction/confidence if available (for yes/no questions)
        if "prediction" in result and result["prediction"] is not None:
            # For yes/no questions, prediction is already extracted
            # But we still want the raw answer for consistency
            if output_model_answer_raw and row["model_answer_raw"] is None:
                row["model_answer_raw"] = result["prediction"]
            else:
                # Store prediction as raw answer if raw extraction failed
                if row["model_answer_raw"] is None:
                    row["model_answer_raw"] = result["prediction"]
        
        if "confidence" in result and result["confidence"] is not None:
            row["model_confidence"] = result["confidence"]
        
        # Extract confidence from outputs if requested
        if extract_confidence_from_outputs and outputs_list and idx < len(outputs_list):
            if tokenizer and row["model_answer_raw"]:
                confidence = extract_confidence_from_output(
                    outputs_list[idx], tokenizer, row["model_answer_raw"]
                )
                if confidence is not None:
                    row["model_confidence"] = confidence
        
        # Determine correctness (comparison will be done separately after normalization)
        # For now, leave it as None - will be computed in the accuracy scripts
        
        # Add any additional metadata
        for key, value in result.items():
            if key not in row and key not in ["response", "prediction", "confidence"]:
                row[key] = value
        
        df_list.append(row)
    
    df = pd.DataFrame(df_list)
    return df


def save_unified_results(df: pd.DataFrame, output_path: Path, experiment_name: str):
    """Save unified results DataFrame to CSV."""
    df.to_csv(output_path, index=False)
    print(f"\nUnified results saved to: {output_path.resolve()}")
    print(f"Dataset: {df['dataset'].iloc[0] if len(df) > 0 else 'N/A'}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {', '.join(df.columns.tolist())}")
