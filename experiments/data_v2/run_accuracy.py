"""
Run accuracy computation for vlm_levels_v2 dataset with visual attribute questions.

This script runs inference on the vlm_levels_v2 dataset using the "visual_attribute" prompt strategy.
It compares model outputs with ground truth answers (color/shape attributes).
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add experiments to path
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.prompt_llava import infer_model_for_levels, MODEL_ID
from utils.plotter import Plotter
from utils.results_processing import extract_raw_answer_from_response
from pathlib import Path


def extract_answer_from_response(response: str) -> str:
    """Extract the answer from the full model response."""
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
            return generated
    
    # Fallback: try to find "QUESTION:" and extract what comes after ASSISTANT
    if "QUESTION:" in response and "ASSISTANT:" in response:
        # Find the last ASSISTANT: marker
        assistant_idx = response.rfind("ASSISTANT:")
        if assistant_idx != -1:
            generated = response[assistant_idx + len("ASSISTANT:"):].strip()
            generated = generated.split("\n")[0].strip()
            return generated
    
    # Last resort: return the full response
    return response.strip()


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    text = str(text).lower().strip()
    # Remove trailing punctuation
    text = text.rstrip(".,!?").strip()
    # Remove common prefixes
    text = text.replace("the ", "").strip()
    text = text.replace("a ", "").strip()
    text = text.replace("an ", "").strip()
    # Remove "is" constructions if present (e.g., "purple" from "the object is purple")
    if " is " in text:
        parts = text.split(" is ", 1)
        if len(parts) == 2:
            # Return the attribute part (after "is")
            text = parts[1].strip()
    return text


def compute_accuracy_for_questions(
    level_ids: list,
    show_output: bool = True,
    use_attention_mask: bool = False,
    always_mask: bool = False,
    mask_type: str = "objects_only",
    results_subdir: str = "accuracy_question_ablation",
) -> pd.Series:
    """Run inference and compute accuracy for visual attribute questions."""
    print(f"\n{'='*60}")
    print(f"Running visual_attribute accuracy experiment")
    print(f"QA Key: qa")
    print(f"Levels: {level_ids}")
    print(f"{'='*60}\n")
    
    # Run inference with optional attention masking
    results_list = infer_model_for_levels(
        level_ids=level_ids,
        prompt_strategy="visual_attribute",
        show_llm_output=show_output,
        qa_key="qa",
        data_dir="data/vlm_levels_v2",
        use_attention_mask=use_attention_mask,
        always_mask=always_mask,
        mask_type=mask_type,
    )
    
    # Convert to DataFrame
    if not results_list:
        print(f"No results returned")
        return pd.Series(dtype=float)
    
    results_df = pd.DataFrame(results_list)
    print(f"Debug: DataFrame columns: {results_df.columns.tolist()}")
    print(f"Debug: DataFrame shape: {results_df.shape}")
    
    # Add dataset name
    results_df["dataset"] = "vlm_levels_v2"
    
    # Extract raw model answers from response (not normalized for storage)
    results_df["model_answer_raw"] = results_df["response"].apply(extract_raw_answer_from_response)
    
    # For correctness checking, normalize answers
    results_df["prediction_clean"] = results_df["model_answer_raw"].str.lower().str.strip()
    results_df["prediction_normalized"] = results_df["prediction_clean"].apply(normalize_answer)
    
    # Normalize ground truth (answers are capitalized in the dataset, e.g., "Pink", "Circle")
    results_df["ground_truth_normalized"] = results_df["ground_truth"].str.lower().str.strip()
    
    # Compare normalized predictions with normalized ground truth
    results_df["is_correct"] = (
        (results_df["prediction_normalized"] == results_df["ground_truth_normalized"]) |
        (results_df["prediction_clean"] == results_df["ground_truth_normalized"])
    )
    
    # Rename confidence column for consistency (if present)
    if "confidence" in results_df.columns:
        results_df["model_confidence"] = results_df["confidence"]
    
    # Ensure required columns exist
    required_cols = ["dataset", "level_id", "image_id", "qa_id", "question", "ground_truth", 
                     "model_answer_raw", "model_confidence", "is_correct"]
    for col in required_cols:
        if col not in results_df.columns:
            results_df[col] = None
    
    # Reorder columns for consistent output
    priority_cols = ["dataset", "level_id", "image_id", "qa_id", "question"]
    if "question_type" in results_df.columns:
        priority_cols.append("question_type")
    priority_cols.extend(["ground_truth", "model_answer_raw", "model_confidence", "is_correct"])
    other_cols = [c for c in results_df.columns if c not in priority_cols]
    results_df = results_df[priority_cols + other_cols]
    
    # Save results to new folder structure
    project_root = Path(__file__).resolve().parents[1]
    model_name = MODEL_ID.split("/")[-1]
    results_path = project_root / "results_llava_hf" / model_name / results_subdir / "data_v2"
    results_path.mkdir(parents=True, exist_ok=True)
    filename = "visual_attribute_results.csv"
    output_path = results_path / filename
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path.resolve()}")
    
    # Compute accuracy by level
    if len(results_df) > 0:
        accuracy_by_level = results_df.groupby("level_id")["is_correct"].mean()
        
        # Also compute accuracy by question_type
        if "question_type" in results_df.columns:
            accuracy_by_type = results_df.groupby("question_type")["is_correct"].mean()
            print(f"\nAccuracy by question type:")
            for qtype, acc in accuracy_by_type.items():
                print(f"  {qtype}: {acc:.2%}")
        
        print(f"\nAccuracy by level:")
        for level, acc in accuracy_by_level.items():
            print(f"  {level}: {acc:.2%}")
        print(f"  Overall: {results_df['is_correct'].mean():.2%}")
        
        return accuracy_by_level
    else:
        print(f"No results found")
        return pd.Series(dtype=float)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run accuracy for vlm_levels_v2 (attribute).")
    parser.add_argument("--levels", nargs="+", default=["level_0", "level_1", "level_2", "level_3", "level_4"])
    parser.add_argument("--show-output", action="store_true", help="Show full model outputs during inference")
    parser.add_argument("--use-attention-mask", action="store_true", help="Enable attention masking")
    parser.add_argument(
        "--mask-type",
        type=str,
        default="objects_only",
        help="Masking strategy (passed to infer_model_for_levels)",
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("VISUAL ATTRIBUTE QUESTIONS ACCURACY")
    print("="*60)
    
    accuracy = compute_accuracy_for_questions(
        level_ids=args.levels,
        show_output=args.show_output,
        use_attention_mask=args.use_attention_mask,
        mask_type=args.mask_type,
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if len(accuracy) > 0:
        for level, acc in accuracy.items():
            print(f"  {level}: {acc:.2%}")

