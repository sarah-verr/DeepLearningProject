"""
Run accuracy computation for vlm_levels dataset (original, without versioning).

This script runs inference on the vlm_levels dataset using the "visual" prompt strategy.
It saves results in a unified format matching other accuracy scripts.
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
from utils.results_processing import extract_raw_answer_from_response


def compute_accuracy_for_questions(
    level_ids: list,
    show_output: bool = True,
    use_attention_mask: bool = False,
    always_mask: bool = False,
    mask_type: str = "objects_only",
    results_subdir: str = "accuracy_question_ablation",
) -> pd.Series:
    """Run inference and compute accuracy for visual yes/no questions."""
    print(f"\n{'='*60}")
    print(f"Running visual yes/no accuracy experiment")
    print(f"QA Key: qa")
    print(f"Levels: {level_ids}")
    print(f"{'='*60}\n")
    
    # Run inference with optional attention masking
    results_list = infer_model_for_levels(
        level_ids=level_ids,
        prompt_strategy="visual",
        show_llm_output=show_output,
        qa_key="qa",
        data_dir="data/vlm_levels",
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
    results_df["dataset"] = "vlm_levels"
    
    # Extract raw model answers from response (not normalized for storage)
    results_df["model_answer_raw"] = results_df["response"].apply(extract_raw_answer_from_response)
    
    # For yes/no questions, use prediction for correctness checking
    if "prediction" in results_df.columns and results_df["prediction"].notna().any():
        # For correctness, compare prediction (yes/no) with ground truth
        results_df["is_correct"] = results_df["prediction"] == results_df["ground_truth"]
        # If model_answer_raw is empty/None, fallback to prediction
        results_df.loc[results_df["model_answer_raw"].isna() | (results_df["model_answer_raw"] == ""), "model_answer_raw"] = results_df["prediction"]
    else:
        # Fallback: simple string comparison
        results_df["prediction_clean"] = results_df["model_answer_raw"].str.lower().str.strip()
        results_df["ground_truth_normalized"] = results_df["ground_truth"].str.lower().str.strip()
        results_df["is_correct"] = results_df["prediction_clean"] == results_df["ground_truth_normalized"]
    
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
    results_path = project_root / "results_llava_hf" / model_name / results_subdir / "data"
    results_path.mkdir(parents=True, exist_ok=True)
    filename = "visual_yesno_results.csv"
    output_path = results_path / filename
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path.resolve()}")
    
    # Compute accuracy by level
    if len(results_df) > 0:
        accuracy_by_level = results_df.groupby("level_id")["is_correct"].mean()
        
        # Also compute accuracy by question_type if present
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
    parser = argparse.ArgumentParser(description="Run accuracy for vlm_levels (yes/no).")
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
    print("ORIGINAL DATASET (vlm_levels) ACCURACY")
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
