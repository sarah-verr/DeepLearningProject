"""
Run existential QA accuracy computation for both yes/no and attribute questions.

This script runs two separate experiments:
1. Yes/no questions (qa_existential_yesno)
2. Attribute questions (qa_existential_attribute)

Both show full model output for level_0.
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

def compute_accuracy_for_questions(
    level_ids: list,
    prompt_strategy: str,
    qa_key: str,
    experiment_name: str,
    show_output: bool = True,
    data_dir: str = "data/vlm_levels_existential_qa",
    use_attention_mask: bool = False,
    always_mask: bool = False,
    mask_type: str = "objects_only",
    results_subdir: str = "accuracy_question_ablation",
) -> pd.Series:
    """Run inference and compute accuracy for a specific question type."""
    print(f"\n{'='*60}")
    print(f"Running {experiment_name} experiment")
    print(f"QA Key: {qa_key}")
    print(f"Levels: {level_ids}")
    print(f"Data directory: {data_dir}")
    print(f"{'='*60}\n")
    
    # Run inference with optional attention masking
    results_list = infer_model_for_levels(
        level_ids=level_ids,
        prompt_strategy=prompt_strategy,
        show_llm_output=show_output,
        qa_key=qa_key,
        data_dir=data_dir,
        use_attention_mask=use_attention_mask,
        always_mask=always_mask,
        mask_type=mask_type,
    )
    
    # Convert to DataFrame
    if not results_list:
        print(f"No results returned for {qa_key}")
        return pd.Series(dtype=float)
    
    # Debug: print first result to see structure
    if results_list:
        print(f"Debug: First result keys: {results_list[0].keys()}")
        print(f"Debug: First result sample: {list(results_list[0].items())[:5]}")
    
    results_df = pd.DataFrame(results_list)
    print(f"Debug: DataFrame columns: {results_df.columns.tolist()}")
    print(f"Debug: DataFrame shape: {results_df.shape}")
    
    # Add dataset name to results
    results_df["dataset"] = "vlm_levels_existential_qa"
    
    # Extract raw model answers from response (not normalized)
    # For yes/no questions, we still want the raw text answer, not just the token prediction
    results_df["model_answer_raw"] = results_df["response"].apply(extract_raw_answer_from_response)
    
    # For yes/no questions, prediction is already extracted, but we keep raw answer separate
    # Use prediction for correctness checking (normalized comparison)
    if qa_key == "qa_existential_yesno":
        # Yes/no questions: use prediction for correctness
        if "prediction" not in results_df.columns:
            print(f"Warning: 'prediction' column not found in results. Available columns: {results_df.columns.tolist()}")
            return pd.Series(dtype=float)
        # For correctness, compare prediction (yes/no) with ground truth
        results_df["is_correct"] = results_df["prediction"] == results_df["ground_truth"]
        # Keep model_answer_raw as the raw extracted text for analysis
        # If model_answer_raw is empty/None, fallback to prediction
        results_df.loc[results_df["model_answer_raw"].isna() | (results_df["model_answer_raw"] == ""), "model_answer_raw"] = results_df["prediction"]
    else:
        # Attribute questions: extract just the generated part from response
        # The response contains the full prompt + answer, we need to extract just the answer
        def extract_generated_text(full_response, question):
            """Extract just the generated answer from the full response."""
            # The response format is: prompt + "QUESTION: {question}\n ASSISTANT: {answer}"
            # Try to find the ASSISTANT marker (most reliable)
            if "ASSISTANT:" in full_response:
                parts = full_response.split("ASSISTANT:", 1)
                if len(parts) > 1:
                    generated = parts[-1].strip()
                    # Remove any trailing special tokens or formatting
                    generated = generated.replace("</s>", "").strip()
                    # Remove trailing newlines and whitespace
                    generated = generated.split("\n")[0].strip()
                    return generated
            
            # Fallback: try to find "QUESTION:" and extract what comes after ASSISTANT
            if "QUESTION:" in full_response and "ASSISTANT:" in full_response:
                # Find the last ASSISTANT: marker
                assitant_idx = full_response.rfind("ASSISTANT:")
                if assitant_idx != -1:
                    generated = full_response[assitant_idx + len("ASSISTANT:"):].strip()
                    generated = generated.split("\n")[0].strip()
                    return generated
            
            # Last resort: return the full response
            return full_response.strip()
        
        results_df["generated_text"] = results_df.apply(
            lambda row: extract_generated_text(row["response"], row["question"]),
            axis=1
        )
        
        # Normalize for comparison - all answers are one-word, so just basic cleaning
        def normalize_answer(text):
            """Normalize model answers: lowercase, strip, remove trailing punctuation."""
            text = str(text).lower().strip()
            text = text.rstrip(".,!?").strip()
            return text
        
        results_df["prediction_clean"] = results_df["generated_text"].apply(normalize_answer)
        
        # Handle ground truth: can be a string or a list of valid answers
        def check_answer(prediction_clean, ground_truth):
            """Check if prediction matches ground truth (both are normalized)."""
            if isinstance(ground_truth, list):
                # If ground truth is a list, check if prediction matches any answer
                gt_clean_list = [normalize_answer(str(gt)) for gt in ground_truth]
                return prediction_clean in gt_clean_list
            else:
                # If ground truth is a string, normalize and compare
                gt_clean = normalize_answer(str(ground_truth))
                return prediction_clean == gt_clean
        
        results_df["is_correct"] = results_df.apply(
            lambda row: check_answer(row["prediction_clean"], row["ground_truth"]),
            axis=1
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
    # Determine dataset folder name based on qa_key
    if qa_key == "qa_existential_yesno":
        dataset_folder = "existential_yesno"
    elif qa_key == "qa_existential_attribute":
        dataset_folder = "existential_attribute"
    else:
        raise ValueError(f"Unknown qa_key: {qa_key}. Expected 'qa_existential_yesno' or 'qa_existential_attribute'")
    results_path = project_root / "results_llava_hf" / model_name / results_subdir / dataset_folder
    results_path.mkdir(parents=True, exist_ok=True)
    filename = f"{experiment_name}_results_{qa_key}.csv"
    output_path = results_path / filename
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path.resolve()}")
    
    # Compute accuracy by level
    if len(results_df) > 0:
        accuracy_by_level = results_df.groupby("level_id")["is_correct"].mean()
        
        print(f"\n{experiment_name} Accuracy by level:")
        for level, acc in accuracy_by_level.items():
            print(f"  {level}: {acc:.2%}")
        if len(accuracy_by_level) > 0:
            print(f"  Overall: {results_df['is_correct'].mean():.2%}")
        
        return accuracy_by_level
    else:
        print(f"No results found for {qa_key}")
        return pd.Series(dtype=float)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run existential QA accuracy (yes/no + attribute).")
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

    # Run for all levels
    level_ids = args.levels
    
    # Experiment 1: Yes/No questions
    print("\n" + "="*60)
    print("EXPERIMENT 1: YES/NO QUESTIONS")
    print("="*60)
    accuracy_yesno = compute_accuracy_for_questions(
        level_ids=level_ids,
        prompt_strategy="existential_yesno",
        qa_key="qa_existential_yesno",
        experiment_name="existential_yesno",
        show_output=args.show_output,  # optional verbose
        use_attention_mask=args.use_attention_mask,
        mask_type=args.mask_type,
    )
    
    # Experiment 2: Attribute questions
    print("\n" + "="*60)
    print("EXPERIMENT 2: ATTRIBUTE QUESTIONS")
    print("="*60)
    accuracy_attribute = compute_accuracy_for_questions(
        level_ids=level_ids,
        prompt_strategy="existential_attribute",
        qa_key="qa_existential_attribute",
        experiment_name="existential_attribute",
        show_output=args.show_output,  # optional verbose
        use_attention_mask=args.use_attention_mask,
        mask_type=args.mask_type,
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nYes/No Questions:")
    if len(accuracy_yesno) > 0:
        for level, acc in accuracy_yesno.items():
            print(f"  {level}: {acc:.2%}")
    print("\nAttribute Questions:")
    if len(accuracy_attribute) > 0:
        for level, acc in accuracy_attribute.items():
            print(f"  {level}: {acc:.2%}")

