"""
Run existential QA accuracy computation for both yes/no and attribute questions.

This script runs two separate experiments:
1. Yes/no questions (qa_existential_yesno)
2. Attribute questions (qa_existential_attribute)

Both show full model output for level_0.
"""

import sys
from pathlib import Path
import pandas as pd

# Add experiments to path
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.prompt_llava import infer_model_for_levels
from utils.plotter import Plotter

def compute_accuracy_for_questions(
    level_ids: list,
    prompt_strategy: str,
    qa_key: str,
    experiment_name: str,
    show_output: bool = True,
) -> pd.Series:
    """Run inference and compute accuracy for a specific question type."""
    print(f"\n{'='*60}")
    print(f"Running {experiment_name} experiment")
    print(f"QA Key: {qa_key}")
    print(f"Levels: {level_ids}")
    print(f"{'='*60}\n")
    
    # Run inference with full output shown
    results_list = infer_model_for_levels(
        level_ids=level_ids,
        prompt_strategy=prompt_strategy,
        show_llm_output=show_output,
        qa_key=qa_key,
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
    
    # For yes/no questions, use the prediction from get_yes_no_probability
    # For attribute questions, use the response field (full output is shown via show_llm_output)
    if qa_key == "qa_existential_yesno":
        # Yes/no questions: prediction is already extracted
        if "prediction" not in results_df.columns:
            print(f"Warning: 'prediction' column not found in results. Available columns: {results_df.columns.tolist()}")
            return pd.Series(dtype=float)
        results_df["correct"] = results_df["prediction"] == results_df["ground_truth"]
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
        
        # Normalize for comparison
        results_df["prediction_clean"] = results_df["generated_text"].str.lower().str.strip()
        
        # Handle variations in model output format
        def normalize_answer(text):
            """Normalize model answers to handle variations like 'The star is purple.' -> 'purple star'"""
            text = str(text).lower().strip()
            # Remove trailing punctuation first
            text = text.rstrip(".,!?").strip()
            
            # Handle "X is Y" format -> "Y X" (e.g., "The star is purple" -> "purple star")
            if " is " in text:
                # Remove "the" prefix if present
                text = text.replace("the ", "").strip()
                parts = text.split(" is ", 1)
                if len(parts) == 2:
                    # "star is purple" -> "purple star"
                    shape = parts[0].strip()
                    color = parts[1].strip()
                    text = f"{color} {shape}"
            
            # Remove other common prefixes
            text = text.replace("the object is a", "").replace("the object is", "").strip()
            # Remove standalone "the" at the start
            if text.startswith("the "):
                text = text[4:].strip()
            
            return text
        
        results_df["prediction_normalized"] = results_df["prediction_clean"].apply(normalize_answer)
        
        # Handle ground truth: can be a string or a list of valid answers
        def check_answer(prediction_norm, prediction_clean, ground_truth):
            """Check if prediction matches ground truth, trying both normalized and clean versions."""
            if isinstance(ground_truth, list):
                # If ground truth is a list, check if prediction matches any answer
                gt_clean_list = [str(gt).lower().strip() for gt in ground_truth]
                gt_norm_list = [normalize_answer(gt) for gt in ground_truth]
                # Try both normalized and clean versions
                return (prediction_clean in gt_clean_list) or (prediction_norm in gt_norm_list) or any(
                    pred in gt or gt in pred for pred in [prediction_clean, prediction_norm] for gt in gt_clean_list
                )
            else:
                # If ground truth is a string, try both normalized and clean
                gt_clean = str(ground_truth).lower().strip()
                gt_norm = normalize_answer(ground_truth)
                return (prediction_clean == gt_clean) or (prediction_norm == gt_norm) or (
                    prediction_clean in gt_clean or gt_clean in prediction_clean
                )
        
        results_df["correct"] = results_df.apply(
            lambda row: check_answer(row["prediction_normalized"], row["prediction_clean"], row["ground_truth"]),
            axis=1
        )
    
    # Save results
    plotter = Plotter(experiment_name="existential_qa")
    filename = f"{experiment_name}_results_{qa_key}.csv"
    results_df.to_csv(plotter.results_dir / filename, index=False)
    
    # Compute accuracy by level
    if len(results_df) > 0:
        accuracy_by_level = results_df.groupby("level_id")["correct"].mean()
        
        print(f"\n{experiment_name} Accuracy by level:")
        for level, acc in accuracy_by_level.items():
            print(f"  {level}: {acc:.2%}")
        if len(accuracy_by_level) > 0:
            print(f"  Overall: {results_df['correct'].mean():.2%}")
        
        return accuracy_by_level
    else:
        print(f"No results found for {qa_key}")
        return pd.Series(dtype=float)

if __name__ == "__main__":
    # Run for level_0 only with full output
    level_ids = ["level_4"]
    
    # Experiment 1: Yes/No questions
    print("\n" + "="*60)
    print("EXPERIMENT 1: YES/NO QUESTIONS")
    print("="*60)
    accuracy_yesno = compute_accuracy_for_questions(
        level_ids=level_ids,
        prompt_strategy="existential_yesno",
        qa_key="qa_existential_yesno",
        experiment_name="existential_yesno",
        show_output=True,
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
        show_output=True,
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

