"""
Run accuracy computation for vlm_levels_v3 dataset with relational position questions.

This script runs inference on the vlm_levels_v3 dataset using the "visual_relational" prompt strategy.
It compares model outputs with ground truth answers (spatial relations).
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
            # Remove trailing punctuation
            generated = generated.rstrip(".,!?").strip()
            return generated
    
    # Fallback: try to find "QUESTION:" and extract what comes after ASSISTANT
    if "QUESTION:" in response and "ASSISTANT:" in response:
        # Find the last ASSISTANT: marker
        assistant_idx = response.rfind("ASSISTANT:")
        if assistant_idx != -1:
            generated = response[assistant_idx + len("ASSISTANT:"):].strip()
            generated = generated.split("\n")[0].strip()
            generated = generated.rstrip(".,!?").strip()
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
    # Handle variations: "top left" -> "top-left", "bottom right" -> "bottom-right", etc.
    text = text.replace("top left", "top-left").replace("top-left", "top-left")
    text = text.replace("top right", "top-right").replace("top-right", "top-right")
    text = text.replace("bottom left", "bottom-left").replace("bottom-left", "bottom-left")
    text = text.replace("bottom right", "bottom-right").replace("bottom-right", "bottom-right")
    return text


def compute_accuracy_for_questions(
    level_ids: list,
    show_output: bool = True,
) -> pd.Series:
    """Run inference and compute accuracy for relational position questions."""
    print(f"\n{'='*60}")
    print(f"Running visual_relational accuracy experiment")
    print(f"QA Key: qa")
    print(f"Levels: {level_ids}")
    print(f"{'='*60}\n")
    
    # Run inference with full output shown
    results_list = infer_model_for_levels(
        level_ids=level_ids,
        prompt_strategy="visual_relational",
        show_llm_output=show_output,
        qa_key="qa",
        data_dir="data/vlm_levels_v3",
    )
    
    # Convert to DataFrame
    if not results_list:
        print(f"No results returned")
        return pd.Series(dtype=float)
    
    results_df = pd.DataFrame(results_list)
    print(f"Debug: DataFrame columns: {results_df.columns.tolist()}")
    print(f"Debug: DataFrame shape: {results_df.shape}")
    
    # Extract answers from model responses
    results_df["generated_text"] = results_df["response"].apply(extract_answer_from_response)
    results_df["prediction_clean"] = results_df["generated_text"].str.lower().str.strip()
    results_df["prediction_normalized"] = results_df["prediction_clean"].apply(normalize_answer)
    
    # Normalize ground truth (answers are lowercase in the dataset, e.g., "above", "below", "top-left")
    results_df["ground_truth_normalized"] = results_df["ground_truth"].str.lower().str.strip()
    
    # Compare normalized predictions with normalized ground truth
    results_df["correct"] = (
        (results_df["prediction_normalized"] == results_df["ground_truth_normalized"]) |
        (results_df["prediction_clean"] == results_df["ground_truth_normalized"])
    )
    
    # Save results
    plotter = Plotter(experiment_name="data_v3")
    filename = "visual_relational_results.csv"
    output_path = plotter.results_dir / filename
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path.resolve()}")
    
    # Compute accuracy by level
    if len(results_df) > 0:
        accuracy_by_level = results_df.groupby("level_id")["correct"].mean()
        
        # Also compute accuracy by question_type
        if "question_type" in results_df.columns:
            accuracy_by_type = results_df.groupby("question_type")["correct"].mean()
            print(f"\nAccuracy by question type:")
            for qtype, acc in accuracy_by_type.items():
                print(f"  {qtype}: {acc:.2%}")
        
        print(f"\nAccuracy by level:")
        for level, acc in accuracy_by_level.items():
            print(f"  {level}: {acc:.2%}")
        print(f"  Overall: {results_df['correct'].mean():.2%}")
        
        return accuracy_by_level
    else:
        print(f"No results found")
        return pd.Series(dtype=float)


if __name__ == "__main__":
    # Run for level_0 only with full output
    level_ids = ["level_4"]
    
    print("\n" + "="*60)
    print("RELATIONAL POSITION QUESTIONS ACCURACY")
    print("="*60)
    
    accuracy = compute_accuracy_for_questions(
        level_ids=level_ids,
        show_output=True,
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if len(accuracy) > 0:
        for level, acc in accuracy.items():
            print(f"  {level}: {acc:.2%}")
