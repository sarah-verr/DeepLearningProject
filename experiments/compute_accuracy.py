import os
from utils.prompt_llava import infer_model_for_levels
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_accuracy_by_level(level_id: str, prompt_strategy: str = "visual") -> float:

    result_list = infer_model_for_levels([level_id])
    # Convert to DataFrame for easier tracking
    results_df = pd.DataFrame(result_list)
    level_df = results_df[results_df['level_id'] == level_id]
    # Create a new column: whether the prediction was correct
    level_df['correct'] = results_df['prediction'] == results_df['ground_truth']

    # Compute accuracy per level
    accuracy_for_level = level_df.groupby('level_id')['correct'].mean()
    
    return accuracy_for_level

def compute_accuracy_for_all_levels(prompt_strategy: str = "visual") -> float:

    levels = ["level_0", "level_1", "level_2", "level_3", "level_4"]
    result_list = infer_model_for_levels(levels)
    # Convert to DataFrame for easier tracking
    results_df = pd.DataFrame(result_list)
    # Create a new column: whether the prediction was correct
    results_df['correct'] = results_df['prediction'] == results_df['ground_truth']

    # Compute accuracy by level
    accuracy_by_level = results_df.groupby('level_id')['correct'].mean()
    
    # write full detailed results
    all_results_path = os.path.join(RESULTS_DIR, "results_all_levels.csv")
    results_df.to_csv(all_results_path, index=False)

    # write per-level accuracies
    acc_by_level_path = os.path.join(RESULTS_DIR, "accuracy_by_level.csv")
    accuracy_by_level.to_frame("accuracy").reset_index().to_csv(
        acc_by_level_path, index=False
    )

    return accuracy_by_level