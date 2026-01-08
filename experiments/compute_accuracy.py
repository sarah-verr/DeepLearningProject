import os
from utils.prompt_llava import infer_model_for_levels
import pandas as pd

def compute_accuracy_by_level(level_id: str, prompt_strategy: str = "visual") -> float:
    # locate all annotation files for this level

    total = 0
    correct = 0

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
    # locate all annotation files for this level

    total = 0
    correct = 0
    levels = ["level_0", "level_1", "level_2", "level_3", "level_4"]
    result_list = infer_model_for_levels(levels)
    # Convert to DataFrame for easier tracking
    results_df = pd.DataFrame(result_list)
    # Create a new column: whether the prediction was correct
    results_df['correct'] = results_df['prediction'] == results_df['ground_truth']

    # Compute accuracy per level
    accuracy_by_level = results_df.groupby('level_id')['correct'].mean()
    
    return accuracy_by_level