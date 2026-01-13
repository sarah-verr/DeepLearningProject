from experiments.compute_accuracy import compute_accuracy_for_all_levels
from experiments.attention_analysis import aggregate_attention_and_save_to_file,  full_detailed_attention_map_for_sample
import random
import os
import json

if __name__ == "__main__":
    levels = ["level_0", "level_1", "level_2", "level_3", "level_4"]
    # levels = ["sample"]
    # compute_accuracy_for_all_levels()
    
    # aggregate_attention_and_save_to_file(levels)

    # For all levels, process 5 random samples each
    full_detailed_attention_map_for_sample(levels, num_samples=3)
    