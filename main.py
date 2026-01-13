from experiments.compute_accuracy import compute_accuracy_for_all_levels
<<<<<<< HEAD
from experiments.attention_analysis import attention_distribution_between_text_and_visual
=======
from experiments.attention_analysis import aggregate_attention_and_save_to_file,  full_detailed_attention_map_for_sample
import random
import os
import json
>>>>>>> 5377b2f0425a36e119609f3a4180b3e1e327ba0c

if __name__ == "__main__":
    levels = ["level_0", "level_1", "level_2", "level_3", "level_4"]
    # levels = ["sample"]
<<<<<<< HEAD
    # compute_accuracy_for_all_levels(levels)
    attention_distribution_between_text_and_visual(levels)
    
=======
    compute_accuracy_for_all_levels("visual", masked=False)
    
    # aggregate_attention_and_save_to_file(levels)

    # For all levels, process 5 random samples each
    # full_detailed_attention_map_for_sample(levels, num_samples=3)
>>>>>>> 5377b2f0425a36e119609f3a4180b3e1e327ba0c
    