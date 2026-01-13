from experiments.compute_accuracy import compute_accuracy_for_all_levels
from experiments.attention_analysis import aggregate_attention_and_save_to_file,  full_detailed_attention_map_for_sample
import random
import os
import json

if __name__ == "__main__":
    levels = ["level_0", "level_1", "level_2", "level_3", "level_4"]
    # levels = ["sample"]
    # compute_accuracy_for_all_levels("visual", masked=False)
    
    # aggregate_attention_and_save_to_file(levels)

    # For all levels, process 5 random samples each
    layers_and_heads_to_plot = [
                    (0, 7),
                    (0, 24),
                    (0, 30),
                    (5, 15),
                    (7, 6),
                    (12, 23),  # layer 12, head 23
                    (11, 17),
                    (14, 6),
                    (16, 25),
                    (19, 6),
                    (26, 19),
                    (29, 30), # layer 29, head 30
                    (31, 11)
                    # Add more as needed
                ]
    full_detailed_attention_map_for_sample(levels, num_samples=1, layers_and_heads_to_plot=layers_and_heads_to_plot)
    