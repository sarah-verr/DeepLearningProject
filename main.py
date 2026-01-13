from experiments.compute_accuracy import compute_accuracy_for_all_levels
<<<<<<< HEAD
<<<<<<< HEAD
from experiments.attention_analysis import attention_distribution_between_text_and_visual
=======
=======
>>>>>>> 4ca68f3c539453b359f02ed566707b0af65ad950
from experiments.attention_analysis import aggregate_attention_and_save_to_file,  full_detailed_attention_map_for_sample
import random
import os
import json
<<<<<<< HEAD
>>>>>>> 5377b2f0425a36e119609f3a4180b3e1e327ba0c
=======
>>>>>>> 4ca68f3c539453b359f02ed566707b0af65ad950

if __name__ == "__main__":
    levels = ["level_0", "level_1", "level_2", "level_3", "level_4"]
    # levels = ["sample"]
<<<<<<< HEAD
<<<<<<< HEAD
    # compute_accuracy_for_all_levels(levels)
    attention_distribution_between_text_and_visual(levels)
    
=======
    compute_accuracy_for_all_levels("visual", masked=False)
=======
    # compute_accuracy_for_all_levels("visual", masked=False)
>>>>>>> 4ca68f3c539453b359f02ed566707b0af65ad950
    
    # aggregate_attention_and_save_to_file(levels)

    # For all levels, process 5 random samples each
<<<<<<< HEAD
    # full_detailed_attention_map_for_sample(levels, num_samples=3)
>>>>>>> 5377b2f0425a36e119609f3a4180b3e1e327ba0c
=======
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
>>>>>>> 4ca68f3c539453b359f02ed566707b0af65ad950
    