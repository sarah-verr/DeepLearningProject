from experiments.compute_accuracy import compute_accuracy_for_all_levels
from experiments.attention_analysis import aggregate_attention_and_save_to_file, full_detailed_attention_map_for_sample, average_attention_over_layer_groups

import random
import os
import json

if __name__ == "__main__":
    print("[DEBUG] Running main.py ...")
    levels = ["level_2"]
    # levels = ["sample"]

    # compute_accuracy_for_all_levels("visual", masked=False)
    # aggregate_attention_and_save_to_file(levels, "attention_results_masked.jsonl", use_masked_attention=True, always_mask=True)

    # layers_and_heads_to_plot = [
    #                 (0, 7),
    #                 (0, 24),
    #                 (0, 30),
    #                 (5, 15),
    #                 (7, 6),
    #                 (11, 17),
    #                 (12, 23),  # layer 12, head 23
    #                 (14, 6),
    #                 (15, 10),
    #                 (15, 14),
    #                 (16, 3),
    #                 (16, 25),
    #                 (19, 6),
    #                 (26, 19),
    #                 (29, 15),
    #                 (29, 30), # layer 29, head 30
    #                 (31, 11)
    #                 # Add more as needed
    #             ]

    layers_and_heads_to_plot = []
    
    # Add all heads for layers 5, 6, and 7
    for layer in [0, 1, 9, 10, 11, 12, 13, 14, 30, 31]:
        for head in range(32):
            if (layer, head) not in layers_and_heads_to_plot:
                layers_and_heads_to_plot.append((layer, head))


    full_detailed_attention_map_for_sample(levels, num_samples=3, layers_and_heads_to_plot=layers_and_heads_to_plot, use_relational_phrase=True, use_masked_attention = True, always_mask = True)

    # layer_groups = [[0,1], [8,9,10,11,12,13,14,15,16, 17], [30,31]]
    # average_attention_over_layer_groups(levels, layer_groups,num_samples=2,use_relational_phrase=True)
    