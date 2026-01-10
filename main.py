from experiments.compute_accuracy import compute_accuracy_for_all_levels
from experiments.attention_analysis import attention_distribution_between_text_and_visual

if __name__ == "__main__":
    levels = ["level_0", "level_1", "level_2", "level_3", "level_4"]
    # levels = ["sample"]
    # compute_accuracy_for_all_levels(levels)
    attention_distribution_between_text_and_visual(levels)
    
    