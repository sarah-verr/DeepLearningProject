from experiments.compute_accuracy import compute_accuracy_for_all_levels
from experiments.attention_analysis import aggregate_attention_and_save_to_file, visualise_full_attention

if __name__ == "__main__":
    levels = ["level_0", "level_1", "level_2", "level_3", "level_4"]
    # levels = ["sample"]
    # compute_accuracy_for_all_levels()
    
    # aggregate_attention_and_save_to_file(levels)

    level_id = "level_0"
    image_id = "00001_b"
    qa_id = 1
    visualise_full_attention(level_id, image_id, qa_id)
    
    