from utils.prompt_llava import infer_model_for_levels
from utils.plotter import Plotter
import pandas as pd

# Single shared Plotter instance manages the results directory
plotter = Plotter(experiment_name="compute_accuracy")

def compute_accuracy_for_all_levels(
    prompt_strategy: str = "visual",
    masked: bool = False,
    qa_key: str = "qa",
    experiment_name: str = None,
    show_llm_output: bool = False,
) -> float:
    """
    Compute accuracy for all levels.
    
    Args:
        prompt_strategy: Prompt strategy ("visual", "caption", "scene", "existential")
        masked: Whether to use plain black/white images
        qa_key: Key in annotation JSON for QA pairs ("qa" or "qa_existential")
        experiment_name: Optional experiment name for output files (defaults to prompt_strategy)
    """
    levels = ["level_0", "level_1", "level_2", "level_3", "level_4"]
    result_list = infer_model_for_levels(
        level_ids=levels,
        prompt_strategy=prompt_strategy,
        use_plain_images=masked,
        qa_key=qa_key,
        show_llm_output=show_llm_output,
    )

    # Convert to DataFrame for easier tracking
    results_df = pd.DataFrame(result_list)
    
    # Save results to CSV
    exp_name = experiment_name or prompt_strategy
    if qa_key != "qa":
        exp_name = f"{exp_name}_{qa_key}"
    
    filename = None
    if masked:
        filename = f"{exp_name}_results_all_levels_with_masked_images.csv"
    else:
        filename = f"{exp_name}_results_all_levels.csv"

    results_df.to_csv(plotter.results_dir / filename, index=False)

    # Create a new column: whether the prediction was correct
    results_df["correct"] = results_df["prediction"] == results_df["ground_truth"]

    # Compute accuracy by level
    accuracy_by_level = results_df.groupby("level_id")["correct"].mean()

    # Accuracy bar plot
    plotter.plot_accuracy_bars(
        accuracy_by_level,
        filename=f"{exp_name}_accuracy_by_level.png",
        title=f"Accuracy by level ({exp_name})",
    )

    # Confusion matrix over all levels
    cm_all = pd.crosstab(
        results_df["ground_truth"],
        results_df["prediction"],
        rownames=["ground_truth"],
        colnames=["prediction"],
        dropna=False,
    ).reindex(index=["yes", "no"], columns=["yes", "no"], fill_value=0)

    plotter.plot_confusion_matrix(
        cm_all,
        filename=f"{exp_name}_confusion_matrix_all_levels.png",
        title=f"Confusion Matrix (yes/no) for all levels ({exp_name})",
    )

    return accuracy_by_level