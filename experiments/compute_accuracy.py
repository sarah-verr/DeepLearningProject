from utils.prompt_llava import infer_model_for_levels
from utils.plotter import Plotter
import pandas as pd

# Single shared Plotter instance manages the results directory
plotter = Plotter(experiment_name="compute_accuracy")

<<<<<<< HEAD
def compute_accuracy_for_all_levels(prompt_strategy: str = "visual") -> float:
    levels = ["level_0", "level_1", "level_2", "level_3", "level_4"]
    result_list = infer_model_for_levels(levels, prompt_strategy=prompt_strategy)

    # Convert to DataFrame for easier tracking
    results_df = pd.DataFrame(result_list)
=======
def compute_accuracy_for_all_levels(prompt_strategy: str = "visual", masked = False) -> float:
    levels = ["level_0", "level_1", "level_2", "level_3", "level_4"]
    result_list = infer_model_for_levels(level_ids=levels, prompt_strategy=prompt_strategy, use_plain_images=masked)

    # Convert to DataFrame for easier tracking
    results_df = pd.DataFrame(result_list)
    # Save results to CSV
    filename = None
    if masked:
        filename = "simple_results_all_levels_with_masked_images.csv"
    else:
        filename = "simple_results_all_levels.csv"

    results_df.to_csv(plotter.results_dir / filename, index=False)
>>>>>>> 5377b2f0425a36e119609f3a4180b3e1e327ba0c
    # Create a new column: whether the prediction was correct
    results_df["correct"] = results_df["prediction"] == results_df["ground_truth"]

    # Compute accuracy by level
    accuracy_by_level = results_df.groupby("level_id")["correct"].mean()

    # plots (accuracy bars + confusion matrix), no CSV output

    # Accuracy bar plot
    plotter.plot_accuracy_bars(
        accuracy_by_level,
        filename="accuracy_by_level.png",
        title="Accuracy by level",
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
        filename="confusion_matrix_all_levels.png",
        title="Confusion Matrix (yes/no) for all levels",
    )

    return accuracy_by_level