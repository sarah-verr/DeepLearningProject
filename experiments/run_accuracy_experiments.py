"""
Run accuracy experiments for all three datasets:
1. vlm_levels (existential yes/no and attribute questions)
2. vlm_levels_v2 (visual attribute questions)
3. vlm_levels_v3 (visual relational questions)

This script runs all experiments and saves results in a unified CSV format.
"""

import sys
from pathlib import Path
import pandas as pd

# Add experiments to path
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import experiment functions
from experiments.data.run_accuracy import compute_accuracy_for_questions as compute_data_accuracy
from experiments.existential_qa.run_accuracy import compute_accuracy_for_questions as compute_existential_accuracy
from experiments.data_v2.run_accuracy import compute_accuracy_for_questions as compute_v2_accuracy
from experiments.data_v3.run_accuracy import compute_accuracy_for_questions as compute_v3_accuracy


def run_all_experiments(level_ids: list = None, show_output: bool = False):
    """Run accuracy experiments for all three datasets.
    
    Args:
        level_ids: List of level IDs to run (default: all levels)
        show_output: Whether to show model outputs during inference
    """
    if level_ids is None:
        level_ids = ["level_0", "level_1", "level_2", "level_3", "level_4"]
    
    print("\n" + "="*80)
    print("RUNNING ALL ACCURACY EXPERIMENTS")
    print("="*80)
    print(f"Levels: {level_ids}")
    print(f"Show output: {show_output}")
    print("="*80)
    
    results_summary = {}
    
    # Experiment 0: vlm_levels (original dataset) - Yes/No questions
    print("\n" + "="*80)
    print("EXPERIMENT 0: vlm_levels (original) - Yes/No Questions")
    print("="*80)
    try:
        accuracy_data = compute_data_accuracy(
            level_ids=level_ids,
            show_output=show_output,
        )
        results_summary["vlm_levels"] = accuracy_data
    except Exception as e:
        print(f"Error in original dataset experiment: {e}")
        import traceback
        traceback.print_exc()
        results_summary["vlm_levels"] = None
    
    # Experiment 1: vlm_levels - Yes/No questions (existential)
    print("\n" + "="*80)
    print("EXPERIMENT 1: vlm_levels - Existential Yes/No Questions")
    print("="*80)
    try:
        accuracy_yesno = compute_existential_accuracy(
            level_ids=level_ids,
            prompt_strategy="existential_yesno",
            qa_key="qa_existential_yesno",
            experiment_name="existential_yesno",
            show_output=show_output,
        )
        results_summary["vlm_levels_yesno"] = accuracy_yesno
    except Exception as e:
        print(f"Error in yes/no experiment: {e}")
        results_summary["vlm_levels_yesno"] = None
    
    # Experiment 2: vlm_levels - Attribute questions
    print("\n" + "="*80)
    print("EXPERIMENT 2: vlm_levels - Existential Attribute Questions")
    print("="*80)
    try:
        accuracy_attribute = compute_existential_accuracy(
            level_ids=level_ids,
            prompt_strategy="existential_attribute",
            qa_key="qa_existential_attribute",
            experiment_name="existential_attribute",
            show_output=show_output,
        )
        results_summary["vlm_levels_attribute"] = accuracy_attribute
    except Exception as e:
        print(f"Error in attribute experiment: {e}")
        results_summary["vlm_levels_attribute"] = None
    
    # Experiment 3: vlm_levels_v2 - Visual Attribute questions
    print("\n" + "="*80)
    print("EXPERIMENT 3: vlm_levels_v2 - Visual Attribute Questions")
    print("="*80)
    try:
        accuracy_v2 = compute_v2_accuracy(
            level_ids=level_ids,
            show_output=show_output,
        )
        results_summary["vlm_levels_v2"] = accuracy_v2
    except Exception as e:
        print(f"Error in v2 experiment: {e}")
        results_summary["vlm_levels_v2"] = None
    
    # Experiment 4: vlm_levels_v3 - Visual Relational questions
    print("\n" + "="*80)
    print("EXPERIMENT 4: vlm_levels_v3 - Visual Relational Questions")
    print("="*80)
    try:
        accuracy_v3 = compute_v3_accuracy(
            level_ids=level_ids,
            show_output=show_output,
        )
        results_summary["vlm_levels_v3"] = accuracy_v3
    except Exception as e:
        print(f"Error in v3 experiment: {e}")
        results_summary["vlm_levels_v3"] = None
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for exp_name, accuracy in results_summary.items():
        print(f"\n{exp_name}:")
        if accuracy is not None and len(accuracy) > 0:
            for level, acc in accuracy.items():
                print(f"  {level}: {acc:.2%}")
        else:
            print("  No results")
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)
    print("\nResults are saved in: results_llava_hf/{model_name}/accuracy_question_ablation/")
    print("  - data/ (original vlm_levels)")
    print("  - data_v2/ (vlm_levels_v2)")
    print("  - data_v3/ (vlm_levels_v3)")
    print("  - existential_yesno/ (existential yes/no questions)")
    print("  - existential_attribute/ (existential attribute questions)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run accuracy experiments for all datasets")
    parser.add_argument("--levels", type=str, nargs="+", 
                       default=["level_0", "level_1", "level_2", "level_3", "level_4"],
                       help="Level IDs to run (default: all levels)")
    parser.add_argument("--show-output", action="store_true",
                       help="Show model outputs during inference")
    
    args = parser.parse_args()
    
    run_all_experiments(
        level_ids=args.levels,
        show_output=args.show_output,
    )
