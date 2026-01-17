"""
Run accuracy experiments with attention masking for all datasets:
1. vlm_levels (yes/no questions) - mask when relation exists
2. vlm_levels (yes/no questions) - always mask (even when relation doesn't exist)
3. vlm_levels_v2 (attribute questions) - always mask (relations always exist)
4. vlm_levels_v3 (relational questions) - always mask (relations always exist)

This script runs all masked experiments and saves results in a unified CSV format.
"""

import sys
from pathlib import Path
import pandas as pd
import argparse

# Add experiments to path
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import experiment functions
from experiments.data.run_accuracy_w_attention_mask import compute_accuracy_for_questions as compute_data_mask_accuracy
from experiments.data.run_accuracy_w_attention_mask_for_existing_relation import compute_accuracy_for_questions as compute_data_mask_always_accuracy
from experiments.data_v2.run_accuracy_w_attention_mask import compute_accuracy_for_questions as compute_v2_mask_accuracy
from experiments.data_v3.run_accuracy_w_attention_mask import compute_accuracy_for_questions as compute_v3_mask_accuracy


def run_all_masked_experiments(level_ids: list = None, show_output: bool = False):
    """Run accuracy experiments with attention masking for all datasets.
    
    Args:
        level_ids: List of level IDs to run (default: all levels)
        show_output: Whether to show model outputs during inference
    """
    if level_ids is None:
        level_ids = ["level_0", "level_1", "level_2", "level_3", "level_4"]
    
    print("\n" + "="*80)
    print("RUNNING ALL MASKED ACCURACY EXPERIMENTS")
    print("="*80)
    print(f"Levels: {level_ids}")
    print(f"Show output: {show_output}")
    print("="*80)
    
    results_summary = {}
    
    # Experiment 1: vlm_levels - Yes/No questions with masking (only when relation exists)
    print("\n" + "="*80)
    print("EXPERIMENT 1: vlm_levels - Yes/No Questions WITH MASKING (when relation exists)")
    print("="*80)
    try:
        accuracy_data_mask = compute_data_mask_accuracy(
            level_ids=level_ids,
            show_output=show_output,
        )
        results_summary["vlm_levels_mask"] = accuracy_data_mask
    except Exception as e:
        print(f"Error in masked yes/no experiment: {e}")
        import traceback
        traceback.print_exc()
        results_summary["vlm_levels_mask"] = None
    
    # Experiment 2: vlm_levels - Yes/No questions with masking (always, even when relation doesn't exist)
    print("\n" + "="*80)
    print("EXPERIMENT 2: vlm_levels - Yes/No Questions WITH MASKING (always, even when relation doesn't exist)")
    print("="*80)
    try:
        accuracy_data_mask_always = compute_data_mask_always_accuracy(
            level_ids=level_ids,
            show_output=show_output,
        )
        results_summary["vlm_levels_mask_always"] = accuracy_data_mask_always
    except Exception as e:
        print(f"Error in always-masked yes/no experiment: {e}")
        import traceback
        traceback.print_exc()
        results_summary["vlm_levels_mask_always"] = None
    
    # Experiment 3: vlm_levels_v2 - Attribute questions with masking
    print("\n" + "="*80)
    print("EXPERIMENT 3: vlm_levels_v2 - Attribute Questions WITH MASKING")
    print("="*80)
    try:
        accuracy_v2_mask = compute_v2_mask_accuracy(
            level_ids=level_ids,
            show_output=show_output,
        )
        results_summary["vlm_levels_v2_mask"] = accuracy_v2_mask
    except Exception as e:
        print(f"Error in masked v2 experiment: {e}")
        import traceback
        traceback.print_exc()
        results_summary["vlm_levels_v2_mask"] = None
    
    # Experiment 4: vlm_levels_v3 - Relational questions with masking
    print("\n" + "="*80)
    print("EXPERIMENT 4: vlm_levels_v3 - Relational Questions WITH MASKING")
    print("="*80)
    try:
        accuracy_v3_mask = compute_v3_mask_accuracy(
            level_ids=level_ids,
            show_output=show_output,
        )
        results_summary["vlm_levels_v3_mask"] = accuracy_v3_mask
    except Exception as e:
        print(f"Error in masked v3 experiment: {e}")
        import traceback
        traceback.print_exc()
        results_summary["vlm_levels_v3_mask"] = None
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL MASKED EXPERIMENTS")
    print("="*80)
    for exp_name, accuracy in results_summary.items():
        if accuracy is not None and len(accuracy) > 0:
            print(f"\n{exp_name}:")
            overall = accuracy.mean() if len(accuracy) > 0 else 0.0
            print(f"  Overall: {overall:.2%}")
            for level, acc in accuracy.items():
                print(f"  {level}: {acc:.2%}")
        else:
            print(f"\n{exp_name}: Failed or no results")
    
    print("\n" + "="*80)
    print("ALL MASKED EXPERIMENTS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run masked accuracy experiments for all datasets")
    parser.add_argument("--levels", nargs="+", default=None, 
                        help="Level IDs to run (e.g., level_0 level_1). Default: all levels")
    parser.add_argument("--show-output", action="store_true", 
                        help="Show full model outputs during inference")
    
    args = parser.parse_args()
    
    level_ids = args.levels if args.levels else None
    
    run_all_masked_experiments(level_ids=level_ids, show_output=args.show_output)
