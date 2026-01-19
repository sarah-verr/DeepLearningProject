"""
Run masked accuracy experiments for all datasets (mirrors run_all_accuracy_experiments.py).

Experiments:
0. vlm_levels (original) - Yes/No questions (qa)
1. vlm_levels_existential_qa - Existential Yes/No questions (qa_existential_yesno)
2. vlm_levels_existential_qa - Existential Attribute questions (qa_existential_attribute)
3. vlm_levels_v2 - Visual Attribute questions (qa)
4. vlm_levels_v3 - Visual Relational questions (qa)

All experiments run with attention masking enabled and save results to:
  results_llava_hf/{model_name}/masked_accuracy_ablation/
"""

import sys
from pathlib import Path
import argparse

# Add experiments to path
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import experiment functions
from experiments.data.run_accuracy import compute_accuracy_for_questions as compute_data_accuracy
from experiments.existential_qa.run_accuracy import compute_accuracy_for_questions as compute_existential_accuracy
from experiments.data_v2.run_accuracy import compute_accuracy_for_questions as compute_v2_accuracy
from experiments.data_v3.run_accuracy import compute_accuracy_for_questions as compute_v3_accuracy
from utils.prompt_llava import MODEL_ID


# Default masking configuration (easy to tweak for all experiments)
DEFAULT_USE_ATTENTION_MASK = True
DEFAULT_ALWAYS_MASK = True  # Always apply masking, even when relations don't exist
DEFAULT_MASK_TYPE = "opposite_side"
DEFAULT_RESULTS_SUBDIR = "always_masked_accuracy_ablation"


def run_all_masked_experiments(
    level_ids: list | None = None,
    show_output: bool = False,
    use_attention_mask: bool = DEFAULT_USE_ATTENTION_MASK,
    always_mask: bool = DEFAULT_ALWAYS_MASK,
    mask_type: str = DEFAULT_MASK_TYPE,
    results_subdir: str = DEFAULT_RESULTS_SUBDIR,
):
    """Run masked accuracy experiments for all datasets."""
    if level_ids is None:
        level_ids = ["level_0", "level_1", "level_2", "level_3", "level_4"]

    model_name = MODEL_ID.split("/")[-1]

    print("\n" + "=" * 80)
    print("RUNNING ALL MASKED ACCURACY EXPERIMENTS")
    print("=" * 80)
    print(f"Levels: {level_ids}")
    print(f"Show output: {show_output}")
    print(f"Use attention mask: {use_attention_mask}")
    print(f"Always mask: {always_mask}")
    print(f"Mask type: {mask_type}")
    print(f"Results subdir: {results_subdir}")
    print("=" * 80)

    results_summary = {}

    # Experiment 0: vlm_levels (original dataset) - Yes/No questions
    print("\n" + "=" * 80)
    print("EXPERIMENT 0 (MASKED): vlm_levels (original) - Yes/No Questions")
    print("=" * 80)
    try:
        accuracy_data = compute_data_accuracy(
            level_ids=level_ids,
            show_output=show_output,
            use_attention_mask=use_attention_mask,
            always_mask=always_mask,
            mask_type=mask_type,
            results_subdir=results_subdir,
        )
        results_summary["vlm_levels"] = accuracy_data
    except Exception as e:
        print(f"Error in original dataset masked experiment: {e}")
        import traceback

        traceback.print_exc()
        results_summary["vlm_levels"] = None

    # Experiment 1: existential yes/no
    print("\n" + "=" * 80)
    print("EXPERIMENT 1 (MASKED): vlm_levels - Existential Yes/No Questions")
    print("=" * 80)
    try:
        accuracy_yesno = compute_existential_accuracy(
            level_ids=level_ids,
            prompt_strategy="existential_yesno",
            qa_key="qa_existential_yesno",
            experiment_name="existential_yesno",
            show_output=show_output,
            use_attention_mask=use_attention_mask,
            always_mask=always_mask,
            mask_type=mask_type,
            results_subdir=results_subdir,
        )
        results_summary["vlm_levels_yesno"] = accuracy_yesno
    except Exception as e:
        print(f"Error in existential yes/no masked experiment: {e}")
        import traceback

        traceback.print_exc()
        results_summary["vlm_levels_yesno"] = None

    # Experiment 2: existential attribute
    print("\n" + "=" * 80)
    print("EXPERIMENT 2 (MASKED): vlm_levels - Existential Attribute Questions")
    print("=" * 80)
    try:
        accuracy_attribute = compute_existential_accuracy(
            level_ids=level_ids,
            prompt_strategy="existential_attribute",
            qa_key="qa_existential_attribute",
            experiment_name="existential_attribute",
            show_output=show_output,
            use_attention_mask=use_attention_mask,
            always_mask=always_mask,
            mask_type=mask_type,
            results_subdir=results_subdir,
        )
        results_summary["vlm_levels_attribute"] = accuracy_attribute
    except Exception as e:
        print(f"Error in existential attribute masked experiment: {e}")
        import traceback

        traceback.print_exc()
        results_summary["vlm_levels_attribute"] = None

    # Experiment 3: vlm_levels_v2 - visual attribute
    print("\n" + "=" * 80)
    print("EXPERIMENT 3 (MASKED): vlm_levels_v2 - Visual Attribute Questions")
    print("=" * 80)
    try:
        accuracy_v2 = compute_v2_accuracy(
            level_ids=level_ids,
            show_output=show_output,
            use_attention_mask=use_attention_mask,
            always_mask=always_mask,
            mask_type=mask_type,
            results_subdir=results_subdir,
        )
        results_summary["vlm_levels_v2"] = accuracy_v2
    except Exception as e:
        print(f"Error in v2 masked experiment: {e}")
        import traceback

        traceback.print_exc()
        results_summary["vlm_levels_v2"] = None

    # Experiment 4: vlm_levels_v3 - visual relational
    print("\n" + "=" * 80)
    print("EXPERIMENT 4 (MASKED): vlm_levels_v3 - Visual Relational Questions")
    print("=" * 80)
    try:
        accuracy_v3 = compute_v3_accuracy(
            level_ids=level_ids,
            show_output=show_output,
            use_attention_mask=use_attention_mask,
            always_mask=always_mask,
            mask_type=mask_type,
            results_subdir=results_subdir,
        )
        results_summary["vlm_levels_v3"] = accuracy_v3
    except Exception as e:
        print(f"Error in v3 masked experiment: {e}")
        import traceback

        traceback.print_exc()
        results_summary["vlm_levels_v3"] = None

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY (MASKED)")
    print("=" * 80)
    for exp_name, accuracy in results_summary.items():
        print(f"\n{exp_name}:")
        if accuracy is not None and len(accuracy) > 0:
            for level, acc in accuracy.items():
                print(f"  {level}: {acc:.2%}")
        else:
            print("  No results")

    print("\n" + "=" * 80)
    print("All masked experiments completed!")
    print("=" * 80)
    print(f"\nResults are saved in: results_llava_hf/{model_name}/{results_subdir}/")
    print("  - data/ (original vlm_levels)")
    print("  - data_v2/ (vlm_levels_v2)")
    print("  - data_v3/ (vlm_levels_v3)")
    print("  - existential_yesno/ (existential yes/no questions)")
    print("  - existential_attribute/ (existential attribute questions)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run masked accuracy experiments for all datasets")
    parser.add_argument(
        "--levels",
        type=str,
        nargs="+",
        default=["level_0", "level_1", "level_2", "level_3", "level_4"],
        help="Level IDs to run (default: all levels)",
    )
    parser.add_argument("--show-output", action="store_true", help="Show model outputs during inference")
    parser.add_argument(
        "--mask-type",
        type=str,
        default=DEFAULT_MASK_TYPE,
        help="Masking strategy (passed to infer_model_for_levels)",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default=DEFAULT_RESULTS_SUBDIR,
        help="Subdirectory under results_llava_hf/<model>/ to store outputs",
    )
    parser.add_argument(
        "--no-always-mask",
        action="store_true",
        help="Disable always_mask (only mask when relations exist)",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable attention masking (useful for A/B runs while keeping same output folder)",
    )

    args = parser.parse_args()

    run_all_masked_experiments(
        level_ids=args.levels,
        show_output=args.show_output,
        use_attention_mask=(not args.no_mask),
        always_mask=(not args.no_always_mask),
        mask_type=args.mask_type,
        results_subdir=args.results_subdir,
    )

