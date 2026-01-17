"""
Analyze accuracy results across all experiments.

This script loads CSV results from all accuracy experiments and generates:
1. Accuracy per level (level_id) per experiment
2. Accuracy per question type per experiment  
3. Detailed statistics for specific questions
4. Visualization plots
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

# Add project root to path
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.prompt_llava import MODEL_ID


def find_all_result_files(base_dir: Path) -> Dict[str, Path]:
    """Find all CSV result files organized by dataset/experiment."""
    results = {}
    model_name = MODEL_ID.split("/")[-1]
    results_base = base_dir / "results_llava_hf" / model_name / "accuracy_question_ablation"
    
    if not results_base.exists():
        print(f"Results directory not found: {results_base}")
        return results
    
    # Search for CSV files in subdirectories
    for csv_file in results_base.rglob("*.csv"):
        # Extract dataset name from path
        # Structure: .../accuracy_question_ablation/{dataset_name}/{filename}.csv
        rel_path = csv_file.relative_to(results_base)
        dataset_name = rel_path.parts[0]  # First directory after accuracy_question_ablation
        filename = csv_file.stem
        
        # Create a key: {dataset_name}_{experiment_type}
        # e.g., "data_visual_yesno", "data_v2_visual_attribute"
        key = f"{dataset_name}_{filename}"
        results[key] = csv_file
    
    return results


def load_results(file_path: Path) -> pd.DataFrame:
    """Load a CSV result file."""
    try:
        df = pd.read_csv(file_path)
        print(f"  Loaded {len(df)} rows from {file_path.name}")
        return df
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return pd.DataFrame()


def compute_accuracy_breakdown(df: pd.DataFrame) -> Dict:
    """Compute various accuracy breakdowns."""
    if len(df) == 0 or "is_correct" not in df.columns:
        return {}
    
    breakdown = {
        "overall": df["is_correct"].mean(),
        "total": len(df),
        "correct": df["is_correct"].sum(),
    }
    
    # By level
    if "level_id" in df.columns:
        breakdown["by_level"] = df.groupby("level_id")["is_correct"].agg(["mean", "count"]).to_dict("index")
    
    # By question type
    if "question_type" in df.columns:
        breakdown["by_question_type"] = df.groupby("question_type")["is_correct"].agg(["mean", "count"]).to_dict("index")
    
    # By dataset
    if "dataset" in df.columns:
        breakdown["by_dataset"] = df.groupby("dataset")["is_correct"].agg(["mean", "count"]).to_dict("index")
    
    return breakdown


def print_summary_table(all_results: Dict[str, pd.DataFrame]):
    """Print a summary table of all experiments."""
    print("\n" + "="*80)
    print("ACCURACY SUMMARY BY EXPERIMENT")
    print("="*80)
    
    summary_data = []
    for key, df in all_results.items():
        if len(df) == 0:
            continue
        if "is_correct" not in df.columns:
            continue
        
        breakdown = compute_accuracy_breakdown(df)
        summary_data.append({
            "Experiment": key,
            "Total": breakdown.get("total", 0),
            "Correct": breakdown.get("correct", 0),
            "Accuracy": f"{breakdown.get('overall', 0):.2%}",
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
    else:
        print("No results found.")


def print_accuracy_by_level(all_results: Dict[str, pd.DataFrame]):
    """Print accuracy broken down by level for each experiment."""
    print("\n" + "="*80)
    print("ACCURACY BY LEVEL (per experiment)")
    print("="*80)
    
    for key, df in all_results.items():
        if len(df) == 0 or "level_id" not in df.columns or "is_correct" not in df.columns:
            continue
        
        print(f"\n{key}:")
        accuracy_by_level = df.groupby("level_id")["is_correct"].agg(["mean", "count"])
        for level, row in accuracy_by_level.iterrows():
            print(f"  {level}: {row['mean']:.2%} ({int(row['count'])} questions)")


def print_accuracy_by_question_type(all_results: Dict[str, pd.DataFrame]):
    """Print accuracy broken down by question type for each experiment."""
    print("\n" + "="*80)
    print("ACCURACY BY QUESTION TYPE (per experiment)")
    print("="*80)
    
    for key, df in all_results.items():
        if len(df) == 0 or "question_type" not in df.columns or "is_correct" not in df.columns:
            continue
        
        print(f"\n{key}:")
        accuracy_by_type = df.groupby("question_type")["is_correct"].agg(["mean", "count"])
        for qtype, row in accuracy_by_type.iterrows():
            print(f"  {qtype}: {row['mean']:.2%} ({int(row['count'])} questions)")


def plot_accuracy_by_level(all_results: Dict[str, pd.DataFrame], output_dir: Path):
    """Create visualization plots for accuracy by level."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect data for plotting
    plot_data = []
    for key, df in all_results.items():
        if len(df) == 0 or "level_id" not in df.columns or "is_correct" not in df.columns:
            continue
        
        accuracy_by_level = df.groupby("level_id")["is_correct"].mean()
        for level, acc in accuracy_by_level.items():
            plot_data.append({
                "experiment": key,
                "level": level,
                "accuracy": acc,
            })
    
    if not plot_data:
        print("No data available for plotting.")
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure with subplots
    n_experiments = plot_df["experiment"].nunique()
    fig, axes = plt.subplots(1, min(3, n_experiments), figsize=(6*min(3, n_experiments), 6))
    if n_experiments == 1:
        axes = [axes]
    
    experiments = sorted(plot_df["experiment"].unique())
    for idx, exp in enumerate(experiments[:3]):  # Plot up to 3 experiments
        exp_data = plot_df[plot_df["experiment"] == exp]
        ax = axes[idx] if n_experiments > 1 else axes[0]
        
        levels = sorted(exp_data["level"].unique(), key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 999)
        accuracies = [exp_data[exp_data["level"] == level]["accuracy"].iloc[0] for level in levels]
        
        ax.bar(range(len(levels)), accuracies, alpha=0.7)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels, rotation=45, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{exp}\n(Overall: {exp_data['accuracy'].mean():.2%})")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "accuracy_by_level.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to: {output_path}")


def plot_accuracy_by_question_type(all_results: Dict[str, pd.DataFrame], output_dir: Path):
    """Create visualization plots for accuracy by question type."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect data for plotting
    plot_data = []
    for key, df in all_results.items():
        if len(df) == 0 or "question_type" not in df.columns or "is_correct" not in df.columns:
            continue
        
        accuracy_by_type = df.groupby("question_type")["is_correct"].mean()
        for qtype, acc in accuracy_by_type.items():
            plot_data.append({
                "experiment": key,
                "question_type": qtype,
                "accuracy": acc,
            })
    
    if not plot_data:
        print("No question type data available for plotting.")
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure
    n_experiments = plot_df["experiment"].nunique()
    fig, axes = plt.subplots(1, min(3, n_experiments), figsize=(6*min(3, n_experiments), 6))
    if n_experiments == 1:
        axes = [axes]
    
    experiments = sorted(plot_df["experiment"].unique())
    for idx, exp in enumerate(experiments[:3]):  # Plot up to 3 experiments
        exp_data = plot_df[plot_df["experiment"] == exp]
        ax = axes[idx] if n_experiments > 1 else axes[0]
        
        qtypes = sorted(exp_data["question_type"].unique())
        accuracies = [exp_data[exp_data["question_type"] == qt]["accuracy"].iloc[0] for qt in qtypes]
        
        ax.bar(range(len(qtypes)), accuracies, alpha=0.7)
        ax.set_xticks(range(len(qtypes)))
        ax.set_xticklabels(qtypes, rotation=45, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{exp}\n(Overall: {exp_data['accuracy'].mean():.2%})")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "accuracy_by_question_type.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_path}")


def plot_question_type_by_level(all_results: Dict[str, pd.DataFrame], output_dir: Path):
    """Create visualization plots for question type accuracy broken down by level."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect data for plotting
    plot_data = []
    for key, df in all_results.items():
        if len(df) == 0 or "question_type" not in df.columns or "level_id" not in df.columns or "is_correct" not in df.columns:
            continue
        
        # Group by both question_type and level_id
        grouped = df.groupby(["question_type", "level_id"])["is_correct"].mean().reset_index()
        for _, row in grouped.iterrows():
            plot_data.append({
                "experiment": key,
                "question_type": row["question_type"],
                "level": row["level_id"],
                "accuracy": row["is_correct"],
            })
    
    if not plot_data:
        print("No question type by level data available for plotting.")
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create separate plots for each experiment that has question types
    experiments_with_types = plot_df["experiment"].unique()
    
    for exp in experiments_with_types:
        exp_data = plot_df[plot_df["experiment"] == exp]
        
        # Get unique question types and levels
        qtypes = sorted(exp_data["question_type"].unique())
        levels = sorted(exp_data["level"].unique(), key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 999)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(levels))
        width = 0.8 / len(qtypes)  # Width of bars
        
        for i, qtype in enumerate(qtypes):
            offsets = (i - len(qtypes)/2 + 0.5) * width
            accuracies = []
            for level in levels:
                subset = exp_data[(exp_data["level"] == level) & (exp_data["question_type"] == qtype)]
                if len(subset) > 0:
                    accuracies.append(subset["accuracy"].iloc[0])
                else:
                    accuracies.append(0.0)
            
            ax.bar(x + offsets, accuracies, width, label=qtype, alpha=0.7)
        
        ax.set_xlabel("Level", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"Question Type Accuracy by Level\n{exp}", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.set_ylim(0, 1.0)
        ax.legend(title="Question Type", loc="best")
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        safe_exp_name = exp.replace("/", "_").replace("\\", "_")
        output_path = output_dir / f"question_type_by_level_{safe_exp_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to: {output_path}")
        
        # Also create a heatmap version
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create pivot table for heatmap
        pivot_data = exp_data.pivot_table(index="question_type", columns="level", values="accuracy", aggfunc="mean")
        # Reorder columns (levels)
        pivot_data = pivot_data[levels]
        
        sns.heatmap(pivot_data, annot=True, fmt=".2%", cmap="YlOrRd", vmin=0, vmax=1, 
                    cbar_kws={"label": "Accuracy"}, ax=ax, linewidths=0.5)
        ax.set_title(f"Question Type Accuracy by Level (Heatmap)\n{exp}", fontsize=14)
        ax.set_xlabel("Level", fontsize=12)
        ax.set_ylabel("Question Type", fontsize=12)
        
        plt.tight_layout()
        output_path_heatmap = output_dir / f"question_type_by_level_heatmap_{safe_exp_name}.png"
        plt.savefig(output_path_heatmap, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Heatmap saved to: {output_path_heatmap}")


def create_masked_vs_unmasked_comparison(all_results: Dict[str, pd.DataFrame], output_dir: Path):
    """Create comprehensive comparison plots showing masked vs unmasked accuracy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find pairs of masked and unmasked results
    # Pattern: {dataset}_{experiment} and {dataset}_{experiment}_w_mask or _w_mask_always
    masked_pairs = {}
    
    for key, df in all_results.items():
        if len(df) == 0 or "is_correct" not in df.columns:
            continue
        
        # Check if this is a masked result
        if "_w_mask" in key:
            # Find corresponding unmasked version
            unmasked_key = key.replace("_w_mask_always", "").replace("_w_mask", "")
            if unmasked_key in all_results:
                pair_name = unmasked_key.replace("_results", "")
                if pair_name not in masked_pairs:
                    masked_pairs[pair_name] = {}
                if "_always" in key:
                    masked_pairs[pair_name]["masked_always"] = df
                else:
                    masked_pairs[pair_name]["masked"] = df
                masked_pairs[pair_name]["unmasked"] = all_results[unmasked_key]
    
    if not masked_pairs:
        print("No masked/unmasked pairs found for comparison.")
        return
    
    # Create comparison plots for each pair
    for pair_name, pair_data in masked_pairs.items():
        if "unmasked" not in pair_data:
            continue
        
        unmasked_df = pair_data["unmasked"]
        
        # Create figure with subplots for each masked variant
        masked_variants = [k for k in pair_data.keys() if k != "unmasked"]
        
        if not masked_variants:
            continue
        
        safe_name = pair_name.replace("/", "_").replace("\\", "_")
        
        # 1. Overall accuracy comparison
        fig, ax = plt.subplots(figsize=(8, 6))
        overall_data = []
        overall_data.append({"Method": "Unmasked", "Accuracy": unmasked_df["is_correct"].mean()})
        for variant in masked_variants:
            masked_df = pair_data[variant]
            variant_label = "Always Mask" if "always" in variant else "Mask (when relation exists)"
            overall_data.append({"Method": variant_label, "Accuracy": masked_df["is_correct"].mean()})
        
        overall_df = pd.DataFrame(overall_data)
        ax.bar(overall_df["Method"], overall_df["Accuracy"], alpha=0.7, color=["blue", "orange", "green"][:len(overall_df)])
        ax.set_ylabel("Overall Accuracy", fontsize=12)
        ax.set_title(f"Overall Accuracy Comparison\n{pair_name}", fontsize=14)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        for i, (method, acc) in enumerate(zip(overall_df["Method"], overall_df["Accuracy"])):
            ax.text(i, acc + 0.02, f"{acc:.2%}", ha="center", fontsize=10)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        output_path = output_dir / f"masked_vs_unmasked_overall_{safe_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Overall comparison plot saved to: {output_path}")
        
        # 2. Accuracy by level comparison
        if "level_id" in unmasked_df.columns:
            levels = sorted(unmasked_df["level_id"].unique(), 
                           key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 999)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(levels))
            width = 0.8 / (len(masked_variants) + 1)
            
            # Unmasked bars
            unmasked_acc = [unmasked_df[unmasked_df["level_id"] == level]["is_correct"].mean() for level in levels]
            ax.bar(x - width * (len(masked_variants) / 2), unmasked_acc, width, label="Unmasked", alpha=0.7, color="blue")
            
            # Masked bars
            colors = ["orange", "green"]
            for idx, variant in enumerate(masked_variants):
                masked_df = pair_data[variant]
                masked_acc = [masked_df[masked_df["level_id"] == level]["is_correct"].mean() for level in levels]
                variant_label = "Always Mask" if "always" in variant else "Mask"
                ax.bar(x - width * (len(masked_variants) / 2) + width * (idx + 1), masked_acc, width, 
                       label=variant_label, alpha=0.7, color=colors[idx % len(colors)])
            
            ax.set_xlabel("Level", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title(f"Accuracy by Level Comparison\n{pair_name}", fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(levels)
            ax.set_ylim(0, 1.0)
            ax.legend(loc="best")
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            output_path = output_dir / f"masked_vs_unmasked_by_level_{safe_name}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"By-level comparison plot saved to: {output_path}")
        
        # 3. Accuracy by question type comparison
        if "question_type" in unmasked_df.columns:
            qtypes = sorted(unmasked_df["question_type"].unique())
            
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(qtypes))
            width = 0.8 / (len(masked_variants) + 1)
            
            # Unmasked bars
            unmasked_acc = [unmasked_df[unmasked_df["question_type"] == qt]["is_correct"].mean() for qt in qtypes]
            ax.bar(x - width * (len(masked_variants) / 2), unmasked_acc, width, label="Unmasked", alpha=0.7, color="blue")
            
            # Masked bars
            colors = ["orange", "green"]
            for idx, variant in enumerate(masked_variants):
                masked_df = pair_data[variant]
                masked_acc = [masked_df[masked_df["question_type"] == qt]["is_correct"].mean() for qt in qtypes]
                variant_label = "Always Mask" if "always" in variant else "Mask"
                ax.bar(x - width * (len(masked_variants) / 2) + width * (idx + 1), masked_acc, width, 
                       label=variant_label, alpha=0.7, color=colors[idx % len(colors)])
            
            ax.set_xlabel("Question Type", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title(f"Accuracy by Question Type Comparison\n{pair_name}", fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(qtypes, rotation=45, ha="right")
            ax.set_ylim(0, 1.0)
            ax.legend(loc="best")
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            output_path = output_dir / f"masked_vs_unmasked_by_question_type_{safe_name}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"By-question-type comparison plot saved to: {output_path}")
        
        # 4. Heatmap comparison (question type × level)
        if "question_type" in unmasked_df.columns and "level_id" in unmasked_df.columns:
            levels = sorted(unmasked_df["level_id"].unique(), 
                           key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 999)
            qtypes = sorted(unmasked_df["question_type"].unique())
            
            for variant in masked_variants:
                masked_df = pair_data[variant]
                
                # Create pivot tables for heatmaps (aggregate by mean to handle duplicates)
                unmasked_pivot = unmasked_df.pivot_table(
                    index="question_type", columns="level_id", values="is_correct", aggfunc="mean"
                )
                masked_pivot = masked_df.pivot_table(
                    index="question_type", columns="level_id", values="is_correct", aggfunc="mean"
                )
                
                # Ensure same order
                unmasked_pivot = unmasked_pivot.reindex(index=qtypes, columns=levels)
                masked_pivot = masked_pivot.reindex(index=qtypes, columns=levels)
                
                # Create side-by-side comparison
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Unmasked heatmap
                sns.heatmap(unmasked_pivot, annot=True, fmt=".2%", cmap="YlOrRd", vmin=0, vmax=1,
                           cbar_kws={"label": "Accuracy"}, ax=axes[0], linewidths=0.5)
                axes[0].set_title(f"Unmasked\n{pair_name}", fontsize=14)
                axes[0].set_xlabel("Level", fontsize=12)
                axes[0].set_ylabel("Question Type", fontsize=12)
                
                # Masked heatmap
                variant_label = "Always Mask" if "always" in variant else "Mask (when relation exists)"
                sns.heatmap(masked_pivot, annot=True, fmt=".2%", cmap="YlOrRd", vmin=0, vmax=1,
                           cbar_kws={"label": "Accuracy"}, ax=axes[1], linewidths=0.5)
                axes[1].set_title(f"Masked ({variant_label})\n{pair_name}", fontsize=14)
                axes[1].set_xlabel("Level", fontsize=12)
                axes[1].set_ylabel("", fontsize=12)
                
                # Difference heatmap (masked - unmasked)
                diff_pivot = masked_pivot - unmasked_pivot
                sns.heatmap(diff_pivot, annot=True, fmt="+.2%", cmap="RdBu_r", center=0,
                           vmin=-0.5, vmax=0.5, cbar_kws={"label": "Accuracy Difference"},
                           ax=axes[2], linewidths=0.5)
                axes[2].set_title(f"Difference (Masked - Unmasked)\n{pair_name}", fontsize=14)
                axes[2].set_xlabel("Level", fontsize=12)
                axes[2].set_ylabel("", fontsize=12)
                
                plt.tight_layout()
                variant_suffix = "_always" if "always" in variant else ""
                output_path = output_dir / f"masked_vs_unmasked_heatmap_{safe_name}{variant_suffix}.png"
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"Heatmap comparison plot saved to: {output_path}")


def analyze_specific_questions(all_results: Dict[str, pd.DataFrame], min_count: int = 5):
    """Analyze accuracy for specific question patterns."""
    print("\n" + "="*80)
    print("QUESTION-SPECIFIC ANALYSIS")
    print("="*80)
    
    for key, df in all_results.items():
        if len(df) == 0 or "question" not in df.columns or "is_correct" not in df.columns:
            continue
        
        print(f"\n{key}:")
        
        # Group by exact question text
        question_stats = df.groupby("question")["is_correct"].agg(["mean", "count"])
        question_stats = question_stats[question_stats["count"] >= min_count].sort_values("mean")
        
        print(f"\n  Worst performing questions (>= {min_count} occurrences):")
        for question, row in question_stats.head(5).iterrows():
            print(f"    {row['mean']:.2%} ({int(row['count'])}x): {question[:80]}...")
        
        print(f"\n  Best performing questions (>= {min_count} occurrences):")
        for question, row in question_stats.tail(5).iterrows():
            print(f"    {row['mean']:.2%} ({int(row['count'])}x): {question[:80]}...")


def main():
    """Main analysis function."""
    base_dir = _REPO_ROOT / "experiments"
    
    print("Loading result files...")
    result_files = find_all_result_files(base_dir)
    
    if not result_files:
        print("No result files found!")
        return
    
    print(f"Found {len(result_files)} result files:")
    for key, path in result_files.items():
        print(f"  {key}: {path}")
    
    # Load all results
    print("\nLoading data...")
    all_results = {}
    for key, path in result_files.items():
        all_results[key] = load_results(path)
    
    # Generate summaries
    print_summary_table(all_results)
    print_accuracy_by_level(all_results)
    print_accuracy_by_question_type(all_results)
    analyze_specific_questions(all_results)
    
    # Create visualizations
    output_dir = base_dir / "results_llava_hf" / MODEL_ID.split("/")[-1] / "accuracy_question_ablation" / "analysis"
    print(f"\nGenerating plots...")
    plot_accuracy_by_level(all_results, output_dir)
    plot_accuracy_by_question_type(all_results, output_dir)
    plot_question_type_by_level(all_results, output_dir)
    create_masked_vs_unmasked_comparison(all_results, output_dir)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Plots saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
