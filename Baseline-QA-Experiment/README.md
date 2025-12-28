# Baseline QA Experiment

Tests LLaVA's basic object recognition using existence questions before spatial reasoning evaluation.

## Overview

The baseline QA experiment tests whether LLaVA can correctly identify the presence or absence of colored objects in images. This establishes a foundation for understanding whether poor performance on spatial reasoning tasks is due to:

1. **Confirmation bias**: Many VQA datasets have imbalanced answer distributions (mostly "yes" answers), which can lead models to develop bias toward positive answers. Our balanced approach uses exactly 2 "yes" and 2 "no" ground truth answers per image to avoid this pitfall.

2. **Basic object recognition failures**: If the model cannot reliably identify whether objects exist in images, then failures in spatial reasoning tasks may simply reflect poor perception rather than reasoning limitations.

## Prompting Strategies

Four different prompting strategies are implemented to test the impact of prompt formulation on model performance:

### Strategy 0: Minimalist
- **Format**: `"Is there a {color} {shape}?"`

### Strategy 1: Contextual
- **Format**: `"Does this image have a {color} {shape}?"`

### Strategy 2: Strict Format
- **Format**: `"Does this image have a {color} {shape}? Answer: Yes or No."`

### Strategy 3: Priming
- **Format**: `"This is a Yes/No question: Does this image have a {color} {shape}?"`

## Question Generation

Each image generates exactly 4 questions with a **balanced distribution**:
- **2 "yes" questions**: Ask about objects that actually exist in the image
- **2 "no" questions**: Ask about objects that don't exist in the image

Questions are randomly sampled from all possible color-shape combinations to ensure variety.

## Files

- `add_ann_baseline_qa_strategies.py` - Adds baseline QA questions to annotation files
- `prompt_baseline_qa.py` - Runs LLaVA inference on baseline questions
- `analyze_baseline_qa.ipynb` - Interactive analysis with confusion matrices and plots
- `submit_analysis.sbatch` - SLURM batch processing script

## Quick Start

### 1. Add Baseline Questions to Synthetic Data

After generating synthetic data with the main pipeline:

```bash
cd Baseline-QA-Experiment
python add_ann_baseline_qa_strategies.py
```

This adds `qa_baseline_0`, `qa_baseline_1`, `qa_baseline_2`, and `qa_baseline_3` keys to all annotation files.

### 2. Run Inference

**Single image, all strategies:**
```bash
python prompt_baseline_qa.py --level 0 --id 00000_b --all-strategies
```

**Batch processing (all images, all strategies):**
```bash
sbatch submit_analysis.sbatch
```

### 3. Analyze Results

**Interactive analysis with confusion matrices:**
```bash
jupyter notebook analyze_baseline_qa.ipynb
```

This notebook provides:
- Confusion matrices for each strategy/level
- Accuracy statistics and comparisons
- Performance analysis on yes vs no questions
- Visual plots and detailed metrics


## Output Structure

Results are saved in: `baseline_results/level_X/image_id/strategy_Y_timestamp/overview/results.json`

Each results file contains accuracy metrics, individual question results, and metadata for analysis.