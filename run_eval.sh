#!/bin/bash

#SBATCH --job-name=vlm_eval          # Job name
#SBATCH --account=deep_learning      # Account name
#SBATCH --output=logs/eval_%j.out         # Standard output log (%j inserts JobID)
#SBATCH --error=logs/eval_%j.err          # Standard error log
#SBATCH --time=01:00:00              # Time limit (hrs:min:sec)

export HF_HOME="/work/scratch/$USER"

# 1. Load modules (adjust based on your cluster's setup)
# module load cuda/12.1

# 2. Activate your virtual environment
# Replace 'venv' with the actual path to your environment
source venv/bin/activate

# 3. Run the evaluation
# Replace the levels with the ones you want to test
python compute_accuracy.py --levels 0 1 2 3 4 5 6 \
    --output_json evaluation_results_with_visual.json \
    --log_file model_calls_log_with_visual.csv