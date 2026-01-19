#!/bin/bash

#SBATCH --job-name=objects_only_mask_accuracy      # Job name
#SBATCH --account=deep_learning      # Account name
#SBATCH --output=logs/accuracy_w_mask_ablation_%j.out    # Standard output log (%j inserts JobID)
#SBATCH --error=logs/accuracy_w_mask_ablation_%j.err     # Standard error log
#SBATCH --time=24:00:00              # Time limit (hrs:min:sec) - adjust as needed

export HF_HOME="/work/scratch/$USER"

# Activate virtual environment
source venv/bin/activate
source setup.sh

echo "Running existential QA accuracy experiments..."
python experiments/run_masked_accuracy_experiments.py

echo "Job completed successfully!"

EOF
