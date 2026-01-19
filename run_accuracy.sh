#!/bin/bash

#SBATCH --job-name=contradicted_qa_accuracy
#SBATCH --account=deep_learning
#SBATCH --output=logs/contradicted_qa_accuracy_%j.out
#SBATCH --error=logs/contradicted_qa_accuracy_%j.err
#SBATCH --time=24:00:00

export HF_HOME="/work/scratch/$USER"

# Activate virtual environment
source venv/bin/activate
source setup.sh

echo "Running contradicted QA accuracy experiment..."
python experiments/contradicted_qa/run_accuracy.py --levels level_0 level_1 level_2 level_3 level_4

echo "Job completed successfully!"
