#!/bin/bash
#SBATCH --job-name=vlm-probe-l1
#SBATCH --output=logs/vlm_probe_l1_%j.out
#SBATCH --error=logs/vlm_probe_l1_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=jobs
#SBATCH --account=deep_learning

# Always good practice: start in project dir
cd /home/tenkhtuvshin/DeepLearningProject

# Create logs dir if it doesn't exist
mkdir -p logs

# Environment
export HF_HOME="/work/scratch/tenkhtuvshin"

# Activate venv
source venv/bin/activate

# Optional: print some debug info
echo "Python: $(which python)"
echo "HF_HOME=$HF_HOME"
echo "Starting probing run at $(date)"

# Run your command
python main.py --level 1 --use_all_images --probe_layers

echo "Finished at $(date)"