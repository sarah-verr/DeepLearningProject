#!/bin/bash

#SBATCH --job-name=vlm_eval          # Job name
#SBATCH --account=deep_learning      # Account name
#SBATCH --output=logs/eval_%j.out         # Standard output log (%j inserts JobID)
#SBATCH --error=logs/eval_%j.err          # Standard error log
<<<<<<< HEAD
#SBATCH --time=02:00:00              # Time limit (hrs:min:sec)
=======
#SBATCH --time=01:00:00              # Time limit (hrs:min:sec)
>>>>>>> 5377b2f0425a36e119609f3a4180b3e1e327ba0c

export HF_HOME="/work/scratch/$USER"

# 1. Load modules (adjust based on your cluster's setup)
# module load cuda/12.1

# 2. Activate your virtual environment
# Replace 'venv' with the actual path to your environment
source venv/bin/activate
<<<<<<< HEAD
=======
source setup.sh
>>>>>>> 5377b2f0425a36e119609f3a4180b3e1e327ba0c

echo "STARTING JOB..."
# 3. Run the experiment
python main.py

echo "JOB ENDED..."