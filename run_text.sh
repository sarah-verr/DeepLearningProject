#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# unset SBATCH_GRES SBATCH_GPUS SBATCH_GRES_PER_TASK SBATCH_TRES_PER_TASK SBATCH_GPUS_PER_TASK

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=logit_lens_raw
#SBATCH --account=deep_learning
#SBATCH --output=${REPO_DIR}/logs/%x-%j.out
#SBATCH --error=${REPO_DIR}/logs/%x-%j.err
SBATCH --time=02:00:00     

set -euo pipefail

cd "$REPO_DIR"

export HF_HOME="/work/scratch/\$USER/hf"
export HF_HUB_CACHE="\$HF_HOME/hub"
export TRANSFORMERS_CACHE="\$HF_HOME/transformers"
export HF_DATASETS_CACHE="\$HF_HOME/datasets"
export TORCH_HOME="\$HF_HOME/torch"
export TMPDIR="/work/scratch/\$USER/tmp"
mkdir -p "\$HF_HUB_CACHE" "\$TRANSFORMERS_CACHE" "\$HF_DATASETS_CACHE" "\$TORCH_HOME" "\$TMPDIR" "${REPO_DIR}/logs"

source venv/bin/activate

# python3 Text-Only/text_only_objective_vlm_prompt.py --levels level_1
# python3 Text-Only/Analysis/analyze_objective_attention.py
# python experiments/logit_lens/run_text_only_objective.py
python experiments/logit_lens/run_visual_logit_lens.py

EOF
