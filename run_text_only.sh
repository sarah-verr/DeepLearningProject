#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=text_only
#SBATCH --account=deep_learning
#SBATCH --output=${REPO_DIR}/logs/%x-%j.out
#SBATCH --error=${REPO_DIR}/logs/%x-%j.err

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

python3 Text-Only/text_only_interactive.py --level level_4 --summary_plots --plot_examples -1 --save_phrase_plots


EOF

