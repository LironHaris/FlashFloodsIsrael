#!/bin/bash
#SBATCH --job-name=ealstm_cv_epochs
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/logs_%x_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/logs_%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00

# Standalone k-fold CV of one config's fixed hyperparameters (no sweep, no early drop),
# e.g. configs/cv_epochs_0_3.yml: every fold trains the full epochs to re-estimate avg_best_epoch.

# 1. Activate Conda
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods

# 2. Keep Matplotlib / wandb caches in lab space (the network home is near-full)
export MPLCONFIGDIR=/sci/labs/efratmorin/liron.haris/.matplotlib_cache
export WANDB_CACHE_DIR=/sci/labs/efratmorin/liron.haris/wandb_cache
export WANDB_CONFIG_DIR=/sci/labs/efratmorin/liron.haris/.config/wandb

# 3. Project directory
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael
mkdir -p runs

CONFIG_PATH="${1:?Usage: sbatch run_cv_epochs.sh <config_path>, e.g. configs/cv_epochs_0_3.yml}"

# 4. Authenticate via WANDB_API_KEY instead of ~/.netrc (see run_cv_sweep.sh)
export WANDB_API_KEY="$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1], encoding='utf-8'))['wandb_api_key'])" "$CONFIG_PATH")"

# 5. Run all folds
python src/cross_validation_sweep.py --config "$CONFIG_PATH"
