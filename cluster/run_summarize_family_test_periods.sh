#!/bin/bash
#SBATCH --job-name=family_summary
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/compare_%x_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/compare_%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Hydrographs, NSE CDF and median statistics of one model family over its leads x test periods
# (src/summarize_family_test_periods.py). The GPU is only used when a model has no per-basin reports yet.
# Usage: sbatch cluster/run_summarize_family_test_periods.sh <config> [--reevaluate]

# 1. Conda environment
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods

# 2. Matplotlib / W&B caches in lab space (not the near-full home)
export MPLCONFIGDIR=/sci/labs/efratmorin/liron.haris/.matplotlib_cache
export WANDB_CACHE_DIR=/sci/labs/efratmorin/liron.haris/wandb_cache
export WANDB_CONFIG_DIR=/sci/labs/efratmorin/liron.haris/.config/wandb

# 3. Project directory
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 4. Run (extra args, e.g. --reevaluate, are passed through)
CONFIG_PATH="${1:?Usage: sbatch run_summarize_family_test_periods.sh <config> [--reevaluate]}"
shift
python src/summarize_family_test_periods.py --config "$CONFIG_PATH" "$@"
