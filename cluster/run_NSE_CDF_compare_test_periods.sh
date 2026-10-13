#!/bin/bash
#SBATCH --job-name=nse_cdf_test_periods
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/compare_%x_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/compare_%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# NSE CDF of one model setup across its test periods (src/NSE_CDF_compare_test_periods.py).
# Reuses existing per-basin test reports; the GPU is only needed when a model must be re-evaluated.
# Usage: sbatch cluster/run_NSE_CDF_compare_test_periods.sh <config> [--reevaluate]

# 1. Conda environment
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods

# 2. Matplotlib / W&B caches in lab space (not the near-full home)
export MPLCONFIGDIR=/sci/labs/efratmorin/liron.haris/.matplotlib_cache
export WANDB_CACHE_DIR=/sci/labs/efratmorin/liron.haris/wandb_cache
export WANDB_CONFIG_DIR=/sci/labs/efratmorin/liron.haris/.config/wandb

# 3. Project directory
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 4. Run the comparison (extra args, e.g. --reevaluate, are passed through)
CONFIG_PATH="${1:?Usage: sbatch run_NSE_CDF_compare_test_periods.sh <config> [--reevaluate]}"
shift
python src/NSE_CDF_compare_test_periods.py --config "$CONFIG_PATH" "$@"
