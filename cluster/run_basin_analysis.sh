#!/bin/bash
#SBATCH --job-name=basin_analysis
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/basin_%x_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/basin_%x_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# Per-basin analysis of one model family from its existing summary outputs
# (src/summarize_family_test_periods.py --basin-only -> src/family_basin_analysis.py). CPU only, logs to W&B.
# Usage: sbatch cluster/run_basin_analysis.sh configs/summarize_family_test_periods_<family>_<variant>.yml

source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods

export MPLCONFIGDIR=/sci/labs/efratmorin/liron.haris/.matplotlib_cache
export WANDB_CACHE_DIR=/sci/labs/efratmorin/liron.haris/wandb_cache
export WANDB_CONFIG_DIR=/sci/labs/efratmorin/liron.haris/.config/wandb
export MPLBACKEND=Agg
unset DISPLAY

cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

CONFIG_PATH="${1:?Usage: sbatch run_basin_analysis.sh <family summary config>}"
python src/summarize_family_test_periods.py --config "$CONFIG_PATH" --basin-only
