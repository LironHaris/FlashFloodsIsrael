#!/bin/bash
#SBATCH --job-name=cumulative_stats
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/cumulative_%x_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/cumulative_%x_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# Cumulative-lead event statistics of one model family from its existing flood_event_comparison CSVs
# (src/summarize_family_test_periods.py --cumulative-only). CPU only, reads CSVs, logs to W&B.
# Usage: sbatch cluster/run_cumulative_stats.sh configs/summarize_family_test_periods_model_0.yml

source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods

export MPLCONFIGDIR=/sci/labs/efratmorin/liron.haris/.matplotlib_cache
export WANDB_CACHE_DIR=/sci/labs/efratmorin/liron.haris/wandb_cache
export WANDB_CONFIG_DIR=/sci/labs/efratmorin/liron.haris/.config/wandb
export MPLBACKEND=Agg
unset DISPLAY

cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

CONFIG_PATH="${1:?Usage: sbatch run_cumulative_stats.sh <family summary config>}"
python src/summarize_family_test_periods.py --config "$CONFIG_PATH" --cumulative-only
