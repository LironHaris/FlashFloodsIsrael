#!/bin/bash
#SBATCH --job-name=flood_compare_eval
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/compare_eval_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/compare_eval_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

# 1. טעינת סביבת העבודה של המעבדה
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods

# 2. מעקפי בעיית הדיסק המלא
export HOME=/sci/labs/efratmorin/liron.haris/
export MPLCONFIGDIR=/sci/labs/efratmorin/liron.haris/.matplotlib_cache
export WANDB_CACHE_DIR=/sci/labs/efratmorin/liron.haris/wandb_cache
export WANDB_CONFIG_DIR=/sci/labs/efratmorin/liron.haris/.config/wandb
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 3. הרצת ניתוח ההשוואה שהופק ע"י model_compare_test.py
# Reads flood_event_comparison.csv + peaks_analysis_comparison_summary.csv
# from that same comparison's output dir and writes model_comparison_eval.csv
# (per-model TP/FP/FN, precision/recall/F1, peak timing/magnitude summary).
# Requires model_compare_test.py to have already been run for this config.
# Usage: sbatch run_model_compare_eval_test.sh <comparison_config_path>
CONFIG_PATH="${1:-configs/compare_model_0_leads.yml}"
python src/model_compare_eval_test.py --config "$CONFIG_PATH"
