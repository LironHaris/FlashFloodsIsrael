#!/bin/bash
#SBATCH --job-name=flood_analyze_results
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/analyze_results_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/analyze_results_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

# 1. טעינת סביבת העבודה של המעבדה
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods

# 2. מעקפי בעיית הדיסק המלא
export HOME=/sci/labs/efratmorin/liron.haris/
export MPLCONFIGDIR=/sci/labs/efratmorin/liron.haris/.matplotlib_cache
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 3. הרצת ניתוח התוצאות שהופקו ע"י quick_test.py/test.py
# Runs both analyses in one go: NSE-vs-static-feature scatter plots
# (config['nse_scatter_static_features']) and precision/recall classification
# metrics (per basin per lead time + lead/model means/variance).
# Usage: sbatch run_analyze_results.sh <config_path>
CONFIG_PATH="${1:-configs/config.yml}"
python src/analyze_results.py --config "$CONFIG_PATH"
