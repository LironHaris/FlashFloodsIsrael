#!/bin/bash
#SBATCH --job-name=flood_replot_nse_cdf
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/replot_nse_cdf_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/replot_nse_cdf_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

# 1. טעינת סביבת העבודה של המעבדה
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods

# 2. מעקפי בעיית הדיסק המלא
export HOME=/sci/labs/efratmorin/liron.haris/
export MPLCONFIGDIR=/sci/labs/efratmorin/liron.haris/.matplotlib_cache
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 3. הרצת השרטוט מחדש בלבד - ללא הרצת המודלים מחדש (מניח model_compare_test.py כבר רץ)
# Usage: sbatch run_replot_nse_cdf.sh <comparison_config_path>
CONFIG_PATH="${1:-configs/compare_model_0_leads.yml}"
python src/replot_nse_cdf_comparison.py --config "$CONFIG_PATH"
