#!/bin/bash
#SBATCH --job-name=return_periods
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/return_periods_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/return_periods_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00

# 1. טעינת הסביבה והקונדה
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods

# 2. מעקפי דיסק ומיקום
export HOME=/sci/labs/efratmorin/liron.haris/
export MPLCONFIGDIR=/sci/labs/efratmorin/liron.haris/.matplotlib_cache
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 3. הרצת הסקריפט
python src/calculate_return_periods.py