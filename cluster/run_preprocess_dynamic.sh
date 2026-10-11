#!/bin/bash
#SBATCH --job-name=flood_preprocess
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/preprocess_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/preprocess_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# 1. טעינת סביבת העבודה של המעבדה
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods

# 2. מעקפי בעיית הדיסק המלא
export HOME=/sci/labs/efratmorin/liron.haris/
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 3. הרצת סקריפט ה-Preprocess שלך
# Optional 1st arg: preprocessing config, e.g. configs/preprocess_test_2009_2010_2011.yml
CONFIG_PATH="${1:-configs/preprocess.yml}"
python src/preprocess_dynamic_data.py --config "$CONFIG_PATH"