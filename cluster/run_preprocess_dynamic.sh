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
python src/preprocess_dynamic_data.py