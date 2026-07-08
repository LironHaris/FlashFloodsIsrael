#!/bin/bash
#SBATCH --job-name=flood_basin_splits
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/basin_splits_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/basin_splits_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

# 1. טעינת סביבת העבודה של המעבדה
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods

# 2. מעקפי בעיית הדיסק המלא
export HOME=/sci/labs/efratmorin/liron.haris/
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 3. יצירת/עדכון רשימות הבסיסים (israel_train/val/test.txt) מתוך דוח הזמינות הקיים
python src/basin_splits.py
