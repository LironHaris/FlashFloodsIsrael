#!/bin/bash
#SBATCH --job-name=flood_analyze_static_features
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/analyze_static_features_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/analyze_static_features_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

# 1. טעינת סביבת העבודה של המעבדה
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods

# 2. מעקפי בעיית הדיסק המלא
export HOME=/sci/labs/efratmorin/liron.haris/
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 3. הרצת ניתוח ה-PCA על התכונות הסטטיות (pca_variance_summary/pca_loadings/pca_projected_data)
python src/analyze_static_features.py
