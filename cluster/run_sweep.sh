#!/bin/bash
#SBATCH --job-name=ealstm_sweep
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/sweep_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/sweep_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# 1. טעינת ה-Conda של המעבדה והפעלת הסביבה
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods

# 2. מעקפי בעיית הדיסק המלא במדעי המחשב
export HOME=/sci/labs/efratmorin/liron.haris/
export MPLCONFIGDIR=/sci/labs/efratmorin/liron.haris/.matplotlib_cache

# 3. מעבר לתיקיית הפרויקט
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 4. הרצת ה-Agent (כל ריצה להחליף את ה ID שבסוף הכתובת)
wandb agent liron-haris-hebrew-university-of-jerusalem/flash-floods-israel/4ey73tda

