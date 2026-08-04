#!/bin/bash
#SBATCH --job-name=ealstm_flood_compare
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/compare_%x_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/compare_%x_%j.err
#SBATCH --gres=gpu:1                    # שריון GPU אחד (גם אם ההרצה עצמה נעולה על CPU)
#SBATCH --cpus-per-task=4              # 4 ליבות עבור טעינת נתונים ופלוטים
#SBATCH --mem=16G                       # 16 ג'יגה זיכרון RAM
#SBATCH --time=24:00:00                 # מגבלת זמן להרצה (ניתן לשנות לפי הצורך)

# 1. טעינת ה-Conda environment
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh

# 2. הפעלת ה-conda environment
conda activate flashfloods

# 3. הגדרת תיקיית קאש ל-Matplotlib (כדי שלא תיכתב לתיקיית הבית ב-sci)
export MPLCONFIGDIR=/sci/labs/efratmorin/liron.haris/.matplotlib_cache

# 4. הגדרת תיקיית קאש/קונפיג עבור W&B (כדי שלא תיכתב לתיקיית הבית)
export WANDB_CACHE_DIR=/sci/labs/efratmorin/liron.haris/wandb_cache
export WANDB_CONFIG_DIR=/sci/labs/efratmorin/liron.haris/.config/wandb

# 5. מעבר לתיקיית הפרויקט
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 6. הרצת סקריפט ההשוואה
CONFIG_PATH="${1:-configs/compare_model_0_leads.yml}"
python src/model_compare_test.py --config "$CONFIG_PATH"
