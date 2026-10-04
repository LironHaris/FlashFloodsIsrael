#!/bin/bash
#SBATCH --job-name=flood_plot_events_by_year
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/plot_events_by_year_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/plot_events_by_year_%j.err
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

# 4. מעבר לתיקיית הפרויקט
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 5. הרצת סקריפט הפלוטים - basin/model checkpoints מגיעים מקובץ ה-comparison
# config (אותו פורמט בדיוק כמו model_compare_test.py); השנים מגיעות כארגומנטים.
# שימוש: sbatch run_plot_events_by_year.sh <comparison_config_path> <year> [year ...]
CONFIG_PATH="${1:?Usage: sbatch run_plot_events_by_year.sh <comparison_config_path> <year> [year ...]}"
shift
python src/plot_events_by_year.py --config "$CONFIG_PATH" --years "$@"
