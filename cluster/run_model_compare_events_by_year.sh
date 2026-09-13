#!/bin/bash
#SBATCH --job-name=flood_compare_events_by_year
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/compare_events_by_year_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/compare_events_by_year_%j.err
#SBATCH --gres=gpu:1                    # שריון GPU אחד (גם אם ההרצה עצמה נעולה על CPU)
#SBATCH --cpus-per-task=4              # 4 ליבות עבור טעינת נתונים ופלוטים
#SBATCH --mem=24G                       # 24 ג'יגה זיכרון RAM - הועלה מ-16G עבור ריצות עם הרבה שנים/בסיסים
#SBATCH --time=72:00:00                 # מגבלת זמן להרצה - הועלה מ-24 ל-72 שעות עבור ריצות כבדות (הרבה שנים * הרבה בסיסים * כמה מודלים בו-זמנית); התאימו לפי הצורך ולפי מגבלת הקלאסטר שלכם

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

# 6. הרצת סקריפט ההשוואה השנתית - basin/model checkpoints מגיעים מקובץ ה-comparison
# config (אותו פורמט בדיוק כמו model_compare_test.py); השנים מגיעות כארגומנטים.
# שימוש: sbatch run_model_compare_events_by_year.sh <comparison_config_path> <year> [year ...]
CONFIG_PATH="${1:?Usage: sbatch run_model_compare_events_by_year.sh <comparison_config_path> <year> [year ...]}"
shift
python src/model_compare_events_by_year.py --config "$CONFIG_PATH" --years "$@"
