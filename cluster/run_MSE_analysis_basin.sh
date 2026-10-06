#!/bin/bash
#SBATCH --job-name=MSE_analysis_basin
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/logs_%x_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/logs_%x_%j.err
#SBATCH --gres=gpu:1                    # שריון GPU אחד (גם אם ההרצה עצמה נעולה על CPU)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# 1. טעינת ה-Conda environment
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh

# 2. הפעלת ה-conda environment
conda activate flashfloods

# 3. הגדרת תיקיית קאש למטפלוטליב
export MPLCONFIGDIR=/sci/labs/efratmorin/liron.haris/.matplotlib_cache
# ללא תצוגה גרפית: מונע קריסה כש-DISPLAY של ssh -X עובר לג'וב ונסגר
export MPLBACKEND=Agg
unset DISPLAY

# 3b. wandb writes its cache/config to $HOME by default, which resolves to the
# near-full /cs/usr network home on compute nodes - force it into lab space instead.
export WANDB_CACHE_DIR=/sci/labs/efratmorin/liron.haris/wandb_cache
export WANDB_CONFIG_DIR=/sci/labs/efratmorin/liron.haris/.config/wandb

# 4. מעבר לתיקיית הפרויקט
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 5. יצירת תיקיית ריצות
mkdir -p runs

# 6. ניתוח MSE לפי שנה הידרולוגית עבור אגנים נבחרים (train/val/test/full) עם מודל מאומן
#    שימוש: sbatch cluster/run_MSE_analysis_basin.sh <config> <basin> [<basin> ...]
CONFIG_PATH="$1"
shift
if [ -z "$CONFIG_PATH" ] || [ $# -eq 0 ]; then
    echo "Usage: sbatch cluster/run_MSE_analysis_basin.sh <config> <basin> [<basin> ...]" >&2
    exit 1
fi
python src/MSE_analysis_basin.py --config "$CONFIG_PATH" --basins "$@"
