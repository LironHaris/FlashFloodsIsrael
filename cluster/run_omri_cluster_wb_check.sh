#!/bin/bash
#SBATCH --job-name=omri_cluster_wb_check
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

# 3b. wandb writes its cache/config to $HOME by default, which resolves to the
# near-full /cs/usr network home on compute nodes - force it into lab space instead.
export WANDB_CACHE_DIR=/sci/labs/efratmorin/liron.haris/wandb_cache
export WANDB_CONFIG_DIR=/sci/labs/efratmorin/liron.haris/.config/wandb

# 4. מעבר לתיקיית הפרויקט
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 5. יצירת תיקיית ריצות
mkdir -p runs

# 6. בדיקת קישוריות cluster<->wandb, בבידוד מלא: משתמש במשקלים המאומנים של
#    best_model_0_2_no_dry_no_nan (checkpoint_path בקונפיג, נקרא בלבד) אבל כותב
#    הכל תחת ניסוי חדש ומבודד runs/omri_cluster_wb_check/ - לא דורס אף ריצה קיימת.
#    שלב א': quick_test.py מפיק את visual_report_basin_il_8155.csv (אגן בודד).
#    שלב ב': plot_hydrographs.py מצייר את אירוע השיטפון שהוגדר ומעלה אותו ל-wandb.
python src/quick_test.py --config configs/omri_cluster_wb_check.yml
python src/plot_hydrographs.py --config configs/omri_cluster_wb_check.yml
