#!/bin/bash
#SBATCH --job-name=plot_hydrographs
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

# 6. הרצת ציור ההידרוגרף העצמאי (basin/window מגיעים מ-config['plot_hydrographs']) -
#    דורש שכבר הופק visual_report_basin_<id>.csv עבור הקונפיג הזה, בין אם ע"י test.py
#    (תחת runs/<experiment_name>/) ובין אם ע"י model_compare_events_by_year.py /
#    plot_events_by_year.py (תחת runs/<experiment_name>_custom_period[_<years_tag>]/) -
#    במקרה השני העבירו את שם התיקייה המלא כארגומנט שני. ארגומנט שלישי אופציונלי
#    (output_experiment_name) שומר את ה-PNG/wandb run תחת תיקייה נפרדת משם, כך
#    שקריאה מדוח קיים לא כותבת לתוך התיקייה של אותו ניסוי.
# שימוש: sbatch run_plot_hydrographs.sh <config_path> [experiment_name_override] [output_experiment_name]
CONFIG_PATH="${1:-configs/config.yml}"
EXPERIMENT_NAME_OVERRIDE="${2:-}"
OUTPUT_EXPERIMENT_NAME="${3:-}"
ARGS=(--config "$CONFIG_PATH")
if [ -n "$EXPERIMENT_NAME_OVERRIDE" ]; then
    ARGS+=(--experiment-name "$EXPERIMENT_NAME_OVERRIDE")
fi
if [ -n "$OUTPUT_EXPERIMENT_NAME" ]; then
    ARGS+=(--output-experiment-name "$OUTPUT_EXPERIMENT_NAME")
fi
python src/plot_hydrographs.py "${ARGS[@]}"
