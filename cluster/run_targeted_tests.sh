#!/bin/bash
#SBATCH --job-name=targeted_test
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/targeted_test_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/targeted_test_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

if [ -z "$1" ]; then
    echo "[ERROR] Please provide a W&B Run ID. Usage: sbatch run_targeted_tests.sh <RUN_ID>"
    exit 1
fi

RUN_ID=$1
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods
export HOME=/sci/labs/efratmorin/liron.haris/
export MPLCONFIGDIR=/sci/labs/efratmorin/liron.haris/.matplotlib_cache

SWEEP_EXP_NAME="model_0_train_2" 
RUN_DIR="./runs/"
CHECKPOINT_SRC="${RUN_DIR}${SWEEP_EXP_NAME}/${RUN_ID}/best_model.pt"

# אנחנו מייצרים את תיקיית היעד המשולבת עם ה-ID של הריצה הנוכחית
TARGET_DIR="${RUN_DIR}${SWEEP_EXP_NAME}_eval_${RUN_ID}"

echo "[INFO] Preparing evaluation environment for Run ID: ${RUN_ID}"
if [ ! -f "$CHECKPOINT_SRC" ]; then
    echo "[ERROR] Checkpoint not found at ${CHECKPOINT_SRC}"
    exit 1
fi

# העתקת המשקולות למיקום הסטרילי שהמודל מצפה לו
mkdir -p "$TARGET_DIR"
cp "$CHECKPOINT_SRC" "$TARGET_DIR/best_model.pt"

# הרצת הצינור המלא (עובד ישירות מול ה-config.yml שערכת ידנית)
echo "[INFO] Spawning test.py pipeline..."
python src/test.py

echo "[SUCCESS] Evaluation finished for Run ID: ${RUN_ID}."