#!/bin/bash
#SBATCH --job-name=ealstm_test
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/test_eval_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/test_eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# 1. טעינת הסביבה
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods

# 2. מעקפי דיסק
export HOME=/sci/labs/efratmorin/liron.haris/
export MPLCONFIGDIR=/sci/labs/efratmorin/liron.haris/.matplotlib_cache
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 3. הרצת הניתוח (תחליף ל-ID של ה-Sweep שלך)
python src/test_sweep.py 4ey73tda --top 3