#!/bin/bash
#SBATCH --job-name=ealstm_test
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/test_eval_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/test_eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# 1. ����� ������
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods

# 2. ����� ����
export HOME=/sci/labs/efratmorin/liron.haris/
export MPLCONFIGDIR=/sci/labs/efratmorin/liron.haris/.matplotlib_cache
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 3. ���� ������ (SWEEP_ID �����, TOP_N/CONFIG_PATH ��������)
SWEEP_ID="${1:?Usage: sbatch run_sweep_test.sh <sweep_id> [top_n] [config_path]}"
TOP_N="${2:-3}"
CONFIG_PATH="${3:-configs/config.yml}"
python src/test_sweep.py "$SWEEP_ID" --top "$TOP_N" --config "$CONFIG_PATH"