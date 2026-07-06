#!/bin/bash
#SBATCH --job-name=ealstm_flood_test
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/test_%x_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/test_%x_%j.err
#SBATCH --gres=gpu:1                    # ���� GPU ��� (��� ������ ����� �-CUDA ��� ������)
#SBATCH --cpus-per-task=4              # 4 ����� ���� ������ ������ �����
#SBATCH --mem=16G                       # 16 ���� ������ RAM (����� ������� ���� ���� ������ ������)
#SBATCH --time=24:00:00                 # ����� ��� �� ������ (�� ������ ��� ���� ������)

# 1. ����� �-Conda ������ ������
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh

# 2. �������� �� ����� �������
conda activate flashfloods

# 3. ����� ����� ����� �� Matplotlib (��� ����� ����� ����� �� ������� �-sci)
export MPLCONFIGDIR=/sci/labs/efratmorin/liron.haris/.matplotlib_cache

# 4. ����� ����� ����� ���� W&B (��� ���� ����� ��� ������ ������)
export HOME=/sci/labs/efratmorin/liron.haris/

# 5. ���� ������� �������
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 6. ���� ������ ������
CONFIG_PATH="${1:-configs/config.yml}"
python src/quick_test.py --config "$CONFIG_PATH"
