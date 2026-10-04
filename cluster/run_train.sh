#!/bin/bash
#SBATCH --job-name=ealstm_flood_train
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/logs_%x_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/logs_%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# 1. ����� �-Conda ������ ������
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh

# 2. �������� �� ����� �������
conda activate flashfloods

# 3. ����� ����� ����� ����: ���� ������ ��� �� Matplotlib ������
export MPLCONFIGDIR=/sci/labs/efratmorin/liron.haris/.matplotlib_cache

# 3b. wandb writes its cache/config to $HOME by default, which resolves to the
# near-full /cs/usr network home on compute nodes - force it into lab space instead.
export WANDB_CACHE_DIR=/sci/labs/efratmorin/liron.haris/wandb_cache
export WANDB_CONFIG_DIR=/sci/labs/efratmorin/liron.haris/.config/wandb

# 4. ���� ������� �������
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 5. ����� ������ ������
mkdir -p runs

# 6. ����� ����� �� ������
CONFIG_PATH="${1:-configs/config.yml}"
python src/train.py --config "$CONFIG_PATH"