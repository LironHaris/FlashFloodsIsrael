#!/bin/bash
#SBATCH --job-name=ealstm_sweep
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/sweep_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/sweep_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# 1. ����� �-Conda �� ������ ������ ������
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods

# 2. ����� ����� ����� ���� ����� �����
export HOME=/sci/labs/efratmorin/liron.haris/
export MPLCONFIGDIR=/sci/labs/efratmorin/liron.haris/.matplotlib_cache

# 3. ���� ������� �������
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 4. SWEEP_ID is required (from `wandb sweep configs/sweep.yaml`); CONFIG_PATH picks
# which model's config each trial sweeps (defaults to configs/config.yml).
SWEEP_ID="${1:?Usage: sbatch run_sweep.sh <sweep_id> [config_path]}"
export FLASHFLOODS_CONFIG="${2:-configs/config.yml}"

# 5. ���� �-Agent
wandb agent liron-haris-hebrew-university-of-jerusalem/flash-floods-israel/$SWEEP_ID

