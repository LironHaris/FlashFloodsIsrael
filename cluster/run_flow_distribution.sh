#!/bin/bash
#SBATCH --job-name=flow_distribution
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/logs_%x_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/logs_%x_%j.err
#SBATCH --gres=gpu:1                    # CPU-only run, but GPU partitions are the ones that accept the job (else BadConstraints)
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

# Normalized-flow and per-month histograms of flow / normal / other Flow_type timesteps (src/flow_distribution.py).
# Needs the all-data preprocessing first:
#   pid=$(sbatch --parsable cluster/run_preprocess_dynamic.sh configs/preprocess_all_data.yml)
#   sbatch --dependency=afterok:$pid cluster/run_flow_distribution.sh configs/preprocess_all_data.yml

# 1. Conda environment
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods

# 2. Matplotlib / W&B caches in lab space (not the near-full home)
export MPLCONFIGDIR=/sci/labs/efratmorin/liron.haris/.matplotlib_cache
export WANDB_CACHE_DIR=/sci/labs/efratmorin/liron.haris/wandb_cache
export WANDB_CONFIG_DIR=/sci/labs/efratmorin/liron.haris/.config/wandb

# 3. Project directory
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 4. Run
CONFIG_PATH="${1:-configs/preprocess_all_data.yml}"
python src/flow_distribution.py --config "$CONFIG_PATH"
