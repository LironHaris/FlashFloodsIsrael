#!/bin/bash
#SBATCH --job-name=flow_type_intervals
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/logs_%x_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/logs_%x_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00

# One-time precompute of per-station Flow_type intervals (flow / flow_normal) for training-time
# masks (train_flow_type_mask in the cross_val_flow_* / train_flow_* configs). Writes
# data/processed/flow_type_intervals.csv from data/water authority/hydrographs_unified.csv.

# 1. Conda environment
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods

# 2. Project directory
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 3. Run
python src/build_flow_type_intervals.py
