#!/bin/bash
# Submits one run_train.sh job per final-training config made by src/make_final_train_configs.py.
# Usage (from the project root):
#   bash cluster/submit_final_trains.sh                          # all 125: train_flow_0_6_* + train_flow_normal_*
#   bash cluster/submit_final_trains.sh 'train_flow_normal_0_6_*'  # a subset (glob inside configs/)

cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

if [ -n "$1" ]; then
    PATTERNS=("configs/$1.yml")
else
    PATTERNS=("configs/train_flow_0_6_20*.yml" "configs/train_flow_normal_*_20*.yml")
fi

for pattern in "${PATTERNS[@]}"; do
    for config in $pattern; do
        [ -f "$config" ] || { echo "no match: $pattern"; continue; }
        echo "$config -> $(sbatch --parsable --job-name="$(basename "$config" .yml)" cluster/run_train.sh "$config")"
    done
done
