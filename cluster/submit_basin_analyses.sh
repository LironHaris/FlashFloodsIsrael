#!/bin/bash
# Submits the per-basin analysis (cluster/run_basin_analysis.sh) of every finished family summary
# (configs/summarize_family_test_periods_<family>_<variant>_eval_<mask>.yml, 22 configs).
# Usage (from the project root): bash cluster/submit_basin_analyses.sh

cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

for config in configs/summarize_family_test_periods_*_eval_*.yml; do
    name=$(basename "$config" .yml | sed 's/^summarize_family_test_periods_/basin_/')
    id=$(sbatch --parsable --job-name="$name" cluster/run_basin_analysis.sh "$config") || exit 1
    echo "$config -> $id"
done
