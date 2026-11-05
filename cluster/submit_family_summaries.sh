#!/bin/bash
# Submits the 22 family summaries (configs/summarize_family_test_periods_<family>_<variant>_eval_<mask>.yml).
# 1. One --reports-only job per (family, model set) evaluates every member model that has no per-basin
#    reports yet - the flow / top1 / best_crossval models are shared by several summaries, so they must
#    not be evaluated by parallel jobs at the same time.
# 2. The 22 summaries start once all of those finished OK (--dependency=afterok).
# Usage (from the project root): bash cluster/submit_family_summaries.sh

cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# avg3 covers every top1-3 member; flow / best_crossval have one config per lead
REPORT_CONFIGS=()
for family in model_0 model_1_3; do
    for variant in flow_eval_all flow_normal_avg3_eval_all best_crossval_eval_flow; do
        REPORT_CONFIGS+=("configs/summarize_family_test_periods_${family}_${variant}.yml")
    done
done

report_ids=()
for config in "${REPORT_CONFIGS[@]}"; do
    name="reports_$(basename "$config" .yml | sed 's/^summarize_family_test_periods_//')"
    id=$(sbatch --parsable --job-name="$name" cluster/run_summarize_family_test_periods.sh "$config" --reports-only) || exit 1
    echo "$config (--reports-only) -> $id"
    report_ids+=("$id")
done
dependency="afterok:$(IFS=:; echo "${report_ids[*]}")"

for config in configs/summarize_family_test_periods_*_eval_*.yml; do
    name=$(basename "$config" .yml | sed 's/^summarize_family_test_periods_/family_/')
    id=$(sbatch --parsable --dependency="$dependency" --job-name="$name" \
         cluster/run_summarize_family_test_periods.sh "$config") || exit 1
    echo "$config -> $id"
done
