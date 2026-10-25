#!/bin/bash
# Keeps submitting final-training configs (cluster/run_train.sh) as queue slots free up, until all are in.
# Runs on the login node - start it once inside tmux or with nohup:
#   tmux new -s autosubmit 'bash cluster/auto_submit_trains.sh'
#   nohup bash cluster/auto_submit_trains.sh > runs/auto_submit.log 2>&1 &
# Optional args: config globs inside configs/ (default: the 65 not yet submitted on 2026-10-25), e.g.
#   bash cluster/auto_submit_trains.sh 'train_flow_normal_1_3_6_*'
# Env overrides: MAX_JOBS (queue cap, default 60), SLEEP_SEC (default 300).
#
# A config counts as already submitted if a job with its name (submit_final_trains.sh / this script name
# jobs after the config) is in squeue or in sacct since SINCE - so re-running it never double-submits.

cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

MAX_JOBS="${MAX_JOBS:-60}"
SLEEP_SEC="${SLEEP_SEC:-300}"
SINCE="${SINCE:-2026-10-25}"

if [ "$#" -gt 0 ]; then
    GLOBS=("$@")
else
    GLOBS=('train_flow_normal_0_6_top3_*' 'train_flow_normal_1_3_0_*' 'train_flow_normal_1_3_12_*'
           'train_flow_normal_1_3_3_*' 'train_flow_normal_1_3_6_*')
fi

PENDING=()
for glob in "${GLOBS[@]}"; do
    for config in configs/$glob.yml; do
        [ -f "$config" ] && PENDING+=("$config") || echo "no match: configs/$glob.yml"
    done
done
echo "$(date '+%F %T') ${#PENDING[@]} configs to submit (MAX_JOBS=$MAX_JOBS, every ${SLEEP_SEC}s)"

while true; do
    known="$( { squeue -u "$USER" -h -o '%j'; sacct -u "$USER" -S "$SINCE" -X -n -P -o JobName; } | sort -u )"
    in_queue=$(squeue -u "$USER" -h | wc -l)
    remaining=0
    for config in "${PENDING[@]}"; do
        name=$(basename "$config" .yml)
        grep -qxF "$name" <<< "$known" && continue
        remaining=$((remaining + 1))
        [ "$in_queue" -ge "$MAX_JOBS" ] && continue
        if jobid=$(sbatch --parsable --job-name="$name" cluster/run_train.sh "$config" 2>&1); then
            echo "$(date '+%F %T') $config -> $jobid"
            in_queue=$((in_queue + 1))
            remaining=$((remaining - 1))
        else
            echo "$(date '+%F %T') sbatch refused $config: $jobid"
            in_queue=$MAX_JOBS  # limit hit earlier than MAX_JOBS - wait for the next round
        fi
    done
    if [ "$remaining" -eq 0 ]; then
        echo "$(date '+%F %T') all configs submitted - done."
        break
    fi
    echo "$(date '+%F %T') $remaining left, $in_queue jobs in queue - sleeping ${SLEEP_SEC}s"
    sleep "$SLEEP_SEC"
done
