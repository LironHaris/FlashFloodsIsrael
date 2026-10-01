#!/bin/bash
# Radar Model B (1a / 1b): prepare inputs (once) -> train -> test.
#
#   sbatch --export=ALL,RUN=radar_1a jobs/radar_modelB.sbatch
#   sbatch --export=ALL,RUN=radar_1b jobs/radar_modelB.sbatch
#
# Lines marked FILL IN depend on the lab's training protocol (partition, GPU request, env).

#SBATCH --job-name=radar_modelB
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=24:00:00            # FILL IN if the protocol uses another limit
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1               # FILL IN (e.g. gpu:a100:1) per protocol
##SBATCH --partition=FILL_IN       # FILL IN and uncomment if a partition is required

set -euo pipefail
RUN="${RUN:-radar_1a}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

# --- environment: FILL IN with the protocol's activation line ---------------------------
# e.g. source /sci/labs/efratmorin/<user>/envs/floods/bin/activate   or   conda activate floods
# ------------------------------------------------------------------------------------------

export FLOODS_CONFIG="configs/${RUN}.yml"
echo "[job] run=${RUN} config=${FLOODS_CONFIG} host=$(hostname) start=$(date)"
nvidia-smi || true

# W&B: needs `wandb login` done once on the cluster, or WANDB_API_KEY in the environment.

# 1) model-ready inputs (shared by 1a and 1b; skipped if already prepared)
INPUTS_DIR=$(python -c "import yaml,os;print(yaml.safe_load(open(os.environ['FLOODS_CONFIG']))['processed_timeseries_dir'])")
if [ ! -f "${INPUTS_DIR}/il_17110.csv" ]; then
  echo "[job] preparing radar inputs -> ${INPUTS_DIR}"
  python src/prepare_radar_inputs.py
fi

# 2) train, 3) test
python src/train.py
python src/test.py
echo "[job] done $(date)"
