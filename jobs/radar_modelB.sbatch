#!/bin/bash
# Radar Model B (1a / 1b) on Moriah: prepare inputs (once) -> train -> test.
# Settings follow Liron's cluster/run_train.sh (1 GPU, 4 CPUs, 32G, 24h, conda env `flashfloods`),
# with all caches and logs under Ariel's lab folder.
#
#   cd /sci/labs/efratmorin/ariel.monzon/FlashFloodsIsrael_radar
#   sbatch --export=ALL,RUN=radar_1a jobs/radar_modelB.sbatch
#   sbatch --export=ALL,RUN=radar_1b jobs/radar_modelB.sbatch      # after 1a has prepared the inputs
#   (other GPU partition: add  --partition=catfish  before jobs/...)

#SBATCH --job-name=radar_modelB
#SBATCH --output=/sci/labs/efratmorin/ariel.monzon/SR_radar/logs/%x_%j.out
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -eo pipefail
RUN="${RUN:-radar_1a}"
LAB=/sci/labs/efratmorin/ariel.monzon
REPO="${SLURM_SUBMIT_DIR:-$LAB/FlashFloodsIsrael_radar}"
cd "$REPO"

# --- environment (Liron's conda env, read-only use) ---------------------------------------
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods
export MPLCONFIGDIR=$LAB/.matplotlib_cache
export WANDB_CACHE_DIR=$LAB/wandb_cache
export WANDB_CONFIG_DIR=$LAB/.config/wandb
export WANDB_DIR=$LAB/SR_radar/wandb
mkdir -p "$MPLCONFIGDIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DIR" "$LAB/SR_radar/logs"
# W&B login: run `wandb login` once (with the same WANDB_CONFIG_DIR exported) or set WANDB_API_KEY.

export FLOODS_CONFIG="configs/${RUN}.yml"
echo "[job] run=${RUN} config=${FLOODS_CONFIG} host=$(hostname) start=$(date)"
echo "[job] python=$(which python)"; nvidia-smi || true

# 1) model-ready inputs (shared by 1a and 1b; skipped if already prepared)
INPUTS_DIR=$(python -c "import yaml,os;print(yaml.safe_load(open(os.environ['FLOODS_CONFIG']))['processed_timeseries_dir'])")
if [ "$(ls "$INPUTS_DIR"/*.csv 2>/dev/null | wc -l)" -lt 53 ]; then
  echo "[job] preparing radar inputs -> ${INPUTS_DIR}"
  python src/prepare_radar_inputs.py
fi

# 2) train, 3) test
python src/train.py
python src/test.py
echo "[job] done $(date)"
