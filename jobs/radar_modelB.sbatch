#!/bin/bash
# Radar Model B (SRradar_B_mean_mask = 1a, SRradar_B_allstats_mask = 1b) on Moriah: prepare inputs (once) -> train -> test.
# Settings follow Liron's cluster/run_train.sh (1 GPU, 4 CPUs, 32G, 24h, conda env `flashfloods`),
# with all caches and logs under Ariel's lab folder.
#
#   cd /sci/labs/efratmorin/ariel.monzon/FlashFloodsIsrael_radar
#   sbatch --export=ALL,RUN=SRradar_B_mean_mask jobs/radar_modelB.sbatch
#   sbatch --export=ALL,RUN=SRradar_B_allstats_mask jobs/radar_modelB.sbatch      # after 1a has prepared the inputs
#   (other GPU partition: add  --partition=catfish  before jobs/...)

#SBATCH --job-name=radar_modelB
#SBATCH --output=/sci/labs/efratmorin/ariel.monzon/SR_radar/logs/%x_%j.out
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -eo pipefail
RUN="${RUN:-SRradar_B_mean_mask}"
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
# W&B login: compute nodes may not see the home folder (/cs/usr/...), so keep the credentials in the lab
# folder:  cp ~/.netrc /sci/labs/efratmorin/ariel.monzon/.netrc && chmod 600 /sci/labs/efratmorin/ariel.monzon/.netrc
if [ -f "$LAB/.netrc" ]; then export NETRC="$LAB/.netrc"; fi

export FLOODS_CONFIG="configs/${RUN}.yml"
echo "[job] run=${RUN} config=${FLOODS_CONFIG} host=$(hostname) start=$(date)"
echo "[job] python=$(which python)"; nvidia-smi || true

# inputs and statics are built once by jobs/radar_prepare.sbatch (submit it first; see its header)
INPUTS_DIR=$(python -c "import yaml,os;print(yaml.safe_load(open(os.environ['FLOODS_CONFIG']))['processed_timeseries_dir'])")
if [ "$(ls "$INPUTS_DIR"/*.csv 2>/dev/null | wc -l)" -lt 1 ]; then
  echo "[job] ERROR: no prepared inputs in ${INPUTS_DIR} - run jobs/radar_prepare.sbatch first"; exit 1
fi

# train, then test
python src/train.py
python src/test.py
echo "[job] done $(date)"
