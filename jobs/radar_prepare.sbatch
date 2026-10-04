#!/bin/bash
# One-time preparation for the radar runs (CPU): model-ready radar inputs + extended static attributes.
# Run before the training jobs; they can depend on it:
#
#   PREP=$(sbatch --parsable jobs/radar_prepare.sbatch)
#   for v in SRradar_B_mean_mask SRradar_B_allstats_mask; do for k in 0 1 2 3; do
#     sbatch --dependency=afterok:$PREP -J ${v}_L$k --export=ALL,RUN=${v}_L$k jobs/radar_modelB.sbatch
#   done; done

#SBATCH --job-name=SRradar_prepare
#SBATCH --output=/sci/labs/efratmorin/ariel.monzon/SR_radar/logs/%x_%j.out
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -eo pipefail
LAB=/sci/labs/efratmorin/ariel.monzon
cd "${SLURM_SUBMIT_DIR:-$LAB/FlashFloodsIsrael_radar}"
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh
conda activate flashfloods
export FLOODS_CONFIG=configs/SRradar_B_mean_mask_L0.yml     # data settings are identical in all 8 configs

python src/prepare_radar_inputs.py
python src/extend_static_attributes.py
echo "[prepare] done $(date)"
