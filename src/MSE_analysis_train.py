"""
Module: MSE_analysis_train.py
Description: Runs an already-trained model (run_dir/experiment_name/best_model.pt)
             once over the train split - every train period clipped to start no
             earlier than hydrological year 2000 - and writes the per-basin /
             per-hydrological-year loss breakdown (CSVs + MSE pies, top-N
             basins) followed by the train-set usage audit. See MSE_analysis.py.
"""

import argparse

from MSE_analysis import run_mse_analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train-split MSE analysis of a trained model: per-basin/per-year loss "
                     "breakdown (CSVs + pie charts) and usage audit (basin-hour pairs used, "
                     "non-zero, prediction-threshold-crossed) for the top-N basins by train loss. "
                     "Train periods are clipped to start no earlier than hydrological year 2000.")
    parser.add_argument("--config", type=str, default="configs/config.yml",
                         help="Path to the YAML config file for this run.")
    args = parser.parse_args()
    run_mse_analysis(args.config, 'train')
