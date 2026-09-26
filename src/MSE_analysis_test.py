"""
Module: MSE_analysis_test.py
Description: Runs an already-trained model (run_dir/experiment_name/best_model.pt)
             once over the test split and writes the per-basin /
             per-hydrological-year loss breakdown (CSVs + MSE pies, top-N
             basins) followed by the test-set usage audit. See MSE_analysis.py.
"""

import argparse

from MSE_analysis import run_mse_analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test-split MSE analysis of a trained model: per-basin/per-year loss "
                     "breakdown (CSVs + pie charts) and usage audit (basin-hour pairs used, "
                     "non-zero, prediction-threshold-crossed) for the top-N basins by test loss.")
    parser.add_argument("--config", type=str, default="configs/config.yml",
                         help="Path to the YAML config file for this run.")
    args = parser.parse_args()
    run_mse_analysis(args.config, 'test')
