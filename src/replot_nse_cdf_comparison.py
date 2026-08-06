"""
Module: replot_nse_cdf_comparison.py
Description: Regenerates the NSE CDF comparison plot, the flood-event
             comparison CSV, and the peaks analysis (detail + summary) CSVs
             from an already-run model_compare_test.py's saved per-model
             report CSVs, without re-running model inference.
"""

import argparse
import os
import matplotlib.pyplot as plt

from model_compare_test import (
    load_config, validate_comparable, compute_model_nse, plot_nse_cdf_comparison,
    run_event_and_peaks_analysis,
)


def main(comparison_config_path="configs/compare_model_0_leads.yml"):
    comparison_config = load_config(comparison_config_path)
    model_config_paths = comparison_config['model_configs']
    model_labels_cfg = comparison_config.get('model_labels')

    model_configs, model_labels, shared_basins, model_leads = validate_comparable(
        model_config_paths, model_labels_cfg
    )

    output_dir = os.path.join(comparison_config.get('run_dir', './runs/model_comparisons/'),
                               comparison_config['comparison_name'])

    model_nse = compute_model_nse(model_configs, model_labels, model_leads, shared_basins)
    cdf_fig = plot_nse_cdf_comparison(model_nse, model_leads, comparison_config, output_dir)
    if cdf_fig is None:
        print("[WARNING] No valid NSE values found in the saved reports - "
              "has model_compare_test.py been run for this config yet?")
    else:
        plt.close(cdf_fig)

    run_event_and_peaks_analysis(model_configs, model_labels, model_leads, shared_basins,
                                  output_dir, use_wandb=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate the NSE CDF comparison plot, flood-event comparison CSV, and "
                     "peaks analysis CSVs from an already-run model_compare_test.py's saved "
                     "per-model report CSVs (no re-inference)."
    )
    parser.add_argument("--config", type=str, default="configs/compare_model_0_leads.yml",
                         help="Path to the same comparison config YAML used with model_compare_test.py.")
    args = parser.parse_args()
    main(args.config)
