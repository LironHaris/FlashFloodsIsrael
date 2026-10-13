"""
Module: NSE_CDF_compare_test_periods.py
Description: NSE empirical CDF comparison of ONE model setup (same lead time)
             trained and tested on DIFFERENT test periods - e.g. the five
             best_crossval_0_3_<test years> models. The counterpart of
             model_compare_test.py, which compares different models on the
             SAME test split: here test periods and test basin lists are
             expected to differ, so each model is scored on its own test
             split and its own test_basin_file. Only the NSE CDF (+ per-basin
             and summary CSVs) is produced - no hydrographs, flood events or
             peaks.

             Per model, the per-basin reports
             (run_dir/experiment_name/visualization_reports/visual_report_basin_*.csv,
             written by test.py / model_compare_test.py) are reused when they
             exist; otherwise - or with --reevaluate - the model is run over its
             test split first (model_compare_test.run_single_model_eval).
             NSE is computed exactly as model_compare_test.py does (report_nse).

             Outputs in comparison run_dir/comparison_name/:
               comparison_plots/nse_cdf_comparison.png
               nse_per_basin.csv   (label, test_years, basin_id, nse)
               nse_summary.csv     (label, test_years, n_basins, median/mean NSE,
                                    fraction of basins with NSE > 0 and > 0.5)

Usage:
  python src/NSE_CDF_compare_test_periods.py --config configs/NSE_CDF_compare_test_periods_0_3.yml [--reevaluate]
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')  # file-only backend: never needs an X display (cluster nodes)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from flow_quality_check import test_periods, get_hydrological_year
from model_compare_test import (load_config, _read_basin_list, run_single_model_eval,
                                report_nse, plot_nse_cdf_comparison)


def describe_test_years(config):
    """Hydrological years covered by the config's test split, e.g. '2015, 2016, 2020'."""
    start_month = config.get('hydro_year_start_month', 10)
    years = []
    for s, e in test_periods(config):
        first, last = get_hydrological_year(pd.DatetimeIndex([s, e]), start_month=start_month)
        years.extend(range(int(first), int(last) + 1))
    return ', '.join(str(y) for y in sorted(set(years)))


def validate_same_lead(model_config_paths, model_labels=None):
    """
    Lighter than model_compare_test.validate_comparable: every model has
    exactly one forecast_lead_times entry and they are all the SAME lead;
    labels are unique. Test periods / basin lists may differ (the point of
    this comparison). Returns (model_configs, model_labels, lead).
    """
    model_configs = [load_config(p) for p in model_config_paths]

    leads = {}
    for path, config in zip(model_config_paths, model_configs):
        config_leads = config.get('forecast_lead_times', [])
        if len(config_leads) != 1:
            raise ValueError(f"'{path}' has forecast_lead_times={config_leads} - expected exactly 1 entry.")
        leads[path] = config_leads[0]
    if len(set(leads.values())) > 1:
        details = ", ".join(f"{p}=+{l}h" for p, l in leads.items())
        raise ValueError(f"All models must share the same lead time for this comparison. Got: {details}")

    if model_labels is None:
        model_labels = [config['experiment_name'] for config in model_configs]
    if len(model_labels) != len(model_configs):
        raise ValueError(f"model_labels has {len(model_labels)} entries for {len(model_configs)} model_configs.")
    if len(set(model_labels)) != len(model_labels):
        raise ValueError(f"model_labels must be unique. Got: {model_labels}")

    return model_configs, model_labels, next(iter(leads.values()))


def main(comparison_config_path, reevaluate=False):
    comparison_config = load_config(comparison_config_path)
    model_configs, model_labels, lead = validate_same_lead(
        comparison_config['model_configs'], comparison_config.get('model_labels'))

    print("=" * 75)
    print(f"      NSE CDF across test periods - {len(model_configs)} models, lead +{lead}h")
    print("=" * 75)

    output_dir = os.path.join(comparison_config.get('run_dir', './runs/model_comparisons/'),
                              comparison_config['comparison_name'])
    os.makedirs(output_dir, exist_ok=True)

    per_basin_rows, summary_rows = [], []
    model_nse = {}
    for label, config in zip(model_labels, model_configs):
        basins = sorted(_read_basin_list(config['test_basin_file']))
        exp_dir = os.path.join(config.get('run_dir', './runs/'), config['experiment_name'])
        years = describe_test_years(config)

        n_reports = sum(os.path.exists(os.path.join(exp_dir, "visualization_reports", f"visual_report_basin_{b}.csv"))
                        for b in basins)
        if reevaluate or n_reports == 0:
            print(f"\n[INFO] '{label}' ({config['experiment_name']}): evaluating test hydro-years {years} "
                  f"on {len(basins)} basins{' (--reevaluate)' if reevaluate else ' (no existing reports)'}...")
            run_single_model_eval(config, basins)
        else:
            print(f"\n[INFO] '{label}' ({config['experiment_name']}): reusing existing reports for "
                  f"{n_reports}/{len(basins)} test basins (test hydro-years {years}).")

        nse_values = []
        for basin in basins:
            nse = report_nse(exp_dir, lead, basin)
            if nse is None:
                continue
            nse_values.append(nse)
            per_basin_rows.append({'label': label, 'test_years': years, 'basin_id': basin, 'nse': nse})
        model_nse[label] = nse_values

        values = np.array(nse_values)
        summary_rows.append({
            'label': label,
            'experiment_name': config['experiment_name'],
            'test_years': years,
            'n_test_basins': len(basins),
            'n_basins_with_nse': len(values),
            'median_nse': float(np.median(values)) if len(values) else np.nan,
            'mean_nse': float(np.mean(values)) if len(values) else np.nan,
            'frac_nse_gt_0': float(np.mean(values > 0)) if len(values) else np.nan,
            'frac_nse_gt_0.5': float(np.mean(values > 0.5)) if len(values) else np.nan,
        })
        print(f"[INFO] '{label}': NSE for {len(values)} basins, median "
              f"{summary_rows[-1]['median_nse']:.3f}")

    per_basin_path = os.path.join(output_dir, "nse_per_basin.csv")
    pd.DataFrame(per_basin_rows, columns=['label', 'test_years', 'basin_id', 'nse']).to_csv(per_basin_path, index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "nse_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[INFO] Wrote {per_basin_path} and {summary_path}")
    print(summary_df[['label', 'test_years', 'n_basins_with_nse', 'median_nse', 'mean_nse',
                      'frac_nse_gt_0', 'frac_nse_gt_0.5']].to_string(index=False))

    model_leads = {label: lead for label in model_labels}
    cdf_fig = plot_nse_cdf_comparison(model_nse, model_leads, comparison_config, output_dir)

    if comparison_config.get('use_wandb', False) and WANDB_AVAILABLE:
        api_key = comparison_config.get('wandb_api_key')
        if api_key:
            wandb.login(key=api_key)
        wandb.init(
            project=comparison_config.get('wandb_project', 'flash-floods-israel'),
            name=f"{comparison_config['comparison_name']}_nse_periods",
            job_type='nse_test_periods_compare',
            config=comparison_config,
        )
        log = {'nse_summary': wandb.Table(dataframe=summary_df)}
        if cdf_fig is not None:
            log['compare/nse_cdf'] = wandb.Image(cdf_fig)
        wandb.log(log)
        wandb.finish()

    if cdf_fig is not None:
        plt.close(cdf_fig)
    print(f"\n[INFO] NSE CDF test-period comparison completed. Outputs in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE CDF comparison of one model setup across different test periods.")
    parser.add_argument("--config", type=str, required=True,
                        help="Comparison config YAML (comparison_name, model_configs, model_labels, ...).")
    parser.add_argument("--reevaluate", action="store_true",
                        help="Re-run every model over its test split even if its reports already exist.")
    args = parser.parse_args()
    main(args.config, reevaluate=args.reevaluate)
