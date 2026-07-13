"""
Module: analyze_results.py
Description: Post-hoc Results Analysis Library for EA-LSTM Flood Forecasting.
             Consumes the visual_report_basin_*.csv files produced by
             test.py / quick_test.py to build cross-basin diagnostic plots.
             Run manually after quick_test.py / test.py has finished.
"""

import argparse
import glob
import os
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_config(yaml_path):
    """Load the YAML configuration file safely."""
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def _get_exp_dir(config):
    run_dir = config.get('run_dir', './runs/')
    return os.path.join(run_dir, config['experiment_name'])


def _save_figure(fig, exp_dir, filename):
    """Save a figure into {exp_dir}/analysis_plots/ and return the output path."""
    plots_dir = os.path.join(exp_dir, "analysis_plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, filename)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    return out_path


def _discover_test_basins(exp_dir):
    """Every basin with a visual_report CSV actually produced by test.py/quick_test.py."""
    report_paths = glob.glob(os.path.join(exp_dir, "visualization_reports", "visual_report_basin_*.csv"))
    pattern = re.compile(r"visual_report_basin_(.+)\.csv$")
    basins = []
    for path in report_paths:
        match = pattern.search(os.path.basename(path))
        if match:
            basins.append(match.group(1))
    return sorted(basins)


def compute_basin_nse(basin_id, exp_dir, lead):
    """
    NSE for one basin at one lead time, computed the same way as
    test.py:generate_basin_summary_table / quick_test.py:compute_nse_per_lead.
    Returns None if the report is missing or NSE is undefined (ss_tot == 0).
    """
    report_path = os.path.join(exp_dir, "visualization_reports", f"visual_report_basin_{basin_id}.csv")
    if not os.path.exists(report_path):
        return None

    df = pd.read_csv(report_path)
    actual_col = f"actual_lead_{lead}h"
    pred_col = f"pred_lead_{lead}h"
    if actual_col not in df.columns or pred_col not in df.columns:
        return None

    actuals_np = df[actual_col].to_numpy()
    preds_np = df[pred_col].to_numpy()
    ss_tot = float(np.sum((actuals_np - actuals_np.mean()) ** 2))
    if ss_tot == 0:
        return None
    ss_res = float(np.sum((actuals_np - preds_np) ** 2))
    return 1.0 - ss_res / ss_tot


def load_static_feature_values(feature_name, config):
    """
    Looks up a static feature's values, keyed by gauge_id. Checks
    config['static_attributes_file'] (raw physical units) first; if the
    feature isn't a column there, falls back to
    config['normalized_static_attributes_file'] - this is what lets a
    PCA-variant config (e.g. model_0_pca.yml, which points
    normalized_static_attributes_file at pca_projected_data.csv) resolve
    features like 'PC1'/'PC2' with no extra config needed.
    """
    checked_paths = []
    for file_key in ('static_attributes_file', 'normalized_static_attributes_file'):
        path = config.get(file_key)
        if not path:
            continue
        static_df = pd.read_csv(path, dtype={'gauge_id': str})
        if feature_name in static_df.columns:
            return dict(zip(static_df['gauge_id'], static_df[feature_name]))
        checked_paths.append((path, [c for c in static_df.columns if c != 'gauge_id']))

    details = "; ".join(f"{path} (available: {cols})" for path, cols in checked_paths)
    raise ValueError(f"Static feature '{feature_name}' not found. Checked: {details}")


def _build_nse_scatter_figure(nse_vals, feature_vals, static_feature, lead, subtitle, config):
    """Static matplotlib scatter: x = static feature value, y = NSE."""
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150, facecolor="#fafafa")
    ax.set_facecolor("#ffffff")

    ax.scatter(feature_vals, nse_vals, s=60, color='#2b8cbe',
               edgecolors='#1e1e1e', linewidths=0.5, zorder=3)

    ax.set_xlabel(static_feature)
    ax.set_ylabel(f"NSE (lead {lead}h)")
    ax.set_title(f"{config.get('experiment_name', '')}: NSE vs '{static_feature}' ({subtitle}), lead {lead}h")
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)

    fig.tight_layout()
    return fig


def plot_nse_by_static_feature(config, basin_ids=None):
    """
    Scatter plot per static feature x per forecast lead time x NSE sign:
    x = the static feature's value, y = NSE. Each (feature, lead) pair
    produces two figures - one for basins with negative NSE, one for the
    rest (NSE >= 0) - so a handful of badly negative basins don't compress
    the y-axis for the well-performing ones. Static features come from
    config['nse_scatter_static_features']. Saves to
    {exp_dir}/analysis_plots/nse_vs_{static_feature}_lead{lead}h_{negative|nonnegative}.png
    Returns a {feature: {lead: {'negative': fig, 'nonnegative': fig}}} dict
    (a group key is absent if that group had no basins to plot).
    """
    exp_dir = _get_exp_dir(config)

    if basin_ids is None:
        basin_ids = _discover_test_basins(exp_dir)
    if not basin_ids:
        print(f"[Warning] No visual_report_basin_*.csv files found under {exp_dir}. "
              f"Run test.py or quick_test.py first.")
        return {}

    all_figs = {}
    for static_feature in config['nse_scatter_static_features']:
        feature_values = load_static_feature_values(static_feature, config)

        figs = {}
        for lead in config['forecast_lead_times']:
            basins, nse_vals, feature_vals = [], [], []
            for basin_id in sorted(basin_ids):
                nse = compute_basin_nse(basin_id, exp_dir, lead)
                if nse is None:
                    print(f"  [Warning] Skipping basin {basin_id}: no usable NSE at lead {lead}h.")
                    continue
                if basin_id not in feature_values or pd.isna(feature_values[basin_id]):
                    print(f"  [Warning] Skipping basin {basin_id}: no '{static_feature}' value.")
                    continue
                basins.append(basin_id)
                nse_vals.append(nse)
                feature_vals.append(feature_values[basin_id])

            if not basins:
                print(f"[Warning] Lead {lead}h: no basins had both an NSE value and a "
                      f"'{static_feature}' value. Skipping.")
                continue

            groups = [
                ('negative', 'NSE < 0'),
                ('nonnegative', 'NSE ≥ 0'),
            ]
            group_figs = {}
            for group_key, subtitle in groups:
                if group_key == 'negative':
                    idxs = [i for i, v in enumerate(nse_vals) if v < 0]
                else:
                    idxs = [i for i, v in enumerate(nse_vals) if v >= 0]

                if not idxs:
                    print(f"[Warning] Lead {lead}h, '{static_feature}': no basins with {subtitle}. Skipping.")
                    continue

                group_nse = [nse_vals[i] for i in idxs]
                group_feat = [feature_vals[i] for i in idxs]

                fig = _build_nse_scatter_figure(group_nse, group_feat,
                                                 static_feature, lead, subtitle, config)

                filename = f"nse_vs_{static_feature}_lead{lead}h_{group_key}.png"
                out_path = _save_figure(fig, exp_dir, filename)
                print(f"[INFO] Saved scatter plot to: {out_path}")

                group_figs[group_key] = fig

            figs[lead] = group_figs

        all_figs[static_feature] = figs

    return all_figs


def _confusion_counts(actual_np, pred_np, threshold):
    """TP/FP/FN counts from per-timestep boolean masks, extending
    test.py:calculate_threshold_metrics's TP/FP logic with the missing FN term."""
    actual_mask = actual_np >= threshold
    pred_mask = pred_np >= threshold
    tp = int((actual_mask & pred_mask).sum())
    fp = int((~actual_mask & pred_mask).sum())
    fn = int((actual_mask & ~pred_mask).sum())
    return tp, fp, fn


def compute_basin_classification_metrics(basin_id, exp_dir, lead, prediction_rp):
    """
    Precision/Recall for one basin at one lead time, using the canonical
    prediction_threshold return period's threshold already baked into the
    report by test.py:build_and_export_report. Returns None if the report
    or required columns are missing.
    """
    report_path = os.path.join(exp_dir, "visualization_reports", f"visual_report_basin_{basin_id}.csv")
    if not os.path.exists(report_path):
        return None

    df = pd.read_csv(report_path)
    actual_col = f"actual_lead_{lead}h"
    pred_col = f"pred_lead_{lead}h"
    thresh_col = f"threshold_{prediction_rp}yr_rp"
    if actual_col not in df.columns or pred_col not in df.columns or thresh_col not in df.columns:
        return None

    threshold = df[thresh_col].iloc[0]
    tp, fp, fn = _confusion_counts(df[actual_col].to_numpy(), df[pred_col].to_numpy(), threshold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')

    return {'precision': precision, 'recall': recall}


def compute_classification_metrics_by_lead(config, basin_ids=None):
    """
    Precision/Recall per basin per lead time, then the mean across basins
    per lead, then the mean across leads (one model-level number per metric).
    Also computes model_variance: the pooled variance of precision and of
    recall across every individual basin x lead value (not the variance of
    the small per-lead-means series). Saves a single CSV to
    {exp_dir}/analysis_plots/classification_metrics.csv with per-basin rows,
    one 'lead_mean' row per lead, and final 'model_mean'/'model_variance'
    rows. Returns (per_basin_df, per_lead_means_df, model_means, model_variance).
    """
    exp_dir = _get_exp_dir(config)

    if basin_ids is None:
        basin_ids = _discover_test_basins(exp_dir)
    if not basin_ids:
        print(f"[Warning] No visual_report_basin_*.csv files found under {exp_dir}. "
              f"Run test.py or quick_test.py first.")
        return None

    prediction_rp = config['prediction_threshold']

    per_basin_rows = []
    per_lead_mean_rows = []
    for lead in config['forecast_lead_times']:
        lead_rows = []
        for basin_id in sorted(basin_ids):
            metrics = compute_basin_classification_metrics(basin_id, exp_dir, lead, prediction_rp)
            if metrics is None:
                print(f"  [Warning] Skipping basin {basin_id} at lead {lead}h: report or columns missing.")
                continue
            row = {'basin_id': basin_id, 'lead': lead, **metrics}
            per_basin_rows.append(row)
            lead_rows.append(metrics)

        if not lead_rows:
            print(f"[Warning] Lead {lead}h: no basins produced usable classification metrics.")
            continue

        lead_df = pd.DataFrame(lead_rows)
        lead_mean = {'basin_id': 'lead_mean', 'lead': lead,
                     'precision': lead_df['precision'].mean(), 'recall': lead_df['recall'].mean()}
        per_lead_mean_rows.append(lead_mean)

    per_basin_df = pd.DataFrame(per_basin_rows)
    per_lead_means_df = pd.DataFrame(per_lead_mean_rows)
    model_means = per_lead_means_df[['precision', 'recall']].mean()
    model_variance = per_basin_df[['precision', 'recall']].var()

    model_mean_row = pd.DataFrame([{'basin_id': 'model_mean', 'lead': None,
                                     'precision': model_means['precision'], 'recall': model_means['recall']}])
    model_variance_row = pd.DataFrame([{'basin_id': 'model_variance', 'lead': None,
                                         'precision': model_variance['precision'], 'recall': model_variance['recall']}])
    combined_df = pd.concat([per_basin_df, per_lead_means_df, model_mean_row, model_variance_row], ignore_index=True)

    plots_dir = os.path.join(exp_dir, "analysis_plots")
    os.makedirs(plots_dir, exist_ok=True)
    csv_path = os.path.join(plots_dir, "classification_metrics.csv")
    combined_df.to_csv(csv_path, index=False)

    print("\nClassification metrics — per-lead means:")
    print(per_lead_means_df.to_string(index=False))
    print(f"\nModel mean — precision: {model_means['precision']:.4f}, recall: {model_means['recall']:.4f}")
    print(f"Model variance — precision: {model_variance['precision']:.4f}, recall: {model_variance['recall']:.4f}")
    print(f"Saved to: {csv_path}\n")

    return per_basin_df, per_lead_means_df, model_means, model_variance


def main(config=None, basin_ids=None):
    if config is None:
        config = load_config("configs/config.yml")

    use_wandb = config.get('use_wandb', False) and WANDB_AVAILABLE
    if use_wandb:
        api_key = config.get('wandb_api_key')
        if api_key:
            wandb.login(key=api_key)
        wandb.init(
            project=config.get('wandb_project', 'flash-floods-israel'),
            name=f"{config['experiment_name']}_analyze_results",
            config=config,
        )

    figs = plot_nse_by_static_feature(config, basin_ids=basin_ids)
    if use_wandb:
        for static_feature, lead_figs in figs.items():
            for lead, group_figs in lead_figs.items():
                for group_key, fig in group_figs.items():
                    key = f"analyze_results/scatter/{static_feature}/lead_{lead}h_{group_key}"
                    wandb.log({key: wandb.Image(fig)})
                    plt.close(fig)

    result = compute_classification_metrics_by_lead(config, basin_ids=basin_ids)
    if use_wandb and result is not None:
        _, per_lead_means_df, model_means, model_variance = result
        for _, row in per_lead_means_df.iterrows():
            lead = int(row['lead'])
            wandb.log({
                f"analyze_results/classification/precision_lead_{lead}h": row['precision'],
                f"analyze_results/classification/recall_lead_{lead}h": row['recall'],
            })
        wandb.run.summary['classification_mean_precision'] = model_means['precision']
        wandb.run.summary['classification_mean_recall'] = model_means['recall']
        wandb.run.summary['classification_variance_precision'] = model_variance['precision']
        wandb.run.summary['classification_variance_recall'] = model_variance['recall']

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze quick_test.py/test.py results: NSE-by-static-feature scatter plots "
                     "(config['nse_scatter_static_features']) and precision/recall classification metrics.")
    parser.add_argument("--config", type=str, default="configs/config.yml",
                         help="Path to the YAML config file matching the run to analyze.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(config=cfg)
