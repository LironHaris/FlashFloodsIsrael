"""
Module: model_compare_test.py
Description: Compares multiple single-lead-time models on the same test split.
             Produces one NSE empirical CDF with all models overlaid, per-basin
             hydrographs with all models' predictions + real flow overlaid (all
             aligned on real-world target time, not forecast-issuance time),
             and flood-event TP/FN/FP/TN classified per model over the union of
             every event any model (or the real flow) produced - the same
             event window can carry a different label per model.
             Every compared model must have exactly one forecast_lead_times
             entry - this script does not support multi-lead-time models.
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from dataset import IsraelBasinsDataset
from quick_test import setup_evaluation_from_checkpoint
from test import evaluate_basin_sequences, build_and_export_report
import find_flood_events as ffe
import plot_hydrographs as ph


def load_config(yaml_path):
    """Load the YAML configuration file safely."""
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def _read_basin_list(path):
    with open(path, encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}


# Config keys that define the evaluation methodology itself (RP thresholds,
# buffer/merge windows, test window) - must be identical across every
# compared model, or the comparison is apples-to-oranges.
VALIDATE_IDENTICAL_KEYS = [
    'test_start_date', 'test_end_date', 'prediction_threshold',
    'return_periods_years', 'event_merge_gap_hours', 'visual_buffer_days',
    'hourly_flow_return_periods_combined',
]


def validate_comparable(model_config_paths, model_labels=None):
    """
    Loads every model config and validates they're comparable:
      - each has exactly one forecast_lead_times entry
      - VALIDATE_IDENTICAL_KEYS match exactly across all of them
      - their test_basin_file basin-ID sets are identical
    Raises ValueError naming the offending config(s) on any mismatch.
    Returns (model_configs, model_labels, shared_basins, model_leads).
    """
    model_configs = [load_config(p) for p in model_config_paths]

    for path, config in zip(model_config_paths, model_configs):
        leads = config.get('forecast_lead_times', [])
        if len(leads) != 1:
            raise ValueError(
                f"model_compare_test.py doesn't support multi lead times right now: "
                f"'{path}' has forecast_lead_times={leads} (expected exactly 1 entry)."
            )

    for key in VALIDATE_IDENTICAL_KEYS:
        values = [config.get(key) for config in model_configs]
        if len(set(str(v) for v in values)) > 1:
            details = ", ".join(f"{p}={v}" for p, v in zip(model_config_paths, values))
            raise ValueError(
                f"model_compare_test.py requires identical '{key}' across all compared "
                f"models for a fair comparison. Got: {details}"
            )

    basin_sets = [(_read_basin_list(config['test_basin_file']), path)
                  for config, path in zip(model_configs, model_config_paths)]
    reference_set, reference_path = basin_sets[0]
    for basin_set, path in basin_sets[1:]:
        if basin_set != reference_set:
            diff = reference_set.symmetric_difference(basin_set)
            raise ValueError(
                f"model_compare_test.py requires identical test_basin_file basin sets across "
                f"all compared models (otherwise the NSE CDF comparison isn't fair). "
                f"'{reference_path}' and '{path}' disagree on: {sorted(diff)}"
            )

    if model_labels is None:
        model_labels = [config['experiment_name'] for config in model_configs]
    if len(set(model_labels)) != len(model_labels):
        raise ValueError(f"model_labels must be unique. Got: {model_labels}")

    shared_basins = sorted(reference_set)
    model_leads = {label: config['forecast_lead_times'][0]
                   for label, config in zip(model_labels, model_configs)}

    return model_configs, model_labels, shared_basins, model_leads


def run_single_model_eval(config, basins):
    """
    Evaluates one model over the given basin list, writing its own report
    CSVs exactly as quick_test.py does (weights from config['checkpoint_path']).
    Returns exp_dir.
    """
    model, device, exp_dir = setup_evaluation_from_checkpoint(config)
    output_dir = os.path.join(exp_dir, "visualization_reports")
    os.makedirs(output_dir, exist_ok=True)

    test_dataset = IsraelBasinsDataset(split_type='test', config=config,
                                       use_basin_splits=config.get('use_basin_splits', True))

    for basin in basins:
        basin_data = evaluate_basin_sequences(basin, test_dataset, model, device, config)
        if basin_data is None:
            print(f"  [WARNING] No continuous test windows found for basin {basin} "
                  f"({config['experiment_name']}). Skipping.")
            continue
        timestamps, actual_leads_dict, pred_leads_dict = basin_data
        build_and_export_report(basin, output_dir, timestamps, actual_leads_dict, pred_leads_dict, config)

    return exp_dir


def compute_model_nse(model_configs, model_labels, model_leads, shared_basins):
    """
    Per-model, per-basin NSE using that model's own raw report
    (actual_lead_{lead}h vs pred_lead_{lead}h from the same row - unaffected
    by the target-time shift used elsewhere in this script, since both
    values already correspond to the same target moment). Returns
    {label: [nse_per_basin...]}.
    """
    model_nse = {label: [] for label in model_labels}
    for label, config in zip(model_labels, model_configs):
        exp_dir = os.path.join(config.get('run_dir', './runs/'), config['experiment_name'])
        lead = model_leads[label]
        for basin in shared_basins:
            report_path = os.path.join(exp_dir, "visualization_reports", f"visual_report_basin_{basin}.csv")
            if not os.path.exists(report_path):
                continue
            df = pd.read_csv(report_path)
            actual_col = f"actual_lead_{lead}h"
            pred_col = f"pred_lead_{lead}h"
            if actual_col not in df.columns or pred_col not in df.columns:
                continue
            actuals_np = df[actual_col].to_numpy()
            preds_np = df[pred_col].to_numpy()
            ss_tot = float(np.sum((actuals_np - actuals_np.mean()) ** 2))
            if ss_tot == 0:
                continue
            ss_res = float(np.sum((actuals_np - preds_np) ** 2))
            model_nse[label].append(1.0 - ss_res / ss_tot)
    return model_nse


def merge_basin_reports(basin, model_configs, model_labels, model_leads):
    """
    Builds one merged, target-time-aligned DataFrame for a basin: each
    model's own report is shifted by its own lead (target_time = timestamp +
    lead) before merging, so every model's prediction lands on the real-world
    moment it actually predicts - directly comparable to the real flow and to
    other models regardless of their individual lead. A model missing a
    target time (e.g. a NaN-tolerance-excluded window) simply has NaN there
    after the outer merge, which matplotlib draws as a gap. Returns None if
    no model has a report for this basin.
    """
    slim_frames = []
    for label, config in zip(model_labels, model_configs):
        exp_dir = os.path.join(config.get('run_dir', './runs/'), config['experiment_name'])
        report_path = os.path.join(exp_dir, "visualization_reports", f"visual_report_basin_{basin}.csv")
        if not os.path.exists(report_path):
            continue

        df = pd.read_csv(report_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        lead = model_leads[label]
        df['target_time'] = df['timestamp'] + pd.Timedelta(hours=lead)

        actual_col = f"actual_lead_{lead}h"
        pred_col = f"pred_lead_{lead}h"
        thresh_cols = [c for c in df.columns if c.startswith('threshold_') and c.endswith('yr_rp')]

        rename_map = {actual_col: f'actual_at_target__{label}', pred_col: f'pred__{label}'}
        rename_map.update({c: f'{c}__{label}' for c in thresh_cols})

        slim = df[['target_time', actual_col, pred_col] + thresh_cols].rename(columns=rename_map)
        slim_frames.append(slim)

    if not slim_frames:
        return None

    merged = slim_frames[0]
    for frame in slim_frames[1:]:
        merged = pd.merge(merged, frame, on='target_time', how='outer')

    # Coalesce ground truth across models - wherever target times overlap the
    # values should agree (same underlying data), so this just maximizes coverage.
    actual_cols = [c for c in merged.columns if c.startswith('actual_at_target__')]
    merged['actual_flow'] = merged[actual_cols[0]]
    for c in actual_cols[1:]:
        merged['actual_flow'] = merged['actual_flow'].combine_first(merged[c])
    merged = merged.drop(columns=actual_cols)

    # Coalesce return-period thresholds back to bare `threshold_{rp}yr_rp`
    # column names (static per basin; validated identical across models).
    rp_years = sorted({c.split('threshold_')[1].split('yr_rp')[0]
                        for c in merged.columns if c.startswith('threshold_') and 'yr_rp' in c})
    for rp in rp_years:
        rp_cols = [c for c in merged.columns if c.startswith(f'threshold_{rp}yr_rp')]
        merged[f'threshold_{rp}yr_rp'] = merged[rp_cols[0]]
        for c in rp_cols[1:]:
            merged[f'threshold_{rp}yr_rp'] = merged[f'threshold_{rp}yr_rp'].combine_first(merged[c])
        merged = merged.drop(columns=[c for c in rp_cols if c != f'threshold_{rp}yr_rp'])

    merged = merged.sort_values('target_time').reset_index(drop=True)
    merged = merged.rename(columns={'target_time': 'timestamp'})
    return merged


def cluster_events(tagged_events):
    """
    Merges overlapping (unpadded core_start/core_end) events from ANY source
    (real flow or any model's predictions) into unified clusters via the
    standard merge-overlapping-intervals sweep (sort by start, extend while
    the next event's start <= the cluster's current end). Each cluster keeps
    its member tagged events for classify_clusters to inspect.
    """
    if not tagged_events:
        return []
    events_sorted = sorted(tagged_events, key=lambda e: e['core_start'])
    clusters = [{'core_start': events_sorted[0]['core_start'],
                 'core_end': events_sorted[0]['core_end'],
                 'members': [events_sorted[0]]}]
    for ev in events_sorted[1:]:
        if ev['core_start'] <= clusters[-1]['core_end']:
            clusters[-1]['core_end'] = max(clusters[-1]['core_end'], ev['core_end'])
            clusters[-1]['members'].append(ev)
        else:
            clusters.append({'core_start': ev['core_start'], 'core_end': ev['core_end'], 'members': [ev]})
    return clusters


def classify_clusters(clusters, model_labels, basin_id):
    """
    For each cluster x each model label, determines TP/FN/FP/TN based on
    whether a 'real' member and/or that model's member is present in the
    cluster. Returns tidy rows: one row per (cluster, model) pair.
    """
    rows = []
    for idx, cluster in enumerate(clusters, start=1):
        real_members = [m for m in cluster['members'] if m['source'] == 'real']
        real_present = len(real_members) > 0
        real_peak = max((m['peak_flow'] for m in real_members), default=None)

        for label in model_labels:
            model_members = [m for m in cluster['members'] if m['source'] == label]
            model_present = len(model_members) > 0
            model_peak = max((m['peak_flow'] for m in model_members), default=None)

            if real_present and model_present:
                classification = 'TP'
            elif real_present and not model_present:
                classification = 'FN'
            elif not real_present and model_present:
                classification = 'FP'
            else:
                classification = 'TN'

            rows.append({
                'basin_id': basin_id,
                'event_idx': idx,
                'core_start': cluster['core_start'].strftime('%Y-%m-%d %H:%M:%S'),
                'core_end': cluster['core_end'].strftime('%Y-%m-%d %H:%M:%S'),
                'real_peak_flow': real_peak,
                'model_label': label,
                'model_peak_flow': model_peak,
                'label': classification,
            })
    return rows


def build_model_color_map(model_labels):
    """Fixes one color per model label, reused consistently across every
    plot in the run (not per-figure), so a model's color never changes."""
    cmap = plt.get_cmap('tab10')
    return {label: cmap(i % 10) for i, label in enumerate(model_labels)}


def plot_nse_cdf_comparison(model_nse, model_leads, comparison_config, output_dir):
    """
    Empirical CDF of basin-level NSE, one curve per model overlaid on a
    single figure so models are directly comparable. Saves a PNG and returns
    the figure, or None if no model had any valid NSE values.
    """
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150, facecolor="#fafafa")
    ax.set_facecolor("#ffffff")

    color_map = build_model_color_map(list(model_nse.keys()))
    any_curve = False
    for label, nse_values in model_nse.items():
        values = sorted(v for v in nse_values if v is not None and not np.isnan(v))
        if not values:
            continue
        any_curve = True
        n = len(values)
        cdf_y = [(i + 1) / n for i in range(n)]
        lead = model_leads[label]
        ax.plot(values, cdf_y, color=color_map[label], linewidth=2, label=f'{label} (+{lead}h, n={n})')

    if not any_curve:
        plt.close(fig)
        return None

    ax.set_title(f"NSE Empirical CDF — Model Comparison ({comparison_config['comparison_name']})",
                 fontsize=12, fontweight='bold', pad=15, color='#2c3e50')
    ax.set_xlabel('NSE', fontsize=10.5, labelpad=8)
    ax.set_ylabel('CDF', fontsize=10.5, labelpad=8)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=':', alpha=0.5, color='#b0b0b0')
    ax.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#e2e2e2', fontsize=9)

    fig.tight_layout()

    plots_dir = os.path.join(output_dir, "comparison_plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, "nse_cdf_comparison.png")
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"[INFO] Saved comparison NSE CDF to: {out_path}")

    return fig


def plot_hydrograph_comparison(window_df, model_labels, model_leads, model_color_map, title,
                                output_dir, filename, event_labels=None, hourly_xticks=False):
    """
    Fixed-size hydrograph: real flow (black) + one line per model's
    pred__{label} column (in its fixed color). window_df's 'timestamp'
    column is target time (real-world clock time each value pertains to),
    so every model's line - regardless of its own lead - is directly
    comparable at the same x position. event_labels: optional
    {label: 'TP'/'FN'/'FP'/'TN'} to append to each model's legend entry.
    """
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150, facecolor="#fafafa")
    ax.set_facecolor("#ffffff")

    ax.plot(window_df['timestamp'], window_df['actual_flow'],
            color='#1e1e1e', linewidth=2.0, label='Actual Streamflow')

    for label in model_labels:
        col = f'pred__{label}'
        if col in window_df.columns:
            lead = model_leads[label]
            tag = f" [{event_labels[label]}]" if event_labels and label in event_labels else ""
            ax.plot(window_df['timestamp'], window_df[col],
                    color=model_color_map[label], linewidth=1.4, alpha=0.9,
                    label=f'{label} (+{lead}h){tag}')

    for c in window_df.columns:
        if c.startswith('threshold_') and c.endswith('yr_rp'):
            rp = c.split('threshold_')[1].split('yr_rp')[0]
            thresh_series = window_df[c].dropna()
            if not thresh_series.empty and thresh_series.iloc[0] > 0:
                ax.axhline(y=float(thresh_series.iloc[0]), color=ph.THRESHOLD_COLORS.get(int(rp), '#d9d9d9'),
                           linestyle='-.', linewidth=2.0, alpha=1.0,
                           label=f'{rp}yr RP ({thresh_series.iloc[0]:.1f} m³/s)')

    ax.set_title(title, fontsize=12, fontweight='bold', pad=15, color='#2c3e50')
    ax.set_xlabel('Time (target)', fontsize=10.5, labelpad=8)
    ax.set_ylabel('Discharge (m³/s)', fontsize=10.5, labelpad=8)
    ax.grid(True, linestyle=':', alpha=0.5, color='#b0b0b0')
    ax.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#e2e2e2', fontsize=9)

    if hourly_xticks:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %Hh'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    fig.tight_layout()

    plots_dir = os.path.join(output_dir, "comparison_plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, filename)
    fig.savefig(out_path, facecolor=fig.get_facecolor())

    return fig


def main(comparison_config_path="configs/compare_model_0_leads.yml"):
    comparison_config = load_config(comparison_config_path)

    model_config_paths = comparison_config['model_configs']
    model_labels_cfg = comparison_config.get('model_labels')

    print("=" * 75)
    print("      Multi-Model Comparison — Single-Lead-Time Models Only")
    print("=" * 75)
    print(f"[INFO] Comparing {len(model_config_paths)} models: {model_config_paths}")

    model_configs, model_labels, shared_basins, model_leads = validate_comparable(
        model_config_paths, model_labels_cfg
    )
    print(f"[INFO] Validation passed. {len(shared_basins)} shared test basins. Leads: {model_leads}")

    output_dir = os.path.join(comparison_config.get('run_dir', './runs/model_comparisons/'),
                               comparison_config['comparison_name'])
    os.makedirs(output_dir, exist_ok=True)

    use_wandb = comparison_config.get('use_wandb', False) and WANDB_AVAILABLE
    if use_wandb:
        api_key = comparison_config.get('wandb_api_key')
        if api_key:
            wandb.login(key=api_key)
        wandb.init(
            project=comparison_config.get('wandb_project', 'flash-floods-israel'),
            name=f"{comparison_config['comparison_name']}_compare",
            config=comparison_config,
        )

    # Step 1: evaluate every model over the shared basin list (writes each
    # model's own report CSVs into its own run_dir/experiment_name, unchanged
    # from quick_test.py's behavior).
    for label, config in zip(model_labels, model_configs):
        print(f"\n[INFO] Evaluating model '{label}' ({config['experiment_name']})...")
        run_single_model_eval(config, shared_basins)

    # Step 2: comparison NSE CDF - one curve per model, same figure.
    model_nse = compute_model_nse(model_configs, model_labels, model_leads, shared_basins)
    cdf_fig = plot_nse_cdf_comparison(model_nse, model_leads, comparison_config, output_dir)
    if cdf_fig is not None:
        if use_wandb:
            wandb.log({"compare/nse_cdf": wandb.Image(cdf_fig)})
        plt.close(cdf_fig)

    # Step 3: per-basin target-time merge, N-way flood-event union/classification, hydrographs.
    prediction_rp = model_configs[0]['prediction_threshold']
    buffer_days = model_configs[0].get('visual_buffer_days', 4)
    merge_gap_hours = model_configs[0].get('event_merge_gap_hours', 0)

    model_color_map = build_model_color_map(model_labels)
    all_rows = []

    print(f"\n[INFO] Scanning {len(shared_basins)} basins for flood events across all models...")
    for basin in shared_basins:
        merged_df = merge_basin_reports(basin, model_configs, model_labels, model_leads)
        if merged_df is None:
            print(f"  [WARNING] No model has data for basin {basin}. Skipping.")
            continue

        threshold_value = ffe.load_basin_threshold(basin, prediction_rp, model_configs[0])
        if threshold_value is None or threshold_value <= 0:
            continue

        tagged = []
        real_events = ffe._scan_series_for_events(merged_df, 'actual_flow', threshold_value,
                                                    buffer_days, merge_gap_hours)
        for ev in real_events:
            tagged.append({**ev, 'source': 'real',
                            'core_start': pd.to_datetime(ev['core_start']),
                            'core_end': pd.to_datetime(ev['core_end'])})

        for label in model_labels:
            col = f'pred__{label}'
            if col not in merged_df.columns:
                continue
            pred_events = ffe._scan_series_for_events(merged_df, col, threshold_value,
                                                        buffer_days, merge_gap_hours)
            for ev in pred_events:
                tagged.append({**ev, 'source': label,
                                'core_start': pd.to_datetime(ev['core_start']),
                                'core_end': pd.to_datetime(ev['core_end'])})

        if not tagged:
            continue

        clusters = cluster_events(tagged)
        rows = classify_clusters(clusters, model_labels, basin)
        all_rows.extend(rows)

        for idx, cluster in enumerate(clusters, start=1):
            padded_start = max(cluster['core_start'] - pd.Timedelta(days=buffer_days), merged_df['timestamp'].min())
            padded_end = min(cluster['core_end'] + pd.Timedelta(days=buffer_days), merged_df['timestamp'].max())
            window_df = merged_df[(merged_df['timestamp'] >= padded_start) & (merged_df['timestamp'] <= padded_end)]
            if window_df.empty:
                continue

            event_labels = {row['model_label']: row['label'] for row in rows if row['event_idx'] == idx}
            title = (f"Basin {basin} — Model Comparison Storm Event\n"
                     f"Core: {cluster['core_start'].strftime('%Y-%m-%d %H:%M')} → "
                     f"{cluster['core_end'].strftime('%Y-%m-%d %H:%M')}")
            filename = f"hydrograph_{basin}_event{idx}.png"
            fig = plot_hydrograph_comparison(window_df, model_labels, model_leads, model_color_map,
                                              title, output_dir, filename, event_labels=event_labels,
                                              hourly_xticks=True)
            if use_wandb:
                wandb.log({f"compare/flood_events/{basin}/event{idx}": wandb.Image(fig)})
            plt.close(fig)

    csv_path = os.path.join(output_dir, "flood_event_comparison.csv")
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"\n[INFO] Wrote {len(all_rows)} comparison event rows to {csv_path}")

    if use_wandb:
        wandb.finish()

    print(f"\n[INFO] Model comparison completed. Outputs in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare multiple single-lead-time models on the same test split.")
    parser.add_argument("--config", type=str, default="configs/compare_model_0_leads.yml",
                         help="Path to the comparison config YAML (lists model_configs to compare).")
    args = parser.parse_args()
    main(args.config)
