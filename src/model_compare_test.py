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
from torch.utils.data import ConcatDataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import dataset
from dataset import IsraelBasinsDataset
from quick_test import setup_evaluation_from_checkpoint
from test import evaluate_basin_sequences, build_and_export_report
import find_flood_events as ffe
from find_flood_events import cluster_events  # re-exported for existing callers (plot_events_by_year.py)
import plot_hydrographs as ph
import peaks_analyze as pa
from flow_quality_check import get_hydrological_year


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


def _cluster_hydro_year(core_start, hydro_year_start_month):
    return int(get_hydrological_year(pd.DatetimeIndex([core_start]), start_month=hydro_year_start_month)[0])


class _PeriodBasinDataset:
    """
    Minimal IsraelBasinsDataset-compatible wrapper - sample_basin_mappings,
    sample_date_mappings, __getitem__/__len__ - built directly from a list
    of SingleBasinDataset objects over caller-supplied periods. Bypasses
    IsraelBasinsDataset's hardcoded train/val/test split-type restriction
    (dataset.py:_get_split_bounds_and_config only knows those three) without
    touching dataset.py: builds the exact same tracking arrays
    IsraelBasinsDataset.__init__ does, from the same basin_datasets shape,
    just fed a custom periods list instead of one derived from a fixed split.
    """
    def __init__(self, basin_datasets):
        self.basin_datasets = basin_datasets
        self.concat_dataset = ConcatDataset(basin_datasets)
        self.sample_basin_mappings = []
        self.sample_date_mappings = []
        for ds in basin_datasets:
            for i in range(len(ds)):
                self.sample_basin_mappings.append(ds.gauge_id)
                actual_idx = ds.valid_indices[i]
                target_idx = actual_idx + ds.seq_length - 1
                self.sample_date_mappings.append(ds.dates[target_idx])

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        return self.concat_dataset[idx]


def evaluate_model_over_years(config, basins, periods):
    """
    Loads config['checkpoint_path'] and runs inference over exactly the
    given periods - not the configured test split. Mirrors
    run_single_model_eval, swapping IsraelBasinsDataset(split_type='test', ...)
    for a dataset built directly via dataset._build_basin_datasets over
    custom periods, wrapped in _PeriodBasinDataset. split_type='test' passed
    to _build_basin_datasets only selects eval_max_nan_pct tolerance inside
    SingleBasinDataset - it doesn't tie this evaluation to the configured
    test window. Always recomputes (no skip-if-exists caching), same
    convention as run_single_model_eval. Returns exp_dir.
    """
    model, device, exp_dir = setup_evaluation_from_checkpoint(config)
    output_dir = os.path.join(exp_dir, "visualization_reports")
    os.makedirs(output_dir, exist_ok=True)

    basin_datasets = dataset._build_basin_datasets(basins, config, periods, split_type='test')
    period_dataset = _PeriodBasinDataset(basin_datasets)

    for basin in basins:
        basin_data = evaluate_basin_sequences(basin, period_dataset, model, device, config)
        if basin_data is None:
            print(f"  [WARNING] No windows found for basin {basin} ({config['experiment_name']}) "
                  f"in the requested period(s). Skipping.")
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
        thresh_cols = [c for c in df.columns if c.startswith('threshold_')]

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

    # Coalesce every threshold column (any type/label - see
    # find_flood_events.threshold_column_name) back to its bare name (static
    # per basin; validated identical across models). Each was tagged
    # f'{base_col}__{label}' above; group by base_col to merge back.
    threshold_cols = [c for c in merged.columns if c.startswith('threshold_')]
    base_names = sorted({c.rsplit('__', 1)[0] for c in threshold_cols})
    for base in base_names:
        matching = [c for c in threshold_cols if c == base or c.startswith(f'{base}__')]
        merged[base] = merged[matching[0]]
        for c in matching[1:]:
            merged[base] = merged[base].combine_first(merged[c])
        merged = merged.drop(columns=[c for c in matching if c != base])

    merged = merged.sort_values('target_time').reset_index(drop=True)
    merged = merged.rename(columns={'target_time': 'timestamp'})

    # Rain is basin ground truth, independent of which model/lead produced a
    # row, so it's attached by real-world timestamp here (once, on the final
    # merged frame) rather than threaded through the per-model slim frames
    # above (which would incorrectly shift it by each model's own lead).
    rain_df = ph.load_basin_rain_series(basin, model_configs[0])
    if rain_df is not None:
        merged = merged.merge(rain_df, on='timestamp', how='left')

    return merged


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

        median_nse = float(np.median(values))
        ax.axvline(x=median_nse, color=color_map[label], linewidth=1.5, linestyle='--',
                   alpha=0.8, label=f'{label} median: {median_nse:.3f}')

    if not any_curve:
        plt.close(fig)
        return None

    ax.set_title(f"NSE Empirical CDF — Model Comparison ({comparison_config['comparison_name']})",
                 fontsize=12, fontweight='bold', pad=15, color='#2c3e50')
    ax.set_xlabel('NSE', fontsize=10.5, labelpad=8)
    ax.set_ylabel('CDF', fontsize=10.5, labelpad=8)
    ax.set_xlim(0, 1)
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
                                output_dir, filename, event_labels=None, hourly_xticks=False,
                                xlim=None, shown_threshold_specs=None):
    """
    Fixed-size hydrograph: real flow (black) + one line per model's
    pred__{label} column (in its fixed color). window_df's 'timestamp'
    column is target time (real-world clock time each value pertains to),
    so every model's line - regardless of its own lead - is directly
    comparable at the same x position. event_labels: optional
    {label: 'TP'/'FN'/'FP'/'TN'} to append to each model's legend entry.
    xlim: optional (start, end) timestamps to fix the x-axis to explicitly.
    Without this, matplotlib autoscales from window_df's data - which
    collapses to a degenerate (0, 1) numeric range when the window contains a
    single unique timestamp (e.g. a one-hour exceedance event with little/no
    buffer padding), and hourly_xticks then blows past Locator.MAXTICKS trying
    to tick across that bogus range instead of rendering a normal hydrograph.
    Pass the caller's own known padded-window bounds to avoid this regardless
    of how many rows fall inside.
    shown_threshold_specs: optional list of threshold specs (see
    find_flood_events.normalize_threshold_specs) naming exactly which
    threshold bars to draw - replaces the old behavior of blanket-drawing
    every threshold_*yr_rp column found in window_df. None draws nothing.
    """
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150, facecolor="#fafafa")
    ax.set_facecolor("#ffffff")

    if xlim is not None:
        ax.set_xlim(*ph._safe_xlim(*xlim))

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

    for spec in (shown_threshold_specs or []):
        label = ffe.threshold_label(spec)
        col = ffe.threshold_column_name(spec)
        if col not in window_df.columns:
            continue
        thresh_series = window_df[col].dropna()
        if thresh_series.empty or thresh_series.iloc[0] <= 0:
            continue
        color = ph.THRESHOLD_COLORS.get(label, ph.THRESHOLD_COLORS.get(spec['type'], '#d9d9d9'))
        ax.axhline(y=float(thresh_series.iloc[0]), color=color,
                   linestyle='-.', linewidth=2.0, alpha=1.0,
                   label=f'{label} ({thresh_series.iloc[0]:.1f} m³/s)')

    ax.set_title(title, fontsize=12, fontweight='bold', pad=15, color='#2c3e50')
    ax.set_xlabel('Time (target)', fontsize=10.5, labelpad=8)
    ax.set_ylabel('Discharge (m³/s)', fontsize=10.5, labelpad=8)
    ax.grid(True, linestyle=':', alpha=0.5, color='#b0b0b0')

    if 'rain_mm' in window_df.columns:
        ax2 = ph.add_rain_overlay(ax, window_df['timestamp'], window_df['rain_mm'])
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right',
                  frameon=True, facecolor='#ffffff', edgecolor='#e2e2e2', fontsize=9)
    else:
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


def plot_hydrograph_difference(window_df, model_labels, model_leads, model_color_map, title,
                                output_dir, filename, hourly_xticks=False, xlim=None):
    """
    Fixed-size plot of observed-minus-predicted flow (actual_flow -
    pred__{label}) per model, to spot where/when predictions diverge most
    from reality - a horizontal zero line marks perfect agreement. Rain
    overlay above for reference, same as plot_hydrograph_comparison. No
    threshold bars: the y-axis is a flow difference, not a flow value, so a
    threshold line wouldn't mean anything here.
    """
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150, facecolor="#fafafa")
    ax.set_facecolor("#ffffff")

    if xlim is not None:
        ax.set_xlim(*ph._safe_xlim(*xlim))

    ax.axhline(y=0, color='#1e1e1e', linewidth=1.5, alpha=0.8, label='Zero (perfect match)')

    for label in model_labels:
        col = f'pred__{label}'
        if col in window_df.columns:
            lead = model_leads[label]
            diff = window_df['actual_flow'] - window_df[col]
            ax.plot(window_df['timestamp'], diff,
                    color=model_color_map[label], linewidth=1.2, alpha=0.9,
                    label=f'{label} (+{lead}h) obs−pred')

    ax.set_title(title, fontsize=12, fontweight='bold', pad=15, color='#2c3e50')
    ax.set_xlabel('Time (target)', fontsize=10.5, labelpad=8)
    ax.set_ylabel('Observed − Predicted Discharge (m³/s)', fontsize=10.5, labelpad=8)
    ax.grid(True, linestyle=':', alpha=0.5, color='#b0b0b0')

    if 'rain_mm' in window_df.columns:
        ax2 = ph.add_rain_overlay(ax, window_df['timestamp'], window_df['rain_mm'])
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right',
                  frameon=True, facecolor='#ffffff', edgecolor='#e2e2e2', fontsize=9)
    else:
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


def plot_yearly_overview_hydrographs(model_configs, model_labels, model_leads, shared_basins,
                                      years, output_dir, use_wandb=False):
    """
    Per basin, per INDIVIDUAL requested hydrological year (never merged
    across adjacent years, unlike dataset.build_year_periods' evaluation
    grouping): two whole-year plots reusing the same target-time-merged,
    rain-attached frame merge_basin_reports already builds for event
    hydrographs - (1) the standard multi-model observed-vs-predicted
    hydrograph (plot_hydrograph_comparison, unchanged, just given the whole
    year as its window instead of a padded event window), and (2) the
    observed-minus-predicted difference plot (plot_hydrograph_difference).
    Both saved into output_dir/comparison_plots/ alongside event hydrographs,
    distinguished by filename.
    """
    hydro_year_start_month = model_configs[0].get('hydro_year_start_month', 10)
    shown_threshold_specs = ffe.normalize_threshold_specs(
        model_configs[0].get('shown_threshold_bars', model_configs[0].get('return_periods_years', []))
    )
    model_color_map = build_model_color_map(model_labels)

    print(f"\n[INFO] Building yearly overview hydrographs for {len(shared_basins)} basins "
          f"x {len(set(years))} year(s)...")
    for basin in shared_basins:
        merged_df = merge_basin_reports(basin, model_configs, model_labels, model_leads)
        if merged_df is None:
            continue

        for year in sorted(set(years)):
            year_start, year_end = dataset.build_year_periods([year], hydro_year_start_month)[0]
            year_start, year_end = pd.Timestamp(year_start), pd.Timestamp(year_end)
            window_df = merged_df[(merged_df['timestamp'] >= year_start) & (merged_df['timestamp'] <= year_end)]
            if window_df.empty:
                continue

            title = (f"Basin {basin} — Hydrological Year {year}\n"
                      f"{year_start.date()} → {year_end.date()}")
            fig = plot_hydrograph_comparison(window_df, model_labels, model_leads, model_color_map,
                                              title, output_dir, f"hydrograph_{basin}_{year}_full.png",
                                              hourly_xticks=False, xlim=(year_start, year_end),
                                              shown_threshold_specs=shown_threshold_specs)
            if use_wandb:
                wandb.log({f"yearly/{basin}/{year}/hydrograph": wandb.Image(fig)})
            plt.close(fig)

            diff_title = (f"Basin {basin} — Observed minus Predicted, Hydrological Year {year}\n"
                          f"{year_start.date()} → {year_end.date()}")
            diff_fig = plot_hydrograph_difference(window_df, model_labels, model_leads, model_color_map,
                                                   diff_title, output_dir, f"hydrograph_diff_{basin}_{year}.png",
                                                   hourly_xticks=False, xlim=(year_start, year_end))
            if use_wandb:
                wandb.log({f"yearly/{basin}/{year}/difference": wandb.Image(diff_fig)})
            plt.close(diff_fig)


def run_event_and_peaks_analysis(model_configs, model_labels, model_leads, shared_basins,
                                  output_dir, use_wandb=False, periods=None, years=None):
    """
    Per-basin target-time merge, N-way flood-event union/classification,
    hydrographs, and peak timing/magnitude analysis. Reuses each model's
    already-saved visual_report_basin_*.csv (via merge_basin_reports) - no
    model/inference needed, so this can be re-run standalone any time after
    run_single_model_eval (or evaluate_model_over_years) has produced those
    reports at least once (see replot_nse_cdf_comparison.py). Writes
    flood_event_comparison.csv, peaks_analysis_comparison.csv, and
    peaks_analysis_comparison_summary.csv to output_dir, plus per-event
    hydrograph PNGs into output_dir/comparison_plots/.

    periods/years: optional, for callers evaluating over arbitrary (possibly
    non-contiguous) hydrological years rather than each model's fixed
    configured test split (see model_compare_events_by_year.py).
      - periods=None (default): unchanged behavior - each basin's whole
        merged report history is scanned for events in one
        _scan_series_for_events call, exactly as when the report already
        covers precisely one contiguous evaluation window (the configured
        test split).
      - periods=[(start, end), ...]: events are scanned separately within
        each period and the tagged results concatenated before clustering,
        so _scan_series_for_events's consecutive-row grouping can never
        merge an event tail from one requested period with an event head
        from a different, temporally distant period.
      - years: optional set/list of hydrological years: when given (only
        meaningful together with periods), clusters are additionally kept
        only if their core_start's hydrological year is in years - a cheap,
        mostly-redundant defensive check mirroring periods' own bounds.

    prediction_threshold (model_configs[0]) may hold multiple threshold specs
    (return_period and/or specific_discharge - see
    find_flood_events.normalize_threshold_specs). Event classification and
    peaks analysis run SEPARATELY per threshold, each writing its own
    flood_event_comparison_{label}.csv / peaks_analysis_comparison_{label}.csv
    / peaks_analysis_comparison_summary_{label}.csv. Hydrograph PNGs are
    DEDUPED across thresholds: a basin+time-window flagged by more than one
    threshold is plotted exactly once (see find_flood_events.merge_events_across_thresholds),
    labeled by whichever contributing threshold has the lower resolved flow
    value for that basin. shown_threshold_bars (falling back to
    return_periods_years) controls which threshold lines are drawn on those
    plots, independent of prediction_threshold.
    """
    prediction_specs = ffe.normalize_threshold_specs(model_configs[0]['prediction_threshold'])
    shown_threshold_specs = ffe.normalize_threshold_specs(
        model_configs[0].get('shown_threshold_bars', model_configs[0].get('return_periods_years', []))
    )
    buffer_days = model_configs[0].get('visual_buffer_days', 4)
    merge_gap_hours = model_configs[0].get('event_merge_gap_hours', 0)
    hydro_year_start_month = model_configs[0].get('hydro_year_start_month', 10)
    period_bounds = ([(pd.Timestamp(start), pd.Timestamp(end)) for start, end in periods]
                      if periods is not None else None)
    years = set(years) if years is not None else None
    area_map = (ffe.load_basin_area_map(model_configs[0])
                if any(s['type'] == 'specific_discharge' for s in prediction_specs) else None)

    model_color_map = build_model_color_map(model_labels)
    flow_std_map = pa.load_flow_std_map(model_configs[0])

    prediction_labels = [ffe.threshold_label(s) for s in prediction_specs]
    all_rows_by_label = {label: [] for label in prediction_labels}
    comparison_pairs_by_label = {label: [] for label in prediction_labels}

    # For the plotting-dedup pass: per basin, each threshold's own clusters
    # (with that cluster's per-model classification rows attached) plus that
    # basin's resolved flow value per threshold.
    plot_events_by_basin = {}    # {basin: {label: [{'core_start','core_end','rows'}]}}
    value_by_label_by_basin = {} # {basin: {label: resolved_flow_value}}
    merged_df_by_basin = {}

    print(f"\n[INFO] Scanning {len(shared_basins)} basins for flood events across all models "
          f"and thresholds {prediction_labels}...")
    for basin in shared_basins:
        merged_df = merge_basin_reports(basin, model_configs, model_labels, model_leads)
        if merged_df is None:
            print(f"  [WARNING] No model has data for basin {basin}. Skipping.")
            continue
        merged_df_by_basin[basin] = merged_df

        events_by_label = {}
        value_by_label = {}

        for spec, label in zip(prediction_specs, prediction_labels):
            threshold_value = ffe.resolve_threshold_value(basin, spec, model_configs[0], area_map)
            if threshold_value is None or threshold_value <= 0:
                continue
            value_by_label[label] = threshold_value

            def _scan_and_tag(df, threshold_value=threshold_value):
                tagged = []
                real_events = ffe._scan_series_for_events(df, 'actual_flow', threshold_value,
                                                            buffer_days, merge_gap_hours)
                for ev in real_events:
                    tagged.append({**ev, 'source': 'real',
                                    'core_start': pd.to_datetime(ev['core_start']),
                                    'core_end': pd.to_datetime(ev['core_end'])})

                for model_label in model_labels:
                    col = f'pred__{model_label}'
                    if col not in df.columns:
                        continue
                    pred_events = ffe._scan_series_for_events(df, col, threshold_value,
                                                                buffer_days, merge_gap_hours)
                    for ev in pred_events:
                        tagged.append({**ev, 'source': model_label,
                                        'core_start': pd.to_datetime(ev['core_start']),
                                        'core_end': pd.to_datetime(ev['core_end'])})
                return tagged

            if period_bounds is None:
                tagged = _scan_and_tag(merged_df)
            else:
                tagged = []
                for period_start, period_end in period_bounds:
                    period_df = merged_df[(merged_df['timestamp'] >= period_start) &
                                           (merged_df['timestamp'] <= period_end)]
                    if period_df.empty:
                        continue
                    tagged.extend(_scan_and_tag(period_df))

            if not tagged:
                continue

            clusters = cluster_events(tagged)
            if years is not None:
                clusters = [c for c in clusters
                            if _cluster_hydro_year(c['core_start'], hydro_year_start_month) in years]
                if not clusters:
                    continue

            rows = classify_clusters(clusters, model_labels, basin)
            all_rows_by_label[label].extend(rows)
            comparison_pairs_by_label[label].extend(
                pa.collect_comparison_peak_pairs(merged_df, rows, basin, model_leads, flow_std_map))

            events_by_label[label] = [
                {'core_start': cluster['core_start'], 'core_end': cluster['core_end'],
                 'rows': [r for r in rows if r['event_idx'] == idx]}
                for idx, cluster in enumerate(clusters, start=1)
            ]

        if events_by_label:
            plot_events_by_basin[basin] = events_by_label
            value_by_label_by_basin[basin] = value_by_label

    # Plotting pass: one PNG per basin per merged (cross-threshold-deduped) window.
    for basin, events_by_label in plot_events_by_basin.items():
        merged_df = merged_df_by_basin[basin]
        merged_windows = ffe.merge_events_across_thresholds(events_by_label, value_by_label_by_basin[basin])

        for idx, window in enumerate(merged_windows, start=1):
            padded_start = max(window['core_start'] - pd.Timedelta(days=buffer_days), merged_df['timestamp'].min())
            padded_end = min(window['core_end'] + pd.Timedelta(days=buffer_days), merged_df['timestamp'].max())
            window_df = merged_df[(merged_df['timestamp'] >= padded_start) & (merged_df['timestamp'] <= padded_end)]
            if window_df.empty:
                continue

            event_labels = {row['model_label']: row['label'] for row in window['rows']}
            title = (f"Basin {basin} — Model Comparison Storm Event [{window['winning_label']}]\n"
                     f"Core: {window['core_start'].strftime('%Y-%m-%d %H:%M')} → "
                     f"{window['core_end'].strftime('%Y-%m-%d %H:%M')}")
            filename = f"hydrograph_{basin}_event{idx}.png"
            fig = plot_hydrograph_comparison(window_df, model_labels, model_leads, model_color_map,
                                              title, output_dir, filename, event_labels=event_labels,
                                              hourly_xticks=True, xlim=(padded_start, padded_end),
                                              shown_threshold_specs=shown_threshold_specs)
            if use_wandb:
                wandb.log({f"compare/flood_events/{basin}/event{idx}": wandb.Image(fig)})
            plt.close(fig)

    event_columns = ['basin_id', 'event_idx', 'core_start', 'core_end',
                      'real_peak_flow', 'model_label', 'model_peak_flow', 'label']
    for label in prediction_labels:
        csv_path = os.path.join(output_dir, f"flood_event_comparison_{label}.csv")
        pd.DataFrame(all_rows_by_label[label], columns=event_columns).to_csv(csv_path, index=False)
        print(f"\n[INFO] [{label}] Wrote {len(all_rows_by_label[label])} comparison event rows to {csv_path}")

        peaks_detail_df = pd.DataFrame(comparison_pairs_by_label[label], columns=pa.DETAIL_COLUMNS + ['model_label'])
        peaks_detail_df, peaks_summary_df = pa.build_comparison_peaks_report(peaks_detail_df, model_labels)

        peaks_csv_path = os.path.join(output_dir, f"peaks_analysis_comparison_{label}.csv")
        peaks_detail_df.to_csv(peaks_csv_path, index=False)
        print(f"[INFO] [{label}] Wrote peaks analysis comparison ({len(peaks_detail_df)} matched events) to {peaks_csv_path}")

        peaks_summary_csv_path = os.path.join(output_dir, f"peaks_analysis_comparison_summary_{label}.csv")
        peaks_summary_df.to_csv(peaks_summary_csv_path, index=False)
        print(f"[INFO] [{label}] Wrote peaks analysis comparison summary to {peaks_summary_csv_path}")


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

    # Step 3: per-basin target-time merge, N-way flood-event union/classification, hydrographs, peaks.
    run_event_and_peaks_analysis(model_configs, model_labels, model_leads, shared_basins,
                                  output_dir, use_wandb)

    if use_wandb:
        wandb.finish()

    print(f"\n[INFO] Model comparison completed. Outputs in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare multiple single-lead-time models on the same test split.")
    parser.add_argument("--config", type=str, default="configs/compare_model_0_leads.yml",
                         help="Path to the comparison config YAML (lists model_configs to compare).")
    args = parser.parse_args()
    main(args.config)
