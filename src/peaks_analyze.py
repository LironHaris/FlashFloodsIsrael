"""
Module: peaks_analyze.py
Description: Peak Timing & Magnitude Analysis for EA-LSTM Flood Forecasting.
             For every matched real/predicted flood event (TP) - and,
             separately, every real event whether or not the model matched it
             (TP+FN) - computes the signed time distance between the model's
             predicted peak and the real peak, and the signed magnitude
             difference between them (normalized by each basin's own flow
             std). Consumed by quick_test.py/test.py (single model) and
             model_compare_test.py (N single-lead-time models).

             Operates entirely in target time (a pred_lead_{k}h value at row
             timestamp t is a prediction FOR t+k, not for t) - never reuses
             find_predicted_flood_events.py's/compare_flood_events.py's
             on-disk CSVs directly, since those scan in raw issuance time.
"""

import os
import numpy as np
import pandas as pd

import find_flood_events as ffe
import compare_flood_events as cfe


def load_flow_std_map(config):
    """{basin_id: flow_std} from config['availability_report_file']. The
    gauge_id column there is already 'il_'-prefixed, identical in format to
    every other basin_id used across this pipeline - a direct string lookup,
    no prefix stripping needed."""
    availability_df = pd.read_csv(config['availability_report_file'], dtype={'gauge_id': str})
    return dict(zip(availability_df['gauge_id'], availability_df['flow_std']))


def _load_basin_report(basin_id, exp_dir):
    report_path = os.path.join(exp_dir, "visualization_reports", f"visual_report_basin_{basin_id}.csv")
    if not os.path.exists(report_path):
        print(f"  [Warning] No visual report for basin {basin_id} at {report_path}. Skipping.")
        return None
    df = pd.read_csv(report_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def _build_target_shifted_pred_series(df, lead):
    """['timestamp', 'pred_flow'] frame with timestamp = df['timestamp'] +
    lead hours (target time), mirroring model_compare_test.py:
    merge_basin_reports' shift but for a single model's own report."""
    col = f"pred_lead_{lead}h"
    shifted = pd.DataFrame({
        'timestamp': df['timestamp'] + pd.Timedelta(hours=lead),
        'pred_flow': df[col],
    })
    return shifted.sort_values('timestamp').reset_index(drop=True)


def _scan_events(df, value_col, threshold_value, buffer_days, merge_gap_hours):
    """Thin wrapper over ffe._scan_series_for_events."""
    return ffe._scan_series_for_events(df, value_col, threshold_value, buffer_days, merge_gap_hours)


def _to_timestamps(events):
    """Copies each _scan_series_for_events dict with core_start/core_end
    parsed to Timestamps (cfe.match_events requires Timestamps, not strings)."""
    return [{**ev, 'core_start': pd.to_datetime(ev['core_start']), 'core_end': pd.to_datetime(ev['core_end'])}
            for ev in events]


def _peak_in_window(df, value_col, core_start, core_end):
    """idxmax/max of df[value_col] within [core_start, core_end]. Returns
    (None, None) if the window has no rows or every value there is NaN."""
    window = df[(df['timestamp'] >= core_start) & (df['timestamp'] <= core_end)]
    if window.empty or window[value_col].isna().all():
        return None, None
    peak_idx = window[value_col].idxmax()
    return window.loc[peak_idx, 'timestamp'], float(window.loc[peak_idx, value_col])


def _build_event_pair_row(core_start, core_end, actual_df, pred_df, pred_col,
                           basin_id, lead, label, flow_std_map):
    """
    Shared by both the single-model and multi-model integration points.
    actual_df/pred_df must each have a 'timestamp' column plus 'actual_flow'
    (actual_df) / pred_col (pred_df) respectively - they may be the same
    DataFrame (model_compare_test.py's merged_df) or different ones
    (single-model: the raw report vs. the lead-shifted pred series).
    No threshold check on the model's peak - same rule for TP and FN.
    """
    real_peak_time, real_peak_flow = _peak_in_window(actual_df, 'actual_flow', core_start, core_end)
    model_peak_time, model_peak_flow = _peak_in_window(pred_df, pred_col, core_start, core_end)

    if real_peak_time is None or model_peak_time is None:
        print(f"  [Warning] Basin {basin_id}, lead {lead}, label {label}: "
              f"no usable peak in window [{core_start}, {core_end}]. Skipping event.")
        return None

    flow_std = flow_std_map.get(basin_id)
    if flow_std is None or pd.isna(flow_std) or flow_std == 0:
        magnitude_diff_norm = float('nan')
    else:
        magnitude_diff_norm = (model_peak_flow - real_peak_flow) / flow_std

    return {
        'basin_id': basin_id,
        'lead': lead,
        'label': label,
        'core_start': core_start.strftime('%Y-%m-%d %H:%M:%S'),
        'core_end': core_end.strftime('%Y-%m-%d %H:%M:%S'),
        'real_peak_time': real_peak_time.strftime('%Y-%m-%d %H:%M:%S'),
        'real_peak_flow': real_peak_flow,
        'model_peak_time': model_peak_time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_peak_flow': model_peak_flow,
        'time_distance_h': (model_peak_time - real_peak_time) / pd.Timedelta(hours=1),
        'magnitude_diff_norm': magnitude_diff_norm,
    }


DETAIL_COLUMNS = ['basin_id', 'lead', 'label', 'core_start', 'core_end',
                  'real_peak_time', 'real_peak_flow', 'model_peak_time', 'model_peak_flow',
                  'time_distance_h', 'magnitude_diff_norm']


def collect_peak_pairs(config, basin_ids=None, exp_dir=None, flow_std_map=None):
    """
    Single-model peak-pair collection: for every basin x every configured
    lead time, matches real vs. (target-time-shifted) predicted flood events
    and returns one row per matched TP/FN event (FP dropped - no real peak to
    compare against). Returns an empty, correctly-columned DataFrame if
    nothing matched anywhere.
    """
    if exp_dir is None:
        exp_dir = os.path.join(config.get('run_dir', './runs/'), config['experiment_name'])
    if flow_std_map is None:
        flow_std_map = load_flow_std_map(config)
    if basin_ids is None:
        with open(config['test_basin_file']) as f:
            basin_ids = [line.strip() for line in f if line.strip()]

    buffer_days = config.get('visual_buffer_days', 4)
    merge_gap_hours = config.get('event_merge_gap_hours', 0)
    prediction_rp = config.get('prediction_threshold', 2)

    rows = []
    for basin_id in basin_ids:
        df = _load_basin_report(basin_id, exp_dir)
        if df is None:
            continue

        threshold_value = ffe.load_basin_threshold(basin_id, prediction_rp, config)
        if threshold_value is None or threshold_value <= 0:
            print(f"  [Warning] Basin {basin_id}: no usable {prediction_rp}yr threshold. Skipping.")
            continue

        real_events = _scan_events(df, 'actual_flow', threshold_value, buffer_days, merge_gap_hours)
        real_events_ts = _to_timestamps(real_events)

        for lead in config.get('forecast_lead_times', [0, 1, 2, 3]):
            pred_series_df = _build_target_shifted_pred_series(df, lead)
            predicted_events = _scan_events(pred_series_df, 'pred_flow', threshold_value,
                                             buffer_days, merge_gap_hours)
            matched = cfe.match_events(real_events_ts, _to_timestamps(predicted_events))

            for ev in matched:
                if ev['label'] == 'FP':
                    continue
                row = _build_event_pair_row(ev['core_start'], ev['core_end'],
                                             df[['timestamp', 'actual_flow']], pred_series_df, 'pred_flow',
                                             basin_id, lead, ev['label'], flow_std_map)
                if row is not None:
                    rows.append(row)

    return pd.DataFrame(rows, columns=DETAIL_COLUMNS)


def _pooled_stats(detail_df, scope, value_col):
    """Shared pooling logic for compute_peak_timing/compute_peak_magnitude:
    filters detail_df to the given scope, then pools value_col across every
    basin (lead_mean rows) and across every basin x lead (model_mean/
    model_variance rows) - no per-basin-then-averaged step anywhere."""
    scoped_df = detail_df[detail_df['label'] == 'TP'] if scope == 'TP' else detail_df

    lead_rows = []
    for lead, lead_df in scoped_df.groupby('lead'):
        lead_rows.append({'basin_id': 'lead_mean', 'lead': lead, 'scope': scope,
                           'n_events': len(lead_df), value_col: lead_df[value_col].mean()})

    model_mean_row = {'basin_id': 'model_mean', 'lead': None, 'scope': scope,
                       'n_events': len(scoped_df), value_col: scoped_df[value_col].mean()}
    model_variance_row = {'basin_id': 'model_variance', 'lead': None, 'scope': scope,
                           'n_events': len(scoped_df), value_col: scoped_df[value_col].var()}

    return pd.DataFrame(lead_rows + [model_mean_row, model_variance_row])


def compute_peak_timing(detail_df, scope):
    """Pooled mean (per lead, and model-level mean/variance) of
    time_distance_h over the given scope ('TP' or 'TP_FN')."""
    return _pooled_stats(detail_df, scope, 'time_distance_h')


def compute_peak_magnitude(detail_df, scope):
    """Pooled mean (per lead, and model-level mean/variance) of
    magnitude_diff_norm over the given scope ('TP' or 'TP_FN')."""
    return _pooled_stats(detail_df, scope, 'magnitude_diff_norm')


SCOPES = ('TP', 'TP_FN')


def build_peaks_report(config, basin_ids=None, exp_dir=None):
    """
    Assembles the full single-model peaks report: detail rows (label-tagged)
    followed by both scopes' summary rows (scope-tagged: lead_mean,
    model_mean, model_variance). Does not write to disk.
    """
    detail_df = collect_peak_pairs(config, basin_ids=basin_ids, exp_dir=exp_dir)

    summary_frames = []
    for scope in SCOPES:
        timing = compute_peak_timing(detail_df, scope)
        magnitude = compute_peak_magnitude(detail_df, scope)
        merged = pd.merge(timing, magnitude, on=['basin_id', 'lead', 'scope', 'n_events'], how='outer')
        summary_frames.append(merged)

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    return pd.concat([detail_df, summary_df], ignore_index=True)


def main(config=None, basin_ids=None):
    if config is None:
        config = ffe.load_config("configs/config.yml")

    exp_dir = os.path.join(config.get('run_dir', './runs/'), config['experiment_name'])
    combined_df = build_peaks_report(config, basin_ids=basin_ids, exp_dir=exp_dir)

    csv_path = os.path.join(exp_dir, "peaks_analysis.csv")
    os.makedirs(exp_dir, exist_ok=True)
    combined_df.to_csv(csv_path, index=False)

    for scope in SCOPES:
        model_mean = combined_df[(combined_df['basin_id'] == 'model_mean') & (combined_df['scope'] == scope)]
        model_var = combined_df[(combined_df['basin_id'] == 'model_variance') & (combined_df['scope'] == scope)]
        if not model_mean.empty:
            print(f"[INFO] Scope {scope} — mean time_distance_h: {model_mean['time_distance_h'].iloc[0]:.3f}, "
                  f"mean magnitude_diff_norm: {model_mean['magnitude_diff_norm'].iloc[0]:.3f}")
        if not model_var.empty:
            print(f"[INFO] Scope {scope} — variance time_distance_h: {model_var['time_distance_h'].iloc[0]:.3f}, "
                  f"variance magnitude_diff_norm: {model_var['magnitude_diff_norm'].iloc[0]:.3f}")

    print(f"[INFO] Wrote peaks analysis to {csv_path}")
    return csv_path


def collect_comparison_peak_pairs(merged_df, classified_rows, basin_id, model_leads, flow_std_map):
    """
    Multi-model peak-pair collection, called from model_compare_test.py's
    per-basin loop right after classify_clusters(). classified_rows is that
    call's tidy TP/FN/FP/TN row list; merged_df is the same basin's already
    target-time-aligned frame from merge_basin_reports (columns 'timestamp',
    'actual_flow', 'pred__{label}' per model). Returns a list of dicts (not a
    DataFrame) so the caller can accumulate across the whole basin loop.
    """
    pairs = []
    for row in classified_rows:
        if row['label'] not in ('TP', 'FN'):
            continue
        core_start = pd.to_datetime(row['core_start'])
        core_end = pd.to_datetime(row['core_end'])
        label_model = row['model_label']
        pred_col = f"pred__{label_model}"
        if pred_col not in merged_df.columns:
            continue

        pair = _build_event_pair_row(core_start, core_end, merged_df, merged_df, pred_col,
                                      basin_id, model_leads[label_model], row['label'], flow_std_map)
        if pair is not None:
            pair['model_label'] = label_model
            pairs.append(pair)

    return pairs


def build_comparison_peaks_report(comparison_detail_df, model_labels):
    """
    Assembles the multi-model peaks report as two separate frames: detail
    rows (grouped by model_label, label-tagged) and summary rows (each
    model's model_mean/model_variance rows per scope - no lead_mean tier,
    since each compared model has exactly one lead by construction, so
    model_mean already is that model's only-lead pooled stat).
    Returns (detail_df, summary_df).
    """
    if comparison_detail_df.empty:
        comparison_detail_df = pd.DataFrame(columns=['model_label'] + DETAIL_COLUMNS)

    summary_rows = []
    for label in model_labels:
        model_df = comparison_detail_df[comparison_detail_df['model_label'] == label]
        model_lead = model_df['lead'].iloc[0] if not model_df.empty else None

        for scope in SCOPES:
            scoped_df = model_df[model_df['label'] == 'TP'] if scope == 'TP' else model_df

            summary_rows.append({
                'model_label': label, 'basin_id': 'model_mean', 'lead': model_lead, 'scope': scope,
                'n_events': len(scoped_df),
                'time_distance_h': scoped_df['time_distance_h'].mean(),
                'magnitude_diff_norm': scoped_df['magnitude_diff_norm'].mean(),
            })
            summary_rows.append({
                'model_label': label, 'basin_id': 'model_variance', 'lead': model_lead, 'scope': scope,
                'n_events': len(scoped_df),
                'time_distance_h': scoped_df['time_distance_h'].var(),
                'magnitude_diff_norm': scoped_df['magnitude_diff_norm'].var(),
            })

    summary_df = pd.DataFrame(summary_rows)
    return comparison_detail_df, summary_df


if __name__ == "__main__":
    main()
