"""
Module: compare_flood_events.py
Description: Matches real (actual-flow) and predicted flood events by time-window
             overlap, classifying each as TP/FP/FN, so test.py/quick_test.py can
             plot one graph per event instead of duplicating real+predicted plots.
"""

import os
import pandas as pd

import find_flood_events as ffe


def _overlaps(a_start, a_end, b_start, b_end):
    return a_start <= b_end and b_start <= a_end


def match_events(real_events, predicted_events):
    """
    real_events, predicted_events: lists of dicts with 'core_start' (Timestamp),
    'core_end' (Timestamp), 'peak_flow' (float).
    Returns a chronologically sorted list of dicts: label ('TP'/'FN'/'FP'),
    core_start, core_end (union of both windows for TP), real_peak_flow,
    predicted_peak_flow (either may be None).
    """
    matched_pred_idxs = set()
    results = []

    for real_ev in real_events:
        match_idx = next(
            (i for i, p in enumerate(predicted_events)
             if i not in matched_pred_idxs and _overlaps(real_ev['core_start'], real_ev['core_end'],
                                                           p['core_start'], p['core_end'])),
            None,
        )
        if match_idx is not None:
            pred_ev = predicted_events[match_idx]
            matched_pred_idxs.add(match_idx)
            results.append({
                'label': 'TP',
                'core_start': min(real_ev['core_start'], pred_ev['core_start']),
                'core_end': max(real_ev['core_end'], pred_ev['core_end']),
                'real_peak_flow': real_ev['peak_flow'],
                'predicted_peak_flow': pred_ev['peak_flow'],
            })
        else:
            results.append({
                'label': 'FN',
                'core_start': real_ev['core_start'],
                'core_end': real_ev['core_end'],
                'real_peak_flow': real_ev['peak_flow'],
                'predicted_peak_flow': None,
            })

    for i, pred_ev in enumerate(predicted_events):
        if i not in matched_pred_idxs:
            results.append({
                'label': 'FP',
                'core_start': pred_ev['core_start'],
                'core_end': pred_ev['core_end'],
                'real_peak_flow': None,
                'predicted_peak_flow': pred_ev['peak_flow'],
            })

    results.sort(key=lambda r: r['core_start'])
    return results


def main(config=None, basin_ids=None):
    """
    Runs the real-vs-predicted event match SEPARATELY for every configured
    prediction_threshold spec (return_period and/or specific_discharge),
    writing one flood_event_comparison_{label}.csv per threshold. Returns
    {label: output_path}.
    """
    if config is None:
        config = ffe.load_config("configs/config.yml")

    prediction_specs = ffe.normalize_threshold_specs(config.get('prediction_threshold', 2))
    longest_lead = max(config.get('forecast_lead_times', [0, 1, 2, 3]))

    run_dir = config.get('run_dir', './runs/')
    exp_dir = os.path.join(run_dir, config['experiment_name'])

    real_path = config['find_flood_events_output']
    predicted_path = os.path.join(exp_dir, "predicted_flood_events.csv")

    all_real_df = pd.read_csv(real_path, dtype={'basin_id': str}) if os.path.exists(real_path) else pd.DataFrame()
    all_predicted_df = pd.read_csv(predicted_path, dtype={'basin_id': str}) if os.path.exists(predicted_path) else pd.DataFrame()

    output_paths = {}
    for spec in prediction_specs:
        label = ffe.threshold_label(spec)

        real_df = all_real_df[all_real_df['threshold_label'] == label] if not all_real_df.empty else all_real_df
        predicted_df = (all_predicted_df[(all_predicted_df['threshold_label'] == label) &
                                          (all_predicted_df['lead_time_h'] == longest_lead)]
                        if not all_predicted_df.empty else all_predicted_df)

        if basin_ids is None:
            basins = sorted(set(real_df.get('basin_id', [])) | set(predicted_df.get('basin_id', [])))
        else:
            basins = list(basin_ids)

        rows = []
        for basin in basins:
            real_events = [
                {'core_start': pd.to_datetime(r['core_start']), 'core_end': pd.to_datetime(r['core_end']), 'peak_flow': r['peak_flow']}
                for _, r in real_df[real_df.get('basin_id', pd.Series(dtype=str)) == basin].iterrows()
            ] if not real_df.empty else []
            predicted_events = [
                {'core_start': pd.to_datetime(r['core_start']), 'core_end': pd.to_datetime(r['core_end']), 'peak_flow': r['peak_flow']}
                for _, r in predicted_df[predicted_df.get('basin_id', pd.Series(dtype=str)) == basin].iterrows()
            ] if not predicted_df.empty else []

            if not real_events and not predicted_events:
                continue

            for idx, ev in enumerate(match_events(real_events, predicted_events), start=1):
                rows.append({
                    'basin_id': basin,
                    'event_idx': idx,
                    'label': ev['label'],
                    'threshold_label': label,
                    'lead_time_h': longest_lead,
                    'core_start': ev['core_start'].strftime('%Y-%m-%d %H:%M:%S'),
                    'core_end': ev['core_end'].strftime('%Y-%m-%d %H:%M:%S'),
                    'real_peak_flow': ev['real_peak_flow'],
                    'predicted_peak_flow': ev['predicted_peak_flow'],
                })

        columns = ['basin_id', 'event_idx', 'label', 'threshold_label', 'lead_time_h',
                   'core_start', 'core_end', 'real_peak_flow', 'predicted_peak_flow']
        output_path = os.path.join(exp_dir, f"flood_event_comparison_{label}.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)
        print(f"[INFO] Wrote {len(rows)} compared flood events (TP/FP/FN) for threshold {label} to {output_path}")
        output_paths[label] = output_path

    return output_paths


if __name__ == "__main__":
    main()
