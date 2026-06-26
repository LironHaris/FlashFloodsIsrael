"""
Module: find_flood_events.py
Description: Automated Flash Flood Event Finder for Selected Return Periods.
             Scans test outputs to locate timestamps where actual discharge 
             crossed critical benchmarks and appends visual padding buffers.
"""

import os
import yaml
import pandas as pd


def load_config(yaml_path):
    """Load the YAML configuration file safely."""
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def load_basin_threshold(basin_id, target_rp, config):
    """Loads the cross-checked streamflow threshold for a given return period from
    the combined return-period file (see new_return_periods.py)."""
    combined_path = config['hourly_flow_return_periods_combined']
    if not os.path.exists(combined_path):
        print(f"  [Warning] Combined return-period file missing at {combined_path}.")
        return None

    combined_df = pd.read_csv(combined_path, dtype={'basin_id': str})
    col = f'RP{target_rp}'
    if col not in combined_df.columns:
        available_rps = [c[2:] for c in combined_df.columns if c.startswith('RP')]
        print(f"  [Warning] Return Period {target_rp}yr not available in combined file. Available: {available_rps}")
        return None

    row = combined_df[combined_df['basin_id'] == str(basin_id)]
    if row.empty:
        print(f"  [Warning] Basin {basin_id} not found in combined return-period file.")
        return None

    value = row.iloc[0][col]
    if pd.isna(value) or value <= 0:
        return None
    return float(value)


def scan_for_events(basin_id, target_rp, buffer_days, config):
    """
    Scans the basin's test report to isolate continuous blocks where the 
    actual streamflow crossed the designated threshold, returning bound parameters.
    """
    run_dir = config.get('run_dir', './runs/')
    exp_dir = os.path.join(run_dir, config['experiment_name'])
    report_path = os.path.join(exp_dir, "visualization_reports", f"visual_report_basin_{basin_id}.csv")
    
    if not os.path.exists(report_path):
        print(f"  [Error] Evaluation report missing for basin {basin_id}. Run test.py first.")
        return []
        
    # Load test results
    df = pd.read_csv(report_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Fetch the critical physical threshold value
    threshold_value = load_basin_threshold(basin_id, target_rp, config)
    if threshold_value is None or threshold_value <= 0:
        print(f"  [Info] Threshold value for {target_rp}yr RP is unstable (<=0). Skipping basin scan.")
        return []

    # Locate indices where actual flow crossed the threshold
    exceedance_mask = df['actual_flow'] >= threshold_value
    if not exceedance_mask.any():
        print(f"  [-] Basin {basin_id} never exceeded the {target_rp}-year threshold during the test split.")
        return []

    # Step 1 — find initial exceedance blocks
    df['is_exceedance'] = exceedance_mask.astype(int)
    df['event_id'] = (df['is_exceedance'] != df['is_exceedance'].shift()).cumsum()

    # Step 2 — collect raw core events (no padding yet)
    raw_events = []
    for _, chunk in df[df['is_exceedance'] == 1].groupby('event_id'):
        raw_events.append({
            'core_start': chunk['timestamp'].min(),
            'core_end':   chunk['timestamp'].max(),
            'peak_flow':  chunk['actual_flow'].max(),
        })

    # Step 3 — merge events whose inter-event gap is within the configured tolerance
    merge_gap_hours = config.get('event_merge_gap_hours', 0)
    if merge_gap_hours > 0 and len(raw_events) > 1:
        merge_td = pd.Timedelta(hours=merge_gap_hours)
        merged = [raw_events[0].copy()]
        for ev in raw_events[1:]:
            gap = ev['core_start'] - merged[-1]['core_end']
            if gap <= merge_td:
                merged[-1]['core_end']  = ev['core_end']
                merged[-1]['peak_flow'] = max(merged[-1]['peak_flow'], ev['peak_flow'])
            else:
                merged.append(ev.copy())
        raw_events = merged

    print(f"  [★] Found {len(raw_events)} distinct exceedance event(s) for {target_rp}yr RP:")

    # Step 4 — apply padding and format output
    discovered_events = []
    for ev in raw_events:
        padded_start = max(ev['core_start'] - pd.Timedelta(days=buffer_days), df['timestamp'].min())
        padded_end   = min(ev['core_end']   + pd.Timedelta(days=buffer_days), df['timestamp'].max())

        discovered_events.append({
            'core_start':       ev['core_start'].strftime('%Y-%m-%d %H:%M:%S'),
            'core_end':         ev['core_end'].strftime('%Y-%m-%d %H:%M:%S'),
            'peak_flow':        float(ev['peak_flow']),
            'plot_ready_start': padded_start.strftime('%Y-%m-%d %H:%M:%S'),
            'plot_ready_end':   padded_end.strftime('%Y-%m-%d %H:%M:%S'),
        })

        print(f"      • Event Peak: {ev['peak_flow']:.2f} m3/s | Core Duration: [{ev['core_start']} -> {ev['core_end']}]")
        print(f"        Padded Window for Plotting: '{padded_start}' TO '{padded_end}'")

    return discovered_events


def main(config=None, basin_ids=None):
    if config is None:
        config = load_config("configs/config.yml")

    buffer_days   = config.get('visual_buffer_days', 4)
    return_periods = config.get('return_periods_years', [2, 5, 10, 15, 20, 30])
    output_path   = config['find_flood_events_output']

    if basin_ids is None:
        with open(config['test_basin_file']) as f:
            basins = [line.strip() for line in f if line.strip()]
    else:
        basins = list(basin_ids)

    print("=" * 75)
    print("      Automated Flash Flood Event Scanner — Test Dataset Evaluation")
    print("=" * 75)
    print(f"[INFO] Basins: {len(basins)} | Return periods: {return_periods} | Merge gap: {config.get('event_merge_gap_hours', 0)}h")
    print("-" * 75 + "\n")

    rows = []
    for basin in basins:
        print(f"Scanning Basin: {basin}...")
        for rp in return_periods:
            events = scan_for_events(basin, rp, buffer_days, config)
            for idx, ev in enumerate(events):
                rows.append({
                    "basin_id":            basin,
                    "return_period_years": rp,
                    "event_idx":           idx + 1,
                    "core_start":          ev["core_start"],
                    "core_end":            ev["core_end"],
                    "peak_flow":           ev["peak_flow"],
                    "plot_ready_start":    ev["plot_ready_start"],
                    "plot_ready_end":      ev["plot_ready_end"],
                })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\n[INFO] Wrote {len(rows)} flood events to {output_path}")


if __name__ == "__main__":
    main()