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
    """Loads the specific GEV streamflow threshold for a given return period."""
    eva_dir = config.get('return_periods_output_dir', 'data/processed/return_periods')
    gev_path = os.path.join(eva_dir, str(basin_id), 'Hourly_Flow_theoretical_GEV.csv')
    
    if not os.path.exists(gev_path):
        print(f"  [Warning] GEV stats missing for basin {basin_id}. Cannot evaluate thresholds.")
        return None
        
    gev_df = pd.read_csv(gev_path)
    row = gev_df[gev_df['Return_Period_Years'] == target_rp]
    
    if row.empty:
        available_rps = gev_df['Return_Period_Years'].tolist()
        print(f"  [Warning] Return Period {target_rp}yr not found for basin {basin_id}. Available: {available_rps}")
        return None
        
    return float(row['Theoretical_Value'].values[0])


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

    # Group consecutive exceedance hours into distinct flood events
    df['is_exceedance'] = exceedance_mask.astype(int)
    # Track continuous events using a cumulative sum on shift changes
    df['event_id'] = (df['is_exceedance'] != df['is_exceedance'].shift()).cumsum()
    flood_events_df = df[df['is_exceedance'] == 1]
    
    discovered_events = []
    grouped = flood_events_df.groupby('event_id')
    
    print(f"  [★] Found {len(grouped)} distinct exceedance event(s) for {target_rp}yr RP:")
    
    for _, event_chunk in grouped:
        core_start = event_chunk['timestamp'].min()
        core_end = event_chunk['timestamp'].max()
        peak_flow = event_chunk['actual_flow'].max()
        
        # Apply the visual padding buffer (expanding backwards and forwards in time)
        padded_start = core_start - pd.Timedelta(days=buffer_days)
        padded_end = core_end + pd.Timedelta(days=buffer_days)
        
        # Constrain padding to dataset boundaries
        padded_start = max(padded_start, df['timestamp'].min())
        padded_end = min(padded_end, df['timestamp'].max())
        
        event_meta = {
            'core_start': core_start.strftime('%Y-%m-%d %H:%M:%S'),
            'core_end': core_end.strftime('%Y-%m-%d %H:%M:%S'),
            'peak_flow': float(peak_flow),
            'plot_ready_start': padded_start.strftime('%Y-%m-%d %H:%M:%S'),
            'plot_ready_end': padded_end.strftime('%Y-%m-%d %H:%M:%S')
        }
        discovered_events.append(event_meta)
        
        print(f"      • Event Peak: {peak_flow:.2f} m3/s | Core Duration: [{core_start} -> {core_end}]")
        print(f"        Padded Window for Plotting: '{padded_start}' TO '{padded_end}'")
        
    return discovered_events


def main():
    config = load_config("configs/config.yml")
    
    # Extract the visual padding buffer from configuration
    buffer_days = config.get('visual_buffer_days', 4)
    
    print("=" * 75)
    print("      Automated Flash Flood Event Scanner — Test Dataset Evaluation")
    print("=" * 75)
    
    # Interactive input prompts with explicit format instructions
    print("\n[!] Basin ID Format Note: Use the naming convention prefix 'il_' followed by digits.")
    print("    Example for single basin: il_123")
    print("    Example for multiple basins: il_04, il_5678, il_991234")
    
    basin_input = input("\n[?] Enter target basin ID(s) (separate multiple items with a comma): ")
    # Clean whitespace and parse into a clean list of strings
    target_basins = [b.strip() for b in basin_input.split(",") if b.strip()]
    
    available_rps = config.get('return_periods_years', [2, 5, 10, 15, 20, 30])
    rp_input = input(f"[?] Enter the target Return Period in years {available_rps}: ")
    try:
        target_return_period = int(rp_input.strip())
    except ValueError:
        print("\n[ERROR] Return Period must be a valid integer. Execution terminated.")
        return

    print("\n" + "-" * 75)
    print(f"[INFO] Initiating automated flood scan for Return Period: {target_return_period} Years")
    print(f"[INFO] Applied visual padding buffer (drawn from config): {buffer_days} days per window side")
    print("-" * 75 + "\n")
    
    all_extracted_windows = {}
    
    # Scan through requested basins sequentially
    for basin in target_basins:
        print(f"Scanning Dataset Context for Basin: {basin}...")
        events = scan_for_events(basin, target_return_period, buffer_days, config)
        if events:
            all_extracted_windows[basin] = events
            
    # Compile and display ready-to-copy parameters
    if all_extracted_windows:
        print("\n" + "=" * 75)
        print("[SUMMARY] Ready Python inputs to copy directly into plot_hydrographs.py:")
        print("=" * 75)
        
        for basin, events in all_extracted_windows.items():
            for idx, ev in enumerate(events):
                print(f"\n# Basin {basin} - Event {idx+1} (Peak: {ev['peak_flow']:.1f} m3/s)")
                print(f"STORM_START = \"{ev['plot_ready_start']}\"")
                print(f"STORM_END = \"{ev['plot_ready_end']}\"")
    else:
        print("\n[-] No threshold exceedance events discovered for the selected configuration in the test split.")


if __name__ == "__main__":
    main()