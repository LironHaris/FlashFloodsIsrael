"""
Module: data_type_mask.py
Description: For every basin timeseries in processed_timeseries_dir, counts how
             many of its hourly time steps fall inside periods the water
             authority labeled with a selected Flow_type in
             hydrographs_unified.csv, and writes one CSV row per basin (plus a
             final TOTAL row) with the total step count, the flow step count and
             the flow percentage. One CSV is written per entry in OUTPUTS:
               - data_type_mask.csv:             Flow_type == 'flow'
               - data_type_mask_flow_normal.csv: Flow_type in {'flow', 'normal'}

             A record's Flow_type holds from its timestamp until the station's
             next record, but never across a hydrograph boundary (an 'end'
             record, or a gap before the next 'start' record). An hourly step t
             covers [t, t+1h) - the same left-closed/left-labeled convention as
             preprocess_dynamic_data.resample_to_hourly - and counts as 'flow'
             if any selected interval overlaps it.
"""

import os

import numpy as np
import pandas as pd
import yaml

UNIFIED_FILE = './data/water authority/hydrographs_unified.csv'
OUTPUTS = {
    './data/processed/data quality reports/data_type_mask.csv': {'flow'},
    './data/processed/data quality reports/data_type_mask_flow_normal.csv': {'flow', 'normal'},
}
ONE_HOUR = np.timedelta64(1, 'h')


def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def load_unified(path):
    df = pd.read_csv(path, usecols=['Station_ID', 'Flow_sampling_time', 'Flow_type', 'Record_type'],
                     dtype=str)
    for col in df.columns:
        df[col] = df[col].str.strip()
    df['Flow_sampling_time'] = pd.to_datetime(df['Flow_sampling_time'], format='%d/%m/%Y %H:%M:%S')
    return df


def build_flow_intervals(station_df, flow_types):
    """
    Returns (starts, ends): sorted, disjoint datetime64 arrays of the merged
    [start, end) intervals during which the station's Flow_type is in flow_types.
    """
    station_df = station_df.sort_values('Flow_sampling_time', kind='stable')
    times = station_df['Flow_sampling_time'].to_numpy()
    is_flow = station_df['Flow_type'].isin(flow_types).to_numpy()
    record_type = station_df['Record_type'].to_numpy()

    # Interval i spans [times[i], times[i+1]) - kept only if record i has a
    # selected Flow_type and the pair doesn't straddle two hydrographs.
    keep = is_flow[:-1] & (record_type[:-1] != 'end') & (record_type[1:] != 'start')
    starts = times[:-1][keep]
    ends = times[1:][keep]
    if len(starts) == 0:
        return starts, ends

    merged_starts, merged_ends = [starts[0]], [ends[0]]
    for s, e in zip(starts[1:], ends[1:]):
        if s <= merged_ends[-1]:
            merged_ends[-1] = max(merged_ends[-1], e)
        else:
            merged_starts.append(s)
            merged_ends.append(e)
    return np.array(merged_starts), np.array(merged_ends)


def flow_mask(hours, starts, ends):
    """True for each hour h whose [h, h+1h) overlaps any [start, end) interval."""
    if len(starts) == 0:
        return np.zeros(len(hours), dtype=bool)
    idx = np.searchsorted(starts, hours + ONE_HOUR, side='left') - 1
    valid = idx >= 0
    mask = np.zeros(len(hours), dtype=bool)
    mask[valid] = ends[idx[valid]] > hours[valid]
    return mask


def _pct(part, whole):
    return round(100.0 * part / whole, 2) if whole else 0.0


def count_basins(timeseries_dir, by_station, flow_types):
    """Returns a per-basin DataFrame (plus a final TOTAL row) of total vs. selected-Flow_type steps."""
    rows = []
    basin_files = sorted(f for f in os.listdir(timeseries_dir) if f.endswith('.csv'))
    for file_name in basin_files:
        basin_id = file_name.replace('.csv', '')
        station_id = basin_id.replace('il_', '', 1)

        ts = pd.read_csv(os.path.join(timeseries_dir, file_name), usecols=['date'])
        hours = pd.to_datetime(ts['date']).to_numpy()

        station_df = by_station.get(station_id)
        if station_df is None:
            print(f"  [Warning] {basin_id}: no records in unified hydrographs. Counting 0 flow steps.")
            n_flow = 0
        else:
            starts, ends = build_flow_intervals(station_df, flow_types)
            n_flow = int(flow_mask(hours, starts, ends).sum())

        n_total = len(hours)
        rows.append({'basin_id': basin_id, 'total_timesteps': n_total,
                     'flow_timesteps': n_flow, 'flow_pct': _pct(n_flow, n_total)})
        print(f"  {basin_id}: {n_flow}/{n_total} flow steps ({_pct(n_flow, n_total)}%)")

    result = pd.DataFrame(rows, columns=['basin_id', 'total_timesteps', 'flow_timesteps', 'flow_pct'])
    total = int(result['total_timesteps'].sum())
    total_flow = int(result['flow_timesteps'].sum())
    result.loc[len(result)] = ['TOTAL', total, total_flow, _pct(total_flow, total)]
    return result


def main(config):
    timeseries_dir = config['processed_timeseries_dir']

    print(f"[INFO] Loading {UNIFIED_FILE} ...")
    unified = load_unified(UNIFIED_FILE)
    by_station = {station_id: df for station_id, df in unified.groupby('Station_ID')}

    for output_file, flow_types in OUTPUTS.items():
        print(f"\n[INFO] Counting steps with Flow_type in {sorted(flow_types)} ...")
        result = count_basins(timeseries_dir, by_station, flow_types)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        result.to_csv(output_file, index=False)
        total_row = result.iloc[-1]
        print(f"[INFO] {len(result) - 1} basins, {total_row['flow_timesteps']}/{total_row['total_timesteps']} "
              f"basin-time pairs in {sorted(flow_types)} ({total_row['flow_pct']}%). Wrote {output_file}")


if __name__ == "__main__":
    CONFIG_PATH = "configs/config.yml"
    yaml_config = load_config(CONFIG_PATH)
    main(yaml_config)
