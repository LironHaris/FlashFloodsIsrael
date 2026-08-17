"""
Per-basin, per-hydrological-year flow summary: max flow, total annual flow
volume, count of real (non-zero, non-NaN) flow hours, and their percentage
of the year. Years a basin isn't actually "used" for (per the same
train/val/test availability-threshold logic as flow_quality_check.py) are
left blank.
"""

import os

import pandas as pd
import yaml
from tqdm import tqdm

from flow_quality_check import (
    get_hydrological_year,
    hydro_year_bounds,
    build_split_windows,
    is_year_used,
)


def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def compute_basin_year_stats(df, start_month):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['Hydro_Year'] = get_hydrological_year(df.index, start_month=start_month)

    stats = {}
    for year, group in df.groupby('Hydro_Year'):
        flow = group['Flow_m3_sec']
        start, end = hydro_year_bounds(year, start_month)
        total_hours_in_year = (end - start).total_seconds() / 3600
        nonzero_hours = (flow > 0).sum()
        stats[year] = {
            'max_flow': flow.max(),
            'total_flow_m3': flow.sum() * 3600,
            'nonzero_hours': nonzero_hours,
            'nonzero_pct': nonzero_hours / total_hours_in_year * 100,
        }
    return stats


def main(config):
    input_dir = config['processed_timeseries_dir']
    start_month = config.get('hydro_year_start_month', 10)
    output_path = config['yearly_analysis_output_file']
    min_pct = config['min_combined_availability_pct']

    split_windows = build_split_windows(config)
    availability_df = pd.read_csv(config['availability_report_file']).set_index('gauge_id')

    basin_files = sorted(f for f in os.listdir(input_dir) if f.endswith('.csv'))

    per_basin_stats = {}
    all_years = set()
    for file_name in tqdm(basin_files, desc="Yearly analysis"):
        basin_id = file_name.replace('.csv', '')
        df = pd.read_csv(os.path.join(input_dir, file_name))
        stats = compute_basin_year_stats(df, start_month)
        per_basin_stats[basin_id] = stats
        all_years.update(stats.keys())

    all_years = sorted(all_years)
    cols = ['max_flow', 'total_flow_m3', 'nonzero_hours', 'nonzero_pct']

    rows = []
    for basin_id, stats in per_basin_stats.items():
        row = {'basin_id': basin_id}
        availability_row = availability_df.loc[basin_id] if basin_id in availability_df.index else None
        for year in all_years:
            used = is_year_used(year, start_month, split_windows, availability_row, min_pct)
            year_stats = stats.get(year) if used else None
            for col in cols:
                row[f'{year}_{col}'] = year_stats[col] if year_stats is not None else pd.NA
        rows.append(row)

    result_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Wrote yearly analysis for {len(rows)} basins to {output_path}")


if __name__ == "__main__":
    CONFIG_PATH = "configs/config.yml"
    yaml_config = load_config(CONFIG_PATH)
    main(yaml_config)
