"""
Adds two columns to station quality.csv: whether each station has at least
one hydrological year actually used (per flow_quality_check.py's train/val/
test availability-threshold logic) with real surviving data, and which years.
"""

import os

import pandas as pd
import yaml

from flow_quality_check import get_hydrological_year, build_split_windows, is_year_used


def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def get_used_years_for_basin(basin_id, processed_dir, start_month, split_windows, availability_row, min_pct):
    file_path = os.path.join(processed_dir, f'{basin_id}.csv')
    if not os.path.exists(file_path):
        return []

    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    hydro_years = get_hydrological_year(pd.DatetimeIndex(df['date']), start_month=start_month)
    years_present = sorted(set(hydro_years))

    return [
        year for year in years_present
        if is_year_used(year, start_month, split_windows, availability_row, min_pct)
    ]


def main(config):
    station_quality_path = config['station_quality_file']
    processed_dir = config['processed_timeseries_dir']
    start_month = config.get('hydro_year_start_month', 10)
    min_pct = config['min_combined_availability_pct']

    split_windows = build_split_windows(config)
    availability_df = pd.read_csv(config['availability_report_file']).set_index('gauge_id')

    station_df = pd.read_csv(station_quality_path)

    has_used_year = []
    used_years_col = []
    for basin_id in station_df['station_id']:
        availability_row = availability_df.loc[basin_id] if basin_id in availability_df.index else None
        used_years = get_used_years_for_basin(
            basin_id, processed_dir, start_month, split_windows, availability_row, min_pct
        )
        has_used_year.append(len(used_years) > 0)
        used_years_col.append(';'.join(str(y) for y in used_years))

    station_df['has_used_year'] = has_used_year
    station_df['used_years'] = used_years_col

    station_df.to_csv(station_quality_path, index=False)
    print(f"Updated {station_quality_path} with used-year columns for {len(station_df)} stations")


if __name__ == "__main__":
    CONFIG_PATH = "configs/config.yml"
    yaml_config = load_config(CONFIG_PATH)
    main(yaml_config)
