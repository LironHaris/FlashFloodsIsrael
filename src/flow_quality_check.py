"""
Per-basin, per-hydrological-year quality check of the processed flow record.

For every basin CSV in `processed_timeseries_dir`, buckets the `Flow_m3_sec`
column by hydrological year into NaN / zero / non-zero hour counts, reports
what fraction of that hydro year's timestamps are missing from the file
entirely (due to long-NaN-stretch rows being dropped upstream), and flags
years with no real positive-flow signal at all.
"""

import os

import pandas as pd
import yaml
from tqdm import tqdm


def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def get_hydrological_year(date_index, start_month=10):
    return date_index.year + (date_index.month >= start_month).astype(int)


def hydro_year_bounds(year, start_month):
    start = pd.Timestamp(year=year - 1, month=start_month, day=1)
    end = pd.Timestamp(year=year, month=start_month, day=1)
    return start, end


def compute_basin_year_counts(df, start_month):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['Hydro_Year'] = get_hydrological_year(df.index, start_month=start_month)

    counts = {}
    for year, group in df.groupby('Hydro_Year'):
        flow = group['Flow_m3_sec']
        start, end = hydro_year_bounds(year, start_month)
        total_hours = (end - start).total_seconds() / 3600
        missing_index_pct = (total_hours - len(group)) / total_hours * 100
        counts[year] = {
            'nan_hours': flow.isna().sum(),
            'zero_hours': (flow == 0).sum(),
            'nonzero_hours': (flow > 0).sum(),
            'missing_index_pct': missing_index_pct,
        }
    return counts


def main(config):
    input_dir = config['processed_timeseries_dir']
    start_month = config.get('hydro_year_start_month', 10)
    output_path = config['flow_quality_check_output_file']

    basin_files = sorted(f for f in os.listdir(input_dir) if f.endswith('.csv'))

    per_basin_counts = {}
    all_years = set()
    for file_name in tqdm(basin_files, desc="Checking flow quality"):
        basin_id = file_name.replace('.csv', '')
        df = pd.read_csv(os.path.join(input_dir, file_name))
        counts = compute_basin_year_counts(df, start_month)
        per_basin_counts[basin_id] = counts
        all_years.update(counts.keys())

    all_years = sorted(all_years)

    rows = []
    for basin_id, counts in per_basin_counts.items():
        row = {'basin_id': basin_id}
        no_flow_years = []
        for year in all_years:
            year_counts = counts.get(year)
            if year_counts is None:
                row[f'{year}_nan_hours'] = pd.NA
                row[f'{year}_zero_hours'] = pd.NA
                row[f'{year}_nonzero_hours'] = pd.NA
                row[f'{year}_missing_index_pct'] = 100.0
                no_flow_years.append(year)
            else:
                row[f'{year}_nan_hours'] = year_counts['nan_hours']
                row[f'{year}_zero_hours'] = year_counts['zero_hours']
                row[f'{year}_nonzero_hours'] = year_counts['nonzero_hours']
                row[f'{year}_missing_index_pct'] = year_counts['missing_index_pct']
                if year_counts['nonzero_hours'] == 0:
                    no_flow_years.append(year)
        row['years_no_flow_record'] = ';'.join(str(y) for y in no_flow_years)
        rows.append(row)

    result_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Wrote flow quality check for {len(rows)} basins to {output_path}")


if __name__ == "__main__":
    CONFIG_PATH = "configs/config.yml"
    yaml_config = load_config(CONFIG_PATH)
    main(yaml_config)
