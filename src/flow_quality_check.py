"""
Per-basin, per-hydrological-year quality check of the flow record.

Runs against both the processed (hourly-resampled) timeseries and the raw
water-authority gauge log. For every basin CSV, buckets the `Flow_m3_sec`
column by hydrological year into NaN / zero / non-zero counts and flags
years with no real positive-flow signal at all. The processed-data check
additionally reports what fraction of that hydro year's timestamps are
missing from the file entirely (due to long-NaN-stretch rows being dropped
upstream) - not meaningful for the raw data, which has no fixed grid.
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


def compute_basin_year_counts(df, start_month, count_suffix='hours', include_missing_index_pct=True):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['Hydro_Year'] = get_hydrological_year(df.index, start_month=start_month)

    counts = {}
    for year, group in df.groupby('Hydro_Year'):
        flow = group['Flow_m3_sec']
        year_counts = {
            f'nan_{count_suffix}': flow.isna().sum(),
            f'zero_{count_suffix}': (flow == 0).sum(),
            f'nonzero_{count_suffix}': (flow > 0).sum(),
        }
        if include_missing_index_pct:
            start, end = hydro_year_bounds(year, start_month)
            total_hours = (end - start).total_seconds() / 3600
            year_counts['missing_index_pct'] = (total_hours - len(group)) / total_hours * 100
        counts[year] = year_counts
    return counts


def build_split_windows(config):
    windows = []
    for period in config['train_periods']:
        start_ts = pd.Timestamp(period['start_date']) if period['start_date'] else pd.Timestamp.min
        end_ts = pd.Timestamp(period['end_date']) if period['end_date'] else pd.Timestamp.max
        windows.append(('train', start_ts, end_ts))
    windows.append(('val', pd.Timestamp(config['validation_start_date']), pd.Timestamp(config['validation_end_date'])))
    windows.append(('test', pd.Timestamp(config['test_start_date']), pd.Timestamp(config['test_end_date'])))
    return windows


def assign_split(year, start_month, split_windows):
    year_start, year_end = hydro_year_bounds(year, start_month)
    best_split, best_overlap = None, pd.Timedelta(0)
    for name, start, end in split_windows:
        overlap = min(year_end, end) - max(year_start, start)
        if overlap > best_overlap:
            best_overlap, best_split = overlap, name
    return best_split


def is_year_used(year, start_month, split_windows, availability_row, min_pct):
    split = assign_split(year, start_month, split_windows)
    if split is None or availability_row is None:
        return False
    return availability_row[f'combined_availability_pct_{split}'] >= min_pct


def run_quality_check(config, input_dir, output_path, count_suffix='hours', include_missing_index_pct=True):
    start_month = config.get('hydro_year_start_month', 10)
    min_pct = config['min_combined_availability_pct']

    split_windows = build_split_windows(config)
    availability_df = pd.read_csv(config['availability_report_file']).set_index('gauge_id')

    basin_files = sorted(f for f in os.listdir(input_dir) if f.endswith('.csv'))

    per_basin_counts = {}
    all_years = set()
    for file_name in tqdm(basin_files, desc=f"Checking {os.path.basename(output_path)}"):
        basin_id = file_name.replace('.csv', '')
        df = pd.read_csv(os.path.join(input_dir, file_name))
        counts = compute_basin_year_counts(df, start_month, count_suffix, include_missing_index_pct)
        per_basin_counts[basin_id] = counts
        all_years.update(counts.keys())

    all_years = sorted(all_years)
    count_keys = [f'nan_{count_suffix}', f'zero_{count_suffix}', f'nonzero_{count_suffix}']
    if include_missing_index_pct:
        count_keys.append('missing_index_pct')

    rows = []
    for basin_id, counts in per_basin_counts.items():
        row = {'basin_id': basin_id}
        availability_row = availability_df.loc[basin_id] if basin_id in availability_df.index else None
        used_no_flow_years = []
        for year in all_years:
            year_counts = counts.get(year)
            if year_counts is None:
                for key in count_keys:
                    row[f'{year}_{key}'] = 100.0 if key == 'missing_index_pct' else pd.NA
                no_flow = True
            else:
                for key in count_keys:
                    row[f'{year}_{key}'] = year_counts[key]
                no_flow = year_counts[f'nonzero_{count_suffix}'] == 0

            if no_flow and is_year_used(year, start_month, split_windows, availability_row, min_pct):
                used_no_flow_years.append(year)

        row['used_years_no_flow_record'] = ';'.join(str(y) for y in used_no_flow_years)
        rows.append(row)

    result_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Wrote quality check for {len(rows)} basins to {output_path}")


def main(config):
    run_quality_check(
        config,
        input_dir=config['processed_timeseries_dir'],
        output_path=config['flow_quality_check_output_file'],
        count_suffix='hours',
        include_missing_index_pct=True,
    )
    run_quality_check(
        config,
        input_dir=config['raw_dynamic_dir'],
        output_path=config['water_authority_analysis_output_file'],
        count_suffix='readings',
        include_missing_index_pct=False,
    )


if __name__ == "__main__":
    CONFIG_PATH = "configs/config.yml"
    yaml_config = load_config(CONFIG_PATH)
    main(yaml_config)
