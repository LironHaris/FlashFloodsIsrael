"""
Module: data_availability_audit.py
Description: Two-part audit of the actual processed training data
             (data/processed/timeseries/), per basin and hydrological year,
             for the basins used in the pipeline (train+val+test union):
               1. How much basin-year data actually exists - total hourly
                  rows vs. rows with a real (non-NaN) Flow_m3_sec value.
               2. Of those non-NaN hourly rows, how many fall on a calendar
                  date that nan_flow_days.py already flagged as missing in
                  the raw water-authority record for that basin - i.e.
                  timesteps the model trains on despite the raw daily
                  source considering that day missing/NaN.
             Reuses nan_flow_days.py's basin-list logic and
             flow_quality_check.py's hydrological-year bucketing.
"""

import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from nan_flow_days import load_target_basin_ids
from flow_quality_check import get_hydrological_year


def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


ROW_FIELDS = [
    'basin_id', 'year', 'total_hours', 'valid_hours',
    'nan_flow_days_in_year', 'used_timesteps_on_nan_flow_days',
]


def load_processed_basin_df(processed_timeseries_dir, il_id, hydro_year_start_month):
    path = os.path.join(processed_timeseries_dir, f"{il_id}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = get_hydrological_year(pd.DatetimeIndex(df['date']), start_month=hydro_year_start_month)
    df['is_valid'] = df['Flow_m3_sec'].notna()
    return df


def load_nan_flow_dates(nan_flow_days_dir, il_id):
    path = os.path.join(nan_flow_days_dir, f"{il_id}.csv")
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return set(pd.to_datetime(df['date']).dt.date)


def audit_basin(il_id, processed_df, nan_flow_dates, hydro_year_start_month):
    processed_df = processed_df.copy()
    processed_df['calendar_date'] = processed_df['date'].dt.date
    processed_df['on_nan_flow_day'] = processed_df['calendar_date'].isin(nan_flow_dates)

    nan_flow_years = pd.Series(sorted(nan_flow_dates)).apply(
        lambda d: get_hydrological_year(pd.DatetimeIndex([d]), start_month=hydro_year_start_month)[0]
    ) if nan_flow_dates else pd.Series([], dtype=int)

    rows = []
    for year, year_df in processed_df.groupby('year'):
        total_hours = len(year_df)
        valid_hours = int(year_df['is_valid'].sum())
        nan_flow_days_in_year = int((nan_flow_years == year).sum())
        used_on_nan_days = int((year_df['is_valid'] & year_df['on_nan_flow_day']).sum())
        rows.append({
            'basin_id': il_id, 'year': year,
            'total_hours': total_hours, 'valid_hours': valid_hours,
            'nan_flow_days_in_year': nan_flow_days_in_year,
            'used_timesteps_on_nan_flow_days': used_on_nan_days,
        })
    return rows


def _summary_row(basin_id, year, rows):
    return {
        'basin_id': basin_id, 'year': year,
        'total_hours': sum(r['total_hours'] for r in rows),
        'valid_hours': sum(r['valid_hours'] for r in rows),
        'nan_flow_days_in_year': sum(r['nan_flow_days_in_year'] for r in rows),
        'used_timesteps_on_nan_flow_days': sum(r['used_timesteps_on_nan_flow_days'] for r in rows),
    }


def main(config):
    processed_timeseries_dir = config['processed_timeseries_dir']
    nan_flow_days_dir = config['nan_flow_days_output_dir']
    output_path = config['data_availability_audit_output_file']
    hydro_year_start_month = config.get('hydro_year_start_month', 10)

    target_basins = load_target_basin_ids(config)
    print(f"[INFO] Auditing {len(target_basins)} basins (train+val+test union).")

    all_rows = []
    for il_id in sorted(target_basins):
        processed_df = load_processed_basin_df(processed_timeseries_dir, il_id, hydro_year_start_month)
        if processed_df is None:
            print(f"  [Warning] {il_id}: no processed timeseries CSV found. Skipping.")
            continue

        nan_flow_dates = load_nan_flow_dates(nan_flow_days_dir, il_id)
        basin_rows = audit_basin(il_id, processed_df, nan_flow_dates, hydro_year_start_month)
        all_rows.extend(basin_rows)

        basin_summary = _summary_row(il_id, 'ALL_YEARS', basin_rows)
        all_rows.append(basin_summary)
        print(f"  {il_id}: {len(basin_rows)} basin-years, valid_hours={basin_summary['valid_hours']}, "
              f"used_timesteps_on_nan_flow_days={basin_summary['used_timesteps_on_nan_flow_days']}")

    detail_rows = [r for r in all_rows if r['year'] != 'ALL_YEARS']
    basin_year_rows_with_data = sum(1 for r in detail_rows if r['valid_hours'] > 0)
    grand_total = _summary_row('ALL_BASINS', 'ALL_YEARS', detail_rows)
    grand_total['basin_years_with_data'] = basin_year_rows_with_data
    all_rows.append(grand_total)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ROW_FIELDS + ['basin_years_with_data']
    pd.DataFrame(all_rows, columns=fieldnames).to_csv(output_path, index=False)

    print(f"\n[INFO] Basin-years with at least one valid hour: {basin_year_rows_with_data} "
          f"(out of {len(detail_rows)} basin-year rows total)")
    print(f"[INFO] Total valid hours: {grand_total['valid_hours']}")
    print(f"[INFO] Total used timesteps on nan_flow_days-flagged days: "
          f"{grand_total['used_timesteps_on_nan_flow_days']}")
    print(f"[INFO] Wrote {len(all_rows)} rows to {output_path}")


if __name__ == "__main__":
    CONFIG_PATH = "configs/config.yml"
    yaml_config = load_config(CONFIG_PATH)
    main(yaml_config)
