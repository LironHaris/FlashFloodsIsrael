"""
Module: nan_flow_days.py
Description: For each basin actually used in the modeling pipeline (the
             union of israel_train.txt/israel_val.txt/israel_test.txt),
             finds every "missing flow day" in the raw water-authority
             daily-flow record - a calendar day within that basin's own
             observed date range that either has no row at all across the
             5 daily_flow_*.csv files, or has a row with an empty daily
             volume / avg daily flow value - and writes one CSV per basin
             listing them.
"""

import glob
import os

import pandas as pd
import yaml


def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def _read_basin_list(path):
    with open(path, encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}


def load_target_basin_ids(config):
    """Union of train/val/test basin lists, stripped of the 'il_' prefix
    to match the raw water-authority data's plain numeric basin IDs."""
    il_ids = (_read_basin_list(config['train_basin_file'])
              | _read_basin_list(config['validation_basin_file'])
              | _read_basin_list(config['test_basin_file']))
    return {il_id: il_id.replace('il_', '', 1) for il_id in il_ids}


def load_raw_daily_flow(daily_flow_dir):
    """Concatenates every daily_flow_*.csv in daily_flow_dir into one
    DataFrame with a parsed 'date' column and a stripped 'basin_id' column."""
    paths = sorted(glob.glob(os.path.join(daily_flow_dir, 'daily_flow_*.csv')))
    frames = [pd.read_csv(path, dtype=str) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined.columns = combined.columns.str.strip()
    combined = combined.rename(columns={'basin ID': 'basin_id'})
    combined['date'] = pd.to_datetime(combined['date'], format='%d/%m/%Y')
    combined['daily volume m**3'] = pd.to_numeric(combined['daily volume m**3'])
    combined['avg daily flow m**3/sec'] = pd.to_numeric(combined['avg daily flow m**3/sec'])
    return combined


def find_missing_days(basin_df):
    """
    basin_df: one basin's rows (any subset of columns, must include 'date',
    'daily volume m**3', 'avg daily flow m**3/sec'), not necessarily sorted.
    Returns a DataFrame with columns ['date', 'reason'], sorted by date -
    one row per missing day within the basin's own [min_date, max_date].
    """
    basin_df = basin_df.set_index('date').sort_index()
    full_range = pd.date_range(basin_df.index.min(), basin_df.index.max(), freq='D')
    reindexed = basin_df.reindex(full_range)

    # A date absent from the original index has no row at all; a date
    # present but with both value columns empty is a NaN-value day.
    present_mask = reindexed.index.isin(basin_df.index)
    row_missing = ~present_mask
    nan_value = present_mask & reindexed['daily volume m**3'].isna() & reindexed['avg daily flow m**3/sec'].isna()

    missing_dates = list(full_range[row_missing])
    nan_dates = list(full_range[nan_value])

    rows = [{'date': d.strftime('%Y-%m-%d'), 'reason': 'row_missing'} for d in missing_dates]
    rows += [{'date': d.strftime('%Y-%m-%d'), 'reason': 'nan_value'} for d in nan_dates]
    rows.sort(key=lambda r: r['date'])
    return pd.DataFrame(rows, columns=['date', 'reason'])


def screen_nan_flow_days(input_dir, output_dir, nan_flow_days_dir):
    """
    Drops rows from every {basin_id}.csv in input_dir whose calendar date is
    flagged in nan_flow_days_dir/{basin_id}.csv, writing the result to
    output_dir (can equal input_dir for an in-place screen - same pattern as
    screen_dry_years.screen_years). A basin with no nan_flow_days file
    (outside the train+val+test union, or absent from the raw
    water-authority data) passes through unfiltered. Returns
    {basin_id: number of rows dropped}.
    """
    os.makedirs(output_dir, exist_ok=True)
    dropped_by_basin = {}
    basin_files = sorted(f for f in os.listdir(input_dir) if f.endswith('.csv'))
    for file_name in basin_files:
        basin_id = file_name.replace('.csv', '')
        df = pd.read_csv(os.path.join(input_dir, file_name))

        nan_flow_days_path = os.path.join(nan_flow_days_dir, file_name)
        if not os.path.exists(nan_flow_days_path):
            df.to_csv(os.path.join(output_dir, file_name), index=False)
            dropped_by_basin[basin_id] = 0
            continue

        flagged_df = pd.read_csv(nan_flow_days_path)
        flagged_dates = set(pd.to_datetime(flagged_df['date']).dt.date) if not flagged_df.empty else set()

        df['date'] = pd.to_datetime(df['date'])
        on_flagged_day = df['date'].dt.date.isin(flagged_dates)
        screened_df = df.loc[~on_flagged_day]
        screened_df.to_csv(os.path.join(output_dir, file_name), index=False)
        dropped_by_basin[basin_id] = int(on_flagged_day.sum())

    return dropped_by_basin


def main(config):
    daily_flow_dir = config['daily_flow_water_authority_dir']
    output_dir = config['nan_flow_days_output_dir']
    os.makedirs(output_dir, exist_ok=True)

    target_basins = load_target_basin_ids(config)
    print(f"[INFO] Processing {len(target_basins)} basins (train+val+test union).")

    combined = load_raw_daily_flow(daily_flow_dir)
    by_basin = {basin_id: df for basin_id, df in combined.groupby('basin_id')}

    total_missing = 0
    n_processed = 0
    for il_id, raw_basin_id in sorted(target_basins.items()):
        basin_df = by_basin.get(raw_basin_id)
        if basin_df is None:
            print(f"  [Warning] {il_id} (raw id {raw_basin_id}): not found in raw water-authority data. Skipping.")
            continue

        missing_df = find_missing_days(basin_df)
        out_path = os.path.join(output_dir, f"{il_id}.csv")
        missing_df.to_csv(out_path, index=False)

        n_processed += 1
        total_missing += len(missing_df)
        print(f"  {il_id}: {len(missing_df)} missing days -> {out_path}")

    print(f"\n[INFO] Wrote {n_processed} basin files, {total_missing} total missing days, to {output_dir}")


if __name__ == "__main__":
    CONFIG_PATH = "configs/config.yml"
    yaml_config = load_config(CONFIG_PATH)
    main(yaml_config)
