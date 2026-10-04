"""
Screens "dry" hydrological years (flagged by `flow_quality_check.py` as used
but flow-free) out of each basin's timeseries, writing the result to a
separate directory without touching the originals, and reports per-basin how
much of each record was dropped. Runs against both the processed timeseries
and the raw water-authority gauge log, then compares which years each one
flags as dry.
"""

import os

import pandas as pd
import yaml
from tqdm import tqdm

from flow_quality_check import get_hydrological_year


def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def parse_years(cell):
    if pd.isna(cell) or cell == '':
        return set()
    return {int(y) for y in str(cell).split(';')}


def screen_years(config, input_dir, output_dir, quality_check_path, summary_path):
    start_month = config.get('hydro_year_start_month', 10)
    quality_df = pd.read_csv(quality_check_path).set_index('basin_id')

    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    dropped_by_basin = {}
    basin_files = sorted(f for f in os.listdir(input_dir) if f.endswith('.csv'))
    for file_name in tqdm(basin_files, desc=f"Screening {os.path.basename(summary_path)}"):
        basin_id = file_name.replace('.csv', '')
        df = pd.read_csv(os.path.join(input_dir, file_name))
        df['date'] = pd.to_datetime(df['date'])
        hydro_year = get_hydrological_year(pd.DatetimeIndex(df['date']), start_month=start_month)

        if basin_id in quality_df.index:
            flagged_years = parse_years(quality_df.loc[basin_id, 'used_years_no_flow_record'])
        else:
            flagged_years = set()
        years_present = set(hydro_year.unique())
        dropped_years = sorted(flagged_years & years_present)
        dropped_by_basin[basin_id] = set(dropped_years)

        screened_df = df.loc[~hydro_year.isin(dropped_years)]
        screened_df.to_csv(os.path.join(output_dir, file_name), index=False)

        total_years = len(years_present)
        num_dropped = len(dropped_years)
        pct_dropped = (num_dropped / total_years * 100) if total_years else 0.0

        summary_rows.append({
            'basin_id': basin_id,
            'dropped_years': ';'.join(str(y) for y in dropped_years),
            'num_dropped_years': num_dropped,
            'total_years': total_years,
            'pct_dropped_years': pct_dropped,
        })

    summary_df = pd.DataFrame(summary_rows)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote screened timeseries for {len(summary_rows)} basins to {output_dir}")
    print(f"Wrote drop summary to {summary_path}")
    return dropped_by_basin


def write_gap_comparison(processed_dropped, water_authority_dropped, output_path):
    basin_ids = sorted(set(processed_dropped) | set(water_authority_dropped))
    rows = []
    for basin_id in basin_ids:
        proc_years = processed_dropped.get(basin_id, set())
        wa_years = water_authority_dropped.get(basin_id, set())
        both = sorted(proc_years & wa_years)
        only_proc = sorted(proc_years - wa_years)
        only_wa = sorted(wa_years - proc_years)
        rows.append({
            'basin_id': basin_id,
            'dropped_both': ';'.join(str(y) for y in both),
            'dropped_only_processed': ';'.join(str(y) for y in only_proc),
            'dropped_only_water_authority': ';'.join(str(y) for y in only_wa),
            'num_mismatched_years': len(only_proc) + len(only_wa),
        })

    gap_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gap_df.to_csv(output_path, index=False)
    print(f"Wrote water authority data gap comparison to {output_path}")


def main(config):
    processed_dropped = screen_years(
        config,
        input_dir=config['processed_timeseries_dir'],
        output_dir=config['no_dry_years_timeseries_dir'],
        quality_check_path=config['flow_quality_check_output_file'],
        summary_path=config['dry_years_screening_summary_file'],
    )
    water_authority_dropped = screen_years(
        config,
        input_dir=config['processed_timeseries_dir'],
        output_dir=config['no_dry_years_timeseries_water_authority_dir'],
        quality_check_path=config['water_authority_analysis_output_file'],
        summary_path=config['dry_years_water_authority_summary_file'],
    )
    write_gap_comparison(processed_dropped, water_authority_dropped, config['water_authority_data_gap_file'])


if __name__ == "__main__":
    CONFIG_PATH = "configs/config.yml"
    yaml_config = load_config(CONFIG_PATH)
    main(yaml_config)
