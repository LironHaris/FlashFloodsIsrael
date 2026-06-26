"""
Module: new_return_periods.py
Description: Plausibility Check for Official Return Period Values.

Compares the official RP2/RP5 flow values in empirical_vs_new_RP_joined.csv against
each basin's own empirical annual-maxima record (produced by calculate_return_periods.py).
For each target return period, the empirical row whose Empirical_Return_Period_Years is
the smallest value still >= the target (the closest-from-above / most conservative bound)
is used as the comparison point. Whenever the official value is missing, blank, or more
than 10x off from that comparison (in either direction), the comparison value is written
instead. -2 is reserved for basins with no comparison value to fall back on at all (missing
from the joined CSV and with no empirical data of their own).
"""

import os
import yaml
import pandas as pd


def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def load_basin_ids(basin_list_path):
    with open(basin_list_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def get_comparison_value(basin_dir, target_years):
    """
    Returns the empirical flow Value whose Empirical_Return_Period_Years is the
    smallest value still >= target_years, or None if no such row exists (record
    too short to reach that return period) or the basin's empirical file is missing.
    """
    emp_path = os.path.join(basin_dir, 'Hourly_Flow_empirical_AM.csv')
    if not os.path.exists(emp_path):
        return None

    emp_df = pd.read_csv(emp_path)
    candidates = emp_df[emp_df['Empirical_Return_Period_Years'] >= target_years]
    if candidates.empty:
        return None

    best_row = candidates.loc[candidates['Empirical_Return_Period_Years'].idxmin()]
    return float(best_row['Value'])


def resolve_rp_value(official_value, comparison_value):
    """
    Pass through official_value unless it's missing or >10x off from comparison_value
    (either direction), in which case fall back to comparison_value itself. If no
    comparison_value is available, official_value is the only thing there is to return.
    """
    if comparison_value is None or pd.isna(comparison_value):
        return official_value

    if pd.isna(official_value):
        return comparison_value

    if official_value == 0 or comparison_value == 0:
        # one zero and one non-zero is an effectively infinite ratio -> fill;
        # both zero is trivially consistent -> keep official
        return official_value if official_value == comparison_value else comparison_value

    ratio = max(official_value, comparison_value) / min(official_value, comparison_value)
    return comparison_value if ratio > 10 else official_value


def main(config):
    return_periods_dir = config['return_periods_output_dir']
    basin_list_path = os.path.join('data', 'basin_lists', 'all_basins.txt')
    joined_path = os.path.join(return_periods_dir, 'empirical_vs_new_RP_joined.csv')
    output_path = os.path.join(return_periods_dir, 'new_return_periods.csv')

    basin_ids = load_basin_ids(basin_list_path)

    # utf-8-sig strips the BOM on the first header column; basin_id is read as str
    # since the joined CSV stores bare numeric IDs (e.g. '2105') without the 'il_' prefix.
    joined_df = pd.read_csv(joined_path, encoding='utf-8-sig', dtype={'basin_id': str})
    joined_df['basin_id'] = joined_df['basin_id'].str.strip()

    targets = {'RP2': 2, 'RP5': 5}
    rows = []

    for basin_id in basin_ids:
        bare_id = basin_id[len('il_'):] if basin_id.startswith('il_') else basin_id
        match = joined_df[joined_df['basin_id'] == bare_id]
        basin_dir = os.path.join(return_periods_dir, basin_id)

        if match.empty:
            # No official row at all - fall back to the basin's own empirical value,
            # or -2 if it has no empirical data either.
            result = {'basin_id': basin_id}
            for col, target_years in targets.items():
                comparison_value = get_comparison_value(basin_dir, target_years)
                result[col] = comparison_value if comparison_value is not None else -2
            rows.append(result)
            continue

        official_row = match.iloc[0]

        result = {'basin_id': basin_id}
        for col, target_years in targets.items():
            official_value = float(official_row[col])
            comparison_value = get_comparison_value(basin_dir, target_years)
            result[col] = resolve_rp_value(official_value, comparison_value)
        rows.append(result)

    out_df = pd.DataFrame(rows, columns=['basin_id', 'RP2', 'RP5'])
    out_df.to_csv(output_path, index=False)
    print(f"[INFO] Wrote {len(out_df)} basins to {output_path}")


if __name__ == "__main__":
    CONFIG_PATH = "configs/config.yml"
    yaml_config = load_config(CONFIG_PATH)
    main(yaml_config)
