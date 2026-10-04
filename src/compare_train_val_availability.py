"""
Module: compare_train_val_availability.py
Description: Compares the model_0 and model_1_3 "no dry years" config
             families on how many actual trainable windows (exactly what
             dataset.py's real windowing/NaN-tolerance logic hands the
             DataLoader - not raw non-NaN row counts) their train/val
             period definitions produce, per basin and per hydrological
             year, per forecast lead time. Reuses the real dataset classes
             directly (IsraelBasinsDataset, build_cv_group_datasets) so the
             counts are guaranteed to match real training exactly.
"""

import csv
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))

from dataset import IsraelBasinsDataset, build_cv_group_datasets
from flow_quality_check import get_hydrological_year


def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


LEAD_PAIRS = [
    (0, 'configs/best_model_0_0.yml', 'configs/best_model_1_3_0_adamw_crossval_no_dry_years.yml'),
    (1, 'configs/best_model_0_1.yml', 'configs/best_model_1_3_1_adamw_crossval_no_dry_years.yml'),
    (2, 'configs/best_model_0_2.yml', 'configs/best_model_1_3_2_adamw_crossval_no_dry_years.yml'),
    (3, 'configs/best_model_0_3.yml', 'configs/best_model_1_3_3_adamw_crossval_no_dry_years.yml'),
]

OUTPUT_PATH = './data/processed/data quality reports/train_val_availability_comparison.csv'

ROW_FIELDS = [
    'lead', 'basin_id', 'year',
    'model_0_train', 'model_0_val', 'model_1_3_train', 'model_1_3_val',
    'train_ratio_1_3_over_0_pct', 'val_ratio_1_3_over_0_pct',
]


def _tally_by_year(basin_datasets, hydro_year_start_month, counts):
    """
    counts[basin_id][year] += 1 for every valid window in every given
    SingleBasinDataset. Each sample is dated at index + seq_length - 1 -
    exactly how IsraelBasinsDataset itself dates samples internally
    (sample_date_mappings), not lead-shifted.
    """
    for ds in basin_datasets:
        if len(ds) == 0:
            continue
        target_positions = [idx + ds.seq_length - 1 for idx in ds.valid_indices]
        years = get_hydrological_year(ds.dates[target_positions], start_month=hydro_year_start_month)
        basin_counts = counts.setdefault(ds.gauge_id, {})
        for year in years:
            basin_counts[year] = basin_counts.get(year, 0) + 1


def collect_model_0_counts(config):
    """Real train/val period + NaN-tolerance logic for the non-CV model_0 family."""
    hydro_start = config.get('hydro_year_start_month', 10)
    use_basin_splits = config.get('use_basin_splits', True)
    train_counts, val_counts = {}, {}
    train_dataset = IsraelBasinsDataset('train', config, use_basin_splits=use_basin_splits)
    val_dataset = IsraelBasinsDataset('val', config, use_basin_splits=use_basin_splits)
    _tally_by_year(train_dataset.basin_datasets, hydro_start, train_counts)
    _tally_by_year(val_dataset.basin_datasets, hydro_start, val_counts)
    return train_counts, val_counts


def collect_model_1_3_counts(config):
    """
    Real cross-validation period + NaN-tolerance logic for the model_1_3
    family. Train = fixed pre-2008 slice + every group's train-tolerance
    variant; val = every group's eval-tolerance variant (the fixed slice is
    never held out as validation, matching real training).
    """
    hydro_start = config.get('hydro_year_start_month', 10)
    use_basin_splits = config.get('use_basin_splits', True)
    train_counts, val_counts = {}, {}
    fixed_train_datasets, group_datasets = build_cv_group_datasets(config, use_basin_splits=use_basin_splits)
    _tally_by_year(fixed_train_datasets, hydro_start, train_counts)
    for group in group_datasets.values():
        _tally_by_year(group['train'], hydro_start, train_counts)
        _tally_by_year(group['val'], hydro_start, val_counts)
    return train_counts, val_counts


def _ratio_pct(numerator, denominator):
    return numerator / denominator * 100 if denominator else float('nan')


def _build_rows(lead, model_0_train, model_0_val, model_1_3_train, model_1_3_val):
    basin_ids = set(model_0_train) | set(model_0_val) | set(model_1_3_train) | set(model_1_3_val)
    rows = []
    for basin_id in sorted(basin_ids):
        years = (set(model_0_train.get(basin_id, {})) | set(model_0_val.get(basin_id, {}))
                 | set(model_1_3_train.get(basin_id, {})) | set(model_1_3_val.get(basin_id, {})))
        for year in sorted(years):
            m0_train = model_0_train.get(basin_id, {}).get(year, 0)
            m0_val = model_0_val.get(basin_id, {}).get(year, 0)
            m13_train = model_1_3_train.get(basin_id, {}).get(year, 0)
            m13_val = model_1_3_val.get(basin_id, {}).get(year, 0)
            rows.append({
                'lead': lead, 'basin_id': basin_id, 'year': year,
                'model_0_train': m0_train, 'model_0_val': m0_val,
                'model_1_3_train': m13_train, 'model_1_3_val': m13_val,
                'train_ratio_1_3_over_0_pct': _ratio_pct(m13_train, m0_train),
                'val_ratio_1_3_over_0_pct': _ratio_pct(m13_val, m0_val),
            })
    return rows


def _summary_row(lead, rows):
    m0_train = sum(r['model_0_train'] for r in rows)
    m0_val = sum(r['model_0_val'] for r in rows)
    m13_train = sum(r['model_1_3_train'] for r in rows)
    m13_val = sum(r['model_1_3_val'] for r in rows)
    return {
        'lead': lead, 'basin_id': 'ALL_BASINS', 'year': 'ALL_YEARS',
        'model_0_train': m0_train, 'model_0_val': m0_val,
        'model_1_3_train': m13_train, 'model_1_3_val': m13_val,
        'train_ratio_1_3_over_0_pct': _ratio_pct(m13_train, m0_train),
        'val_ratio_1_3_over_0_pct': _ratio_pct(m13_val, m0_val),
    }


def main():
    all_rows = []
    lead_summaries = []

    for lead, model_0_path, model_1_3_path in LEAD_PAIRS:
        print(f"[INFO] Lead {lead}: {model_0_path} vs {model_1_3_path}")
        config_0 = load_config(model_0_path)
        config_1_3 = load_config(model_1_3_path)

        model_0_train, model_0_val = collect_model_0_counts(config_0)
        model_1_3_train, model_1_3_val = collect_model_1_3_counts(config_1_3)

        lead_rows = _build_rows(lead, model_0_train, model_0_val, model_1_3_train, model_1_3_val)
        all_rows.extend(lead_rows)

        lead_summary = _summary_row(lead, lead_rows)
        lead_summaries.append(lead_summary)
        all_rows.append(lead_summary)

        print(f"  -> model_0: train={lead_summary['model_0_train']}, val={lead_summary['model_0_val']} | "
              f"model_1_3: train={lead_summary['model_1_3_train']}, val={lead_summary['model_1_3_val']} | "
              f"ratio(1_3/0): train={lead_summary['train_ratio_1_3_over_0_pct']:.1f}%, "
              f"val={lead_summary['val_ratio_1_3_over_0_pct']:.1f}%")

    grand_total = _summary_row('ALL', lead_summaries)
    all_rows.append(grand_total)
    print(f"\n[INFO] Grand total -> model_0: train={grand_total['model_0_train']}, val={grand_total['model_0_val']} | "
          f"model_1_3: train={grand_total['model_1_3_train']}, val={grand_total['model_1_3_val']} | "
          f"ratio(1_3/0): train={grand_total['train_ratio_1_3_over_0_pct']:.1f}%, "
          f"val={grand_total['val_ratio_1_3_over_0_pct']:.1f}%")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[INFO] Wrote {len(all_rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
