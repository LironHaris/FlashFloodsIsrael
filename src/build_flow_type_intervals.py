"""
Module: build_flow_type_intervals.py
Description: One-time precompute of the water authority's Flow_type intervals per
             station, for training-time masking (config['train_flow_type_mask'],
             see dataset.py). hydrographs_unified.csv is ~15M rows - too heavy to
             load inside every training job - so the merged [start, end)
             intervals are written once to a small CSV:
               columns: station_id, mask, start, end
             with one block per mask in MASKS ('flow' = {flow},
             'flow_normal' = {flow, normal}). Interval construction is
             data_type_mask.build_flow_intervals unchanged (same hydrograph-boundary
             rules), so the training mask and data_type_mask.py / flow_distribution.py
             agree exactly.

Usage:
  python src/build_flow_type_intervals.py [--output data/processed/flow_type_intervals.csv]
"""

import argparse
import os

import pandas as pd

from data_type_mask import UNIFIED_FILE, load_unified, build_flow_intervals

MASKS = {'flow': {'flow'}, 'flow_normal': {'flow', 'normal'}}
DEFAULT_OUTPUT = 'data/processed/flow_type_intervals.csv'


def build_intervals_table(unified):
    frames = []
    for station_id, station_df in unified.groupby('Station_ID'):
        for mask_name, flow_types in MASKS.items():
            starts, ends = build_flow_intervals(station_df, flow_types)
            if len(starts):
                frames.append(pd.DataFrame({'station_id': station_id, 'mask': mask_name,
                                            'start': starts, 'end': ends}))
    return pd.concat(frames, ignore_index=True)


def main(output_path):
    print(f"[INFO] Loading {UNIFIED_FILE} ...")
    unified = load_unified(UNIFIED_FILE)
    table = build_intervals_table(unified)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    table.to_csv(output_path, index=False)
    summary = table.groupby('mask').agg(stations=('station_id', 'nunique'), intervals=('start', 'size'))
    print(summary.to_string())
    print(f"[INFO] Wrote {len(table)} intervals to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute per-station Flow_type intervals for training masks.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main(args.output)
