"""
Module: prepare_radar_inputs.py
Description: Turns the extracted IMS SR radar basin series (one CSV per basin, produced by
             basin_mapping/SR_basin_mapping/extract/extract_sr_basin_hourly.py) into model-ready
             per-basin CSVs for dataset.py, in the same spirit as preprocess_dynamic_data.py:
             - radar features z-scored per basin with train-period statistics (baked into the CSV);
             - gap hours (radar_flag 3/4/5) filled with the value of 0 mm and marked radar_mask = 0;
             - window_ok[t] = the input window ending at t obeys the gap rule
               (every gap run <= gap_max_run hours AND <= gap_max_total gap hours in the window);
             - per-basin flow mean/std from train periods written to the availability report.

radar_flag (from the extraction): 0 observed | 1 missing radar day judged dry -> 0 |
3 invalid hour | 4 valid basin area < 0.9 | 5 missing day that was wet/undecidable (outage).

Config keys used (see configs/radar_1a.yml):
  radar_extraction_dir, processed_timeseries_dir, availability_report_file,
  radar_features (columns to normalize/keep), seq_length, gap_max_run, gap_max_total, train_periods

    FLOODS_CONFIG=configs/radar_1a.yml python src/prepare_radar_inputs.py [--basins il_17110 il_8155]
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml

GAP_FLAGS = (3, 4, 5)


def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def gap_run_length(gap):
    """Length of the current run of consecutive gap hours ending at each hour (0 if not a gap)."""
    idx = np.arange(len(gap))
    last_non_gap = np.maximum.accumulate(np.where(gap == 0, idx, -1))
    return np.where(gap == 1, idx - last_non_gap, 0)


def window_ok_mask(radar_flag, seq_length, max_run, max_total):
    """True at t if the window t-seq_length+1..t has no gap run > max_run and <= max_total gap hours."""
    gap = np.isin(radar_flag, GAP_FLAGS).astype(np.int64)
    run_max = pd.Series(gap_run_length(gap)).rolling(seq_length, min_periods=seq_length).max().to_numpy()
    total = pd.Series(gap).rolling(seq_length, min_periods=seq_length).sum().to_numpy()
    return (run_max <= max_run) & (total <= max_total)          # NaN (first hours) -> False


def prepare_basin(path, config):
    df = pd.read_csv(path, parse_dates=['date']).set_index('date').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    full = pd.date_range(df.index.min(), df.index.max(), freq='h')   # guarantee a contiguous hourly index
    df = df.reindex(full)
    df['radar_flag'] = df['radar_flag'].fillna(5).astype(int)

    feats = config['radar_features']
    gap = np.isin(df['radar_flag'].to_numpy(), GAP_FLAGS)
    df['radar_mask'] = (~gap).astype(np.float32)
    for f in feats:
        df.loc[gap, f] = 0.0                                          # gap -> 0 mm (mask tells the model)

    train_slice = pd.concat([df.loc[p.get('start_date'):p['end_date']] for p in config['train_periods']])
    observed = train_slice[train_slice['radar_mask'] == 1]
    stats = {}
    for f in feats:
        mean, std = float(np.nanmean(observed[f])), float(np.nanstd(observed[f]))
        if not np.isfinite(std) or std == 0.0:
            mean, std = 0.0, 1.0
        df[f] = (df[f] - mean) / std
        stats[f'{f}_mean'], stats[f'{f}_std'] = mean, std

    # gauge rain (the pipeline's original input), z-scored per basin from train periods as in
    # preprocess_dynamic_data.py - kept for rain overlays, gauge-control runs and model 2
    if 'hourly_precipitation' in df.columns:
        g_mean = float(np.nanmean(train_slice['hourly_precipitation']))
        g_std = float(np.nanstd(train_slice['hourly_precipitation']))
        if not np.isfinite(g_std) or g_std == 0.0:
            g_mean, g_std = 0.0, 1.0
        df['hourly_precipitation'] = (df['hourly_precipitation'] - g_mean) / g_std
        stats['hourly_precipitation_mean'], stats['hourly_precipitation_std'] = g_mean, g_std

    flow_mean = float(np.nanmean(train_slice['Flow_m3_sec']))
    flow_std = float(np.nanstd(train_slice['Flow_m3_sec']))
    if not np.isfinite(flow_std) or flow_std == 0.0:
        flow_mean, flow_std = float(np.nanmean(df['Flow_m3_sec'])), float(np.nanstd(df['Flow_m3_sec']))

    df['window_ok'] = window_ok_mask(df['radar_flag'].to_numpy(), config['seq_length'],
                                     config.get('gap_max_run', 2), config.get('gap_max_total', 6))
    keep = ['Flow_m3_sec', *feats, 'radar_mask', 'radar_flag', 'window_ok']
    if 'hourly_precipitation' in df.columns:
        keep.insert(1, 'hourly_precipitation')
    out = df[keep]
    report = {'availability_pct': float(df['Flow_m3_sec'].notna().mean() * 100),
              'flow_mean': flow_mean, 'flow_std': flow_std,
              'window_ok_share': float(out['window_ok'].mean()), **stats}
    return out, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=os.environ.get('FLOODS_CONFIG', 'configs/config.yml'))
    ap.add_argument('--basins', nargs='*', help='subset of gauge ids (default: all files in the extraction)')
    a = ap.parse_args()
    config = load_config(a.config)

    src_dir = os.path.join(config['radar_extraction_dir'], 'basin_hourly')
    out_dir = config['processed_timeseries_dir']
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config['availability_report_file']) or '.', exist_ok=True)

    excluded = set(config.get('exclude_basins') or [])
    basins = a.basins or sorted(f[:-4] for f in os.listdir(src_dir) if f.endswith('.csv') and f[:-4] not in excluded)
    records = []
    for i, b in enumerate(basins, 1):
        out, report = prepare_basin(os.path.join(src_dir, f'{b}.csv'), config)
        out.to_csv(os.path.join(out_dir, f'{b}.csv'), index_label='date', float_format='%.6g')
        records.append({'gauge_id': b, **report})
        print(f"[{i}/{len(basins)}] {b}: window_ok {report['window_ok_share']:.1%}, flow {report['availability_pct']:.1f}%")
    pd.DataFrame(records).to_csv(config['availability_report_file'], index=False)
    print(f"[INFO] wrote {len(records)} basins to {out_dir} and report {config['availability_report_file']}")


if __name__ == '__main__':
    main()
