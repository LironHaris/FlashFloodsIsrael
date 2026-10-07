"""
Module: normalization.py
Description: Single source of the per-basin dynamic-input / target
             normalization statistics. The processed timeseries CSVs hold RAW
             values (preprocess_dynamic_data.py no longer bakes in a z-score);
             normalization is applied on the fly by dataset.SingleBasinDataset
             using the statistics resolved here.

             Statistics are computed per basin, per feature
             (config['dynamic_inputs'] + config['target_variables']), strictly
             from that basin's rows inside config['train_periods'] - never from
             validation/test time. A non-finite or ~zero std becomes 1.0 (the
             train mean is kept); an empty / all-NaN train slice becomes
             mean=0, std=1. There is no fallback to the full record.

             Persistence: train.py / train_sanity.py / run_sweep.py write the
             statistics to exp_dir/normalization_stats.json (and into every
             checkpoint they save). Every other script resolves them via
             resolve_normalization_stats(config), in this order:
               1. config['normalization_stats'] (already attached in-memory)
               2. normalization_stats.json next to config['checkpoint_path'],
                  else in run_dir/experiment_name/
               3. recompute from the raw CSVs + train_periods (with a warning) -
                  deterministic, so it reproduces what training computed as
                  long as the data and train_periods are unchanged.
"""

import json
import os

import numpy as np
import pandas as pd

STATS_FILENAME = 'normalization_stats.json'
MIN_STD = 1e-6
NEGATIVE_TOL = 1e-6  # raw values below -NEGATIVE_TOL can only come from an old z-scored CSV


def stats_features(config):
    """Features that get per-basin statistics: dynamic inputs + targets."""
    return list(config['dynamic_inputs']) + list(config['target_variables'])


def _train_slice(df, config):
    periods = [(p.get('start_date') or None, p['end_date']) for p in config['train_periods']]
    return pd.concat([df.loc[start:end] for start, end in periods])


def compute_basin_stats(config, basin_id):
    """{feature: {'mean', 'std'}} for one basin, from its train-period rows only."""
    path = os.path.join(config['processed_timeseries_dir'], f"{basin_id}.csv")
    features = stats_features(config)
    df = pd.read_csv(path, usecols=['date'] + features, index_col='date', parse_dates=True)
    train_df = _train_slice(df, config)

    stats = {}
    for feat in features:
        values = train_df[feat].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            print(f"  [Warning] {basin_id}/{feat}: no train-period data - using mean=0, std=1.")
            mean, std = 0.0, 1.0
        else:
            mean = float(finite.mean())
            std = float(finite.std())
            if not np.isfinite(std) or std < MIN_STD:
                print(f"  [Warning] {basin_id}/{feat}: train-period std={std} - using std=1 "
                      f"(mean kept at train mean).")
                std = 1.0
            # Raw rain / flow are never negative - clear negatives (beyond float
            # round-off) mean the CSV still holds the old pre-normalized values.
            if finite.min() < -NEGATIVE_TOL:
                print(f"  [Warning] {basin_id}/{feat}: negative values in the processed CSV - "
                      f"it still looks z-scored. Re-run preprocess_dynamic_data.py.")
        stats[feat] = {'mean': mean, 'std': std}
    return stats


def compute_normalization_stats(config, basin_ids):
    """{basin_id: {feature: {'mean', 'std'}}} for every basin with a processed CSV."""
    stats = {}
    for basin_id in basin_ids:
        if os.path.exists(os.path.join(config['processed_timeseries_dir'], f"{basin_id}.csv")):
            stats[basin_id] = compute_basin_stats(config, basin_id)
    return stats


def all_split_basin_ids(config):
    """Union of the train/val/test basin lists that the config defines (every processed
    CSV if basin splits are off). A validation list is optional - cross-validation
    configs have none."""
    if not config.get('use_basin_splits', True):
        return sorted(f[:-4] for f in os.listdir(config['processed_timeseries_dir']) if f.endswith('.csv'))
    basin_ids = set()
    for key in ('train_basin_file', 'validation_basin_file', 'test_basin_file'):
        path = config.get(key)
        if not path:
            continue
        with open(path, 'r') as f:
            basin_ids.update(line.strip() for line in f if line.strip())
    return sorted(basin_ids)


def stats_path_for(config):
    """Where a run's normalization_stats.json lives: next to checkpoint_path if set, else exp_dir."""
    checkpoint_path = config.get('checkpoint_path')
    if checkpoint_path:
        return os.path.join(os.path.dirname(checkpoint_path), STATS_FILENAME)
    return os.path.join(config.get('run_dir', './runs/'), config['experiment_name'], STATS_FILENAME)


def save_normalization_stats(stats, exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    path = os.path.join(exp_dir, STATS_FILENAME)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=1)
    return path


def load_normalization_stats(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_normalization(config, exp_dir):
    """
    Training-side entry point: computes fresh train-period statistics for every
    train/val/test basin, attaches them to config['normalization_stats'] (so
    every loader built from this config uses them) and writes
    exp_dir/normalization_stats.json. Returns the stats dict (callers also
    store it inside saved checkpoints).
    """
    print("[INFO] Computing per-basin normalization statistics from train_periods only...")
    stats = compute_normalization_stats(config, all_split_basin_ids(config))
    config['normalization_stats'] = stats
    path = save_normalization_stats(stats, exp_dir)
    print(f"[INFO] Normalization statistics for {len(stats)} basins saved to {path}")
    return stats


def resolve_normalization_stats(config):
    """Evaluation-side entry point - see the module docstring for the lookup order."""
    stats = config.get('normalization_stats')
    if stats is not None:
        return stats

    path = stats_path_for(config)
    if os.path.exists(path):
        stats = load_normalization_stats(path)
        print(f"[INFO] Loaded normalization statistics from {path}")
    else:
        print(f"[Warning] {path} not found - recomputing normalization statistics from "
              f"train_periods (matches training only if data/train_periods are unchanged).")
        stats = compute_normalization_stats(config, all_split_basin_ids(config))
    config['normalization_stats'] = stats
    return stats


def basin_stats(config, basin_id):
    """One basin's {feature: {'mean', 'std'}}, computing (and caching) it if absent."""
    stats = resolve_normalization_stats(config)
    if basin_id not in stats:
        print(f"[Warning] {basin_id}: no stored normalization statistics - computing from train_periods.")
        stats[basin_id] = compute_basin_stats(config, basin_id)
    return stats[basin_id]
