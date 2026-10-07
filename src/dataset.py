"""
Module: dataset.py
Description: Hourly Multi-Basin Data Pipeline for Entity-Aware LSTM (EA-LSTM) Flood Forecasting.
             Optimized to isolate and align static catchment attributes dynamically per basin.
             The processed timeseries CSVs hold raw values; dynamic inputs and targets are
             z-scored on the fly with per-basin train-period statistics resolved by
             normalization.py.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from normalization import basin_stats

class SingleBasinDataset(Dataset):
    """
    A PyTorch Dataset that handles the dynamic and static data for a SINGLE basin.
    dyn_mean/dyn_std (one value per config['dynamic_inputs'] feature) and
    flow_mean/flow_std are this basin's train-period statistics (normalization.py).
    """
    def __init__(self, dynamic_path, static_path, config, periods, flow_std: float = 1.0,
                 flow_mean: float = 0.0, split_type: str = 'train', dyn_mean=None, dyn_std=None):
        # Extract basin ID dynamically from filename
        self.gauge_id = str(os.path.basename(dynamic_path).replace('.csv', ''))

        # Load dynamic and static data
        dyn_df = pd.read_csv(dynamic_path)
        dyn_df['date'] = pd.to_datetime(dyn_df['date'])
        dyn_df.set_index('date', inplace=True)
        stat_df = pd.read_csv(static_path)

        # Filter the dynamic data to the union of the provided (possibly disjoint) periods
        dyn_df = pd.concat([dyn_df[start_date:end_date] for start_date, end_date in periods])

        # Store index dates for evaluation alignment
        self.dates = dyn_df.index
        
        self.flow_std = np.float32(flow_std)
        self.flow_mean = np.float32(flow_mean)

        # Extract configurations
        self.seq_length = config['seq_length']
        self.dynamic_feature_names = config['dynamic_inputs']
        self.static_feature_names = config['static_attributes']
        self.target_cols = config['target_variables']
        self.forecast_lead_times = config['forecast_lead_times']
        
        # Convert to fast NumPy arrays. The CSV holds raw dynamic inputs - z-score them
        # here with this basin's train-period statistics (NaNs stay NaN).
        n_dyn = len(self.dynamic_feature_names)
        dyn_mean = np.zeros(n_dyn, dtype=np.float32) if dyn_mean is None else np.asarray(dyn_mean, dtype=np.float32)
        dyn_std = np.ones(n_dyn, dtype=np.float32) if dyn_std is None else np.asarray(dyn_std, dtype=np.float32)
        x_dynamic_raw = dyn_df[self.dynamic_feature_names].values.astype(np.float32)
        self.x_dynamic = (x_dynamic_raw - dyn_mean) / dyn_std
        self.y = dyn_df[self.target_cols].values.astype(np.float32)

        # Slice specific basin row inside the master static matrix file
        basin_static_row = stat_df[stat_df['gauge_id'].astype(str) == self.gauge_id]
        if basin_static_row.empty:
            raise KeyError(f"Gauge ID '{self.gauge_id}' missing in static attributes file: {static_path}")

        # Static attributes are pre-normalized (z-scored per feature across all basins)
        # by preprocess_static_attributes.py before this file is loaded.
        self.x_static = basin_static_row[self.static_feature_names].iloc[0].values.astype(np.float32)

        # Zero-rainfall-equivalent value per dynamic feature (raw 0 in the z-score space
        # above), used to fill in NaNs that fall within a window's per-feature tolerance
        # instead of excluding the whole window.
        zero_impute_values = (0.0 - dyn_mean) / dyn_std
        self.x_dynamic_filled = np.where(np.isnan(self.x_dynamic), zero_impute_values, self.x_dynamic)

        # Build valid sample indices. A window is excluded if:
        #  - the observed target flow is NaN at any forecast lead (always, regardless of
        #    split - there is no ground truth to fabricate or score against), or
        #  - any dynamic input feature has more NaN timesteps in the window than that
        #    feature's configured tolerance allows for this split (train vs. val/test).
        # train_max_nan_pct=0 (the default when a feature has no config entry)
        # reproduces the original "any NaN excludes" behavior exactly.
        tolerance_cfg = config.get('dynamic_input_nan_tolerance', {}) or {}
        tol_key = 'train_max_nan_pct' if split_type == 'train' else 'eval_max_nan_pct'
        allowed_nan_pct = np.array([
            tolerance_cfg.get(feat, {}).get(tol_key, 0) for feat in self.dynamic_feature_names
        ], dtype=np.float32)

        max_lead = max(self.forecast_lead_times)
        n_potential = max(0, len(self.x_dynamic) - self.seq_length - max_lead + 1)

        nan_mask = np.isnan(self.x_dynamic).astype(np.float32)
        csum = np.vstack([np.zeros((1, nan_mask.shape[1]), dtype=np.float32), np.cumsum(nan_mask, axis=0)])
        starts = np.arange(n_potential)
        window_nan_pct = (csum[starts + self.seq_length] - csum[starts]) / self.seq_length * 100
        input_ok = np.all(window_nan_pct <= allowed_nan_pct[None, :], axis=1)

        # A window (seq_length inputs + furthest lead) must cover consecutive hours.
        # Rows are only gap-free where the CSV is: disjoint periods (e.g. a held-out
        # CV fold's years) and rows dropped in preprocessing (dry years, long NaN
        # stretches) leave gaps that a window must never be built across.
        hour_steps = np.diff(self.dates.values).astype('timedelta64[h]').astype(np.int64)
        break_csum = np.concatenate([[0], np.cumsum(hour_steps != 1)])
        span = self.seq_length - 1 + max_lead
        contiguous = break_csum[starts + span] - break_csum[starts] == 0

        self.valid_indices = [
            i for i in range(n_potential)
            if input_ok[i] and contiguous[i]
            and not any(
                np.isnan(self.y[i + self.seq_length - 1 + lead]).any()
                for lead in self.forecast_lead_times
            )
        ]
        self.num_samples = len(self.valid_indices)

    def __len__(self):
        return max(0, self.num_samples)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        target_day = actual_idx + self.seq_length - 1
        start_day = target_day - self.seq_length + 1
        
        window_x_dynamic = self.x_dynamic_filled[start_day : target_day + 1]
        
        target_list = []
        for lead in self.forecast_lead_times:
            future_hour = target_day + lead
            target_list.append(self.y[future_hour])

        # self.y stays raw; the normalized target is derived from this same window
        # on the fly, so raw/normalized values can never disagree on NaN positions.
        target_raw_y = np.array(target_list, dtype=np.float32).flatten()
        target_y = (target_raw_y - self.flow_mean) / self.flow_std

        return {
            'dynamic': torch.tensor(window_x_dynamic),
            'static': torch.tensor(self.x_static),
            'target': torch.tensor(target_y),
            'target_raw': torch.tensor(target_raw_y),
            'basin_std': torch.tensor(self.flow_std, dtype=torch.float32),
            'basin_mean': torch.tensor(self.flow_mean, dtype=torch.float32),
        }

class IsraelBasinsDataset(Dataset):
    """
    Wrapper dataset that concatenates multiple individual SingleBasinDatasets 
    and extracts global tracking arrays for sequential basin-by-basin testing.

    Optional overrides: basin_ids replaces the split's basin list file, and
    periods ([(start_date, end_date), ...]) replaces the split's time bounds.
    split_type still selects the NaN tolerance (train vs. eval).
    """
    def __init__(self, split_type, config, use_basin_splits=True, basin_ids=None, periods=None):
        # Step 1: Extract paths, time bounds, and shuffling rules
        basin_list_file, split_periods, _ = _get_split_bounds_and_config(split_type, config, use_basin_splits)
        if periods is None:
            periods = split_periods

        # Step 2: Load the target basin IDs
        if basin_ids is None:
            basin_ids = _load_basin_ids(basin_list_file, config, use_basin_splits, split_type)

        # Step 3: Construct dataset objects for each individual basin
        self.basin_datasets = _build_basin_datasets(basin_ids, config, periods, split_type)
        
        # Step 4: Combine using PyTorch ConcatDataset
        self.concat_dataset = ConcatDataset(self.basin_datasets)
        
        # Step 5: Extract metadata tracking arrays for evaluation
        self.basins = [ds.gauge_id for ds in self.basin_datasets]
        self.sample_basin_mappings = []
        self.sample_date_mappings = []
        
        # Build global index mappings to map any index to its basin and exact datetime
        for ds in self.basin_datasets:
            for i in range(len(ds)):
                self.sample_basin_mappings.append(ds.gauge_id)
                actual_idx = ds.valid_indices[i]
                target_idx = actual_idx + ds.seq_length - 1
                self.sample_date_mappings.append(ds.dates[target_idx])

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        return self.concat_dataset[idx]


def _get_split_bounds_and_config(split_type, config, use_basin_splits):
    buffer_hours = config['seq_length']

    if split_type == 'train':
        periods = [(p.get('start_date'), p['end_date']) for p in config['train_periods']]
        basin_list_file = config['train_basin_file']
        return basin_list_file, periods, True

    elif split_type in ['val', 'test']:
        prefix = 'validation' if split_type == 'val' else 'test'
        # When basin splits are disabled, fall back to the train basin list (master list / directory scan).
        # A config without a validation_basin_file (e.g. a CV fold) validates on the train basins.
        basin_list_file = (config.get(f'{prefix}_basin_file') or config['train_basin_file']
                           if use_basin_splits else config['train_basin_file'])

        if split_type == 'val' and config.get('validation_periods'):
            # Explicit (possibly disjoint) validation periods, e.g. a CV fold's held-out years
            raw_periods = [(s, e) for s, e in config['validation_periods']]
        elif f'{prefix}_start_date' in config and f'{prefix}_end_date' in config:
            raw_periods = [(config[f'{prefix}_start_date'], config[f'{prefix}_end_date'])]
        else:
            raise ValueError(
                f"This config has no {prefix} split ({prefix}_start_date/{prefix}_end_date"
                + (" or validation_periods" if split_type == 'val' else "") + " missing). "
                "Cross-validation configs validate per fold - see cross_validation_sweep.py.")

        # Each period gets a seq_length warm-up buffer so its first hours can be targets
        periods = [
            ((pd.to_datetime(s) - pd.Timedelta(hours=buffer_hours)).strftime('%Y-%m-%d %H:%M:%S'), e)
            for s, e in raw_periods
        ]
        return basin_list_file, periods, False

    raise ValueError("split_type must be either 'train', 'val', or 'test'")


def has_validation_split(config):
    """True if the config defines a validation split (dates or explicit validation_periods)."""
    return bool(config.get('validation_periods')) or (
        'validation_start_date' in config and 'validation_end_date' in config)


# --------------------------------------------------------------------------------
# Cross-validation folds (used by cross_validation_sweep.py)
# --------------------------------------------------------------------------------

def _to_ts(value, default):
    return pd.Timestamp(value) if value else default


def subtract_periods(periods, removed):
    """
    periods minus removed, both lists of (start, end) datetime strings (start may be
    empty/None = open start). Returns the remaining (start, end) pieces in order;
    a removed range [s, e] leaves pieces ending 1h before s and starting 1h after e.
    """
    one_hour = pd.Timedelta(hours=1)
    pieces = [(_to_ts(s, pd.Timestamp.min), _to_ts(e, pd.Timestamp.max)) for s, e in periods]
    for rs, re in removed:
        rs, re = pd.Timestamp(rs), pd.Timestamp(re)
        next_pieces = []
        for ps, pe in pieces:
            if re < ps or rs > pe:
                next_pieces.append((ps, pe))
                continue
            if ps < rs:
                next_pieces.append((ps, rs - one_hour))
            if pe > re:
                next_pieces.append((re + one_hour, pe))
        pieces = next_pieces

    fmt = '%Y-%m-%d %H:%M:%S'
    return [(None if s == pd.Timestamp.min else s.strftime(fmt), e.strftime(fmt)) for s, e in pieces]


def build_fold_configs(config):
    """
    One config per cross_validation.groups entry. Fold j holds out group j's
    hydrological years as validation (validation_periods, validated on the train
    basin list) and trains on train_periods minus those years. Pool years in no
    group are always trained on. Each fold config has the cross_validation block
    removed, so it is an ordinary single-split training config - and gets its own
    normalization statistics from its own train_periods (normalization.py).
    """
    import copy

    groups = config['cross_validation']['groups']
    hydro_start = config.get('hydro_year_start_month', 10)
    pool = [(p.get('start_date') or None, p['end_date']) for p in config['train_periods']]
    test_start, test_end = pd.Timestamp(config['test_start_date']), pd.Timestamp(config['test_end_date'])

    seen = set()
    for idx, years in enumerate(groups):
        dup = seen.intersection(years)
        if dup:
            raise ValueError(f"cross_validation.groups: year(s) {sorted(dup)} appear in more than one group.")
        seen.update(years)

    fold_configs = []
    for idx, years in enumerate(groups):
        group_periods = build_year_periods(years, hydro_start)
        for s, e in group_periods:
            s, e = pd.Timestamp(s), pd.Timestamp(e)
            if s <= test_end and e >= test_start:
                raise ValueError(f"cross_validation.groups[{idx}] {years} overlaps the test period.")
            inside = any(_to_ts(ps, pd.Timestamp.min) <= s and e <= _to_ts(pe, pd.Timestamp.max)
                         for ps, pe in pool)
            if not inside:
                raise ValueError(f"cross_validation.groups[{idx}] {years} is not inside train_periods - "
                                 "train_periods must be the whole non-test pool the groups are drawn from.")

        fold = copy.deepcopy(config)
        fold.pop('cross_validation', None)
        fold.pop('normalization_stats', None)
        fold['train_periods'] = [{'start_date': s, 'end_date': e}
                                 for s, e in subtract_periods(pool, group_periods)]
        fold['validation_periods'] = [[s, e] for s, e in group_periods]
        fold['validation_basin_file'] = config['train_basin_file']
        fold_configs.append(fold)
    return fold_configs


def _load_basin_ids(basin_list_file, config, use_basin_splits, split_type='train'):
    # If temporal split is selected, skip files and load every single basin dynamically
    if not use_basin_splits:
        dyn_dir = config['processed_timeseries_dir']
        all_basins = [f.replace('.csv', '') for f in os.listdir(dyn_dir) if f.endswith('.csv')]
        if split_type == 'train':
            print(f"[Info] Spatial splits disabled. Automatically loaded all {len(all_basins)} basins for temporal split.")
        return all_basins

    if not os.path.exists(basin_list_file):
        raise FileNotFoundError(f"Basin split list file missing at: {basin_list_file}")
    with open(basin_list_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def _build_basin_datasets(basin_ids, config, periods, split_type='train'):
    dyn_dir = config['processed_timeseries_dir'] # Points to the clean resampled data
    static_file_path = config['normalized_static_attributes_file']
    basin_datasets = []
    dynamic_feature_names = config['dynamic_inputs']
    target_col = config['target_variables'][0]

    for basin_id in basin_ids:
        # Dynamic files are stored as [gauge_id].csv based on preprocessing script
        dyn_path = os.path.join(dyn_dir, f"{basin_id}.csv")

        # Safeguard verification: ensuring both dynamic sequence data and master static metrics exist
        if os.path.exists(dyn_path) and os.path.exists(static_file_path):
            # Per-basin train-period statistics (normalization.py) - inputs and target
            # are z-scored with these inside SingleBasinDataset.
            stats = basin_stats(config, basin_id)
            dyn_mean = [stats[feat]['mean'] for feat in dynamic_feature_names]
            dyn_std = [stats[feat]['std'] for feat in dynamic_feature_names]
            # Slice specific basin row inside SingleBasinDataset initialization
            basin_ds = SingleBasinDataset(dyn_path, static_file_path, config, periods,
                                           flow_std=stats[target_col]['std'],
                                           flow_mean=stats[target_col]['mean'], split_type=split_type,
                                           dyn_mean=dyn_mean, dyn_std=dyn_std)
            if len(basin_ds) > 0:
                basin_datasets.append(basin_ds)
        else:
            print(f"[Warning] Missing file paths for basin {basin_id} (Dynamic or Static data missing). Skipping.")

    if len(basin_datasets) == 0:
        raise RuntimeError("No valid basin datasets were generated from the provided split list.")
    return basin_datasets

def _resolve_num_workers(config):
    if config.get('num_workers', -1) != -1:
        return config['num_workers']
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        return int(os.environ['SLURM_CPUS_PER_TASK'])
    if 'google.colab' in sys.modules:
        return 2
    return min(max((os.cpu_count() or 1) // 2, 1), 8)


def get_dataloader(split_type, config, use_basin_splits=True):
    """
    Creates and packages multi-basin datasets for training or standard batch validation.
    """
    # Instantiate the wrapper dataset
    israel_dataset = IsraelBasinsDataset(split_type, config, use_basin_splits=use_basin_splits)
    
    _, _, is_shuffle = _get_split_bounds_and_config(split_type, config, use_basin_splits)
    
    loader = DataLoader(
        israel_dataset, 
        batch_size=config['batch_size'], 
        shuffle=is_shuffle, 
        num_workers=_resolve_num_workers(config),
        drop_last=False
    )
    
    loader.static_feature_names = config['static_attributes']
    loader.dynamic_feature_names = config['dynamic_inputs']

    return loader


# --------------------------------------------------------------------------------
# Hydrological-year period helpers
# --------------------------------------------------------------------------------

def _consecutive_year_runs(years):
    """Groups a list of years into maximal runs of consecutive years, e.g.
    [2010, 2011, 2013, 2014, 2015] -> [(2010, 2011), (2013, 2015)]."""
    sorted_years = sorted(years)
    runs = []
    run_start = run_end = sorted_years[0]
    for y in sorted_years[1:]:
        if y == run_end + 1:
            run_end = y
        else:
            runs.append((run_start, run_end))
            run_start = run_end = y
    runs.append((run_start, run_end))
    return runs


def build_year_range_period(start_year, end_year, hydro_year_start_month=10):
    """
    (start_date, end_date) datetime-string tuple spanning hydrological years
    start_year through end_year inclusive, matching this project's other
    hydro-year convention (get_hydrological_year in flow_quality_check.py):
    hydro-year N = Oct 1 of (N-1) through Sep 30 of N. E.g. a single-year
    range 2016 = '2015-10-01 08:00:00' -> '2016-09-30 07:00:00'; a
    multi-year range 2013-2015 = '2012-10-01 08:00:00' -> '2015-09-30
    07:00:00'.
    """
    start = pd.Timestamp(year=start_year - 1, month=hydro_year_start_month, day=1, hour=8)
    next_start = pd.Timestamp(year=end_year, month=hydro_year_start_month, day=1, hour=8)
    end = next_start - pd.Timedelta(days=1, hours=1)
    return start.strftime('%Y-%m-%d %H:%M:%S'), end.strftime('%Y-%m-%d %H:%M:%S')


def build_year_periods(years, hydro_year_start_month=10):
    """
    Converts a (possibly non-consecutive) list of years into the minimal set
    of disjoint (start_date, end_date) period tuples - one per maximal run
    of consecutive years - so adjacent years merge into a single continuous
    range instead of leaving a spurious 1-day gap at each internal
    year-boundary (the "day before next Oct 1" end-of-year formula, applied
    per year rather than per run, would otherwise drop the last day of every
    year but the run's last).
    """
    return [build_year_range_period(s, e, hydro_year_start_month)
            for s, e in _consecutive_year_runs(years)]
