"""
Module: dataset.py
Description: Hourly Multi-Basin Data Pipeline for Entity-Aware LSTM (EA-LSTM) Flood Forecasting.
             Optimized to isolate and align static catchment attributes dynamically per basin.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

class SingleBasinDataset(Dataset):
    """
    A PyTorch Dataset that handles the dynamic and static data for a SINGLE basin.
    """
    def __init__(self, dynamic_path, static_path, config, periods, flow_std: float = 1.0,
                 flow_mean: float = 0.0, split_type: str = 'train', zero_impute_values=None):
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
        
        # Convert to fast NumPy arrays
        self.x_dynamic = dyn_df[self.dynamic_feature_names].values.astype(np.float32)
        self.y = dyn_df[self.target_cols].values.astype(np.float32)

        # Slice specific basin row inside the master static matrix file
        basin_static_row = stat_df[stat_df['gauge_id'].astype(str) == self.gauge_id]
        if basin_static_row.empty:
            raise KeyError(f"Gauge ID '{self.gauge_id}' missing in static attributes file: {static_path}")

        # Static attributes are pre-normalized (z-scored per feature across all basins)
        # by preprocess_static_attributes.py before this file is loaded.
        self.x_static = basin_static_row[self.static_feature_names].iloc[0].values.astype(np.float32)

        # Zero-rainfall-equivalent value per dynamic feature (in the same z-score space
        # the saved CSV already uses), used to fill in NaNs that fall within a window's
        # per-feature tolerance instead of excluding the whole window.
        if zero_impute_values is None:
            zero_impute_values = np.zeros(len(self.dynamic_feature_names), dtype=np.float32)
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

        self.valid_indices = [
            i for i in range(n_potential)
            if input_ok[i]
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
    """
    def __init__(self, split_type, config, use_basin_splits=True):
        # Step 1: Extract paths, time bounds, and shuffling rules
        basin_list_file, periods, _ = _get_split_bounds_and_config(split_type, config, use_basin_splits)

        # Step 2: Load the target basin IDs
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
        # When basin splits are disabled, fall back to the train basin list (master list / directory scan)
        basin_list_file = config[f'{prefix}_basin_file'] if use_basin_splits else config['train_basin_file']

        raw_start = pd.to_datetime(config[f'{prefix}_start_date'])
        start_date = (raw_start - pd.Timedelta(hours=buffer_hours)).strftime('%Y-%m-%d %H:%M:%S')
        end_date = config[f'{prefix}_end_date']

        return basin_list_file, [(start_date, end_date)], False

    raise ValueError("split_type must be either 'train', 'val', or 'test'")


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

    # Load per-basin flow mean/std (train years only) written by preprocess_dynamic_data.py
    stats_df = pd.read_csv(config['availability_report_file'])
    flow_std_map = dict(zip(stats_df['gauge_id'].astype(str), stats_df['flow_std']))
    flow_mean_map = dict(zip(stats_df['gauge_id'].astype(str), stats_df['flow_mean']))

    # Per-basin zero-rainfall-equivalent z-score for each dynamic input, using the same
    # per-feature mean/std preprocess_dynamic_data.py already baked into the report -
    # used to impute NaNs that fall within a window's configured tolerance (see
    # SingleBasinDataset's valid_indices/x_dynamic_filled).
    dynamic_feature_names = config['dynamic_inputs']
    stats_by_gauge = stats_df.set_index(stats_df['gauge_id'].astype(str))
    zero_impute_maps = {}
    for gauge_id, row in stats_by_gauge.iterrows():
        zero_impute_maps[gauge_id] = np.array([
            (0.0 - row.get(f'{feat}_mean', 0.0)) / row.get(f'{feat}_std', 1.0)
            for feat in dynamic_feature_names
        ], dtype=np.float32)

    for basin_id in basin_ids:
        # Dynamic files are stored as [gauge_id].csv based on preprocessing script
        dyn_path = os.path.join(dyn_dir, f"{basin_id}.csv")

        # Safeguard verification: ensuring both dynamic sequence data and master static metrics exist
        if os.path.exists(dyn_path) and os.path.exists(static_file_path):
            flow_std = flow_std_map.get(basin_id, 1.0)
            flow_mean = flow_mean_map.get(basin_id, 0.0)
            zero_impute_values = zero_impute_maps.get(basin_id)
            # Slice specific basin row inside SingleBasinDataset initialization
            basin_ds = SingleBasinDataset(dyn_path, static_file_path, config, periods, flow_std,
                                           flow_mean=flow_mean, split_type=split_type,
                                           zero_impute_values=zero_impute_values)
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
# K-Fold Cross-Validation Support
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
    (start_date, end_date) datetime-string tuple spanning start_year through
    end_year inclusive, in this project's user-facing convention: "year N"
    starts Oct 1 of N and runs through Sep 30 of N+1 (e.g. a single-year
    range 2016 = '2016-10-01 08:00:00' -> '2017-09-30 07:00:00'; a
    multi-year range 2013-2015 = '2013-10-01 08:00:00' -> '2016-09-30
    07:00:00') - matches the literal digits already used in every config's
    train_periods/validation/test date bounds (verified against
    configs/best_model_0_0.yml's 2013-2015 and 2019-2022 train_periods
    entries and its 2016-2018 test span).
    """
    start = pd.Timestamp(year=start_year, month=hydro_year_start_month, day=1, hour=8)
    next_start = pd.Timestamp(year=end_year + 1, month=hydro_year_start_month, day=1, hour=8)
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


def build_cv_group_datasets(config, use_basin_splits=True):
    """
    Precomputes everything a k-fold cross-validation training run needs, once:
      - the fixed/always-train slice, i.e. whatever train_periods already
        resolves to (e.g. an open-start "years before the CV pool" entry) -
        never rotated out as validation.
      - for every configured cross_validation.groups entry, both a
        train-tolerance and an eval-tolerance set of per-basin datasets (the
        NaN-tolerance policy - train_max_nan_pct vs eval_max_nan_pct - depends
        on which role a group plays for a given epoch, so both are built
        up front rather than re-read from disk every epoch).
    Raises ValueError if any group's years are claimed by another group,
    overlap the test period, or overlap train_periods (the fixed slice must
    be trimmed to exclude any year handed to cross_validation.groups).
    Returns (fixed_train_datasets, group_datasets) where group_datasets is
    {group_idx: {'train': [...], 'val': [...]}}.
    """
    cv_config = config['cross_validation']
    groups = cv_config['groups']
    hydro_start = config.get('hydro_year_start_month', 10)
    basin_ids = _load_basin_ids(config['train_basin_file'], config, use_basin_splits, 'train')

    seen_years = set()
    for g in groups:
        dup = seen_years.intersection(g)
        if dup:
            raise ValueError(f"cross_validation.groups: year(s) {dup} appear in more than one group.")
        seen_years.update(g)

    # Fixed/always-train slice: whatever train_periods already resolves to -
    # unconditionally included in every fold's training set.
    _, fixed_periods, _ = _get_split_bounds_and_config('train', config, use_basin_splits)
    fixed_train_datasets = _build_basin_datasets(basin_ids, config, fixed_periods, split_type='train')

    test_start = pd.Timestamp(config['test_start_date'])
    test_end = pd.Timestamp(config['test_end_date'])

    group_datasets = {}
    for idx, years in enumerate(groups):
        periods = build_year_periods(years, hydro_start)
        for start_str, end_str in periods:
            s, e = pd.Timestamp(start_str), pd.Timestamp(end_str)
            if s <= test_end and e >= test_start:
                raise ValueError(f"cross_validation.groups[{idx}] year period "
                                  f"[{start_str}, {end_str}] overlaps the test period.")
            for fstart, fend in fixed_periods:
                if fstart is not None and s <= pd.Timestamp(fend) and e >= pd.Timestamp(fstart):
                    raise ValueError(f"cross_validation.groups[{idx}] year period "
                                      f"[{start_str}, {end_str}] overlaps train_periods "
                                      f"- trim train_periods down to just the fixed portion.")
        group_datasets[idx] = {
            'train': _build_basin_datasets(basin_ids, config, periods, split_type='train'),
            'val': _build_basin_datasets(basin_ids, config, periods, split_type='val'),
        }
    return fixed_train_datasets, group_datasets


def get_cv_fold_dataloaders(fixed_train_datasets, group_datasets, val_group_idx, config):
    """
    Cheaply recombines the datasets precomputed by build_cv_group_datasets
    into this epoch's train/val DataLoaders - no CSV re-reads. val_group_idx
    is held out as validation (its eval-tolerance variant); the fixed slice
    plus every other group's train-tolerance variant form the training set.
    """
    train_datasets = list(fixed_train_datasets) + [
        ds for g, variants in group_datasets.items()
        if g != val_group_idx for ds in variants['train']
    ]
    val_datasets = group_datasets[val_group_idx]['val']

    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=config['batch_size'],
                               shuffle=True, num_workers=_resolve_num_workers(config), drop_last=False)
    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=config['batch_size'],
                             shuffle=False, num_workers=_resolve_num_workers(config), drop_last=False)

    for loader in (train_loader, val_loader):
        loader.static_feature_names = config['static_attributes']
        loader.dynamic_feature_names = config['dynamic_inputs']

    return train_loader, val_loader