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


# ==============================================================================
# 1. Single Basin Dataset Class
# ==============================================================================
class SingleBasinDataset(Dataset):
    """
    A PyTorch Dataset that handles the dynamic and static data for a SINGLE basin.
    """
    def __init__(self, dynamic_path, static_path, config, periods, flow_std: float = 1.0,
                 flow_mean: float = 0.0):
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

        # Build valid sample indices: skip windows where any forecast target is still NaN
        max_lead = max(self.forecast_lead_times)
        n_potential = len(self.x_dynamic) - self.seq_length - max_lead + 1
        self.valid_indices = [
            i for i in range(max(0, n_potential))
            if not np.isnan(self.x_dynamic[i : i + self.seq_length]).any()
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
        
        window_x_dynamic = self.x_dynamic[start_day : target_day + 1]
        
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


# ==============================================================================
# 2. Main Wrapper Dataset for Multiple Basins (IsraelBasinsDataset)
# ==============================================================================
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
        self.basin_datasets = _build_basin_datasets(basin_ids, config, periods)
        
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


# ==============================================================================
# 3. Private Helper Sub-Functions
# ==============================================================================
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


def _build_basin_datasets(basin_ids, config, periods):
    dyn_dir = config['processed_timeseries_dir'] # Points to the clean resampled data
    static_file_path = config['normalized_static_attributes_file']
    basin_datasets = []

    # Load per-basin flow mean/std (train years only) written by preprocess_dynamic_data.py
    stats_df = pd.read_csv(config['availability_report_file'])
    flow_std_map = dict(zip(stats_df['gauge_id'].astype(str), stats_df['flow_std']))
    flow_mean_map = dict(zip(stats_df['gauge_id'].astype(str), stats_df['flow_mean']))

    for basin_id in basin_ids:
        # Dynamic files are stored as [gauge_id].csv based on preprocessing script
        dyn_path = os.path.join(dyn_dir, f"{basin_id}.csv")

        # Safeguard verification: ensuring both dynamic sequence data and master static metrics exist
        if os.path.exists(dyn_path) and os.path.exists(static_file_path):
            flow_std = flow_std_map.get(basin_id, 1.0)
            flow_mean = flow_mean_map.get(basin_id, 0.0)
            # Slice specific basin row inside SingleBasinDataset initialization
            basin_ds = SingleBasinDataset(dyn_path, static_file_path, config, periods, flow_std,
                                           flow_mean=flow_mean)
            if len(basin_ds) > 0:
                basin_datasets.append(basin_ds)
        else:
            print(f"[Warning] Missing file paths for basin {basin_id} (Dynamic or Static data missing). Skipping.")

    if len(basin_datasets) == 0:
        raise RuntimeError("No valid basin datasets were generated from the provided split list.")
    return basin_datasets


# ==============================================================================
# 4. Main Loader Builder Function
# ==============================================================================
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