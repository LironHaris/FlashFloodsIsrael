"""
Module: dataset.py
Description: Hourly Multi-Basin Data Pipeline for Entity-Aware LSTM (EA-LSTM) Flood Forecasting.
             Optimized to isolate and align static catchment attributes dynamically per basin.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

# ==============================================================================
# 0. NaN Fill Helper
# ==============================================================================
def _fill_flow_nan_by_interpolation(flow_array, min_surrounding, max_gap):
    """
    Linearly interpolate NaN gaps in a 1-D flow array when:
      - gap length <= max_gap
      - at least min_surrounding consecutive non-NaN values exist on both sides
    Returns a new array with qualifying gaps filled.
    """
    flow = flow_array.copy()
    nan_mask = np.isnan(flow)

    if not nan_mask.any():
        return flow

    padded = np.concatenate([[False], nan_mask, [False]])
    starts = np.where(~padded[:-1] &  padded[1:])[0]
    ends   = np.where( padded[:-1] & ~padded[1:])[0]

    n = len(starts)
    for i, (s, e) in enumerate(zip(starts, ends)):
        gap_len = e - s
        if gap_len > max_gap:
            continue

        left_count  = s         if i == 0     else s - ends[i - 1]
        right_count = len(flow) - e if i == n - 1 else starts[i + 1] - e

        if left_count < min_surrounding or right_count < min_surrounding:
            continue

        v_before = flow[s - 1]
        v_after  = flow[e]
        flow[s:e] = np.linspace(v_before, v_after, gap_len + 2)[1:-1]

    return flow


# ==============================================================================
# 1. Single Basin Dataset Class
# ==============================================================================
class SingleBasinDataset(Dataset):
    """
    A PyTorch Dataset that handles the dynamic and static data for a SINGLE basin.
    """
    def __init__(self, dynamic_path, static_path, config, start_date, end_date):
        # Extract basin ID dynamically from filename
        self.gauge_id = str(os.path.basename(dynamic_path).replace('.csv', ''))
        
        # Load dynamic and static data
        dyn_df = pd.read_csv(dynamic_path)
        dyn_df['date'] = pd.to_datetime(dyn_df['date'])
        dyn_df.set_index('date', inplace=True)
        stat_df = pd.read_csv(static_path)
        
        # Filter the dynamic data according to the provided period
        if start_date is None:
            dyn_df = dyn_df[:end_date]
        else:
            dyn_df = dyn_df[start_date : end_date]
            
        # Store index dates for evaluation alignment
        self.dates = dyn_df.index
        
        # Extract configurations
        self.seq_length = config['seq_length']
        self.dynamic_feature_names = config['dynamic_inputs']
        self.static_feature_names = config['static_attributes']
        self.target_cols = config['target_variables']
        self.forecast_lead_times = config['forecast_lead_times']
        
        # Convert to fast NumPy arrays
        self.x_dynamic = dyn_df[self.dynamic_feature_names].values.astype(np.float32)
        self.y = dyn_df[self.target_cols].values.astype(np.float32)

        # Fill qualifying NaN gaps in flow with linear interpolation
        min_surrounding = config.get('flow_nan_fill_min_surrounding', 24)
        max_gap = config.get('flow_nan_fill_max_gap', 72)
        if 'Flow_m3_sec' in self.target_cols:
            flow_col_idx = self.target_cols.index('Flow_m3_sec')
            self.y[:, flow_col_idx] = _fill_flow_nan_by_interpolation(
                self.y[:, flow_col_idx], min_surrounding, max_gap
            )

        # Slice specific basin row inside the master static matrix file
        basin_static_row = stat_df[stat_df['gauge_id'].astype(str) == self.gauge_id]
        if basin_static_row.empty:
            raise KeyError(f"Gauge ID '{self.gauge_id}' missing in static attributes file: {static_path}")

        self.x_static = basin_static_row[self.static_feature_names].iloc[0].values.astype(np.float32)

        # Build valid sample indices: skip windows where any forecast target is still NaN
        max_lead = max(self.forecast_lead_times)
        n_potential = len(self.x_dynamic) - self.seq_length - max_lead + 1
        self.valid_indices = [
            i for i in range(max(0, n_potential))
            if not any(
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
            
        target_y = np.array(target_list, dtype=np.float32).flatten()
        
        return {
            'dynamic': torch.tensor(window_x_dynamic),
            'static': torch.tensor(self.x_static),
            'target': torch.tensor(target_y)
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
        basin_list_file, start_date, end_date, _ = _get_split_bounds_and_config(split_type, config, use_basin_splits)

        # Step 2: Load the target basin IDs
        basin_ids = _load_basin_ids(basin_list_file, config, use_basin_splits, split_type)

        # Step 3: Construct dataset objects for each individual basin
        self.basin_datasets = _build_basin_datasets(basin_ids, config, start_date, end_date)
        
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

    if use_basin_splits:
        if split_type == 'train':
            return config['train_basin_file'], config.get('train_start_date'), config['train_end_date'], True
            
        elif split_type in ['val', 'test']:
            prefix = 'validation' if split_type == 'val' else 'test'
            basin_list_file = config[f'{prefix}_basin_file']
            
            raw_start = pd.to_datetime(config[f'{prefix}_start_date'])
            start_date = (raw_start - pd.Timedelta(hours=buffer_hours)).strftime('%Y-%m-%d %H:%M:%S')
            end_date = config[f'{prefix}_end_date']
            
            return basin_list_file, start_date, end_date, False
    else:
        # Strict temporal split: ignore val/test basin files, use master list or fallback directory scan
        basin_list_file = config['train_basin_file']
        
        if split_type == 'train':
            return basin_list_file, config.get('train_start_date'), config['train_end_date'], True
            
        elif split_type in ['val', 'test']:
            prefix = 'validation' if split_type == 'val' else 'test'
            
            raw_start = pd.to_datetime(config[f'{prefix}_start_date'])
            start_date = (raw_start - pd.Timedelta(hours=buffer_hours)).strftime('%Y-%m-%d %H:%M:%S')
            end_date = config[f'{prefix}_end_date']
            
            return basin_list_file, start_date, end_date, False
            
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


def _build_basin_datasets(basin_ids, config, start_date, end_date):
    dyn_dir = config['processed_timeseries_dir'] # Points to the clean resampled data
    static_file_path = config['static_attributes_file']
    basin_datasets = []

    for basin_id in basin_ids:
        # Dynamic files are stored as [gauge_id].csv based on preprocessing script
        dyn_path = os.path.join(dyn_dir, f"{basin_id}.csv")
        
        # Safeguard verification: ensuring both dynamic sequence data and master static metrics exist
        if os.path.exists(dyn_path) and os.path.exists(static_file_path):
            # Slice specific basin row inside SingleBasinDataset initialization
            basin_ds = SingleBasinDataset(dyn_path, static_file_path, config, start_date, end_date)
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
def get_dataloader(split_type, config, use_basin_splits=True):
    """
    Creates and packages multi-basin datasets for training or standard batch validation.
    """
    # Instantiate the wrapper dataset
    israel_dataset = IsraelBasinsDataset(split_type, config, use_basin_splits=use_basin_splits)
    
    _, _, _, is_shuffle = _get_split_bounds_and_config(split_type, config, use_basin_splits)
    
    loader = DataLoader(
        israel_dataset, 
        batch_size=config['batch_size'], 
        shuffle=is_shuffle, 
        num_workers=config['num_workers'],
        drop_last=False
    )
    
    loader.static_feature_names = config['static_attributes']
    loader.dynamic_feature_names = config['dynamic_inputs']
    
    return loader