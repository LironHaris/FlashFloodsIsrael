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
        
        # Slice specific basin row inside the master static matrix file
        basin_static_row = stat_df[stat_df['gauge_id'].astype(str) == self.gauge_id]
        if basin_static_row.empty:
            raise KeyError(f"Gauge ID '{self.gauge_id}' missing in static attributes file: {static_path}")
            
        self.x_static = basin_static_row[self.static_feature_names].iloc[0].values.astype(np.float32)        
        
        # Calculate valid window boundaries
        self.num_samples = len(self.x_dynamic) - self.seq_length - max(self.forecast_lead_times) + 1

    def __len__(self):
        return max(0, self.num_samples)

    def __getitem__(self, idx):
        target_day = idx + self.seq_length - 1
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
    def __init__(self, split_type, config):
        # Step 1: Extract paths, time bounds, and shuffling rules
        basin_list_file, start_date, end_date, _ = _get_split_bounds_and_config(split_type, config)

        # Step 2: Load the target basin IDs
        basin_ids = _load_basin_ids(basin_list_file)

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
            ds_len = len(ds)
            for idx in range(ds_len):
                self.sample_basin_mappings.append(ds.gauge_id)
                # The prediction time corresponds to the target_day (the 365th step)
                target_idx = idx + ds.seq_length - 1
                self.sample_date_mappings.append(ds.dates[target_idx])

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        return self.concat_dataset[idx]


# ==============================================================================
# 3. Private Helper Sub-Functions
# ==============================================================================
def _get_split_bounds_and_config(split_type, config):
    buffer_hours = config['seq_length'] 

    if split_type == 'train':
        return config['train_basin_file'], None, config['train_end_date'], True
        
    elif split_type in ['val', 'test']:
        prefix = 'validation' if split_type == 'val' else 'test'
        basin_list_file = config[f'{prefix}_basin_file']
        
        raw_start = pd.to_datetime(config[f'{prefix}_start_date'])
        start_date = (raw_start - pd.Timedelta(hours=buffer_hours)).strftime('%Y-%m-%d %H:%M:%S')
        end_date = config[f'{prefix}_end_date']
        
        return basin_list_file, start_date, end_date, False
    else:
        raise ValueError("split_type must be either 'train', 'val', or 'test'")


def _load_basin_ids(basin_list_file):
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
def get_dataloader(split_type, config):
    """
    Creates and packages multi-basin datasets for training or standard batch validation.
    """
    # Instantiate the wrapper dataset
    israel_dataset = IsraelBasinsDataset(split_type, config)
    
    _, _, _, is_shuffle = _get_split_bounds_and_config(split_type, config)
    
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