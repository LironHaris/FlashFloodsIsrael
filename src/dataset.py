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
    It slices continuous hourly time series into look-back sequences of length `seq_length`.
    Preserves feature name order for downstream interpretability and analysis.
    """
    def __init__(self, dynamic_path, static_path, config, start_date, end_date):
        """
        Args:
            dynamic_path (str): Path to the hourly dynamic data CSV.
            static_path (str): Path to the static features CSV.
            config (dict): Configuration dictionary loaded from config.yml.
            start_date (str): Start date string for filtering (including buffer).
            end_date (str): End date string for filtering.
        """
        # Load dynamic and static data
        dyn_df = pd.read_csv(dynamic_path)
        dyn_df['date'] = pd.to_datetime(dyn_df['date'])
        dyn_df.set_index('date', inplace=True)
        stat_df = pd.read_csv(static_path)
        
        # Filter the dynamic data according to the provided period (includes warm-up buffer)
        dyn_df = dyn_df[start_date : end_date]
        
        # Extract configurations and preserve metadata feature names
        self.seq_length = config['seq_length']
        self.dynamic_feature_names = config['dynamic_inputs']
        self.static_feature_names = config['static_attributes']
        self.target_cols = config['target_variables']
        
        # Convert to fast NumPy arrays (float32 is optimized for PyTorch)
        # Explicitly indexing by list enforces strict adherence to config-defined feature order
        self.x_dynamic = dyn_df[self.dynamic_feature_names].values.astype(np.float32)
        self.y = dyn_df[self.target_cols].values.astype(np.float32)
        
        # Extract the single row of static attributes as a 1D vector matching config order
        self.x_static = stat_df[self.static_feature_names].iloc[0].values.astype(np.float32)
        
        # Calculate how many valid look-back sequences can be extracted
        self.num_samples = len(self.x_dynamic) - self.seq_length + 1

    def __len__(self):
        """Returns the total number of valid sequence windows in this dataset."""
        return max(0, self.num_samples)

    def __getitem__(self, idx):
        """
        Generates one sequence sample using explicit Look-back logic.
        """
        # Define the target prediction hour based on the index and window length
        target_day = idx + self.seq_length - 1
        
        # Look back from the target hour to establish the sequence starting hour
        start_day = target_day - self.seq_length + 1
        
        # Slice the past sequence window (Includes start_day up to target_day)
        window_x_dynamic = self.x_dynamic[start_day : target_day + 1]
        
        # Extract the corresponding target value for this exact look-back window
        target_y = self.y[target_day]
        
        return {
            'dynamic': torch.tensor(window_x_dynamic),
            'static': torch.tensor(self.x_static),
            'target': torch.tensor(target_y)
        }


# ==============================================================================
# 2. Private Helper Sub-Functions (Refactored Components)
# ==============================================================================
def _get_split_bounds_and_config(split_type, config):
    """
    Determines date boundaries (including warm-up buffer), basin file paths, 
    and shuffling requirements based on the split type.
    """
    buffer_hours = config['seq_length'] 

    if split_type == 'train':
        return (
            config['train_basin_file'],
            config['train_start_date'],
            config['train_end_date'],
            True # is_shuffle
        )
        
    elif split_type in ['val', 'test']:
        # Dynamically adapt prefixes for validation vs test keys
        prefix = 'validation' if split_type == 'val' else 'test'
        basin_list_file = config[f'{prefix}_basin_file']
        
        raw_start = pd.to_datetime(config[f'{prefix}_start_date'])
        # Expand start date backwards into the previous period to build history for the first sample
        start_date = (raw_start - pd.Timedelta(hours=buffer_hours)).strftime('%Y-%m-%d %H:%M:%S')
        end_date = config[f'{prefix}_end_date']
        
        return basin_list_file, start_date, end_date, False # is_shuffle = False
        
    else:
        raise ValueError("split_type must be either 'train', 'val', or 'test'")


def _load_basin_ids(basin_list_file):
    """Parses, cleans, and returns individual basin IDs from the target text file."""
    if not os.path.exists(basin_list_file):
        raise FileNotFoundError(f"Basin split list file missing at: {basin_list_file}")
        
    with open(basin_list_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def _build_basin_datasets(basin_ids, config, start_date, end_date):
    """
    Scans the disk for available files and instantiates SingleBasinDataset 
    objects for all verified basins.
    """
    dyn_dir = config['raw_dynamic_dir'] 
    stat_dir = config['processed_static_dir']
    basin_datasets = []

    for basin_id in basin_ids:
        dyn_path = os.path.join(dyn_dir, f"{basin_id}.csv")
        stat_path = os.path.join(stat_dir, f"{basin_id}_static.csv")
        
        if os.path.exists(dyn_path) and os.path.exists(stat_path):
            basin_ds = SingleBasinDataset(dyn_path, stat_path, config, start_date, end_date)
            
            # Append only if the basin contains enough data points to form at least one sequence window
            if len(basin_ds) > 0:
                basin_datasets.append(basin_ds)
        else:
            print(f"[Warning] Missing file paths for basin {basin_id}. Skipping.")

    if len(basin_datasets) == 0:
        raise RuntimeError("No valid basin datasets were generated from the provided list.")
        
    return basin_datasets


# ==============================================================================
# 3. Main Function
# ==============================================================================
def get_dataloader(split_type, config):
    """
    Controls the data pipeline by stitching together modular sub-functions.
    Creates, concatenates, packages multi-basin datasets, and attaches evaluation metadata.
    
    Args:
        split_type (str): One of ['train', 'val', 'test'].
        config (dict): Configuration dictionary loaded from config.yml.
        
    Returns:
        DataLoader: A PyTorch DataLoader handling multi-basin batches with metadata attached.
    """
    # Step 1: Extract paths, time bounds (with buffers), and shuffling rules
    basin_list_file, start_date, end_date, is_shuffle = _get_split_bounds_and_config(split_type, config)

    # Step 2: Load the target basin IDs
    basin_ids = _load_basin_ids(basin_list_file)

    # Step 3: Iterate and construct dataset objects for each individual basin
    basin_datasets = _build_basin_datasets(basin_ids, config, start_date, end_date)

    # Step 4: Concatenate all validated basin datasets into a single unified dataset
    full_dataset = ConcatDataset(basin_datasets)
    
    # Step 5: Package the unified dataset into a PyTorch DataLoader
    loader = DataLoader(
        full_dataset, 
        batch_size=config['batch_size'], 
        shuffle=is_shuffle, 
        num_workers=config['num_workers'],
        drop_last=False
    )
    
    # Step 6: Inject feature metadata into the DataLoader object for convenient evaluation access
    loader.static_feature_names = config['static_attributes']
    loader.dynamic_feature_names = config['dynamic_inputs']
    
    return loader