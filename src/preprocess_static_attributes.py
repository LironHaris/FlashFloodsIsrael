"""
Module: preprocess_static_attributes.py
Description: Static Basin Attributes Ingestion, Merging, and Standardization Pipeline.

This script implements a modular preprocessing workflow for static catchment attributes, 
consolidating raw geographic, physiographic, and climatic characteristics from disparate data 
sources (Caravan, HydroATLAS, and localized auxiliary records) into a singular, clean dataset.

Pipeline Architecture & Functional Stages:
1. Heterogeneous Data Merging (Inner Join):
   Loads raw attribute dataframes and performs consecutive relational inner joins mapped on the 
   unique basin identifier ('gauge_id'). This structural choice enforces strict alignment, retaining 
   only the catchments that possess verified records across all metadata source platforms. 
   From the auxiliary dataset, it isolates and appends only the 'area' feature.

2. Feature Filtration (Missing Data Removal):
   Identifies and drops entire feature columns containing any missing values (NaNs). 
   This is a strict mathematical prerequisite for NeuralHydrology and structural architectures 
   like the EA-LSTM; since static attributes directly parameterize the constant static input gate ,
   any undefined value would propagate throughout the recurrent hourly time steps, 
   corrupting the hidden and cell state space updates.

3. Central Tendency & Dispersion Calculation (Feature Statistics):
   Isolates remaining numerical columns to evaluate localized spatial statistics across all basins:
   - Feature Mean (mu)
   - Feature Standard Deviation (sigma)
   These metrics are exported as a report and serve as the baseline scaling factors for downstream 
   Z-score feature scaling, preventing features with large numeric ranges (e.g., basin area) 
   from dominating the loss functions.

Operational Context:
- This script processes spatial metadata boundaries independent of temporal parameters or 
  multi-horizon prediction configurations.
"""

import pandas as pd
import numpy as np
import os
import yaml

def load_config(yaml_path):
    """Load the YAML configuration file."""
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def load_and_merge(raw_static_dir):
    """
    Load source CSV files from the raw static directory and merge them on 'gauge_id', 
    including only the 'area' feature from the third source.
    """
    caravan_path = os.path.join(raw_static_dir, 'attributes_caravan_il.csv')
    hydroatlas_path = os.path.join(raw_static_dir, 'attributes_hydroatlas_il.csv')
    other_path = os.path.join(raw_static_dir, 'attributes_other_il.csv')
    
    df_caravan = pd.read_csv(caravan_path)
    df_hydroatlas = pd.read_csv(hydroatlas_path)
    df_other = pd.read_csv(other_path)[['gauge_id', 'area']]
    
    # Inner join ensures we only keep basins present across all metadata files
    merged = pd.merge(df_caravan, df_hydroatlas, on='gauge_id', how='inner')
    merged = pd.merge(merged, df_other, on='gauge_id', how='inner')
    
    return merged

def clean_missing_features(df):
    """
    Remove all columns containing any missing values to ensure compatibility with NeuralHydrology.
    Logs a warning with the names of any dropped features.
    """
    ids = df[['gauge_id']]
    features = df.drop(columns=['gauge_id'])
    
    # Identify and log features that contain NaNs before dropping them
    missing_cols = features.columns[features.isna().any()].tolist()
    if missing_cols:
        print(f"\n[WARNING] Dropping {len(missing_cols)} static feature(s) due to missing values:")
        print(f"Dropped columns: {missing_cols}\n")
    else:
        print("\n[INFO] No missing values detected in static attributes. All features retained.")
        
    clean_features = features.dropna(axis=1, how='any')
    
    return pd.concat([ids, clean_features], axis=1)

def main(config):
    """
    Execute the full preprocessing pipeline for static attributes: 
    merge sources, clean missing features, calculate stats, and save outputs.
    """
    # Extract paths from config
    raw_static_dir = config['raw_static_dir']
    output_dir = config['processed_static_dir']
    output_nh = config['static_attributes_file']
    output_stats = config['feature_stats_file']
    output_normalized = config['normalized_static_attributes_file']

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Core pipeline execution
    merged = load_and_merge(raw_static_dir)
    final_df = clean_missing_features(merged)

    # Compute mean and standard deviation per feature, across all basins
    numeric_df = final_df.select_dtypes(include=[np.number])
    mean = numeric_df.mean()
    std = numeric_df.std()
    stats = pd.DataFrame({
        'feature': numeric_df.columns,
        'mean': mean,
        'std': std
    })

    # Z-score normalize each feature (per-column mean/std over all basins).
    # Guard against near-zero std (constant features) to avoid divide-by-zero.
    std_safe = std.where(std >= 1e-6, 1.0)
    normalized_numeric = (numeric_df - mean) / std_safe
    normalized_df = pd.concat([final_df[['gauge_id']], normalized_numeric], axis=1)

    # Save processed files
    final_df.to_csv(output_nh, index=False)
    stats.to_csv(output_stats, index=False)
    normalized_df.to_csv(output_normalized, index=False)

    print(f"Static attributes saved to: {output_nh}")
    print(f"Feature statistics saved to: {output_stats}")
    print(f"Normalized static attributes saved to: {output_normalized}")

if __name__ == "__main__":
    CONFIG_PATH = "configs/config.yml"
    
    yaml_config = load_config(CONFIG_PATH)
    main(yaml_config)