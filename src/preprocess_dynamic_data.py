"""
Module: preprocess_dynamic_data.py
Description: Hourly Hydro-Meteorological Data Preprocessing Pipeline.

This script implements a modular preprocessing pipeline designed to ingest raw, irregular 
catchment observations and standardize them into clean, continuous, hourly time series. 
The outputs serve as a robust foundational dataset for deep learning rainfall-runoff models.

Pipeline Architecture & Functional Stages:
1. Hourly Resampling:
   Transforms raw temporal records into uniform 1-hour block intervals. To preserve the 
   physical constraints of hydro-meteorological variables, cumulative features (precipitation) 
   are aggregated using a localized temporal sum, while continuous state features (discharge/flow) 
   are aggregated via a temporal mean. Bounds are defined using 'left' inclusive indexing.

2. Timeline Realignment & Gap Injection:
   Enforces a strict, continuous hourly index mapped between the basin's earliest available 
   dynamic timestamp ('basin_start_date') through the global experiment boundary ('test_end_date'). 
   Any missing temporal steps or data gaps are explicitly instantiated as NaN cells. 
   This maximizes historical training data per individual basin.

3. Quantitative Quality Analysis:
   Evaluates gauge reporting reliability by computing the percentage of valid, non-NaN flow 
   observations. Crucially, this operation runs on the aligned dataset BEFORE any data filling 
   takes place. This prevents down-stream artificial inflation of reporting health scores.

4. Domain-Specific Imputation:
   Applies a physics-informed approach to missing data reconstruction:
   - Precipitation ('hourly_precipitation'): Missing values are imputed with 0.0. This assumes no 
     unrecorded meteorological forcing events took place during transmission drops.
   - Streamflow ('Flow_m3_sec'): Left explicitly as NaN. Imputing artificial river discharge 
     values would severely compromise the learning capabilities of the neural model and induce 
     false physical behaviors.
"""

import os
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_config(yaml_path):
    """Load the YAML configuration file."""
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# --------------------------------------------------------------------------------
# 1. Resampling Function
# --------------------------------------------------------------------------------
def resample_to_hourly(df):
    """
    Resample raw data to hourly resolution.
    Flow is averaged, rain is summed.
    """
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Standardized to use 'hourly_precipitation' as the core dynamic feature name
    resampled = df.resample('h', closed='left', label='left').agg({
        'Flow_m3_sec': 'mean',
        'hourly_precipitation': 'sum'
    })
    resampled = resampled.rename(columns={'mean_rain': 'hourly_precipitation'})
    return resampled

# --------------------------------------------------------------------------------
# 2. Alignment Function
# --------------------------------------------------------------------------------
def align_to_timeline(resampled_df, start_date, end_date):
    """
    Force the dataframe into a continuous hourly timeline.
    This creates explicit NaNs for missing hours or data gaps.
    """
    all_hours = pd.date_range(start=start_date, end=end_date, freq='h')
    aligned_df = resampled_df.reindex(all_hours)
    return aligned_df

# --------------------------------------------------------------------------------
# 3. Quality Analysis Function
# --------------------------------------------------------------------------------
def analyze_data_quality(aligned_df):
    """
    Calculate the percentage of available (non-NaN) flow data.
    Runs on the aligned data BEFORE imputation to maintain accuracy.
    """
    flow_available_pct = aligned_df['Flow_m3_sec'].notna().mean() * 100
    return flow_available_pct

# --------------------------------------------------------------------------------
# 4. Imputation and Cleaning Function
# --------------------------------------------------------------------------------
def impute_missing_rain(aligned_df):
    """
    Fill missing precipitation values with 0. 
    Flow data is left as NaN to avoid training on artificial discharge data.
    """
    clean_df = aligned_df.copy()
    # FIXED: Changed from 'mean_rain' to 'hourly_precipitation' to match the pipeline
    clean_df['hourly_precipitation'] = clean_df['hourly_precipitation'].fillna(0)
    return clean_df

# --------------------------------------------------------------------------------
# Central Processing Function
# --------------------------------------------------------------------------------
def process_dynamic_data(config):
    """
    Orchestrates the data pipeline using modular helper functions.
    All paths and parameters are drawn directly from the config.
    """
    # Extract global end date boundary
    END_DATE = config['test_end_date']
    
    # Extract all paths from the configuration
    input_dir = config['raw_dynamic_dir']
    output_dir = config['processed_timeseries_dir']
    report_path = config['availability_report_file']

    # Create directories if they do not exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    availability_records = []
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    # Run the pipeline on each file
    for file_name in tqdm(csv_files, desc="Processing Gauges", unit="file"):
        raw_df = pd.read_csv(os.path.join(input_dir, file_name))
        
        # Step 1: Resample to hourly resolution
        hourly_df = resample_to_hourly(raw_df)
        
        # Step 2: Extract the earliest available dynamic timestamp specific to this basin
        # This replaces the hardcoded global START_DATE to ensure no historical data is discarded
        basin_start_date = hourly_df.index.min()
        
        # Align to the full timeline starting from the basin's individual birth date
        aligned_df = align_to_timeline(hourly_df, basin_start_date, END_DATE)
        
        # Step 3: Check original data quality (before imputation)
        flow_available = analyze_data_quality(aligned_df)
        
        # Step 4: Impute missing rain values with 0
        clean_df = impute_missing_rain(aligned_df)
        
        # Record data for the summary report
        availability_records.append({
            'gauge_id': file_name.replace('.csv', ''), 
            'availability_pct': flow_available
        })
        
        # Save the processed CSV file
        output_path = os.path.join(output_dir, file_name)
        clean_df.to_csv(output_path, index_label='date')
        
        tqdm.write(f"Done {file_name}: Processed from {basin_start_date} to {END_DATE}. {flow_available:.2f}% flow data.")

    # Create and save the final availability report
    report_df = pd.DataFrame(availability_records)
    report_df.to_csv(report_path, index=False)
    print(f"\nSummary report saved to: {report_path}")

if __name__ == "__main__":
    CONFIG_PATH = "configs/config.yml"
    yaml_config = load_config(CONFIG_PATH)
    
    process_dynamic_data(yaml_config)