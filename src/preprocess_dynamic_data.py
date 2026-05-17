import pandas as pd
import os
import yaml
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
    
    resampled = df.resample('h', closed='left', label='left').agg({
        'Flow_m3_sec': 'mean',
        'mean_rain': 'sum'
    })
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
    clean_df['mean_rain'] = clean_df['mean_rain'].fillna(0)
    return clean_df

# --------------------------------------------------------------------------------
# Central Processing Function
# --------------------------------------------------------------------------------
def process_dynamic_data(config):
    """
    Orchestrates the data pipeline using modular helper functions.
    All paths and parameters are drawn directly from the config.
    """
    # Extract boundary dates
    START_DATE = config['train_start_date']
    END_DATE = config['test_end_date']
    
    print(f"Overall timeseries span: {START_DATE} to {END_DATE}")

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
        
        # Step 2: Align to the full timeline (creates explicit NaNs)
        aligned_df = align_to_timeline(hourly_df, START_DATE, END_DATE)
        
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
        
        tqdm.write(f"Done {file_name}: {flow_available:.2f}% flow data.")

    # Create and save the final availability report
    report_df = pd.DataFrame(availability_records)
    report_df.to_csv(report_path, index=False)
    print(f"\nSummary report saved to: {report_path}")

if __name__ == "__main__":
    CONFIG_PATH = "config.yml" 
    yaml_config = load_config(CONFIG_PATH)
    
    process_dynamic_data(yaml_config)