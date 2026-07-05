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
   observations, and the percentage of hours with BOTH flow and rain simultaneously available.
   Basins below a configured combined-availability threshold are excluded entirely rather than
   processed further (see min_combined_availability_pct in config).

4. No Fabrication Policy:
   Neither precipitation ('hourly_precipitation') nor streamflow ('Flow_m3_sec') gaps are
   imputed - both are left explicitly as NaN. Inventing values for either would compromise the
   learning capabilities of the neural model and induce false physical behaviors. Any training
   window touching a NaN in either is excluded downstream by dataset.py's validity check.
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
        'mean_rain': 'sum'
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
# 3. Quality Analysis Functions
# --------------------------------------------------------------------------------
def analyze_data_quality(aligned_df):
    """
    Calculate the percentage of available (non-NaN) flow data.
    """
    flow_available_pct = aligned_df['Flow_m3_sec'].notna().mean() * 100
    return flow_available_pct

def analyze_combined_availability(aligned_df):
    """
    Calculate the percentage of hours where both Flow_m3_sec and
    hourly_precipitation are simultaneously non-NaN. Used to decide whether a
    basin has enough real (non-fabricated) data to be worth training on at all.
    """
    combined_available_pct = (
        aligned_df['Flow_m3_sec'].notna() & aligned_df['hourly_precipitation'].notna()
    ).mean() * 100
    return combined_available_pct

# --------------------------------------------------------------------------------
# 4. Cumulative Rain Feature Function
# --------------------------------------------------------------------------------
def add_cumulative_rain_features(df, windows):
    """
    Add trailing rolling-sum cumulative rain columns to an hourly-resolution
    dataframe. hourly_precipitation may contain real NaN gaps (no imputation
    upstream). min_periods=hours means the full trailing window must be
    NaN-free for a value to be produced - any missing hour inside the window,
    or insufficient history at a basin's start of record, yields NaN, which
    propagates downstream so dataset.py excludes that training window instead
    of silently summing over a gap.
    """
    result = df.copy()
    for w in windows:
        result[w['name']] = (
            result['hourly_precipitation']
            .rolling(window=w['hours'], min_periods=w['hours'])
            .sum()
        )
    return result

# --------------------------------------------------------------------------------
# 5. Long NaN Stretch Removal Function
# --------------------------------------------------------------------------------
def drop_long_flow_nan_stretches(df, seq_length):
    """
    Remove rows belonging to consecutive NaN runs in Flow_m3_sec that exceed seq_length.
    Returns the cleaned DataFrame and the number of dropped rows.
    """
    flow = df['Flow_m3_sec'].values
    nan_mask = np.isnan(flow)

    padded = np.concatenate([[False], nan_mask, [False]])
    starts = np.where(~padded[:-1] &  padded[1:])[0]
    ends   = np.where( padded[:-1] & ~padded[1:])[0]

    drop_positions = []
    for s, e in zip(starts, ends):
        if (e - s) > seq_length:
            drop_positions.extend(range(s, e))

    if not drop_positions:
        return df, 0

    labels_to_drop = df.index[drop_positions]
    return df.drop(index=labels_to_drop), len(drop_positions)


# --------------------------------------------------------------------------------
# Central Processing Function
# --------------------------------------------------------------------------------
def process_dynamic_data(config):
    """
    Orchestrates the data pipeline using modular helper functions.
    All paths and parameters are drawn directly from the config.
    """
    # Extract global end date boundary: the latest end date across all configured
    # train/validation/test periods, since train periods can now extend past test.
    all_end_dates = (
        [p['end_date'] for p in config['train_periods']]
        + [config['validation_end_date'], config['test_end_date']]
    )
    END_DATE = max(pd.to_datetime(d) for d in all_end_dates)
    
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
        
        # Step 3: Check original data quality
        flow_available = analyze_data_quality(aligned_df)

        # Step 3b: Exclude basins without enough combined (flow AND rain) real data.
        # Checked before any further processing so excluded basins skip the rest of
        # the pipeline entirely and never get a processed timeseries CSV written -
        # dataset.py already treats a missing CSV as "basin not available."
        combined_available = analyze_combined_availability(aligned_df)
        if combined_available < config['min_combined_availability_pct']:
            availability_records.append({
                'gauge_id': file_name.replace('.csv', ''),
                'availability_pct': flow_available,
                'combined_availability_pct': combined_available,
                'excluded': True,
            })
            tqdm.write(
                f"Skipped {file_name}: {combined_available:.2f}% combined flow+rain "
                f"availability, below the {config['min_combined_availability_pct']}% threshold."
            )
            continue

        # Step 4: Rain gaps are left as real NaN here (no imputation), the same way
        # Flow_m3_sec already is. dataset.py's existing valid_indices check already
        # excludes any training window touching a NaN in either, so this is the only
        # change needed to make rain "skip" instead of fabricating 0.0 for gaps.
        clean_df = aligned_df.copy()

        # Step 4b: Add long-window cumulative (trailing rolling-sum) rain features.
        # Must run before drop_long_flow_nan_stretches (so rolling sees a fully
        # contiguous hourly index).
        clean_df = add_cumulative_rain_features(clean_df, config['cumulative_rain_windows'])

        # Step 5: Remove NaN stretches longer than seq_length (untrainable dead weight)
        clean_df, n_dropped = drop_long_flow_nan_stretches(clean_df, seq_length=config['seq_length'])

        # Compute per-basin flow mean/std from the training periods only (np.nanmean/nanstd ignore
        # NaN entries). These are fixed per-basin normalization constants used to z-score the flow
        # target in dataset.py, so they must reflect train-period variability, not the full
        # train+val+test series. Both are drawn from the same slice/fallback branch so they always
        # describe the same underlying sample.
        train_slice = pd.concat([
            clean_df.loc[p.get('start_date'):p['end_date']] for p in config['train_periods']
        ])
        flow_mean = float(np.nanmean(train_slice['Flow_m3_sec'].values))
        flow_std = float(np.nanstd(train_slice['Flow_m3_sec'].values))
        if not np.isfinite(flow_std) or flow_std == 0.0:
            flow_mean = float(np.nanmean(clean_df['Flow_m3_sec'].values))
            flow_std = float(np.nanstd(clean_df['Flow_m3_sec'].values))

        # Compute per-basin mean/std for hourly_precipitation and every configured
        # cumulative-rain feature, each from the same training-period slice. Unlike flow,
        # these features have no downstream use for raw values, so the z-score is baked
        # directly into the saved series below rather than applied on-the-fly in dataset.py.
        rain_feature_names = ['hourly_precipitation'] + [w['name'] for w in config['cumulative_rain_windows']]
        rain_norm_stats = {}
        for feature_name in rain_feature_names:
            feature_mean = float(np.nanmean(train_slice[feature_name].values))
            feature_std = float(np.nanstd(train_slice[feature_name].values))
            if not np.isfinite(feature_std) or feature_std == 0.0:
                feature_mean = float(np.nanmean(clean_df[feature_name].values))
                feature_std = float(np.nanstd(clean_df[feature_name].values))

            # Bake the normalization into the series that gets saved to disk
            clean_df[feature_name] = (clean_df[feature_name] - feature_mean) / feature_std

            rain_norm_stats[f'{feature_name}_mean'] = feature_mean
            rain_norm_stats[f'{feature_name}_std'] = feature_std

        # Record data for the summary report
        availability_records.append({
            'gauge_id': file_name.replace('.csv', ''),
            'availability_pct': flow_available,
            'combined_availability_pct': combined_available,
            'excluded': False,
            'flow_mean': flow_mean,
            'flow_std': flow_std,
            **rain_norm_stats,
        })

        # Save the processed CSV file
        output_path = os.path.join(output_dir, file_name)
        clean_df.to_csv(output_path, index_label='date')

        tqdm.write(f"Done {file_name}: Processed from {basin_start_date} to {END_DATE}. {flow_available:.2f}% flow data. Dropped {n_dropped} rows (long NaN stretches).")

    # Create and save the final availability report
    report_df = pd.DataFrame(availability_records)
    report_df.to_csv(report_path, index=False)
    print(f"\nSummary report saved to: {report_path}")

if __name__ == "__main__":
    CONFIG_PATH = "configs/config.yml"
    yaml_config = load_config(CONFIG_PATH)
    
    process_dynamic_data(yaml_config)