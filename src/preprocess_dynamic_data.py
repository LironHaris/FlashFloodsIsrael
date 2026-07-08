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
   are aggregated via a temporal mean. Both aggregations require every raw sub-hourly reading in
   the hour to be present and non-NaN; if any reading is missing, the hour's result is NaN rather
   than a value silently computed from incomplete data. Bounds are defined using 'left' inclusive
   indexing.

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

import basin_splits as bs

def load_config(yaml_path):
    """Load the YAML configuration file."""
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# --------------------------------------------------------------------------------
# 1. Resampling Function
# --------------------------------------------------------------------------------
def resample_to_hourly(df):
    """
    Resample raw data to hourly resolution. An hour's flow or rain value is only
    computed when every raw sub-hourly reading in that hour is present and non-NaN;
    otherwise the result is NaN. This avoids pandas' default aggregation behavior,
    which would otherwise fabricate a plausible-looking value from incomplete data:
    sum() treats an all-NaN/empty group as 0.0, and mean() with its default
    skipna=True silently averages over just the present sub-readings.
    """
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    def strict_sum(s):
        return np.nan if len(s) == 0 or s.isna().any() else s.sum()

    def strict_mean(s):
        return np.nan if len(s) == 0 or s.isna().any() else s.mean()

    # Standardized to use 'hourly_precipitation' as the core dynamic feature name
    resampled = df.resample('h', closed='left', label='left').agg({
        'Flow_m3_sec': strict_mean,
        'mean_rain': strict_sum
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

def combined_availability_for_periods(aligned_df, periods):
    """
    Same as analyze_combined_availability, but restricted to the union of the
    given (start_date_or_None, end_date) slices of aligned_df. Used to check
    combined availability separately per split (train/val/test), since a
    basin can be well-observed in one split and sparse in another.
    """
    slices = [aligned_df.loc[start:end] for start, end in periods]
    combined = pd.concat(slices)
    return analyze_combined_availability(combined) if not combined.empty else 0.0

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
        
        # Step 3: Check original data quality (informational only, used in the
        # per-basin console log below - not written to the availability report).
        flow_available = analyze_data_quality(aligned_df)

        # Step 3b: Exclude basins with no viable split at all - if a basin fails the
        # combined (flow AND rain) availability threshold in EVERY one of train/val/test,
        # there's nothing useful to do with it. A basin passing in just one or two splits
        # still gets processed here; basin_splits.py decides per-split membership from
        # these same three values, so e.g. good-train/good-test/no-val data is still used
        # for train and test, just left out of israel_val.txt.
        # Checked before any further processing so excluded basins skip the rest of
        # the pipeline entirely and never get a processed timeseries CSV written -
        # dataset.py already treats a missing CSV as "basin not available."
        train_bounds = [(p.get('start_date'), p['end_date']) for p in config['train_periods']]
        combined_train = combined_availability_for_periods(aligned_df, train_bounds)
        combined_val = combined_availability_for_periods(
            aligned_df, [(config['validation_start_date'], config['validation_end_date'])])
        combined_test = combined_availability_for_periods(
            aligned_df, [(config['test_start_date'], config['test_end_date'])])

        min_pct = config['min_combined_availability_pct']
        if combined_train < min_pct and combined_val < min_pct and combined_test < min_pct:
            availability_records.append({
                'gauge_id': file_name.replace('.csv', ''),
                'combined_availability_pct_train': combined_train,
                'combined_availability_pct_val': combined_val,
                'combined_availability_pct_test': combined_test,
                'excluded': True,
            })
            tqdm.write(
                f"Skipped {file_name}: combined availability train={combined_train:.2f}% "
                f"val={combined_val:.2f}% test={combined_test:.2f}%, below the {min_pct}% threshold in every split."
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
        train_slice = pd.concat([clean_df.loc[start:end] for start, end in train_bounds])
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
            'combined_availability_pct_train': combined_train,
            'combined_availability_pct_val': combined_val,
            'combined_availability_pct_test': combined_test,
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

    # Regenerate the basin split lists from exactly the basins that passed the
    # availability gate this run, so israel_train/val/test.txt can never
    # reference a basin with no processed CSV on disk. Membership per split is
    # decided inside create_basin_splits from each basin's own per-split
    # combined_availability_pct - a basin may appear in multiple lists, or none.
    included_records = [r for r in availability_records if not r['excluded']]
    basin_lists_dir = os.path.dirname(config['train_basin_file'])
    bs.create_basin_splits(included_records, basin_lists_dir, config['min_combined_availability_pct'])

if __name__ == "__main__":
    CONFIG_PATH = "configs/config.yml"
    yaml_config = load_config(CONFIG_PATH)
    
    process_dynamic_data(yaml_config)