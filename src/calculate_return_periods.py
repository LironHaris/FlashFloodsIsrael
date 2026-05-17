import pandas as pd
import numpy as np
import os
import yaml
from scipy.stats import genextreme
from tqdm import tqdm
import warnings

# Ignore scipy curve-fitting warnings for small datasets
warnings.filterwarnings("ignore")

def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def get_hydrological_year(date_index, start_month=10):
    """
    Assign a hydrological year based on the starting month.
    If the start month is October (10), Oct 2008 belongs to Hydrological Year 2009.
    """
    return date_index.year + (date_index.month >= start_month).astype(int)

# --------------------------------------------------------------------------------
# 1. Core Mathematical Preprocessing Functions
# --------------------------------------------------------------------------------
def calculate_empirical_return_periods(series):
    """
    Calculate empirical return periods using the Weibull plotting position formula (P = m / (n+1)).
    Assumes data has already been sorted and cleaned.
    """
    sorted_series = series.sort_values(ascending=False)
    n = len(sorted_series)
    ranks = np.arange(1, n + 1)
    
    prob_empirical = ranks / (n + 1)
    t_empirical = 1 / prob_empirical
    
    empirical_df = pd.DataFrame({
        'Value': sorted_series.values,
        'Rank': ranks,
        'Exceedance_Probability': prob_empirical,
        'Empirical_Return_Period_Years': t_empirical
    }, index=sorted_series.index)
    
    return empirical_df

def calculate_theoretical_gev_return_periods(series, target_return_periods):
    """
    Fit a Generalized Extreme Value (GEV) distribution to the series and
    compute theoretical values for target return periods.
    """
    c, loc, scale = genextreme.fit(series.values)
    
    theoretical_levels = []
    for t in target_return_periods:
        p_exceedance = 1 / t
        theoretical_val = genextreme.isf(p_exceedance, c, loc=loc, scale=scale)
        theoretical_levels.append(max(0, theoretical_val))
        
    theoretical_df = pd.DataFrame({
        'Return_Period_Years': target_return_periods,
        'Theoretical_Value': theoretical_levels
    })
    
    return theoretical_df

def fit_gev_and_calculate_returns(series, target_return_periods, min_years):
    """
    Coordinates the calculation of empirical and theoretical GEV return periods
    for an Annual Maxima series. Safeguarded by a data sufficiency check.
    """
    series = series.dropna()
    
    if len(series) < min_years:
        return None, None
        
    empirical_df = calculate_empirical_return_periods(series)
    theoretical_df = calculate_theoretical_gev_return_periods(series, target_return_periods)
    
    return empirical_df, theoretical_df

# --------------------------------------------------------------------------------
# 2. Basin Preprocessing Sub-Functions
# --------------------------------------------------------------------------------
def load_and_prepare_basin_data(file_path, hydro_start_month):
    """
    Load a basin time series CSV file, format the date index, and 
    append the calculated Hydrological Year.
    """
    gauge_id = os.path.basename(file_path).replace('.csv', '')
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['Hydro_Year'] = get_hydrological_year(df.index, start_month=hydro_start_month)
    return df, gauge_id

def extract_annual_maxima(df):
    """
    Perform duration aggregations to extract hourly and daily annual maxima 
    series for both flow and precipitation.
    """
    # --- A. Hourly Maxima ---
    hourly_am = df.groupby('Hydro_Year').max()
    
    # --- B. Daily Maxima (Flow is averaged daily, rain is summed daily) ---
    daily_resampled = df.resample('D').agg({
        'Flow_m3_sec': 'mean',
        'mean_rain': 'sum',
        'Hydro_Year': 'first'
    })
    daily_am = daily_resampled.groupby('Hydro_Year').max()
    
    # Package into a dictionary for clean looping
    return {
        'Hourly_Flow': hourly_am['Flow_m3_sec'],
        'Hourly_Rain': hourly_am['mean_rain'],
        'Daily_Flow': daily_am['Flow_m3_sec'],
        'Daily_Rain': daily_am['mean_rain']
    }

def save_basin_results(annual_maxima_dict, basin_out_dir, target_periods, min_years):
    """
    Iterates through extracted maxima series, passes them to the statistical engine,
    and writes results to disk if thresholds are met.
    """
    os.makedirs(basin_out_dir, exist_ok=True)
    valid_computations = 0
    
    for name, series in annual_maxima_dict.items():
        emp_df, theo_df = fit_gev_and_calculate_returns(series, target_periods, min_years)
        
        if emp_df is not None:
            emp_df.to_csv(os.path.join(basin_out_dir, f'{name}_empirical_AM.csv'), index_label='Hydro_Year')
            theo_df.to_csv(os.path.join(basin_out_dir, f'{name}_theoretical_GEV.csv'), index=False)
            valid_computations += 1
            
    return valid_computations > 0

# --------------------------------------------------------------------------------
# 3. Main Functions
# --------------------------------------------------------------------------------
def process_basin_extremes(file_path, output_dir, config):
    """
    Controls the entire extreme value analysis workflow for a single basin file
    by stitching together modular sub-functions.
    """
    # Extract constants from configuration
    hydro_start = config.get('hydro_year_start_month', 10)
    target_periods = config['return_periods_years']
    min_years = config.get('min_years_for_gev', 5)
    
    # Step 1: Load data and tag hydrological years
    df, gauge_id = load_and_prepare_basin_data(file_path, hydro_start)
    
    # Step 2: Resample and extract the 4 extreme series
    annual_maxima_dict = extract_annual_maxima(df)
    
    # Step 3: Fit statistical distributions and export data
    basin_out_dir = os.path.join(output_dir, gauge_id)
    is_successful = save_basin_results(annual_maxima_dict, basin_out_dir, target_periods, min_years)
    
    return is_successful

def main(config):
    input_dir = config['processed_timeseries_dir']
    output_dir = config['return_periods_output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    success_count = 0
    for file_name in tqdm(csv_files, desc="Calculating Return Periods", unit="basin"):
        file_path = os.path.join(input_dir, file_name)
        if process_basin_extremes(file_path, output_dir, config):
            success_count += 1
            
    print(f"\n[INFO] Return periods calculated successfully for {success_count} basins.")
    print(f"[INFO] Results saved to: {output_dir}")

if __name__ == "__main__":
    CONFIG_PATH = "configs/config.yml"
    yaml_config = load_config(CONFIG_PATH)
    main(yaml_config)