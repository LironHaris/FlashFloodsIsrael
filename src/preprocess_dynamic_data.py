import pandas as pd
import os
from tqdm import tqdm

def process_dynamic_data(input_dir, output_dir, report_path):
    """
    Resample gauge data, align hydraulic years, and save an availability report with progress tracking.
    """
    START_DATE = '2008-10-01 08:00:00'
    END_DATE = '2023-10-01 07:00:00'
    
    availability_records = []
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    for file_name in tqdm(csv_files, desc="Processing Gauges", unit="file"):
        df = pd.read_csv(os.path.join(input_dir, file_name))
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        resampled_df = df.resample('h', closed='left', label='left').agg({
            'Flow_m3_sec': 'mean',
            'mean_rain': 'sum'
        })
        
        all_hours = pd.date_range(start=START_DATE, end=END_DATE, freq='h')
        processed_df = resampled_df.reindex(all_hours)
        
        flow_available = processed_df['Flow_m3_sec'].notna().mean() * 100
        availability_records.append({
            'gauge_id': file_name.replace('.csv', ''), 
            'availability_pct': flow_available
        })
        
        processed_df['mean_rain'] = processed_df['mean_rain'].fillna(0)
        
        output_path = os.path.join(output_dir, file_name)
        processed_df.to_csv(output_path, index_label='date')
        
        tqdm.write(f"Done {file_name}: {flow_available:.2f}% flow data.")

    report_df = pd.DataFrame(availability_records)
    report_df.to_csv(report_path, index=False)
    print(f"\nSummary report saved to: {report_path}")

if __name__ == "__main__":
    RAW_DIR = os.path.join('data', 'raw', 'dynamic')
    OUT_DIR = os.path.join('data', 'processed', 'timeseries')
    REPORT_FILE = os.path.join('data', 'processed', 'data_availability_report.csv')
    
    process_dynamic_data(RAW_DIR, OUT_DIR, REPORT_FILE)