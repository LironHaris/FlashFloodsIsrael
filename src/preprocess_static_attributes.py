import pandas as pd
import numpy as np
import os

def load_and_merge(caravan_path="data/raw/attributes_caravan_il.csv", hydroatlas_path="data/raw/attributes_hydroatlas_il.csv"):
    """
    Load source CSV files from the raw directory and perform an inner merge on 'gauge_id'.
    """
    if not os.path.exists(caravan_path) or not os.path.exists(hydroatlas_path):
        raise FileNotFoundError("Source files missing in data/raw/")
    
    df1 = pd.read_csv(caravan_path)
    df2 = pd.read_csv(hydroatlas_path)
    return pd.merge(df1, df2, on='gauge_id', how='inner')

def clean_missing_features(df):
    """
    Remove all columns containing any missing values to ensure compatibility with NeuralHydrology.
    """
    ids = df[['gauge_id']]
    features = df.drop(columns=['gauge_id'])
    clean_features = features.dropna(axis=1, how='any')
    
    return pd.concat([ids, clean_features], axis=1)

def main():
    """
    Execute the full preprocessing pipeline: merge, clean, and save final attributes and statistics.
    """
    RAW_DIR = os.path.join('data', 'raw')
    PROCESSED_DIR = os.path.join('data', 'processed')
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    input_caravan = os.path.join(RAW_DIR, 'attributes_caravan_il.csv')
    input_hydroatlas = os.path.join(RAW_DIR, 'attributes_hydroatlas_il.csv')
    output_nh = os.path.join(PROCESSED_DIR, 'static_attributes_nh.csv')
    output_stats = os.path.join(PROCESSED_DIR, 'feature_statistics.csv')

    merged = load_and_merge(input_caravan, input_hydroatlas)
    final_df = clean_missing_features(merged)
    
    numeric_df = final_df.select_dtypes(include=[np.number])
    stats = pd.DataFrame({
        'feature': numeric_df.columns,
        'mean': numeric_df.mean(),
        'std': numeric_df.std()
    })
    
    final_df.to_csv(output_nh, index=False)
    stats.to_csv(output_stats, index=False)

if __name__ == "__main__":
    main()