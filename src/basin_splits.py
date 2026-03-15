import pandas as pd
import numpy as np
import os
import yaml

def create_basin_splits(input_path, output_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Partition basin IDs into training, validation, test, and master lists using seed from config.
    """
    with open('configs/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    seed_value = config.get('seed', 42)
    
    df = pd.read_csv(input_path)
    stations = df['gauge_id'].unique().tolist()
    
    np.random.seed(seed_value)
    np.random.shuffle(stations)
    
    n = len(stations)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    splits = {
        'israel_train.txt': stations[:train_end],
        'israel_val.txt': stations[train_end:val_end],
        'israel_test.txt': stations[val_end:],
        'all_basins.txt': stations  
    }
    
    for filename, ids in splits.items():
        with open(os.path.join(output_dir, filename), 'w') as f:
            for s_id in ids:
                f.write(f"{s_id}\n")
    
    print(f"Splits created in {output_dir}")
    print(f"Total: {len(stations)} (Train: {len(splits['israel_train.txt'])}, Val: {len(splits['israel_val.txt'])}, Test: {len(splits['israel_test.txt'])})")

if __name__ == "__main__":
    INPUT_CSV = os.path.join('data', 'processed', 'static', 'static_attributes_nh.csv')
    BASIN_DIR = os.path.join('data', 'basin_lists')
    create_basin_splits(INPUT_CSV, BASIN_DIR)