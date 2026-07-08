import pandas as pd
import numpy as np
import os
import yaml

def create_basin_splits(basin_ids, output_dir, seed, train_ratio=0.7, val_ratio=0.15):
    """
    Partition the given basin IDs into training, validation, test, and master
    lists (seeded shuffle), writing israel_train.txt/israel_val.txt/
    israel_test.txt/all_basins.txt into output_dir.
    """
    stations = list(basin_ids)

    np.random.seed(seed)
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
        with open(os.path.join(output_dir, filename), 'w', newline='\n') as f:
            for s_id in ids:
                f.write(f"{s_id}\n")
    
    print(f"Splits created in {output_dir}")
    print(f"Total: {len(stations)} (Train: {len(splits['israel_train.txt'])}, Val: {len(splits['israel_val.txt'])}, Test: {len(splits['israel_test.txt'])})")

if __name__ == "__main__":
    with open('configs/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    report_df = pd.read_csv(config['availability_report_file'])
    basin_ids = report_df.loc[~report_df['excluded'], 'gauge_id'].tolist()
    create_basin_splits(basin_ids, os.path.dirname(config['train_basin_file']), config.get('seed', 42))