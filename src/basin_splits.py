import pandas as pd
import numpy as np
import os
import yaml
def create_basin_splits(csv_path="data/attributes_caravan_il.csv", output_dir="configs/basin_lists/", train_ratio=0.7, val_ratio=0.15):
    """
    Reads the static attributes file containing station IDs and randomly partitions them 
    into training, validation, and test sets based on the provided ratios. 
    The resulting lists are saved as three separate text files in the specified destination directory.

    Args:
        csv_path (str): Path to the static attributes CSV file (e.g., 'data/attributes_caravan_il.csv').
        output_dir (str): Directory where the 'train.txt', 'val.txt', and 'test.txt' files will be saved.
        train_ratio (float): Proportion of stations to use for training (default is 0.7).
        val_ratio (float): Proportion of stations to use for validation (default is 0.15).
    """
    with open('configs/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    seed_value = config.get('seed', 42)
    df = pd.read_csv(csv_path)
    stations = df['gauge_id'].unique().tolist()
    
    np.random.seed(seed_value)
    np.random.shuffle(stations)
    
    n = len(stations)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    splits = {
        'israel_train.txt': stations[:train_end],
        'israel_val.txt': stations[train_end:val_end],
        'israel_test.txt': stations[val_end:]
    }
    
    os.makedirs(output_dir, exist_ok=True)
    for filename, ids in splits.items():
        with open(os.path.join(output_dir, filename), 'w') as f:
            for s_id in ids:
                f.write(f"{s_id}\n")
    
    print(f"Done! Created splits in {output_dir}")
    print(f"Train: {len(splits['israel_train.txt'])}, Val: {len(splits['israel_val.txt'])}, Test: {len(splits['israel_test.txt'])}")

if __name__ == "__main__":
    create_basin_splits('data/attributes_caravan_il.csv', 'configs/basin_lists/')