import pandas as pd
import os
import yaml

def create_basin_splits(basin_records, output_dir, min_pct):
    """
    basin_records: list of dicts with 'gauge_id', 'combined_availability_pct_train',
    'combined_availability_pct_val', 'combined_availability_pct_test'. A basin is
    written into israel_train.txt/israel_val.txt/israel_test.txt independently based
    on whether its combined availability for that specific split meets min_pct - it
    may appear in multiple lists (e.g. train+test but not val), or none. No shuffling,
    no ratios: membership is entirely determined by measured data availability.
    """
    train_ids, val_ids, test_ids = [], [], []
    for r in basin_records:
        gid = r['gauge_id']
        if r['combined_availability_pct_train'] >= min_pct:
            train_ids.append(gid)
        if r['combined_availability_pct_val'] >= min_pct:
            val_ids.append(gid)
        if r['combined_availability_pct_test'] >= min_pct:
            test_ids.append(gid)

    all_ids = sorted(set(train_ids) | set(val_ids) | set(test_ids))

    splits = {
        'israel_train.txt': train_ids,
        'israel_val.txt': val_ids,
        'israel_test.txt': test_ids,
        'all_basins.txt': all_ids,
    }
    for filename, ids in splits.items():
        with open(os.path.join(output_dir, filename), 'w', newline='\n') as f:
            for s_id in ids:
                f.write(f"{s_id}\n")

    print(f"Splits created in {output_dir}")
    print(f"Total usable: {len(all_ids)} (Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)})")


if __name__ == "__main__":
    with open('configs/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    report_df = pd.read_csv(config['availability_report_file'])
    records = report_df.loc[~report_df['excluded'],
                            ['gauge_id', 'combined_availability_pct_train',
                             'combined_availability_pct_val', 'combined_availability_pct_test']
                            ].to_dict('records')
    create_basin_splits(records, os.path.dirname(config['train_basin_file']),
                         config['min_combined_availability_pct'])
