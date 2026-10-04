"""
Module: extend_static_attributes.py
Description: Adds basins that are missing from an existing normalized static-attributes file, using the
             SAME normalization constants (feature_statistics.csv: z = (raw - mean) / std), so the added rows
             are on exactly the scale of the existing ones. Before writing, it recomputes one basin that is
             already in the normalized file and refuses to continue unless it reproduces the stored values.

Config keys:
  static_source_dir                 folder with static_attributes_nh.csv, feature_statistics.csv,
                                    static_attributes_normalized.csv (e.g. Liron's processed/static)
  static_extend_basins              list of gauge ids to add
  normalized_static_attributes_file output path (the extended file the runs read)

    FLOODS_CONFIG=configs/SRradar_B_mean_mask_L0.yml python src/extend_static_attributes.py
"""

import os

import numpy as np
import pandas as pd
import yaml


def main():
    cfg = yaml.safe_load(open(os.environ.get('FLOODS_CONFIG', 'configs/config.yml'), encoding='utf-8'))
    src = cfg['static_source_dir']
    out = cfg['normalized_static_attributes_file']
    add_ids = [str(b) for b in cfg.get('static_extend_basins', [])]

    norm = pd.read_csv(os.path.join(src, 'static_attributes_normalized.csv'))
    raw = pd.read_csv(os.path.join(src, 'static_attributes_nh.csv'))
    stats = pd.read_csv(os.path.join(src, 'feature_statistics.csv')).set_index('feature')
    for df in (norm, raw):
        df['gauge_id'] = df['gauge_id'].astype(str)
    raw = raw.set_index('gauge_id')
    cols = [c for c in norm.columns if c != 'gauge_id' and c in stats.index and c in raw.columns]

    def z(gid):
        return (raw.loc[gid, cols] - stats.loc[cols, 'mean']) / stats.loc[cols, 'std']

    ref = norm['gauge_id'].iloc[0]
    diff = float(np.nanmax(np.abs(z(ref).values.astype(float) - norm.set_index('gauge_id').loc[ref, cols].values.astype(float))))
    print(f"[check] recomputed {ref}: max difference vs stored normalized values = {diff:.3g}")
    if diff > 1e-6:
        raise SystemExit("[ERROR] normalization constants do not reproduce the stored file - not writing anything")

    add_ids = [b for b in add_ids if b not in set(norm['gauge_id'])]
    added = pd.DataFrame([{'gauge_id': b, **z(b).to_dict()} for b in add_ids])
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    pd.concat([norm, added], ignore_index=True).to_csv(out, index=False)
    print(f"[INFO] wrote {out}: {len(norm)} existing + {len(added)} added basins {add_ids}")


if __name__ == '__main__':
    main()
