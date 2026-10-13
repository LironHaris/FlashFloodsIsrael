"""
Module: flow_distribution.py
Description: Distribution of observed flow over basin-timestep pairs, restricted
             to the hours the water authority labeled with a selected Flow_type
             (data_type_mask.py's masks: 'flow', and 'flow' OR 'normal').

             Input: the ALL-DATA preprocessed timeseries (configs/preprocess_all_data.yml -
             no test split, the full preprocess_dynamic_data.py pipeline incl. dry-year
             and nan_flow_days screening, written to its own *_all_data locations so no
             training data is touched).

             Per basin, Flow_m3_sec is z-scored with that basin's mean/std over ALL
             its non-NaN rows (normalization.compute_basin_stats with train_periods =
             the whole record) - the model's per-basin target normalization, computed
             here in memory only, never written next to any training run.

             Kept pairs: in-mask hours with non-NaN raw flow > 0 (zero flow excluded,
             x range (0, max]); the same pairs are used for raw and normalized values.
             Per mask x {raw m3/s, normalized}:
               - PDF: histogram (y = number of pairs per flow bin), 200 equal-width bins
               - CDF: exact cumulative count (y = number of pairs with flow <= x)
             each with a linear-y and a log-y figure, plus a histogram CSV and a
             summary.csv; everything is also logged to one W&B run
             (flow_distribution_all_data) when use_wandb is set.

Usage:
  python src/flow_distribution.py --config configs/preprocess_all_data.yml
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')  # file-only backend: never needs an X display (cluster nodes)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from data_type_mask import UNIFIED_FILE, load_unified, build_flow_intervals, flow_mask
from normalization import compute_basin_stats

MASKS = {'flow': {'flow'}, 'flow_normal': {'flow', 'normal'}}
N_BINS = 200
MAX_CDF_POINTS = 5000  # the CDF is exact; only the drawn line is subsampled
TARGET = 'Flow_m3_sec'


def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def collect_pairs(config, by_station):
    """
    Returns ({mask_name: (raw_values, normalized_values)}, {mask_name: counts dict})
    over every processed basin CSV. Kept pairs: in-mask hours with raw flow > 0.
    """
    timeseries_dir = config['processed_timeseries_dir']
    stats_config = {
        'processed_timeseries_dir': timeseries_dir,
        'dynamic_inputs': [],
        'target_variables': [TARGET],
        # whole record: each basin's normalization comes from all its own rows
        'train_periods': [{'start_date': None, 'end_date': config['record_end_date']}],
    }
    raw = {m: [] for m in MASKS}
    norm = {m: [] for m in MASKS}
    counts = {m: {'basins': 0, 'in_mask_steps': 0, 'nan_flow_steps': 0, 'zero_flow_steps': 0, 'kept_pairs': 0}
              for m in MASKS}

    basin_files = sorted(f for f in os.listdir(timeseries_dir) if f.endswith('.csv'))
    for file_name in basin_files:
        basin_id = file_name[:-4]
        station_df = by_station.get(basin_id.replace('il_', '', 1))
        if station_df is None:
            print(f"  [Warning] {basin_id}: no records in unified hydrographs - skipped.")
            continue
        df = pd.read_csv(os.path.join(timeseries_dir, file_name), usecols=['date', TARGET])
        hours = pd.to_datetime(df['date']).to_numpy()
        flow = df[TARGET].to_numpy(dtype=np.float64)
        st = compute_basin_stats(stats_config, basin_id)[TARGET]

        for mask_name, flow_types in MASKS.items():
            starts, ends = build_flow_intervals(station_df, flow_types)
            in_mask = flow_mask(hours, starts, ends)
            if not in_mask.any():
                continue
            values = flow[in_mask]
            is_nan = np.isnan(values)
            keep = ~is_nan & (values > 0)
            c = counts[mask_name]
            c['basins'] += 1
            c['in_mask_steps'] += int(in_mask.sum())
            c['nan_flow_steps'] += int(is_nan.sum())
            c['zero_flow_steps'] += int((values[~is_nan] == 0).sum())
            c['kept_pairs'] += int(keep.sum())
            raw[mask_name].append(values[keep])
            norm[mask_name].append((values[keep] - st['mean']) / st['std'])
        print(f"  {basin_id}: " + ", ".join(f"{m}={len(raw[m][-1]) if raw[m] else 0}" for m in MASKS))

    pairs = {m: (np.concatenate(raw[m]) if raw[m] else np.array([]),
                 np.concatenate(norm[m]) if norm[m] else np.array([])) for m in MASKS}
    return pairs, counts


def histogram_table(values, lower):
    """200 equal-width bins over [lower, max]; returns DataFrame with counts + cumulative counts."""
    counts, edges = np.histogram(values, bins=N_BINS, range=(lower, float(values.max())))
    return pd.DataFrame({'bin_left': edges[:-1], 'bin_right': edges[1:], 'count': counts,
                         'cumulative_count': np.cumsum(counts)})


def plot_pdf(hist_df, title, xlabel, log_y):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    widths = hist_df['bin_right'] - hist_df['bin_left']
    ax.bar(hist_df['bin_left'], hist_df['count'], width=widths, align='edge', color='#2f6f9f', edgecolor='none')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Number of basin-timestep pairs per bin')
    if log_y:
        ax.set_yscale('log')
    ax.set_title(title + (' (log y)' if log_y else ''), fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    fig.tight_layout()
    return fig


def plot_cdf(values, title, xlabel, log_y):
    sorted_values = np.sort(values)
    cum = np.arange(1, len(sorted_values) + 1)
    idx = np.unique(np.linspace(0, len(sorted_values) - 1, MAX_CDF_POINTS).astype(int))
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.step(sorted_values[idx], cum[idx], where='post', color='#b5462f', linewidth=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Number of basin-timestep pairs with flow <= x')
    if log_y:
        ax.set_yscale('log')
    ax.set_title(title + (' (log y)' if log_y else ''), fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    fig.tight_layout()
    return fig


def main(config):
    output_dir = config['flow_distribution_output_dir']
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading {UNIFIED_FILE} ...")
    unified = load_unified(UNIFIED_FILE)
    by_station = {station_id: df for station_id, df in unified.groupby('Station_ID')}

    print(f"[INFO] Collecting basin-timestep pairs from {config['processed_timeseries_dir']} ...")
    pairs, counts = collect_pairs(config, by_station)

    use_wandb = config.get('use_wandb', False)
    if use_wandb:
        import wandb
        if config.get('wandb_api_key'):
            wandb.login(key=config['wandb_api_key'])
        wandb.init(project=config.get('wandb_project', 'flash-floods-israel'),
                   name='flow_distribution_all_data', job_type='flow_distribution',
                   config={'masks': {m: sorted(t) for m, t in MASKS.items()}, 'n_bins': N_BINS,
                           'processed_timeseries_dir': config['processed_timeseries_dir'],
                           'zero_flow_excluded': True})
    log = {}

    summary_rows = []
    for mask_name, (raw_values, norm_values) in pairs.items():
        c = counts[mask_name]
        assert c['kept_pairs'] + c['zero_flow_steps'] + c['nan_flow_steps'] == c['in_mask_steps']
        row = {'mask': mask_name, 'flow_types': '+'.join(sorted(MASKS[mask_name])), **c}
        if len(raw_values) == 0:
            print(f"[WARNING] mask '{mask_name}': no pairs with flow > 0 - nothing to plot.")
            summary_rows.append(row)
            continue
        row.update(raw_min=raw_values.min(), raw_median=float(np.median(raw_values)), raw_max=raw_values.max(),
                   normalized_min=norm_values.min(), normalized_median=float(np.median(norm_values)),
                   normalized_max=norm_values.max())
        summary_rows.append(row)

        for kind, values, lower, xlabel in (
                ('raw', raw_values, 0.0, 'Observed flow [m3/s]'),
                ('normalized', norm_values, float(norm_values.min()), 'Normalized flow (per-basin z-score, all data)')):
            hist_df = histogram_table(values, lower)
            assert hist_df['cumulative_count'].iloc[-1] == len(values)
            hist_df.to_csv(os.path.join(output_dir, f"{mask_name}_{kind}_hist.csv"), index=False)
            title = (f"{kind.capitalize()} flow - Flow_type {row['flow_types']} - "
                     f"{len(values):,} pairs (flow > 0; {c['zero_flow_steps']:,} zero-flow steps excluded)")
            if use_wandb:
                log[f"{mask_name}/{kind}/hist_table"] = wandb.Table(dataframe=hist_df)
            for plot_name, fig in (('pdf', plot_pdf(hist_df, 'PDF: ' + title, xlabel, False)),
                                   ('pdf_logy', plot_pdf(hist_df, 'PDF: ' + title, xlabel, True)),
                                   ('cdf', plot_cdf(values, 'CDF: ' + title, xlabel, False)),
                                   ('cdf_logy', plot_cdf(values, 'CDF: ' + title, xlabel, True))):
                fig.savefig(os.path.join(output_dir, f"{mask_name}_{kind}_{plot_name}.png"))
                if use_wandb:
                    log[f"{mask_name}/{kind}/{plot_name}"] = wandb.Image(fig)
                plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, 'summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))

    if use_wandb:
        log['summary'] = wandb.Table(dataframe=summary_df)
        wandb.log(log)
        wandb.finish()
    print(f"\n[INFO] Flow distribution outputs in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF/CDF of observed flow over flow / flow+normal timesteps.")
    parser.add_argument("--config", type=str, default="configs/preprocess_all_data.yml",
                        help="All-data preprocessing config (default configs/preprocess_all_data.yml).")
    args = parser.parse_args()
    main(load_config(args.config))
