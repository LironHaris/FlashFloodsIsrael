"""
Module: flow_distribution.py
Description: Distribution of observed flow over basin-timestep pairs, split by the
             Flow_type the water authority labeled each hour with (data_type_mask.py's
             intervals): 'flow' (blue), 'normal' (green) and everything else (red) -
             any hour labeled neither, including hours with no label at all.

             Input: the ALL-DATA preprocessed timeseries (configs/preprocess_all_data.yml -
             no test split, the full preprocess_dynamic_data.py pipeline incl. dry-year
             and nan_flow_days screening, written to its own *_all_data locations so no
             training data is touched).

             Per basin, Flow_m3_sec is z-scored with that basin's mean/std over ALL
             its non-NaN rows (normalization.compute_basin_stats with train_periods =
             the whole record) - the model's per-basin target normalization, computed
             here in memory only, never written next to any training run.

             Kept pairs: non-NaN raw flow > 0 (zero flow excluded). Two figures, log-y,
             the three categories overlaid:
               - normalized_flow_hist.png: y = number of pairs per normalized-flow bin
                 (200 equal-width bins shared by all categories)
               - month_hist.png: y = number of pairs per calendar month, Oct..Sep
             plus their CSVs and a summary.csv; everything is also logged to one W&B run
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

# labeled categories; 'other' = every hour in neither
LABELED = {'flow': {'flow'}, 'normal': {'normal'}}
CATEGORIES = ['flow', 'normal', 'other']
COLORS = {'flow': 'tab:blue', 'normal': 'tab:green', 'other': 'tab:red'}
N_BINS = 200
HYDRO_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
MONTH_LABELS = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
TARGET = 'Flow_m3_sec'


def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def collect_pairs(config, by_station):
    """
    Returns ({category: (raw_values, normalized_values, months)}, {category: counts dict})
    over every processed basin CSV. Kept pairs: raw flow > 0.
    """
    timeseries_dir = config['processed_timeseries_dir']
    stats_config = {
        'processed_timeseries_dir': timeseries_dir,
        'dynamic_inputs': [],
        'target_variables': [TARGET],
        # whole record: each basin's normalization comes from all its own rows
        'train_periods': [{'start_date': None, 'end_date': config['record_end_date']}],
    }
    raw = {c: [] for c in CATEGORIES}
    norm = {c: [] for c in CATEGORIES}
    months = {c: [] for c in CATEGORIES}
    counts = {c: {'basins': 0, 'in_category_steps': 0, 'nan_flow_steps': 0, 'zero_flow_steps': 0,
                  'kept_pairs': 0} for c in CATEGORIES}

    basin_files = sorted(f for f in os.listdir(timeseries_dir) if f.endswith('.csv'))
    for file_name in basin_files:
        basin_id = file_name[:-4]
        station_df = by_station.get(basin_id.replace('il_', '', 1))
        if station_df is None:
            print(f"  [Warning] {basin_id}: no records in unified hydrographs - skipped.")
            continue
        df = pd.read_csv(os.path.join(timeseries_dir, file_name), usecols=['date', TARGET])
        hours = pd.to_datetime(df['date'])
        hour_values = hours.to_numpy()
        month_of_hour = hours.dt.month.to_numpy()
        flow = df[TARGET].to_numpy(dtype=np.float64)
        st = compute_basin_stats(stats_config, basin_id)[TARGET]

        masks = {c: flow_mask(hour_values, *build_flow_intervals(station_df, types)) for c, types in LABELED.items()}
        masks['other'] = ~(masks['flow'] | masks['normal'])

        for category in CATEGORIES:
            in_category = masks[category]
            if not in_category.any():
                continue
            values = flow[in_category]
            is_nan = np.isnan(values)
            keep = ~is_nan & (values > 0)
            c = counts[category]
            c['basins'] += 1
            c['in_category_steps'] += int(in_category.sum())
            c['nan_flow_steps'] += int(is_nan.sum())
            c['zero_flow_steps'] += int((values[~is_nan] == 0).sum())
            c['kept_pairs'] += int(keep.sum())
            raw[category].append(values[keep])
            norm[category].append((values[keep] - st['mean']) / st['std'])
            months[category].append(month_of_hour[in_category][keep])
        print(f"  {basin_id}: " + ", ".join(f"{c}={int(masks[c].sum())}" for c in CATEGORIES))

    def cat(arrays):
        return np.concatenate(arrays) if arrays else np.array([])

    pairs = {c: (cat(raw[c]), cat(norm[c]), cat(months[c])) for c in CATEGORIES}
    return pairs, counts


def normalized_hist_table(pairs):
    """200 equal-width bins over the pooled [min, max] of all categories, one count column per category."""
    pooled = np.concatenate([pairs[c][1] for c in CATEGORIES])
    edges = np.histogram_bin_edges(pooled, bins=N_BINS, range=(float(pooled.min()), float(pooled.max())))
    table = pd.DataFrame({'bin_left': edges[:-1], 'bin_right': edges[1:]})
    for c in CATEGORIES:
        table[f'count_{c}'] = np.histogram(pairs[c][1], bins=edges)[0]
    return table, edges


def month_count_table(pairs):
    table = pd.DataFrame({'month': MONTH_LABELS})
    for c in CATEGORIES:
        table[f'count_{c}'] = [int((pairs[c][2] == m).sum()) for m in HYDRO_MONTHS]
    return table


def plot_normalized_hist(hist_df, edges, counts):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for c in CATEGORIES:
        ax.stairs(hist_df[f'count_{c}'], edges, color=COLORS[c], linewidth=1.6,
                  label=f"{c} ({counts[c]['kept_pairs']:,} pairs)")
    ax.set_xlabel('Normalized flow (per-basin z-score, all data)')
    ax.set_ylabel('Number of basin-timestep pairs per bin')
    ax.set_yscale('log')
    ax.set_ylim(top=ax.get_ylim()[1] * 5)  # log-y headroom so the legend clears the data
    ax.set_title('Normalized flow by Flow_type (flow > 0)', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', ncol=3)
    fig.tight_layout()
    return fig


def plot_month_hist(month_df, counts):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    x = np.arange(len(HYDRO_MONTHS))
    width = 0.27
    for i, c in enumerate(CATEGORIES):
        ax.bar(x + (i - 1) * width, month_df[f'count_{c}'], width=width, color=COLORS[c],
               label=f"{c} ({counts[c]['kept_pairs']:,} pairs)")
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlabel('Month (hydrological year, Oct - Sep)')
    ax.set_ylabel('Number of basin-timestep pairs per month')
    ax.set_yscale('log')
    ax.set_ylim(top=ax.get_ylim()[1] * 5)  # log-y headroom so the legend clears the data
    ax.set_title('Basin-timestep pairs per month by Flow_type (flow > 0)', fontsize=11)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', ncol=3)
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

    summary_rows = []
    for c in CATEGORIES:
        raw_values, norm_values, _ = pairs[c]
        n = counts[c]
        assert n['kept_pairs'] + n['zero_flow_steps'] + n['nan_flow_steps'] == n['in_category_steps']
        row = {'category': c, 'flow_types': '+'.join(sorted(LABELED[c])) if c in LABELED else 'not flow/normal', **n}
        if len(raw_values):
            row.update(raw_min=raw_values.min(), raw_median=float(np.median(raw_values)), raw_max=raw_values.max(),
                       normalized_min=norm_values.min(), normalized_median=float(np.median(norm_values)),
                       normalized_max=norm_values.max())
        else:
            print(f"[WARNING] category '{c}': no pairs with flow > 0.")
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
    print(summary_df.to_string(index=False))

    if sum(counts[c]['kept_pairs'] for c in CATEGORIES) == 0:
        print("[WARNING] no pairs with flow > 0 - nothing to plot.")
        return

    hist_df, edges = normalized_hist_table(pairs)
    month_df = month_count_table(pairs)
    for c in CATEGORIES:
        assert hist_df[f'count_{c}'].sum() == month_df[f'count_{c}'].sum() == counts[c]['kept_pairs']
    hist_df.to_csv(os.path.join(output_dir, 'normalized_flow_hist.csv'), index=False)
    month_df.to_csv(os.path.join(output_dir, 'month_counts.csv'), index=False)

    figures = {'normalized_flow_hist': plot_normalized_hist(hist_df, edges, counts),
               'month_hist': plot_month_hist(month_df, counts)}
    for name, fig in figures.items():
        fig.savefig(os.path.join(output_dir, f"{name}.png"))

    if config.get('use_wandb', False):
        import wandb
        if config.get('wandb_api_key'):
            wandb.login(key=config['wandb_api_key'])
        wandb.init(project=config.get('wandb_project', 'flash-floods-israel'),
                   name='flow_distribution_all_data', job_type='flow_distribution',
                   config={'categories': CATEGORIES, 'n_bins': N_BINS,
                           'processed_timeseries_dir': config['processed_timeseries_dir'],
                           'zero_flow_excluded': True})
        log = {name: wandb.Image(fig) for name, fig in figures.items()}
        log.update(normalized_flow_hist_table=wandb.Table(dataframe=hist_df),
                   month_counts_table=wandb.Table(dataframe=month_df),
                   summary=wandb.Table(dataframe=summary_df))
        wandb.log(log)
        wandb.finish()
    for fig in figures.values():
        plt.close(fig)
    print(f"\n[INFO] Flow distribution outputs in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalized-flow and per-month distributions of flow / normal / "
                                                 "other Flow_type timesteps.")
    parser.add_argument("--config", type=str, default="configs/preprocess_all_data.yml",
                        help="All-data preprocessing config (default configs/preprocess_all_data.yml).")
    args = parser.parse_args()
    main(load_config(args.config))
