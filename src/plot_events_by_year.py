"""
Module: plot_events_by_year.py
Description: Year-scoped flood-event hydrograph plotter. Loads one or more
             trained models' checkpoints, runs inference, scans real and
             predicted flood events (the same union/classification logic
             model_compare_test.py uses for its full-test-period comparison),
             restricts them to the requested hydrological year(s), and plots
             an observed-vs-predicted hydrograph (with rain overlay and RP
             threshold lines) for each surviving event. Works for a single
             model or many - reuses model_compare_test.py's building blocks
             unchanged, just adds the year filter and a date-named plotting
             loop instead of scanning/plotting the whole test period.
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from model_compare_test import (
    load_config, validate_comparable, run_single_model_eval,
    merge_basin_reports, cluster_events, classify_clusters,
    build_model_color_map, plot_hydrograph_comparison,
)
import find_flood_events as ffe
from flow_quality_check import get_hydrological_year


def _cluster_hydro_year(core_start, hydro_year_start_month):
    return int(get_hydrological_year(pd.DatetimeIndex([core_start]), start_month=hydro_year_start_month)[0])


def find_and_plot_events(model_configs, model_labels, model_leads, shared_basins, years, output_dir,
                          use_wandb=False):
    """
    Per basin: merges real+predicted+rain (merge_basin_reports), scans real
    and every model's predicted flood events, unions overlapping events
    across sources (cluster_events), keeps only clusters whose core_start
    falls in one of the requested hydrological years, classifies the
    survivors TP/FN/FP/TN per model (classify_clusters), and plots one
    hydrograph per surviving cluster. Writes a companion CSV of exactly the
    plotted events. Returns the number of hydrographs plotted.
    """
    prediction_rp = model_configs[0]['prediction_threshold']
    buffer_days = model_configs[0].get('visual_buffer_days', 4)
    merge_gap_hours = model_configs[0].get('event_merge_gap_hours', 0)
    hydro_year_start_month = model_configs[0].get('hydro_year_start_month', 10)
    years = set(years)

    model_color_map = build_model_color_map(model_labels)
    all_rows = []
    n_plotted = 0

    print(f"\n[INFO] Scanning {len(shared_basins)} basins for flood events in hydro-year(s) {sorted(years)}...")
    for basin in shared_basins:
        merged_df = merge_basin_reports(basin, model_configs, model_labels, model_leads)
        if merged_df is None:
            continue

        threshold_value = ffe.load_basin_threshold(basin, prediction_rp, model_configs[0])
        if threshold_value is None or threshold_value <= 0:
            continue

        tagged = []
        real_events = ffe._scan_series_for_events(merged_df, 'actual_flow', threshold_value,
                                                    buffer_days, merge_gap_hours)
        for ev in real_events:
            tagged.append({**ev, 'source': 'real',
                            'core_start': pd.to_datetime(ev['core_start']),
                            'core_end': pd.to_datetime(ev['core_end'])})

        for label in model_labels:
            col = f'pred__{label}'
            if col not in merged_df.columns:
                continue
            pred_events = ffe._scan_series_for_events(merged_df, col, threshold_value,
                                                        buffer_days, merge_gap_hours)
            for ev in pred_events:
                tagged.append({**ev, 'source': label,
                                'core_start': pd.to_datetime(ev['core_start']),
                                'core_end': pd.to_datetime(ev['core_end'])})

        if not tagged:
            continue

        clusters = cluster_events(tagged)
        clusters = [c for c in clusters
                    if _cluster_hydro_year(c['core_start'], hydro_year_start_month) in years]
        if not clusters:
            continue

        rows = classify_clusters(clusters, model_labels, basin)
        all_rows.extend(rows)

        for idx, cluster in enumerate(clusters, start=1):
            padded_start = max(cluster['core_start'] - pd.Timedelta(days=buffer_days), merged_df['timestamp'].min())
            padded_end = min(cluster['core_end'] + pd.Timedelta(days=buffer_days), merged_df['timestamp'].max())
            window_df = merged_df[(merged_df['timestamp'] >= padded_start) & (merged_df['timestamp'] <= padded_end)]
            if window_df.empty:
                continue

            event_labels = {row['model_label']: row['label'] for row in rows if row['event_idx'] == idx}
            title = (f"Basin {basin} — Storm Event\n"
                     f"Core: {cluster['core_start'].strftime('%Y-%m-%d %H:%M')} → "
                     f"{cluster['core_end'].strftime('%Y-%m-%d %H:%M')}")
            # Filename keyed by the event's own date, not a positional index -
            # this run's filtered cluster list numbers differently than a
            # full model_compare_test.py run over the same basins/models
            # would, so a positional "event{idx}" name would be ambiguous if
            # the two ever write into the same output folder.
            date_tag = cluster['core_start'].strftime('%Y-%m-%d')
            filename = f"hydrograph_{basin}_{date_tag}.png"
            fig = plot_hydrograph_comparison(window_df, model_labels, model_leads, model_color_map,
                                              title, output_dir, filename, event_labels=event_labels,
                                              hourly_xticks=True, xlim=(padded_start, padded_end))
            if use_wandb:
                wandb.log({f"events_by_year/{basin}/{date_tag}": wandb.Image(fig)})
            plt.close(fig)
            n_plotted += 1
            print(f"  {basin} {date_tag}: saved {filename}")

    years_tag = "_".join(str(y) for y in sorted(years))
    csv_path = os.path.join(output_dir, f"flood_event_comparison_years_{years_tag}.csv")
    columns = ['basin_id', 'event_idx', 'core_start', 'core_end',
               'real_peak_flow', 'model_label', 'model_peak_flow', 'label']
    pd.DataFrame(all_rows, columns=columns).to_csv(csv_path, index=False)
    print(f"\n[INFO] Plotted {n_plotted} hydrographs. Wrote {len(all_rows)} rows to {csv_path}")

    return n_plotted


def main(comparison_config_path, years):
    comparison_config = load_config(comparison_config_path)
    model_config_paths = comparison_config['model_configs']
    model_labels_cfg = comparison_config.get('model_labels')

    print("=" * 75)
    print("      Flood-Event Hydrograph Plotter — Year-Scoped")
    print("=" * 75)
    print(f"[INFO] Models: {model_config_paths}")
    print(f"[INFO] Hydrological year(s): {years}")

    model_configs, model_labels, shared_basins, model_leads = validate_comparable(
        model_config_paths, model_labels_cfg
    )
    print(f"[INFO] Validation passed. {len(shared_basins)} shared test basins. Leads: {model_leads}")

    output_dir = os.path.join(comparison_config.get('run_dir', './runs/model_comparisons/'),
                               comparison_config['comparison_name'])
    os.makedirs(output_dir, exist_ok=True)

    use_wandb = comparison_config.get('use_wandb', False) and WANDB_AVAILABLE
    if use_wandb:
        api_key = comparison_config.get('wandb_api_key')
        if api_key:
            wandb.login(key=api_key)
        wandb.init(
            project=comparison_config.get('wandb_project', 'flash-floods-israel'),
            name=f"{comparison_config['comparison_name']}_events_by_year_{'_'.join(str(y) for y in sorted(set(years)))}",
            config={**comparison_config, 'years': years},
        )

    # Load each checkpoint and run inference, writing/refreshing each
    # model's own visual_report_basin_*.csv - same step model_compare_test.py
    # always performs (no skip-if-exists caching, to avoid stale results).
    print("\n[INFO] Evaluating model(s)...")
    for label, config in zip(model_labels, model_configs):
        print(f"  Evaluating '{label}' ({config['experiment_name']})...")
        run_single_model_eval(config, shared_basins)

    find_and_plot_events(model_configs, model_labels, model_leads, shared_basins, years, output_dir,
                          use_wandb=use_wandb)

    if use_wandb:
        wandb.finish()

    print(f"\n[INFO] Done. Outputs in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load one or more trained models' checkpoints, scan real/predicted flood events, "
                     "restrict to the given hydrological year(s), and plot observed-vs-predicted "
                     "hydrographs (with rain overlay) for each.")
    parser.add_argument("--config", type=str, required=True,
                         help="Path to a comparison config YAML (same format as model_compare_test.py; "
                              "a single-entry model_configs list works too).")
    parser.add_argument("--years", type=int, nargs='+', required=True,
                         help="One or more hydrological years (Oct->Sep) to restrict plotted events to.")
    args = parser.parse_args()
    main(args.config, args.years)
