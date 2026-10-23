"""
Module: summarize_family_test_periods.py
Description: One summary per model FAMILY (e.g. model_0 = leads 0/3/6/12) over
             all its lead times x test periods (the best_crossval_<model>_<test years>
             final models). Produces, in run_dir/summary_name/:

               test_<years>/comparison_plots/hydrograph_*.png
                   Event hydrographs (current hydrograph design) per test period,
                   comparing ONLY the family's leads that were tested on those same
                   years (models of other test sets trained on these events), via
                   model_compare_test.run_event_and_peaks_analysis - plus that
                   test set's flood_event_comparison_<thr>.csv / peaks CSVs.
               nse_cdf_leads_test_periods.png
                   NSE empirical CDF: one bright color per lead, a thin curve per
                   test period, and per lead a thick darker MEDIAN curve (the
                   per-CDF-level median of the test-period curves, i.e. the median
                   of their quantile functions) with a dashed vertical line at its
                   median NSE (only the median curves get one).
               nse_per_basin.csv, nse_summary.csv
               stats_by_lead_test_period_<thr>.csv, median_stats_by_lead_<thr>.csv
                   Event statistics (model_comparison_eval columns: TP/FP/FN,
                   precision/recall/F1, peak timing/magnitude) per lead x test
                   period, and their median over test periods per lead, per
                   threshold label (e.g. 2yr, sq0.1).
               availability.csv
                   Which (model, test period) were found / skipped and every
                   input's source.

             Reuses existing outputs as much as possible and works with whatever
             exists (e.g. right after training, before any other analysis):
               - per-basin reports runs/<exp>/visualization_reports/ (test.py /
                 model_compare_test.py) - else the model is evaluated here (GPU);
               - NSE: existing NSE_CDF_compare_test_periods nse_per_basin.csv -
                 else computed from the reports (model_compare_test.report_nse);
               - statistics: existing model_comparison_eval_<thr>.csv of the
                 8-model comparison - else build_eval_summary over that
                 comparison's CSVs - else over this run's own family comparison
                 (FP counts can differ slightly there: events are the union over
                 the compared models; the 'source' column shows which was used).
             A (model, test period) without a config or trained best_model.pt is
             skipped and listed in availability.csv.

Usage:
  python src/summarize_family_test_periods.py --config configs/summarize_family_test_periods_model_0.yml [--reevaluate]
"""

import argparse
import colorsys
import glob
import os

import matplotlib
matplotlib.use('Agg')  # file-only backend: never needs an X display (cluster nodes)
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import find_flood_events as ffe
from model_compare_test import (load_config, _read_basin_list, run_single_model_eval, report_nse,
                                run_event_and_peaks_analysis)
from model_compare_eval_test import build_eval_summary
from NSE_CDF_compare_test_periods import describe_test_years

QUANTILE_GRID = np.linspace(0.0, 1.0, 201)


def _exp_dir(config):
    return os.path.join(config.get('run_dir', './runs/'), config['experiment_name'])


def _report_path(config, basin):
    return os.path.join(_exp_dir(config), "visualization_reports", f"visual_report_basin_{basin}.csv")


def lead_colors(models):
    """Bright base color per model (tab10 order) + a darker shade of the same hue for its median."""
    cmap = plt.get_cmap('tab10')
    colors = {}
    for i, model in enumerate(models):
        r, g, b, _ = cmap(i % 10)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        colors[model] = (mcolors.to_hex((r, g, b)), mcolors.to_hex(colorsys.hls_to_rgb(h, l * 0.55, s)))
    return colors


def load_family(cfg):
    """
    {(model, test): config} for every trained (model, test period), plus
    availability rows for all of them (found or why skipped).
    """
    found, rows = {}, []
    for model in cfg['models']:
        for test in cfg['test_sets']:
            path = cfg['model_config_pattern'].format(model=model, test=test)
            row = {'model': model, 'test_set': test, 'config': path}
            if not os.path.exists(path):
                row['status'] = 'skipped: no config'
            else:
                config = load_config(path)
                weights = config.get('checkpoint_path') or os.path.join(_exp_dir(config), 'best_model.pt')
                if not os.path.exists(weights):
                    row['status'] = f'skipped: not trained ({weights} missing)'
                else:
                    row['status'] = 'used'
                    found[(model, test)] = config
            if row['status'] != 'used':
                print(f"[WARNING] {model} / {test}: {row['status']}")
            rows.append(row)
    return found, rows


def ensure_reports(config, basins, reevaluate):
    """Reuses existing per-basin reports; evaluates the model only if none exist (or --reevaluate)."""
    n_existing = sum(os.path.exists(_report_path(config, b)) for b in basins)
    if reevaluate or n_existing == 0:
        why = '--reevaluate' if reevaluate else 'no existing reports'
        print(f"[INFO] {config['experiment_name']}: evaluating on {len(basins)} test basins ({why})...")
        run_single_model_eval(config, basins)
        return 'evaluated here'
    print(f"[INFO] {config['experiment_name']}: reusing reports for {n_existing}/{len(basins)} test basins.")
    return 'existing reports'


def collect_nse(cfg, model, test, config, basins, lead):
    """
    (rows, source) of per-basin NSE for this model's test set: existing
    NSE_CDF_compare_test_periods values where present, the remaining basins
    computed from the reports (model_compare_test.report_nse - same math).
    """
    nse_dir = cfg.get('existing_nse_dir_pattern', '').format(model=model, test=test)
    path = os.path.join(nse_dir, 'nse_per_basin.csv') if nse_dir else ''
    existing = {}
    if path and os.path.exists(path):
        df = pd.read_csv(path, dtype={'basin_id': str})
        df = df[(df['test_years'] == describe_test_years(config)) & df['basin_id'].isin(basins)]
        existing = dict(zip(df['basin_id'], df['nse']))
    rows, n_computed = [], 0
    for basin in basins:
        if basin in existing:
            rows.append({'basin_id': basin, 'nse': existing[basin]})
            continue
        nse = report_nse(_exp_dir(config), lead, basin)
        if nse is not None:
            rows.append({'basin_id': basin, 'nse': nse})
            n_computed += 1
    if existing and n_computed:
        source = f'NSE_CDF_compare_test_periods ({len(existing)}) + computed from reports ({n_computed})'
    elif existing:
        source = 'NSE_CDF_compare_test_periods'
    else:
        source = 'computed from reports'
    return rows, source


def median_quantile_curve(value_lists):
    """Per-CDF-level median of several empirical CDFs: median of their quantile functions."""
    quantiles = [np.quantile(np.sort(v), QUANTILE_GRID) for v in value_lists if len(v)]
    return np.median(np.vstack(quantiles), axis=0)


def plot_nse_cdf(nse_df, models, leads, summary_name, out_path):
    colors = lead_colors(models)
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150, facecolor="#fafafa")
    ax.set_facecolor("#ffffff")
    median_rows = []
    for model in models:
        model_df = nse_df[nse_df['model'] == model]
        if model_df.empty:
            continue
        bright, dark = colors[model]
        period_values = []
        for test_years, period_df in model_df.groupby('test_years'):
            values = np.sort(period_df['nse'].to_numpy())
            period_values.append(values)
            ax.plot(values, np.arange(1, len(values) + 1) / len(values), color=bright, linewidth=1.0, alpha=0.6)
        median_curve = median_quantile_curve(period_values)
        median_nse = float(np.interp(0.5, QUANTILE_GRID, median_curve))
        ax.plot(median_curve, QUANTILE_GRID, color=dark, linewidth=2.6,
                label=f"{model} (+{leads[model]}h) median of {len(period_values)} test periods: {median_nse:.3f}")
        ax.axvline(x=median_nse, color=dark, linewidth=1.5, linestyle='--', alpha=0.9)
        median_rows.append({'model': model, 'lead': leads[model], 'test_years': 'median curve',
                            'n_test_periods': len(period_values), 'median_nse': median_nse})
    ax.plot([], [], color='#9e9e9e', linewidth=1.0, alpha=0.6, label='thin lines: individual test periods')
    ax.set_title(f"NSE Empirical CDF — leads x test periods ({summary_name})",
                 fontsize=12, fontweight='bold', pad=15, color='#2c3e50')
    ax.set_xlabel('NSE', fontsize=10.5, labelpad=8)
    ax.set_ylabel('CDF', fontsize=10.5, labelpad=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=':', alpha=0.5, color='#b0b0b0')
    ax.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#e2e2e2', fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"[INFO] Saved {out_path}")
    return fig, median_rows


def stats_for(cfg, model, test, thr, family_out_dir):
    """
    (row dict, source) of this model's eval statistics for one test set and
    threshold label, from the first source that has a row for it: the existing
    comparison's eval CSV, else build_eval_summary over that comparison's CSVs
    (all three label models as existing_compare_label_pattern, e.g.
    best_crossval_0_3), else build_eval_summary over this run's family comparison.
    """
    compare_dir = cfg.get('existing_compare_dir_pattern', '').format(model=model, test=test)
    compare_label = cfg.get('existing_compare_label_pattern', '{model}').format(model=model, test=test)
    candidates = []
    if compare_dir:
        eval_csv = os.path.join(compare_dir, f"model_comparison_eval_{thr}.csv")
        if os.path.exists(eval_csv):
            candidates.append(('existing eval CSV', compare_label, lambda p=eval_csv: pd.read_csv(p)))
        if os.path.exists(os.path.join(compare_dir, f"flood_event_comparison_{thr}.csv")):
            candidates.append(('existing comparison CSVs', compare_label,
                               lambda d=compare_dir: build_eval_summary(d, thr)))
    candidates.append(('this run (family only)', compare_label,
                       lambda d=family_out_dir: build_eval_summary(d, thr)))
    for source, label, load in candidates:
        df = load()
        if df is None or df.empty:
            continue
        match = df[df['model_label'].astype(str) == label]
        if not match.empty:
            return match.iloc[0].to_dict(), source
    return None, None


def main(config_path, reevaluate=False):
    cfg = load_config(config_path)
    for key in ('models', 'test_sets'):
        if not all(isinstance(v, str) for v in cfg[key]):
            raise ValueError(f"'{key}' entries must be quoted strings in {config_path} - YAML reads unquoted "
                             f"0_12 / 2009_2010_2011 as numbers. Got: {cfg[key]}")
    out_dir =os.path.join(cfg.get('run_dir', './runs/model_comparisons/'), cfg['summary_name'])
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 75)
    print(f"      Family summary '{cfg['summary_name']}': models {cfg['models']} x test sets {cfg['test_sets']}")
    print("=" * 75)

    family, availability = load_family(cfg)
    if not family:
        raise RuntimeError("No trained (model, test period) found - nothing to summarize. See the warnings above.")
    leads = {}
    for (model, _), config in family.items():
        leads[model] = config['forecast_lead_times'][0]
    models = [m for m in cfg['models'] if m in leads]
    avail_by_key = {(r['model'], r['test_set']): r for r in availability}

    # Step A + B: reports, then hydrographs / event comparison per test period (family's leads only)
    nse_rows = []
    for test in cfg['test_sets']:
        test_models = [m for m in models if (m, test) in family]
        if not test_models:
            continue
        configs = [family[(m, test)] for m in test_models]
        basins = sorted(_read_basin_list(configs[0]['test_basin_file']))
        for model, config in zip(test_models, configs):
            avail_by_key[(model, test)]['reports'] = ensure_reports(config, basins, reevaluate)

        test_out = os.path.join(out_dir, f"test_{test}")
        os.makedirs(test_out, exist_ok=True)
        # Short legend labels (e.g. best_crossval_0_3) - the test years are already in the folder name
        labels = [cfg.get('existing_compare_label_pattern', '{model}').format(model=m, test=test) for m in test_models]
        print(f"\n[INFO] Test set {test}: event hydrographs + comparison for {test_models}")
        run_event_and_peaks_analysis(configs, labels, {lab: leads[m] for lab, m in zip(labels, test_models)},
                                     basins, test_out, use_wandb=False)

        # Step C: NSE per basin
        for model, config in zip(test_models, configs):
            rows, source = collect_nse(cfg, model, test, config, basins, leads[model])
            avail_by_key[(model, test)]['nse_source'] = source
            years = describe_test_years(config)
            nse_rows += [{'model': model, 'lead': leads[model], 'test_years': years, 'test_set': test,
                          'source': source, **r} for r in rows]

    # Step D: NSE CDF
    nse_df = pd.DataFrame(nse_rows)
    nse_df.to_csv(os.path.join(out_dir, 'nse_per_basin.csv'), index=False)
    cdf_fig, median_rows = plot_nse_cdf(nse_df, models, leads, cfg['summary_name'],
                                        os.path.join(out_dir, 'nse_cdf_leads_test_periods.png'))
    summary_rows = []
    for (model, years), g in nse_df.groupby(['model', 'test_years'], sort=False):
        v = g['nse'].to_numpy()
        summary_rows.append({'model': model, 'lead': leads[model], 'test_years': years, 'n_basins': len(v),
                             'median_nse': float(np.median(v)), 'frac_nse_gt_0': float(np.mean(v > 0)),
                             'frac_nse_gt_0.5': float(np.mean(v > 0.5))})
    nse_summary = pd.DataFrame(summary_rows + median_rows)
    nse_summary.to_csv(os.path.join(out_dir, 'nse_summary.csv'), index=False)

    # Step E: event statistics per lead x test period, and their median per lead
    any_config = next(iter(family.values()))
    thr_labels = [ffe.threshold_label(s) for s in ffe.normalize_threshold_specs(any_config['prediction_threshold'])]
    stats_tables = {}
    for thr in thr_labels:
        long_rows = []
        for (model, test) in sorted(family, key=lambda k: (cfg['models'].index(k[0]), k[1])):
            row, source = stats_for(cfg, model, test, thr, os.path.join(out_dir, f"test_{test}"))
            avail_by_key[(model, test)][f'stats_source_{thr}'] = source or 'none'
            if row is None:
                print(f"[WARNING] {model} / {test} / {thr}: no event statistics found.")
                continue
            row.update(model=model, lead=leads[model], test_set=test, source=source)
            long_rows.append(row)
        if not long_rows:
            continue
        long_df = pd.DataFrame(long_rows)
        long_df.to_csv(os.path.join(out_dir, f"stats_by_lead_test_period_{thr}.csv"), index=False)
        stat_cols = [c for c in long_df.columns
                     if c not in ('model', 'lead', 'test_set', 'source', 'model_label')
                     and pd.api.types.is_numeric_dtype(long_df[c])]
        median_rows_thr = []
        for model, g in long_df.groupby('model', sort=False):
            r = {'model': model, 'lead': leads[model], 'n_test_periods': len(g),
                 'test_periods': ';'.join(g['test_set']), 'sources': ';'.join(sorted(set(g['source'])))}
            r.update({c: float(g[c].median()) for c in stat_cols})
            median_rows_thr.append(r)
        median_df = pd.DataFrame(median_rows_thr)
        median_df.to_csv(os.path.join(out_dir, f"median_stats_by_lead_{thr}.csv"), index=False)
        stats_tables[thr] = median_df
        print(f"\n[INFO] Median statistics over test periods ({thr}):")
        print(median_df[['model', 'lead', 'n_test_periods'] + [c for c in ('precision', 'recall', 'f1')
                                                              if c in median_df]].to_string(index=False))

    availability_df = pd.DataFrame(availability)
    availability_df.to_csv(os.path.join(out_dir, 'availability.csv'), index=False)

    if cfg.get('use_wandb', False) and WANDB_AVAILABLE:
        if cfg.get('wandb_api_key'):
            wandb.login(key=cfg['wandb_api_key'])
        wandb.init(project=cfg.get('wandb_project', 'flash-floods-israel'), name=cfg['summary_name'],
                   job_type='family_test_periods_summary', config=cfg)
        log = {'nse_cdf': wandb.Image(cdf_fig), 'nse_summary': wandb.Table(dataframe=nse_summary),
               'availability': wandb.Table(dataframe=availability_df.astype(str))}
        for thr, df in stats_tables.items():
            log[f'median_stats_by_lead_{thr}'] = wandb.Table(dataframe=df)
        wandb.log(log)
        for test in cfg['test_sets']:
            for png in sorted(glob.glob(os.path.join(out_dir, f"test_{test}", "comparison_plots", "hydrograph_*.png"))):
                wandb.log({f"hydrographs/test_{test}/{os.path.basename(png)[:-4]}": wandb.Image(png)})
        wandb.finish()
    plt.close(cdf_fig)
    print(f"\n[INFO] Family summary completed. Outputs in: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hydrographs, NSE CDF and median statistics of one model family "
                                                 "over its lead times x test periods.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--reevaluate", action="store_true",
                        help="Re-run every model over its test split even if its reports already exist.")
    args = parser.parse_args()
    main(args.config, reevaluate=args.reevaluate)
