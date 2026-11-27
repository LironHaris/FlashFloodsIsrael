"""
Module: family_basin_analysis.py
Description: Per-basin analysis of one model family summary (summarize_family_test_periods.py),
             built only from that summary's outputs - nothing is re-evaluated. Writes, in
             run_dir/summary_name/basin_analysis/:

               basin_scores_<thr>.csv
                   One row per test basin: basin_no (stable 1..N by basin_id, the number drawn
                   next to its point in the plots), basin_id, basin_name, gauge_lat, gauge_lon, ari_ix_sav, area, PC1,
                   n_test_periods / test_periods (periods with NSE or events for the basin - basins
                   in fewer than all of them are drawn hollow in the plots), then per lead L:
                     nse_median_<L>h, nse_n_periods_<L>h   median NSE over the test periods
                                                           (nse_per_basin.csv; same in every thr file)
                     tp/fp/fn/precision/recall/f1_<L>h      event counts of that lead pooled over the
                                                           test periods, then precision/recall/F1
                     cum_tp/.../cum_recall/cum_f1_<L>h      the same with cumulative leads (an event
                                                           counts as predicted at L if any lead >= L
                                                           predicted it), pooled over the test periods
               <thr>/<metric>_vs_<x>.png
                   metric (nse_median, recall, precision, cum_recall, cum_precision) against
                   x (ari_ix_sav, area, PC1): one panel per lead, a box per quartile of x (equal basin
                   counts; tick = x range and n), each basin a jittered point with its basin_no on top.
                   Panel titles give n and Spearman rho / Pearson r with p-values on the continuous x
                   (Pearson on unclipped values).
               <thr>/basins_legend.png
                   The basin_no -> basin_id / name key for all plots and maps of this threshold.
               <thr>/map_<metric>.png
                   Map of Israel per metric: one panel per lead, each basin polygon filled by its value
                   (shared colorbar), largest basins drawn first so nested sub-basins stay visible, dashed
                   outline = fewer test periods, basin_no inside each polygon, red dot = the gauge station
                   (basin outlet, gauge_lat/lon - a basin ends at its gauge, not at the sea). Polygons: basin_shapes_file
                   (default data/raw/shapefiles/il/il_basin_shapes.shp); borders: basemap_file (Natural
                   Earth 10m countries, clipped). Skipped with a warning without geopandas / the shapefile.

             Events come from each test period's flood_event_comparison_<thr>.csv (the same file
             summarize_family_test_periods' cumulative statistics use, see find_event_csv). Basin
             attributes: static_attributes_file (ari_ix_sav, area), pca_output_dir/pca_projected_data.csv
             (PC1), raw_static_dir/attributes_other_il.csv (gauge_name), all from a member train config.

             summarize_family_test_periods.py runs this at the end of every full summary (Step E3).

Usage (on an existing summary - same as summarize_family_test_periods.py --basin-only):
  python src/family_basin_analysis.py --config configs/summarize_family_test_periods_model_0_flow_normal_avg3_eval_all.yml
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')  # file-only backend: never needs an X display (cluster nodes)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

import plot_hydrographs as ph
from model_compare_eval_test import _prf, compute_cumulative_confusion

X_FEATURES = [('ari_ix_sav', 'Aridity index (ari_ix_sav)', False),
              ('area', 'Basin area [km²] (log scale)', True),
              ('PC1', 'PC1 of the static attributes', False)]
PLOT_METRICS = [('nse_median', 'Median NSE over test periods'),
                ('recall', 'Recall (pooled over test periods)'),
                ('precision', 'Precision (pooled over test periods)'),
                ('cum_recall', 'Cumulative-lead recall (pooled)'),
                ('cum_precision', 'Cumulative-lead precision (pooled)')]
NSE_FLOOR = -1.0  # NSE below this is drawn at the floor (triangle) so a few bad basins don't flatten the axis
BASIN_SHAPES_FILE = 'data/raw/shapefiles/il/il_basin_shapes.shp'  # gauge_id + basin polygon
# Natural Earth 10m admin-0 countries clipped to 33-37E / 28.8-34.2N (country borders under the basins)
BASEMAP_FILE = 'data/raw/shapefiles/natural_earth/ne_10m_countries_israel_region.shp'
MAP_CRS = 'EPSG:2039'  # Israel TM Grid - metric, undistorted over Israel


def find_event_csv(cfg, out_dir, test, thr, labels):
    """
    (source, events_df) of the first flood_event_comparison_<thr>.csv for this test period that
    has all `labels`: the existing comparison's (existing_compare_dir_pattern - summarize drops it
    in derived mode), else this run's test_<test>/ folder. (None, None) when neither has them.
    """
    candidates = []
    if cfg.get('existing_compare_dir_pattern'):
        compare_dir = cfg['existing_compare_dir_pattern'].format(model='', test=test)
        candidates.append(('existing comparison', os.path.join(compare_dir, f"flood_event_comparison_{thr}.csv")))
    candidates.append(('this run (family only)', os.path.join(out_dir, f"test_{test}", f"flood_event_comparison_{thr}.csv")))
    for source, path in candidates:
        if not os.path.exists(path):
            continue
        events_df = pd.read_csv(path, dtype={'basin_id': str, 'model_label': str})
        if set(labels) <= set(events_df['model_label']):
            return source, events_df
    return None, None


def _read_ids_csv(path, columns):
    if not path or not os.path.exists(path):
        print(f"[WARNING] Basin analysis: {path or '(no path)'} not found - {columns} left empty.")
        return pd.DataFrame(columns=['basin_id'] + columns)
    df = pd.read_csv(path, dtype={'gauge_id': str})
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"[WARNING] Basin analysis: {missing} not in {path}.")
    return df[['gauge_id'] + [c for c in columns if c in df.columns]].rename(columns={'gauge_id': 'basin_id'})


def load_basin_attributes(member_config, basins):
    """basin_no, basin_id, basin_name, gauge_lat/lon, ari_ix_sav, area, PC1 for `basins` (sorted, numbered 1..N)."""
    df = pd.DataFrame({'basin_id': sorted(basins)})
    df.insert(0, 'basin_no', np.arange(1, len(df) + 1))
    names = _read_ids_csv(os.path.join(member_config.get('raw_static_dir', ''), ph.BASIN_NAMES_FILE),
                          ['gauge_name', 'gauge_lat', 'gauge_lon'])
    static = _read_ids_csv(member_config.get('static_attributes_file'), ['ari_ix_sav', 'area'])
    pca = _read_ids_csv(os.path.join(member_config.get('pca_output_dir', './data/processed/pca_results'),
                                     'pca_projected_data.csv'), ['PC1'])
    for extra in (names.rename(columns={'gauge_name': 'basin_name'}), static, pca):
        df = df.merge(extra, on='basin_id', how='left')
    for col in ('basin_name', 'gauge_lat', 'gauge_lon', 'ari_ix_sav', 'area', 'PC1'):
        if col not in df:
            df[col] = np.nan
    df['basin_name'] = df['basin_name'].fillna('').astype(str).str.strip()
    return df[['basin_no', 'basin_id', 'basin_name', 'gauge_lat', 'gauge_lon', 'ari_ix_sav', 'area', 'PC1']]


def _pooled_counts(cfg, out_dir, models, leads, tests, thr):
    """
    {(basin, lead): [tp, fp, fn]} per lead and cumulative, summed over the test periods, and
    {test: basins seen in that test period's event CSV}.
    """
    label_pattern = cfg.get('existing_compare_label_pattern', '{model}')
    per_lead, cumulative, basins = {}, {}, {}
    for test, test_models in tests.items():
        label_leads = {label_pattern.format(model=m, test=test): leads[m] for m in test_models}
        source, events_df = find_event_csv(cfg, out_dir, test, thr, label_leads)
        if events_df is None:
            print(f"[WARNING] Basin analysis: no flood_event_comparison_{thr}.csv with labels "
                  f"{sorted(label_leads)} for test set {test} - its events are left out.")
            continue
        events_df = events_df[events_df['model_label'].isin(label_leads)]
        basins[test] = set(events_df['basin_id'])
        counts = events_df.groupby(['basin_id', 'model_label'])['label'].value_counts().unstack(fill_value=0)
        for (basin, label), row in counts.iterrows():
            acc = per_lead.setdefault((basin, label_leads[label]), [0, 0, 0])
            for i, k in enumerate(('TP', 'FP', 'FN')):
                acc[i] += int(row.get(k, 0))
        for basin, basin_df in events_df.groupby('basin_id'):
            for _, r in compute_cumulative_confusion(basin_df, label_leads).iterrows():
                acc = cumulative.setdefault((basin, int(r['lead'])), [0, 0, 0])
                acc[0] += r['tp']
                acc[1] += r['fp']
                acc[2] += r['fn']
    return per_lead, cumulative, basins


def build_basin_scores(cfg, out_dir, models, leads, tests, thr, member_config, nse_df):
    per_lead, cumulative, event_basins = _pooled_counts(cfg, out_dir, models, leads, tests, thr)
    # A basin was tested in a period when it has an NSE value or events there
    period_basins = {test: set(event_basins.get(test, set())) for test in tests}
    for test, g in nse_df.groupby('test_set'):
        period_basins.setdefault(test, set()).update(g['basin_id'])
    scores = load_basin_attributes(member_config, set().union(*period_basins.values()))
    scores['n_test_periods'] = [sum(basin in b for b in period_basins.values()) for basin in scores['basin_id']]
    scores['test_periods'] = [';'.join(t for t in sorted(period_basins) if basin in period_basins[t])
                              for basin in scores['basin_id']]
    nse_groups = {k: g for k, g in nse_df.groupby(['basin_id', 'lead'])} if not nse_df.empty else {}

    metric_cols = {}
    for lead in sorted({leads[m] for m in models}):
        nse_med, nse_n = [], []
        for basin in scores['basin_id']:
            g = nse_groups.get((basin, lead))
            nse_med.append(float(g['nse'].median()) if g is not None else np.nan)
            nse_n.append(len(g) if g is not None else 0)
        metric_cols[f'nse_median_{lead}h'] = nse_med
        metric_cols[f'nse_n_periods_{lead}h'] = nse_n
        for prefix, counts in (('', per_lead), ('cum_', cumulative)):
            rows = [counts.get((basin, lead), [0, 0, 0]) for basin in scores['basin_id']]
            prf = [_prf(*r) for r in rows]
            for i, name in enumerate(('tp', 'fp', 'fn')):
                metric_cols[f'{prefix}{name}_{lead}h'] = [r[i] for r in rows]
            for i, name in enumerate(('precision', 'recall', 'f1')):
                metric_cols[f'{prefix}{name}_{lead}h'] = [p[i] for p in prf]
    return pd.concat([scores, pd.DataFrame(metric_cols, index=scores.index)], axis=1)


def plot_basin_legend(scores, summary_name, thr, out_path):
    """The basin_no -> basin_id / name key shared by all plots and maps of this threshold, as its own image
    (columns of ~25 rows) so the plots don't each repeat it."""
    rows = list(zip(scores['basin_no'], scores['basin_id'], scores['basin_name']))
    per_col = 25
    ncols = max(1, int(np.ceil(len(rows) / per_col)))
    col_w, row_h = 4.3, 0.22
    fig = plt.figure(figsize=(col_w * ncols + 0.4, row_h * min(len(rows), per_col) + 1.0), dpi=150,
                     facecolor='#ffffff')
    fig.suptitle(f"Basins ({summary_name}, {thr})", fontsize=11, fontweight='bold', color='#2c3e50',
                 y=1 - 0.25 / fig.get_figheight())
    top = 1 - 0.65 / fig.get_figheight()
    step = row_h / fig.get_figheight()
    for i, (no, bid, name) in enumerate(rows):
        c, r = divmod(i, per_col)
        x = (0.2 + c * col_w) / fig.get_figwidth()
        y = top - r * step
        fig.text(x, y, f"{no:>2}", fontsize=8, fontweight='bold', color='#37474f', ha='left', va='top',
                 family='monospace')
        fig.text(x + 0.3 / fig.get_figwidth(), y, f"{bid}  {name}", fontsize=7.5, color='#263238', ha='left',
                 va='top')
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def _quartile_groups(df, feature, n_bins=4):
    """Splits df into up to n_bins equal-count groups of `feature` (fewer on ties / small n): list of
    (sub_df, tick label 'lo–hi\\nn=k')."""
    k = min(n_bins, len(df))
    if k < 2 or df[feature].nunique() < 2:
        bins = pd.Series(0, index=df.index)
    else:
        bins = pd.qcut(df[feature].rank(method='first'), k, labels=False)
    integer = feature == 'area'
    groups = []
    for b in sorted(bins.unique()):
        sub = df[bins == b]
        lo, hi = sub[feature].min(), sub[feature].max()
        fmt = (lambda v: f"{v:,.0f}") if integer else (lambda v: f"{v:.2g}")
        groups.append((sub, f"{fmt(lo)}–{fmt(hi)}\nn={len(sub)}"))
    return groups


def plot_metric_vs_feature(scores, models, leads, metric, metric_title, feature, feature_title, log_x,
                           summary_name, thr, out_path, n_tests):
    """One panel per lead: boxes of the metric over quartiles of the feature, each basin a jittered point
    with its basin_no on top (key in basins_legend.png). Correlations use the continuous feature values."""
    colors = _lead_colors(models)
    lead_models = sorted(models, key=lambda m: leads[m])
    n = len(lead_models)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig_h = 4.6 * nrows + 1.9
    fig = plt.figure(figsize=(12, fig_h), dpi=150, facecolor="#fafafa")
    axes = fig.subplots(nrows, ncols, squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.98, top=1 - 1.05 / fig_h, bottom=1.1 / fig_h, hspace=0.42,
                        wspace=0.16)
    is_nse = metric == 'nse_median'
    feature_label = feature_title.replace(' (log scale)', '')
    any_partial = any_clipped = False

    for ax, model in zip(axes.flat, lead_models):
        lead = leads[model]
        ax.set_facecolor('#ffffff')
        col = f'{metric}_{lead}h'
        df = scores[['basin_no', 'n_test_periods', feature, col]].dropna()
        if log_x:
            df = df[df[feature] > 0]
        bright, dark = colors[model]
        if df.empty:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center', va='center', color='#888888')
            ax.set_title(f"lead +{lead}h ({model})", fontsize=10)
            continue
        groups = _quartile_groups(df, feature)
        box_vals = [np.maximum(g[col].to_numpy(dtype=float), NSE_FLOOR) if is_nse else g[col].to_numpy(dtype=float)
                    for g, _ in groups]
        bp = ax.boxplot(box_vals, positions=range(len(groups)), widths=0.55, showfliers=False,
                        patch_artist=True, zorder=2)
        for patch in bp['boxes']:
            patch.set(facecolor=bright, alpha=0.28, edgecolor=dark, linewidth=0.9)
        for part in ('whiskers', 'caps'):
            for line in bp[part]:
                line.set(color=dark, linewidth=0.9)
        for line in bp['medians']:
            line.set(color=dark, linewidth=1.8)
        for pos, (g, _) in enumerate(groups):
            y = g[col].to_numpy(dtype=float)
            clipped = is_nse & (y < NSE_FLOOR)
            y_draw = np.where(clipped, NSE_FLOOR, y)
            partial = g['n_test_periods'].to_numpy() < n_tests
            any_partial |= partial.any()
            any_clipped |= clipped.any()
            jitter = np.array([np.random.default_rng(int(no)).uniform(-0.2, 0.2) for no in g['basin_no']])
            x = pos + jitter
            for sel_clip, marker in ((False, 'o'), (True, 'v')):
                for sel_part in (False, True):
                    sel = (clipped == sel_clip) & (partial == sel_part)
                    if sel.any():
                        ax.scatter(x[sel], y_draw[sel], s=26, marker=marker,
                                   facecolor='white' if sel_part else bright, edgecolor=dark,
                                   linewidth=1.1 if sel_part else 0.6, zorder=3)
            for xi, yi, no in zip(x, y_draw, g['basin_no']):
                ax.annotate(str(no), (xi, yi), xytext=(3, 2), textcoords='offset points', fontsize=6,
                            color='#37474f', zorder=4)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([label for _, label in groups], fontsize=8)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        x_all = df[feature].to_numpy(dtype=float)
        ax.set_title(f"lead +{lead}h ({model}) — n={len(df)}\n{_corr_text(x_all, df[col].to_numpy(dtype=float))}",
                     fontsize=9.5)
        ax.set_ylim((NSE_FLOOR - 0.05, 1.02) if is_nse else (-0.05, 1.05))
        ax.grid(True, axis='y', linestyle=':', alpha=0.5, color='#b0b0b0')
        ax.set_xlabel(f"{feature_label} — quartiles", fontsize=9)
        ax.set_ylabel(metric_title, fontsize=9)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    handles = [Line2D([], [], marker='o', linestyle='none', markerfacecolor='#9e9e9e', markeredgecolor='#424242',
                      label=f'all {n_tests} test periods')]
    if any_partial:
        handles.append(Line2D([], [], marker='o', linestyle='none', markerfacecolor='white',
                              markeredgecolor='#424242', label=f'< {n_tests} test periods'))
    if any_clipped:
        handles.append(Line2D([], [], marker='v', linestyle='none', markerfacecolor='#9e9e9e',
                              markeredgecolor='#424242', label=f'NSE < {NSE_FLOOR:g} (drawn at {NSE_FLOOR:g})'))
    handles.append(Line2D([], [], linestyle='none', label='number = basin_no (see basins_legend.png)'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.005), ncol=len(handles),
               fontsize=8, frameon=False)
    fig.suptitle(f"{metric_title} vs {feature_label} — per basin\n({summary_name}, {thr})",
                 fontsize=12, fontweight='bold', color='#2c3e50', y=1 - 0.12 / fig_h, va='top')
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def _corr_text(x, y):
    """'Spearman ρ=…(p=…), Pearson r=…(p=…)' - Pearson on the true (unclipped) values, so outliers weigh in."""
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return 'too few / constant values for correlation'
    rho, p_rho = stats.spearmanr(x, y)
    r, p_r = stats.pearsonr(x, y)
    return f"Spearman ρ={rho:.2f} (p={p_rho:.2g}), Pearson r={r:.2f} (p={p_r:.2g})"


def _lead_colors(models):
    # Same colors per lead as summarize_family_test_periods (imported lazily: it imports this module)
    from summarize_family_test_periods import lead_colors
    return lead_colors(models)


def run_basin_analysis(cfg, out_dir, models, leads, tests, thr_labels, member_config):
    """
    tests: {test_set: [models tested on it]}. Writes basin_analysis/ (see module docstring) and returns
    {'tables': {thr: scores_df}, 'images': {'basin_analysis/<thr>/<metric>_vs_<x>': png path}}.
    """
    analysis_dir = os.path.join(out_dir, 'basin_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    nse_path = os.path.join(out_dir, 'nse_per_basin.csv')
    if os.path.exists(nse_path):
        nse_df = pd.read_csv(nse_path, dtype={'basin_id': str})
    else:
        print(f"[WARNING] Basin analysis: {nse_path} not found - NSE columns left empty.")
        nse_df = pd.DataFrame(columns=['basin_id', 'lead', 'test_set', 'nse'])

    results = {'tables': {}, 'images': {}}
    shapes, basemap, maps_tried = None, None, False
    for thr in thr_labels:
        scores = build_basin_scores(cfg, out_dir, models, leads, tests, thr, member_config, nse_df)
        csv_path = os.path.join(analysis_dir, f"basin_scores_{thr}.csv")
        scores.to_csv(csv_path, index=False)
        print(f"[INFO] Basin analysis ({thr}): {len(scores)} basins -> {csv_path}")
        results['tables'][thr] = scores
        plot_dir = os.path.join(analysis_dir, thr)
        os.makedirs(plot_dir, exist_ok=True)
        for metric, metric_title in PLOT_METRICS:
            for feature, feature_title, log_x in X_FEATURES:
                if scores[feature].isna().all():
                    continue
                name = f"{metric}_vs_{feature}"
                png = plot_metric_vs_feature(scores, models, leads, metric, metric_title, feature, feature_title,
                                             log_x, cfg['summary_name'], thr, os.path.join(plot_dir, f"{name}.png"),
                                             n_tests=len(tests))
                results['images'][f"basin_analysis/{thr}/{name}"] = png
        if shapes is None and not maps_tried:
            shapes, basemap = load_map_layers(cfg, set(scores['basin_id']))
            maps_tried = True
        if shapes is not None and not shapes.empty:
            for metric, metric_title in PLOT_METRICS:
                name = f"map_{metric}"
                png = plot_metric_map(scores, shapes, basemap, models, leads, metric, metric_title,
                                      cfg['summary_name'], thr, os.path.join(plot_dir, f"{name}.png"), len(tests))
                results['images'][f"basin_analysis/{thr}/{name}"] = png
        png = plot_basin_legend(scores, cfg['summary_name'], thr, os.path.join(plot_dir, 'basins_legend.png'))
        results['images'][f"basin_analysis/{thr}/basins_legend"] = png
        print(f"[INFO] Basin analysis ({thr}): plots in {plot_dir}")
    return results


def load_map_layers(cfg, basin_ids):
    """
    (basin polygons, basemap countries) in Israel TM (EPSG:2039), or (None, None) when geopandas or the
    basin shapefile is missing. basin_shapes_file / basemap_file may be set in the family summary config.
    """
    try:
        import geopandas as gpd
    except ImportError:
        print("[WARNING] Basin analysis: geopandas not installed - maps skipped.")
        return None, None
    shapes_path = cfg.get('basin_shapes_file', BASIN_SHAPES_FILE)
    if not os.path.exists(shapes_path):
        print(f"[WARNING] Basin analysis: {shapes_path} not found - maps skipped.")
        return None, None
    shapes = gpd.read_file(shapes_path).astype({'gauge_id': str}).rename(columns={'gauge_id': 'basin_id'})
    shapes = shapes[shapes['basin_id'].isin(basin_ids)].to_crs(MAP_CRS)
    missing = sorted(set(basin_ids) - set(shapes['basin_id']))
    if missing:
        print(f"[WARNING] Basin analysis: no polygon for {missing} - left off the maps.")
    basemap_path = cfg.get('basemap_file', BASEMAP_FILE)
    basemap = gpd.read_file(basemap_path).to_crs(MAP_CRS) if os.path.exists(basemap_path) else None
    if basemap is None:
        print(f"[WARNING] Basin analysis: {basemap_path} not found - maps drawn without country borders.")
    return shapes, basemap


def plot_metric_map(scores, shapes, basemap, models, leads, metric, metric_title, summary_name, thr, out_path,
                    n_tests):
    """One map panel per lead: each basin filled by its metric value (shared colorbar), large basins drawn
    first so nested sub-basins stay visible on top, dashed outline = fewer than n_tests test periods."""
    import matplotlib.patheffects as pe
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch

    is_nse = metric == 'nse_median'
    cmap = plt.get_cmap('RdYlBu' if is_nse else 'viridis').copy()
    norm = Normalize(vmin=NSE_FLOOR if is_nse else 0.0, vmax=1.0)
    import geopandas as gpd
    gdf = shapes.merge(scores, on='basin_id', how='inner')
    gdf = gdf.assign(_area=gdf.geometry.area).sort_values('_area', ascending=False)
    minx, miny, maxx, maxy = gdf.total_bounds
    # Gauge stations (basin outlets): each basin ends at its gauge, not at the sea
    gauges = scores.dropna(subset=['gauge_lat', 'gauge_lon'])
    gauge_xy = gpd.GeoSeries(gpd.points_from_xy(gauges['gauge_lon'], gauges['gauge_lat']), crs='EPSG:4326').to_crs(MAP_CRS)
    pad = 12_000

    lead_models = sorted(models, key=lambda m: leads[m])
    # Panel width from the map's own aspect (Israel is ~3x taller than wide); basin key in basins_legend.png
    fig_h, axes_h_frac = 11.0, 0.78
    panel_w = fig_h * axes_h_frac * (maxx - minx + 2 * pad) / (maxy - miny + 2 * pad) + 0.25
    fig_w = max(panel_w * len(lead_models) + 0.6, 9.0)  # wide enough for the bottom legend
    right = 1 - 0.3 / fig_w
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150, facecolor="#fafafa")
    axes = fig.subplots(1, len(lead_models), squeeze=False)[0]
    fig.subplots_adjust(left=0.3 / fig_w, right=right, top=0.90, bottom=0.145, wspace=0.04)
    for ax, model in zip(axes, lead_models):
        col = f'{metric}_{leads[model]}h'
        ax.set_facecolor('#dfeaf2')  # sea
        if basemap is not None:
            israel = basemap['ISO_A3'].isin(['ISR', 'PSE']) if 'ISO_A3' in basemap else np.zeros(len(basemap), bool)
            basemap[~israel].plot(ax=ax, color='#e4e4e4', edgecolor='#9e9e9e', linewidth=0.5)
            basemap[israel].plot(ax=ax, color='#f5f2ea', edgecolor='#757575', linewidth=0.7)
        n_valid = 0
        for _, row in gdf.iterrows():
            value = row[col]
            partial = row['n_test_periods'] < n_tests
            style = dict(edgecolor='#263238', linewidth=0.9 if partial else 0.4, linestyle='--' if partial else '-')
            if pd.isna(value):
                gpd_plot(row, ax, facecolor='#ffffff', hatch='////', **style)
            else:
                n_valid += 1
                gpd_plot(row, ax, facecolor=cmap(norm(max(value, NSE_FLOOR) if is_nse else value)), **style)
        ax.scatter(gauge_xy.x, gauge_xy.y, s=10, color='#d50000', edgecolor='white', linewidth=0.4, zorder=6)
        for _, row in gdf.iterrows():
            p = row.geometry.representative_point()
            ax.annotate(str(row['basin_no']), (p.x, p.y), ha='center', va='center', fontsize=5.2, color='#111111',
                        path_effects=[pe.withStroke(linewidth=1.6, foreground='white')], zorder=5)
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"lead +{leads[model]}h ({model}) — n={n_valid}", fontsize=10)
    cax = fig.add_axes([0.15 * right, 0.085, 0.7 * right, 0.018])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal',
                        extend='min' if is_nse else 'neither')
    cbar.set_label(metric_title + (f" (below {NSE_FLOOR:g} shown as {NSE_FLOOR:g})" if is_nse else ''), fontsize=9)
    fig.legend(handles=[Patch(facecolor='#cccccc', edgecolor='#263238', label=f'all {n_tests} test periods'),
                        Patch(facecolor='#cccccc', edgecolor='#263238', linestyle='--', linewidth=0.9,
                              label=f'< {n_tests} test periods'),
                        Patch(facecolor='#ffffff', edgecolor='#263238', hatch='////', label='no value'),
                        Line2D([], [], marker='o', linestyle='none', markerfacecolor='#d50000',
                               markeredgecolor='white', markersize=5, label='gauge station (basin outlet)')],
               loc='lower left', bbox_to_anchor=(0.02, 0.005), ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f"{metric_title} — per basin map ({summary_name}, {thr})", fontsize=12, fontweight='bold',
                 color='#2c3e50')
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def gpd_plot(row, ax, **kwargs):
    """Draws one basin polygon (row of a GeoDataFrame) - per row, so each basin keeps its own edge style."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    geom = row.geometry
    polys = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
    for poly in polys:
        rings = [poly.exterior] + list(poly.interiors)
        verts, codes = [], []
        for ring in rings:
            xy = np.asarray(ring.coords)
            verts.extend(xy)
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY])
        ax.add_patch(PathPatch(Path(verts, codes), zorder=3, **kwargs))


def wandb_log_dict(results):
    import wandb
    log = {f"basin_scores_{thr}": wandb.Table(dataframe=df) for thr, df in results['tables'].items()}
    log.update({key: wandb.Image(png) for key, png in results['images'].items()})
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-basin scores, boxplots and maps of one family summary "
                                                 "(from its existing outputs).")
    parser.add_argument("--config", type=str, required=True, help="Family summary config "
                        "(configs/summarize_family_test_periods_*.yml)")
    args = parser.parse_args()
    import summarize_family_test_periods
    summarize_family_test_periods.main(args.config, basin_only=True)
