"""
Module: plot_hydrographs.py
Description: Static Visualization Engine for Multi-Horizon EA-LSTM Flood Forecasting.
             Parses evaluation reports to render fixed-size Matplotlib hydrograph
             and analytics figures consumed by test.py and callable standalone
             for targeted storm event inspection.
"""

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


LEAD_COLORS = {0: '#2b8cbe', 1: '#8856a7', 2: '#cb181d', 3: '#86592d', 6: '#cb181d', 12: '#86592d', 24: '#f16913'}
LEAD_STYLES = {0: '-',       1: '--',      2: '--',      3: ':',       6: '--',      12: ':',      24: ':'}
THRESHOLD_COLORS = {2: '#238b45', 5: '#238b45', 10: '#ef3b2c', 20: '#990000', 30: '#67000d'}


def load_config(yaml_path):
    """Load YAML configuration variables safely."""
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def _get_exp_dir(config):
    run_dir = config.get('run_dir', './runs/')
    return os.path.join(run_dir, config['experiment_name'])


def _save_figure(fig, exp_dir, filename):
    """Save a figure into {exp_dir}/hydrograph_plots/ and return the output path."""
    plots_dir = os.path.join(exp_dir, "hydrograph_plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, filename)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    return out_path


def _build_hydrograph_figure(plot_df, title, config, rp_filter=None, hourly_xticks=False):
    """
    Shared Matplotlib figure builder for hydrograph functions.
    Fixed-size canvas (12 x 6.5 in @ 150 dpi).
    rp_filter: if set (int), only draw that return period's threshold line.
    hourly_xticks: if True, format the time axis with hourly ticks (storm windows).
    """
    active_leads = config.get('forecast_lead_times', [0, 1, 2, 3])
    rp_years = [rp_filter] if rp_filter is not None else config.get('return_periods_years', [2, 5, 10])

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150, facecolor="#fafafa")
    ax.set_facecolor("#ffffff")

    ax.plot(plot_df['timestamp'], plot_df['actual_flow'],
            color='#1e1e1e', linewidth=2.0, label='Actual Streamflow')

    for lead in active_leads:
        col = f"pred_lead_{lead}h"
        if col in plot_df.columns:
            ax.plot(plot_df['timestamp'], plot_df[col],
                    color=LEAD_COLORS.get(lead, '#7f7f7f'),
                    linestyle=LEAD_STYLES.get(lead, '-'),
                    linewidth=1.4, alpha=0.85, label=f'+{lead}h Lead')

    # Threshold lines
    for rp in rp_years:
        thresh_col = f"threshold_{rp}yr_rp"
        if thresh_col in plot_df.columns:
            thresh_val = float(plot_df[thresh_col].iloc[0])
            if thresh_val > 0:
                ax.axhline(y=thresh_val, color=THRESHOLD_COLORS.get(rp, '#d9d9d9'),
                           linestyle='-.', linewidth=2.2, alpha=1.0,
                           label=f'{rp}yr RP ({thresh_val:.1f} m³/s)')

    ax.set_title(title, fontsize=12, fontweight='bold', pad=15, color='#2c3e50')
    ax.set_xlabel('Time', fontsize=10.5, labelpad=8)
    ax.set_ylabel('Discharge (m³/s)', fontsize=10.5, labelpad=8)
    ax.grid(True, linestyle=':', alpha=0.5, color='#b0b0b0')
    ax.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#e2e2e2', fontsize=9)

    if hourly_xticks:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %Hh'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    fig.tight_layout()
    return fig


def plot_basin_storm_event(basin_id, start_window, end_window, config, label=None, event_idx=None):
    """
    Fixed-size hydrograph for a targeted storm event window, with an hourly time axis.
    Loads the basin's visual_report CSV, applies visual_buffer_days padding,
    saves a PNG, and returns the figure. Returns None if data is missing or empty.
    label/event_idx: when both are given (e.g. 'TP'/'FN'/'FP' from
    compare_flood_events.py), they set the filename and a title suffix so
    real/predicted/matched events are distinguishable at a glance.
    """
    exp_dir = _get_exp_dir(config)
    report_path = os.path.join(exp_dir, "visualization_reports", f"visual_report_basin_{basin_id}.csv")

    if not os.path.exists(report_path):
        print(f"[ERROR] Visual report missing for basin {basin_id} at {report_path}. Run test.py first.")
        return None

    df = pd.read_csv(report_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    buffer_days = config.get('visual_buffer_days', 4)
    core_start  = pd.to_datetime(start_window)
    core_end    = pd.to_datetime(end_window)
    padded_start = max(core_start - pd.Timedelta(days=buffer_days), df['timestamp'].min())
    padded_end   = min(core_end   + pd.Timedelta(days=buffer_days), df['timestamp'].max())

    plot_df = df[(df['timestamp'] >= padded_start) & (df['timestamp'] <= padded_end)].reset_index(drop=True)

    if plot_df.empty:
        print(f"[ERROR] No data in window for basin {basin_id}.")
        return None

    title = (f"Basin {basin_id} — Storm Event" + (f" [{label}]" if label else "") + "\n"
             f"Core: {core_start.strftime('%Y-%m-%d %H:%M')} → {core_end.strftime('%Y-%m-%d %H:%M')}")

    fig = _build_hydrograph_figure(plot_df, title, config, hourly_xticks=True)
    if event_idx is not None and label is not None:
        filename = f"hydrograph_{basin_id}_event{event_idx}_{label}.png"
    else:
        filename = f"hydrograph_basin_{basin_id}_storm.png"
    _save_figure(fig, exp_dir, filename)
    return fig


def plot_basin_full_history(basin, df, config):
    """
    Fixed-size hydrograph for the full test period of a basin.
    Accepts an in-memory DataFrame (as produced by build_and_export_report in test.py).
    Saves a PNG and returns the figure.
    """
    title = f"Test Period — Basin {basin}"
    fig = _build_hydrograph_figure(df, title, config)
    _save_figure(fig, _get_exp_dir(config), f"hydrograph_basin_{basin}_full.png")
    return fig


def plot_nse_cdf(lead, nse_values, config):
    """
    Empirical CDF of basin-level NSE for one lead time.
    X-axis starts at 0; median is marked with a dashed vertical line.
    Saves a PNG and returns the figure, or None if no valid values.
    """
    values = sorted([v for v in nse_values if v is not None and not np.isnan(v)])
    if not values:
        return None

    n = len(values)
    cdf_y = [(i + 1) / n for i in range(n)]
    median_nse = float(np.median(values))

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150, facecolor="#fafafa")
    ax.set_facecolor("#ffffff")

    ax.plot(values, cdf_y, color='#2b8cbe', linewidth=2, label=f'Lead +{lead}h')
    ax.axvline(x=median_nse, color='#cb181d', linewidth=1.5, linestyle='--',
               label=f'Median: {median_nse:.3f}')

    ax.set_title(f'NSE Empirical CDF — Lead +{lead}h ({n} basins)',
                  fontsize=12, fontweight='bold', pad=15, color='#2c3e50')
    ax.set_xlabel('NSE', fontsize=10.5, labelpad=8)
    ax.set_ylabel('CDF', fontsize=10.5, labelpad=8)
    ax.set_xlim(0, max(1.0, max(values) + 0.05))
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=':', alpha=0.5, color='#b0b0b0')
    ax.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#e2e2e2', fontsize=9)

    fig.tight_layout()
    _save_figure(fig, _get_exp_dir(config), f"nse_cdf_lead_{lead}h.png")
    return fig


def main():
    config = load_config("configs/config.yml")

    print("=" * 75)
    print("      Hydrograph Generation & Visualization Engine — Storm Event Slice")
    print("=" * 75)

    plot_cfg     = config["plot_hydrographs"]
    basin_id     = plot_cfg["basin_id"]
    start_window = plot_cfg["start_time"]
    end_window   = plot_cfg["end_time"]

    print(f"\n[INFO] Basin: {basin_id}")
    print(f"[INFO] Core Window: [{start_window}] → [{end_window}]")
    print(f"[INFO] Visual padding: {config.get('visual_buffer_days', 4)} days")
    print("-" * 75)

    fig = plot_basin_storm_event(basin_id, start_window, end_window, config)

    if fig is not None:
        out_path = os.path.join(_get_exp_dir(config), "hydrograph_plots", f"hydrograph_basin_{basin_id}_storm.png")
        plt.close(fig)

        print("\n" + "=" * 75)
        print(f"[✓] Hydrograph saved to: {out_path}")
        print("=" * 75)


if __name__ == "__main__":
    main()
