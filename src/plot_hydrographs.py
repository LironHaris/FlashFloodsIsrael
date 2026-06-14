"""
Module: plot_hydrographs.py
Description: Interactive Visualization Engine for Multi-Horizon EA-LSTM Flood Forecasting.
             Provides Plotly-based hydrograph and analytics plots consumed by test.py
             and callable standalone for targeted storm event inspection.
"""

import os
import yaml
import pandas as pd
import numpy as np
import plotly.graph_objects as go


LEAD_COLORS = {0: '#2b8cbe', 1: '#8856a7', 2: '#cb181d', 3: '#86592d', 6: '#cb181d', 12: '#86592d', 24: '#f16913'}
LEAD_DASH   = {0: 'solid',   1: 'dash',    2: 'dash',    3: 'dot',     6: 'dash',    12: 'dot',    24: 'dot'}
THRESHOLD_COLORS = {2: '#bae4b3', 5: '#74c476', 10: '#ef3b2c', 20: '#990000', 30: '#67000d'}


def load_config(yaml_path):
    """Load YAML configuration variables safely."""
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def _build_hydrograph_figure(plot_df, title, config, rp_filter=None):
    """
    Shared Plotly figure builder for hydrograph functions.
    rp_filter: if set (int), only draw that return period's threshold line.
    """
    active_leads = config.get('forecast_lead_times', [0, 1, 2, 3])
    rp_years = [rp_filter] if rp_filter is not None else config.get('return_periods_years', [2, 5, 10])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_df['timestamp'], y=plot_df['actual_flow'],
        mode='lines', name='Actual Streamflow',
        line=dict(color='#1e1e1e', width=2.0),
        hovertemplate='%{x}<br>Actual: %{y:.3f} m³/s<extra></extra>',
    ))

    for lead in active_leads:
        col = f"pred_lead_{lead}h"
        if col in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=plot_df['timestamp'], y=plot_df[col],
                mode='lines', name=f'+{lead}h Lead',
                line=dict(color=LEAD_COLORS.get(lead, '#7f7f7f'),
                          dash=LEAD_DASH.get(lead, 'solid'), width=1.4),
                opacity=0.85,
                hovertemplate=f'%{{x}}<br>+{lead}h: %{{y:.3f}} m³/s<extra></extra>',
            ))

    for rp in rp_years:
        thresh_col = f"threshold_{rp}yr_rp"
        if thresh_col in plot_df.columns:
            thresh_val = float(plot_df[thresh_col].iloc[0])
            if thresh_val > 0:
                fig.add_hline(
                    y=thresh_val,
                    line=dict(color=THRESHOLD_COLORS.get(rp, '#d9d9d9'), width=1.0, dash='dashdot'),
                    opacity=0.9,
                    annotation_text=f'{rp}yr RP ({thresh_val:.1f} m³/s)',
                    annotation_position='bottom right',
                )

    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Discharge (m³/s)',
        plot_bgcolor='#ffffff',
        paper_bgcolor='#fafafa',
        legend=dict(font=dict(size=10)),
        hovermode='x unified',
    )
    return fig


def plot_basin_storm_event(basin_id, start_window, end_window, config):
    """
    Interactive Plotly hydrograph for a targeted storm event window.
    Loads the basin's visual_report CSV, applies visual_buffer_days padding,
    and returns a go.Figure. Returns None if data is missing or empty.
    """
    run_dir = config.get('run_dir', './runs/')
    exp_dir = os.path.join(run_dir, config['experiment_name'])
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

    title = (f"Basin {basin_id} — Storm Event<br>"
             f"<sup>Core: {core_start.strftime('%Y-%m-%d %H:%M')} → {core_end.strftime('%Y-%m-%d %H:%M')}</sup>")

    return _build_hydrograph_figure(plot_df, title, config)


def plot_basin_full_history(basin, df, config):
    """
    Interactive Plotly hydrograph for the full test period of a basin.
    Accepts an in-memory DataFrame (as produced by build_and_export_report in test.py).
    """
    title = f"Test Period — Basin {basin}"
    return _build_hydrograph_figure(df, title, config)


def plot_nse_cdf(lead, nse_values):
    """
    Empirical CDF of basin-level NSE for one lead time.
    X-axis starts at 0; median is marked with a dashed vertical line.
    Returns a go.Figure, or None if no valid values.
    """
    values = sorted([v for v in nse_values if v is not None and not np.isnan(v)])
    if not values:
        return None

    n = len(values)
    cdf_y = [(i + 1) / n for i in range(n)]
    median_nse = float(np.median(values))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=values, y=cdf_y,
        mode='lines',
        line=dict(color='#2b8cbe', width=2),
        name=f'Lead +{lead}h',
        hovertemplate='NSE: %{x:.3f}<br>CDF: %{y:.2f}<extra></extra>',
    ))
    fig.add_vline(
        x=median_nse,
        line=dict(color='#cb181d', width=1.5, dash='dash'),
        annotation_text=f'Median: {median_nse:.3f}',
        annotation_position='top right',
    )
    fig.update_layout(
        title=f'NSE Empirical CDF — Lead +{lead}h ({n} basins)',
        xaxis_title='NSE',
        yaxis_title='Cumulative Probability',
        xaxis=dict(range=[0, max(1.0, max(values) + 0.05)]),
        yaxis=dict(range=[0, 1]),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#fafafa',
    )
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
        run_dir  = config.get('run_dir', './runs/')
        exp_dir  = os.path.join(run_dir, config['experiment_name'])
        plots_dir = os.path.join(exp_dir, "hydrograph_plots")
        os.makedirs(plots_dir, exist_ok=True)

        out_path = os.path.join(plots_dir, f"hydrograph_basin_{basin_id}_storm.html")
        fig.write_html(out_path)

        print("\n" + "=" * 75)
        print(f"[✓] Interactive hydrograph saved to: {out_path}")
        print("=" * 75)


if __name__ == "__main__":
    main()
