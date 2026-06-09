"""
Module: plot_hydrographs.py
Description: Interactive Visualization Engine for Multi-Horizon EA-LSTM Flood Forecasting.
             Parses evaluation reports to render formal, publication-ready hydrographs
             overlaying multi-lead predictions, GEV thresholds, and calculated hit rates.
"""

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_config(yaml_path):
    """Load YAML configuration variables safely."""
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def plot_basin_storm_event(basin_id, start_window, end_window, config):
    """
    Generates a localized, engineering-grade hydrograph for a specific basin context.
    Applies the visual buffer configured in config.yml dynamically around the core window.
    """
    run_dir = config.get('run_dir', './runs/')
    exp_dir = os.path.join(run_dir, config['experiment_name'])
    report_path = os.path.join(exp_dir, "visualization_reports", f"visual_report_basin_{basin_id}.csv")
    
    if not os.path.exists(report_path):
        print(f"\n[ERROR] Visual report missing for basin {basin_id} at {report_path}.")
        print("        Please make sure you have executed test.py first.")
        return

    # Read evaluation report dataframe
    df = pd.read_csv(report_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Extract padding buffer configurations from config
    buffer_days = config.get('visual_buffer_days', 4)
    
    # Apply padding to expand the user's core storm dates backwards and forwards
    core_start = pd.to_datetime(start_window)
    core_end = pd.to_datetime(end_window)
    padded_start = core_start - pd.Timedelta(days=buffer_days)
    padded_end = core_end + pd.Timedelta(days=buffer_days)

    # Constrain padding inside the actual boundaries of the test split
    padded_start = max(padded_start, df['timestamp'].min())
    padded_end = min(padded_end, df['timestamp'].max())

    # Slice the visualization dataframe using the padded dates
    mask = (df['timestamp'] >= padded_start) & (df['timestamp'] <= padded_end)
    plot_df = df[mask].reset_index(drop=True)

    if plot_df.empty:
        print(f"\n[ERROR] Sliced timeframe contains no valid data points for basin {basin_id}.")
        return

    # Initialize Figure Canvas Dimensions (Clean Light Aesthetic)
    plt.figure(figsize=(12, 6.5), facecolor="#fafafa")
    ax = plt.axes()
    ax.set_facecolor("#ffffff")

    # Plot Ground Truth Streamflow Vector
    plt.plot(plot_df['timestamp'], plot_df['actual_flow'], color="#1e1e1e", 
             linewidth=2.5, label="Actual Streamflow (Observed)")

    # Iteratively plot available Forecast Lead Horizons
    lead_colors = {0: '#2b8cbe', 1: '#8856a7', 6: '#cb181d', 12: '#86592d', 24: '#f16913'}
    lead_styles = {0: '-', 1: '--', 6: '--', 12: ':', 24: ':'}
    
    active_leads = config.get('forecast_lead_times', [0, 6, 24])
    for lead in active_leads:
        col_name = f"pred_lead_{lead}h"
        if col_name in plot_df.columns:
            c = lead_colors.get(lead, '#7f7f7f')
            s = lead_styles.get(lead, '-')
            plt.plot(plot_df['timestamp'], plot_df[col_name], color=c, linestyle=s,
                     linewidth=1.6, alpha=0.85, label=f"EA-LSTM Forecast (+{lead}h Lead)")

    # Map Static Horizontal Return Period Lines & Extract Hit Rates
    rp_years = config.get('return_periods_years', [2, 5, 10, 20])
    threshold_colors = {2: '#bae4b3', 5: '#74c476', 10: '#ef3b2c', 20: '#990000', 30: '#67000d'}
    
    hit_rate_summary_lines = []
    
    for rp in rp_years:
        thresh_col = f"threshold_{rp}yr_rp"
        if thresh_col in plot_df.columns:
            thresh_val = plot_df[thresh_col].iloc[0]
            
            # Avoid plotting non-existent or failed GEV thresholds (<=0)
            if thresh_val <= 0:
                continue
                
            c_thresh = threshold_colors.get(rp, '#d9d9d9')
            plt.axhline(y=thresh_val, color=c_thresh, linestyle="-.", linewidth=1.2, alpha=0.9,
                        label=f"{rp}-Year Critical Threshold ({thresh_val:.1f} $m^3/s$)")
            
            # Extract associated verification scores for text rendering using a primary evaluation lead
            primary_lead = active_leads[min(2, len(active_leads)-1)] # Default to 6h/24h position
            count_col = f"hit_rate_pred_lead_{primary_lead}h_{rp}yr_count"
            score_col = f"hit_rate_pred_lead_{primary_lead}h_{rp}yr_score"
            
            if count_col in plot_df.columns:
                cnt_str = plot_df[count_col].iloc[0]
                pct_val = plot_df[score_col].iloc[0] * 100
                hit_rate_summary_lines.append(f"{rp}yr Threshold ({primary_lead}h Lead): {cnt_str} ({pct_val:.1f}%)")

    # Embed a formal Verification Hit Rates Text Box inside the figure
    if hit_rate_summary_lines:
        box_content = "Global Categorical Hit Rates:\n" + "─" * 29 + "\n" + "\n".join(hit_rate_summary_lines)
        box_styling = dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#dddddd', alpha=0.9)
        ax.text(0.02, 0.96, box_content, transform=ax.transAxes, fontsize=9.5,
                verticalalignment='top', bbox=box_styling, fontfamily='monospace', color='#2c3e50')

    # Layout Adjustments & Metric Labeling
    plt.title(f"Multi-Horizon Streamflow Hydrograph — Basin Code Context: {basin_id}\nTarget Storm Event Evaluation Analysis (Padded Grid)", 
              fontsize=12, fontweight='bold', pad=15, color="#2c3e50")
    plt.xlabel("Timeline (Date-Hour Resolution)", fontsize=10.5, labelpad=8)
    plt.ylabel("Volumetric Discharge Rate ($m^3 / sec$)", fontsize=10.5, labelpad=8)
    plt.grid(True, linestyle=":", alpha=0.5, color="#b0b0b0")
    
    plt.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e2e2e2", fontsize=9)
    plt.tight_layout()

    # Save Figure Asset to disk
    plots_dir = os.path.join(exp_dir, "hydrograph_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    output_path = os.path.join(plots_dir, f"hydrograph_basin_{basin_id}_interactive_storm.png")
    plt.savefig(output_path, dpi=300, facecolor="#fafafa")
    plt.close()
    
    print("\n" + "="*75)
    print(f"[✓] Hydrograph visualization successfully rendered and exported!")
    print(f"    Output Path: {output_path}")
    print("="*75)


def main():
    config = load_config("configs/config.yml")
    
    print("=" * 75)
    print("      Hydrograph Generation & Visualization Engine — Multi-Horizon Slices")
    print("=" * 75)
    
    plot_cfg = config["plot_hydrographs"]
    basin_id = plot_cfg["basin_id"]
    start_window = plot_cfg["start_time"]
    end_window = plot_cfg["end_time"]

    print("\n" + "-" * 75)
    print(f"[INFO] Initializing visual layout compiler for Basin: {basin_id}")
    print(f"[INFO] Core Window: [{start_window}] TO [{end_window}]")
    print(f"[INFO] Dynamic visual padding applied from configuration: {config.get('visual_buffer_days', 4)} days")
    print("-" * 75)

    # Trigger plotting workflow
    plot_basin_storm_event(basin_id, start_window, end_window, config)


if __name__ == "__main__":
    main()