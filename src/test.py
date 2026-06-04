"""
Module: test.py
Description: Evaluation Pipeline for Multi-Horizon EA-LSTM. Loads trained weights,
             extracts historic return period thresholds, calculates threshold hit rates,
             and exports comprehensive evaluation reports basin-by-basin.
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import local pipeline components
from model import EALSTMModel
from dataset import IsraelBasinsDataset


def load_config(yaml_path):
    """Load the YAML configuration file safely."""
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def setup_evaluation(config):
    """
    Handles model reconstruction and loads the optimal trained weights.
    Inference is forced on CPU to maintain simplicity and memory stability.
    """
    device = torch.device('cpu')
    print(f"[INFO] Evaluation environment locked on hardware target: {device}")

    print("[INFO] Reconstructing EA-LSTM architecture dynamically...")
    model = EALSTMModel(config).to(device)
    
    run_dir = config.get('run_dir', './runs/')
    exp_dir = os.path.join(run_dir, config['experiment_name'])
    best_checkpoint_path = os.path.join(exp_dir, "best_model.pt")
    
    if not os.path.exists(best_checkpoint_path):
        raise FileNotFoundError(f"Missing trained weights at {best_checkpoint_path}. Train the model first.")
        
    print(f"[INFO] Loading trained weights from {best_checkpoint_path}...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Lock dropout and state statistics
    print(f"[INFO] Model successfully deployed from validation Epoch {checkpoint.get('epoch', 'N/A')}")
    
    return model, device, exp_dir


def load_basin_return_periods(basin_id, config):
    """
    Loads historical GEV return period thresholds computed during preprocessing.
    Maps return period years to their physical streamflow value (m3/sec).
    """
    eva_dir = config.get('return_periods_output_dir', 'data/processed/return_periods')
    gev_path = os.path.join(eva_dir, str(basin_id), 'Hourly_Flow_theoretical_GEV.csv')
    
    if not os.path.exists(gev_path):
        return {}
        
    # Read GEV file and convert to dictionary mapping {RP_Year: Flow_Value}
    gev_df = pd.read_csv(gev_path)
    return dict(zip(gev_df['Return_Period_Years'], gev_df['Theoretical_Value']))


def calculate_threshold_metrics(predictions_np, actuals_np, threshold_value):
    """
    Calculates Hit Rate (TPR), False Alarms, and hit counts for a given threshold.
    Returns: (count_string, hit_rate_score, false_alarm_count)
      - count_string: "hits/total_actual_events" e.g. "25/30"
      - hit_rate_score: true_hits / total_actual_events
      - false_alarm_count: predicted exceedances where actual did NOT exceed threshold
    """
    actual_exceedances_mask = (actuals_np >= threshold_value)
    total_actual_events = int(actual_exceedances_mask.sum())

    if total_actual_events == 0:
        return "0/0", 1.0, 0

    predicted_exceedances_mask = (predictions_np >= threshold_value)
    true_hits = int((actual_exceedances_mask & predicted_exceedances_mask).sum())
    false_alarms = int((~actual_exceedances_mask & predicted_exceedances_mask).sum())

    hit_rate_score = true_hits / total_actual_events
    count_string = f"{true_hits}/{total_actual_events}"

    return count_string, hit_rate_score, false_alarms


def evaluate_basin_sequences(basin, test_dataset, model, device, config):
    """
    Runs blinded sequential forward passes over a single targeted basin.
    Extracts predictions across all multi-horizon lead times and gathers core vectors.
    """
    # Isolate look-back windows mapping strictly to this basin identifier
    basin_indices = [i for i, b in enumerate(test_dataset.sample_basin_mappings) if b == basin]
    
    if not basin_indices:
        return None

    timestamps = []
    actual_flows = []
    # Instantiate internal collection arrays for each lead time
    pred_leads_dict = {f"pred_lead_{lead}h": [] for lead in config['forecast_lead_times']}

    with torch.no_grad():
        for idx in tqdm(basin_indices, desc=f"  Evaluating Basin {basin}", leave=False):
            sample = test_dataset[idx]
            
            # Enforce batch dimension [1, seq_length, features]
            x_dynamic = sample['dynamic'].unsqueeze(0).to(device)
            x_static = sample['static'].unsqueeze(0).to(device)
            target = sample['target']  # Extracted ground truth vector
            
            # Forward Pass (Blinded from IDs and timestamps)
            prediction = model(x_dynamic, x_static).squeeze(0).numpy()
            
            # Metadata tracking collection (Outside model execution space)
            timestamps.append(test_dataset.sample_date_mappings[idx])
            actual_flows.append(target[0].item())  # Ground truth flow at current hour
            
            # Unpack lead-time prediction slices
            for i, lead in enumerate(config['forecast_lead_times']):
                pred_leads_dict[f"pred_lead_{lead}h"].append(prediction[i])
                
    return timestamps, actual_flows, pred_leads_dict


def build_and_export_report(basin, output_dir, timestamps, actual_flows, pred_leads_dict, config):
    """
    Assembles predictions, injects static GEV thresholds, computes localized hit-rate 
    metrics per lead time, and commits a clean sorted report dataframe to a CSV file.
    """
    # Initialize basic dataframe structure
    df = pd.DataFrame({
        'timestamp': timestamps,
        'actual_flow': actual_flows
    })
    
    # Append multi-horizon outputs
    for col_name, values in pred_leads_dict.items():
        df[col_name] = values
        
    # Ingest historical extreme value return period benchmarks for this basin
    thresholds = load_basin_return_periods(basin, config)
    actuals_np = df['actual_flow'].to_numpy()
    
    # Inject thresholds and map hit rate statistics dynamically
    for rp_year in config.get('return_periods_years', [2, 5, 10, 15, 20, 30]):
        if rp_year in thresholds:
            thresh_val = thresholds[rp_year]
            df[f'threshold_{rp_year}yr_rp'] = thresh_val  # Static horizontal bar mapping
            
            # Evaluate hit rates across every active prediction horizon
            for col_name in list(pred_leads_dict.keys()):
                preds_np = df[col_name].to_numpy()
                count_str, score, false_alarms = calculate_threshold_metrics(preds_np, actuals_np, thresh_val)

                # Append metrics directly onto the dataframe structure
                df[f'hit_rate_{col_name}_{rp_year}yr_count'] = count_str
                df[f'hit_rate_{col_name}_{rp_year}yr_score'] = score
                df[f'false_alarms_{col_name}_{rp_year}yr'] = false_alarms

    # Enforce chronological sorting to ensure clean hydrograph continuity
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Commit report to disk
    csv_filename = os.path.join(output_dir, f"visual_report_basin_{basin}.csv")
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    print(f"Evaluation report compiled and exported to: {csv_filename}")
    return df


def generate_basin_summary_table(basin, df, config, output_dir):
    """
    Builds a per-basin summary table (one row per lead-time × return-period pair).
    Columns: Lead Time (h), Return Period (yr), Threshold (m³/s),
             Flow Exceedances, Hits, Hit Rate, False Alarms, NSE.
    Saves to CSV and prints to stdout. Returns the summary DataFrame.
    """
    active_leads = config.get('forecast_lead_times', [0, 6, 24])
    rp_years = config.get('return_periods_years', [2, 5, 10, 15, 20, 30])
    actuals_np = df['actual_flow'].to_numpy()
    mean_actual = actuals_np.mean()
    ss_tot = float(np.sum((actuals_np - mean_actual) ** 2))

    # Pre-compute NSE per lead time (full test period, not threshold-specific)
    nse_per_lead = {}
    for lead in active_leads:
        col = f"pred_lead_{lead}h"
        if col in df.columns:
            preds_np = df[col].to_numpy()
            ss_res = float(np.sum((actuals_np - preds_np) ** 2))
            nse_per_lead[lead] = round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else float('nan')

    rows = []
    for lead in active_leads:
        for rp in rp_years:
            thresh_col = f"threshold_{rp}yr_rp"
            count_col = f"hit_rate_pred_lead_{lead}h_{rp}yr_count"
            score_col = f"hit_rate_pred_lead_{lead}h_{rp}yr_score"
            fa_col = f"false_alarms_pred_lead_{lead}h_{rp}yr"

            if count_col not in df.columns:
                continue

            thresh_val = round(float(df[thresh_col].iloc[0]), 2) if thresh_col in df.columns else None
            count_str = str(df[count_col].iloc[0])
            score = float(df[score_col].iloc[0])
            false_alarms = int(df[fa_col].iloc[0]) if fa_col in df.columns else None

            hits, total_events = (int(x) for x in count_str.split('/'))

            rows.append({
                'Lead Time (h)': lead,
                'Return Period (yr)': rp,
                'Threshold (m³/s)': thresh_val,
                'Flow Exceedances': total_events,
                'Hits': hits,
                'Hit Rate': f"{score:.1%}",
                'False Alarms': false_alarms,
                'NSE': nse_per_lead.get(lead),
            })

    summary_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"summary_table_basin_{basin}.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  Basin {basin} — Summary Table:")
    print(summary_df.to_string(index=False))
    print(f"  Saved to: {csv_path}\n")
    return summary_df


def _plot_basin_timeseries(basin, df, config):
    """Generates a full-period time-series figure for a basin and returns it as a wandb.Image."""
    lead_colors = {0: '#2b8cbe', 1: '#8856a7', 6: '#cb181d', 12: '#86592d', 24: '#f16913'}
    lead_styles = {0: '-', 1: '--', 6: '--', 12: ':', 24: ':'}
    threshold_colors = {2: '#bae4b3', 5: '#74c476', 10: '#ef3b2c', 20: '#990000', 30: '#67000d'}

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#fafafa")
    ax.set_facecolor("#ffffff")

    ax.plot(df['timestamp'], df['actual_flow'], color="#1e1e1e", linewidth=1.8,
            label="Actual Streamflow")

    active_leads = config.get('forecast_lead_times', [0, 6, 24])
    for lead in active_leads:
        col = f"pred_lead_{lead}h"
        if col in df.columns:
            ax.plot(df['timestamp'], df[col],
                    color=lead_colors.get(lead, '#7f7f7f'),
                    linestyle=lead_styles.get(lead, '-'),
                    linewidth=1.2, alpha=0.8, label=f"+{lead}h Lead")

    rp_years = config.get('return_periods_years', [2, 5, 10, 20])
    for rp in rp_years:
        thresh_col = f"threshold_{rp}yr_rp"
        if thresh_col in df.columns:
            thresh_val = df[thresh_col].iloc[0]
            if thresh_val > 0:
                ax.axhline(y=thresh_val, color=threshold_colors.get(rp, '#d9d9d9'),
                           linestyle="-.", linewidth=1.0, alpha=0.85,
                           label=f"{rp}yr RP ({thresh_val:.1f} m³/s)")

    ax.set_title(f"Test Period — Basin {basin}", fontsize=11, fontweight='bold', color="#2c3e50")
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Discharge (m³/s)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4, color="#b0b0b0")
    ax.legend(fontsize=8, loc="upper right", frameon=True)
    plt.tight_layout()

    image = wandb.Image(fig, caption=f"Basin {basin} — full test period")
    plt.close(fig)
    return image


def main():
    # Step 1: Config ingestion and model setup
    config = load_config("configs/config.yml")
    model, device, exp_dir = setup_evaluation(config)

    # Step 2: Initialize sterile test split tracking arrays
    print("[INFO] Constructing test datasets and extracting sequential metadata...")
    test_dataset = IsraelBasinsDataset(split_type='test', config=config, use_basin_splits=False)

    # Set up dedicated output folder inside run directory
    output_dir = os.path.join(exp_dir, "visualization_reports")
    os.makedirs(output_dir, exist_ok=True)

    use_wandb = config.get('use_wandb', False) and WANDB_AVAILABLE
    if use_wandb:
        api_key = config.get('wandb_api_key')
        if api_key:
            wandb.login(key=api_key)
        wandb.init(
            project=config.get('wandb_project', 'flash-floods-israel'),
            name=f"{config['experiment_name']}_test",
            config=config,
        )

    # Step 3: Core loop executing sequential analysis basin-by-basin
    test_basins = test_dataset.basins
    print(f"[INFO] Identified {len(test_basins)} test basins. Starting execution loop...\n")

    all_hit_rate_rows = []   # accumulates rows for wandb.Table
    score_accumulator = {}   # {col_name: [values]} for computing means

    for basin in test_basins:
        print(f"Processing Basin Context: {basin}")

        # Sequence inference processing loop
        basin_data = evaluate_basin_sequences(basin, test_dataset, model, device, config)

        if basin_data is None:
            print(f"  [WARNING] No continuous test windows found for basin {basin}. Skipping.")
            continue

        timestamps, actual_flows, pred_leads_dict = basin_data

        # Compile final consolidated data report
        df = build_and_export_report(basin, output_dir, timestamps, actual_flows, pred_leads_dict, config)

        if df is None:
            continue

        # Generate per-basin summary table (always, not gated on W&B)
        summary_df = generate_basin_summary_table(basin, df, config, output_dir)

        if not use_wandb:
            continue

        # Extract hit rate score columns for global aggregation table
        score_cols = [c for c in df.columns if c.endswith('_score') and c.startswith('hit_rate_')]
        basin_metric_dict = {'basin': str(basin)}
        for col in score_cols:
            val = float(df[col].iloc[0])
            basin_metric_dict[col] = val
            score_accumulator.setdefault(col, []).append(val)
        all_hit_rate_rows.append(basin_metric_dict)

        # Upload per-basin summary table to W&B
        cols = list(summary_df.columns)
        wtable = wandb.Table(columns=cols)
        for _, row in summary_df.iterrows():
            wtable.add_data(*row.tolist())
        wandb.log({f"test/summary_table/{basin}": wtable})

        # Generate and upload time-series hydrograph image
        image = _plot_basin_timeseries(basin, df, config)
        wandb.log({f"test/hydrograph/{basin}": image})

    if use_wandb:
        # Log hit rates table
        if all_hit_rate_rows:
            columns = list(all_hit_rate_rows[0].keys())
            table = wandb.Table(columns=columns)
            for row in all_hit_rate_rows:
                table.add_data(*[row[c] for c in columns])
            wandb.log({"test/hit_rates_table": table})

            # Log aggregate means to summary
            for col, values in score_accumulator.items():
                wandb.run.summary[f"test_mean/{col}"] = float(np.mean(values))

        wandb.finish()

    print(f"\n[INFO] Testing loop completed successfully.")
    print(f"[INFO] All inference ready CSVs compiled inside: {output_dir}")


if __name__ == "__main__":
    main()