"""
Module: test.py
Description: Evaluation Pipeline for Multi-Horizon EA-LSTM. Loads trained weights,
             extracts historic return period thresholds, calculates threshold hit rates,
             and exports comprehensive evaluation reports basin-by-basin.
"""

import argparse
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
import plot_hydrographs as ph
import find_flood_events as ffe
import find_predicted_flood_events as pfe
import compare_flood_events as cfe
import peaks_analyze as pa


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
    Loads cross-checked return period thresholds from the combined return-period file
    (see new_return_periods.py). Maps return period years to their physical streamflow
    value (m3/sec). Skips RP columns with no usable value (missing row/column, NaN, or
    a non-positive sentinel like the -2 "no data" marker).
    """
    combined_path = config['hourly_flow_return_periods_combined']
    if not os.path.exists(combined_path):
        return {}

    combined_df = pd.read_csv(combined_path, dtype={'basin_id': str})
    row = combined_df[combined_df['basin_id'] == str(basin_id)]
    if row.empty:
        return {}
    row = row.iloc[0]

    thresholds = {}
    for rp_year in config.get('return_periods_years', [2, 5]):
        col = f'RP{rp_year}'
        if col in row.index and pd.notna(row[col]) and row[col] > 0:
            thresholds[rp_year] = float(row[col])
    return thresholds


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
    # Per-lead actual flows: actual_lead_kh stores observed flow at t+k
    actual_leads_dict = {f"actual_lead_{lead}h": [] for lead in config['forecast_lead_times']}
    pred_leads_dict   = {f"pred_lead_{lead}h":   [] for lead in config['forecast_lead_times']}

    with torch.no_grad():
        for idx in tqdm(basin_indices, desc=f"  Evaluating Basin {basin}", leave=False):
            sample = test_dataset[idx]

            # Enforce batch dimension [1, seq_length, features]
            x_dynamic = sample['dynamic'].unsqueeze(0).to(device)
            x_static = sample['static'].unsqueeze(0).to(device)
            target_raw = sample['target_raw']  # raw m3/s: [actual_t+0, actual_t+1, actual_t+2, ...]
            basin_mean = sample['basin_mean'].item()
            basin_std = sample['basin_std'].item()

            # Forward Pass (Blinded from IDs and timestamps). The model predicts in
            # per-basin z-scored space, so de-normalize before clamping to physical
            # non-negative streamflow (clamp here rather than in the model so training
            # and its gradients remain unconstrained).
            raw_prediction = model(x_dynamic, x_static).squeeze(0) * basin_std + basin_mean
            prediction = torch.clamp(raw_prediction, min=0).cpu().numpy()

            # Metadata tracking collection (Outside model execution space)
            timestamps.append(test_dataset.sample_date_mappings[idx])

            # Unpack both actual and predicted flows per lead time
            for i, lead in enumerate(config['forecast_lead_times']):
                actual_leads_dict[f"actual_lead_{lead}h"].append(target_raw[i].item())
                pred_leads_dict[f"pred_lead_{lead}h"].append(prediction[i])

    return timestamps, actual_leads_dict, pred_leads_dict


def build_and_export_report(basin, output_dir, timestamps, actual_leads_dict, pred_leads_dict, config):
    """
    Assembles predictions, injects static GEV thresholds, computes localized hit-rate
    metrics per lead time, and commits a clean sorted report dataframe to a CSV file.
    """
    # Initialize dataframe with per-lead actual columns
    df = pd.DataFrame({'timestamp': timestamps})
    for col_name, values in actual_leads_dict.items():
        df[col_name] = values
    # 'actual_flow' alias for the model's primary (first configured) lead -
    # used by hydrograph plots and threshold metrics
    primary_lead = config['forecast_lead_times'][0]
    df['actual_flow'] = df[f'actual_lead_{primary_lead}h']

    # Append multi-horizon prediction outputs
    for col_name, values in pred_leads_dict.items():
        df[col_name] = values

    # Ingest historical extreme value return period benchmarks for this basin
    thresholds = load_basin_return_periods(basin, config)
    actuals_np = df['actual_flow'].to_numpy()
    
    # Inject thresholds and map hit rate statistics dynamically
    for rp_year in config.get('return_periods_years', [2, 5, 10]):
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
    active_leads = config.get('forecast_lead_times', [0, 1, 2, 3])
    rp_years = config.get('return_periods_years', [2, 5, 10])

    # NSE per lead time: compare pred_lead_kh against actual_lead_kh (observed flow at t+k)
    nse_per_lead = {}
    for lead in active_leads:
        actual_col = f"actual_lead_{lead}h"
        pred_col   = f"pred_lead_{lead}h"
        if actual_col in df.columns and pred_col in df.columns:
            actuals_np  = df[actual_col].to_numpy()
            preds_np    = df[pred_col].to_numpy()
            ss_tot = float(np.sum((actuals_np - actuals_np.mean()) ** 2))
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


def main(config_path="configs/config.yml"):
    # Step 1: Config ingestion and model setup
    config = load_config(config_path)
    model, device, exp_dir = setup_evaluation(config)

    # Step 2: Initialize sterile test split tracking arrays
    print("[INFO] Constructing test datasets and extracting sequential metadata...")
    test_dataset = IsraelBasinsDataset(split_type='test', config=config,
                                       use_basin_splits=config.get('use_basin_splits', True))

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
    nse_accumulator = {lead: [] for lead in config.get('forecast_lead_times', [0, 1, 2, 3])}

    for basin in test_basins:
        print(f"Processing Basin Context: {basin}")

        # Sequence inference processing loop
        basin_data = evaluate_basin_sequences(basin, test_dataset, model, device, config)

        if basin_data is None:
            print(f"  [WARNING] No continuous test windows found for basin {basin}. Skipping.")
            continue

        timestamps, actual_leads_dict, pred_leads_dict = basin_data

        # Compile final consolidated data report
        df = build_and_export_report(basin, output_dir, timestamps, actual_leads_dict, pred_leads_dict, config)

        if df is None:
            continue

        # Generate per-basin summary table (always, not gated on W&B)
        summary_df = generate_basin_summary_table(basin, df, config, output_dir)

        # Collect per-lead NSE for cross-basin CDF plots
        for lead in nse_accumulator:
            lead_rows = summary_df[summary_df['Lead Time (h)'] == lead]
            if not lead_rows.empty:
                nse_val = lead_rows['NSE'].iloc[0]
                if nse_val is not None and not (isinstance(nse_val, float) and np.isnan(nse_val)):
                    nse_accumulator[lead].append(nse_val)

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

        # Generate and upload static hydrograph
        fig = ph.plot_basin_full_history(basin, df, config)
        wandb.log({f"test/hydrograph/{basin}": wandb.Image(fig)})
        plt.close(fig)

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

        # Scan real + predicted flood events, then compare them at prediction_threshold
        # RP to classify each as TP/FP/FN - one graph per event, no duplicates.
        print("\n[INFO] Scanning for flood events across all basins...")
        ffe.main(config=config, basin_ids=test_basins)
        print("\n[INFO] Scanning for predicted flood events across all basins...")
        pfe.main(config=config, basin_ids=test_basins)
        print("\n[INFO] Comparing real vs. predicted flood events...")
        comparison_path = cfe.main(config=config, basin_ids=test_basins)

        print("\n[INFO] Computing peak timing/magnitude analysis...")
        pa.main(config=config, basin_ids=test_basins)

        if os.path.exists(comparison_path):
            comparison_df = pd.read_csv(comparison_path, dtype={'basin_id': str})
            for _, event in comparison_df.iterrows():
                basin = str(event['basin_id'])
                label = event['label']
                idx = int(event['event_idx'])
                fig = ph.plot_basin_storm_event(basin, event['core_start'], event['core_end'],
                                                 config, label=label, event_idx=idx)
                if fig is not None:
                    key = f"test/flood_events/{basin}/event{idx}_{label}"
                    wandb.log({key: wandb.Image(fig)})
                    plt.close(fig)

        # Generate and upload NSE empirical CDF charts (one per lead time)
        print("\n[INFO] Generating NSE CDF charts per lead time...")
        for lead, nse_vals in nse_accumulator.items():
            cdf_fig = ph.plot_nse_cdf(lead, nse_vals, config)
            if cdf_fig:
                wandb.log({f"test/nse_cdf/lead_{lead}h": wandb.Image(cdf_fig)})
                plt.close(cdf_fig)

        wandb.finish()

    print(f"\n[INFO] Testing loop completed successfully.")
    print(f"[INFO] All inference ready CSVs compiled inside: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained EA-LSTM model.")
    parser.add_argument("--config", type=str, default="configs/config.yml",
                         help="Path to the YAML config file matching the run to evaluate.")
    args = parser.parse_args()
    main(args.config)