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
from tqdm import tqdm

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


def calculate_threshold_hit_rate(predictions_np, actuals_np, threshold_value):
    """
    Calculates the Hit Rate (Sensitivity/TPR) for a given threshold.
    Formula: Hits (True Positives) / Total Actual Exceedances (True Positives + False Negatives)
    """
    # Find positions where the real river flow actually reached or crossed the threshold
    actual_exceedances_mask = (actuals_np >= threshold_value)
    total_actual_events = int(actual_exceedances_mask.sum())
    
    if total_actual_events == 0:
        return "0/0", 1.0  # Perfect score conceptually if no floods occurred

    # Count how many of those actual events were correctly predicted by the model
    predicted_exceedances_mask = (predictions_np >= threshold_value)
    true_hits = int((actual_exceedances_mask & predicted_exceedances_mask).sum())
    
    hit_rate_score = true_hits / total_actual_events
    count_string = f"{true_hits}/{total_actual_events}"
    
    return count_string, hit_rate_score


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
                count_str, score = calculate_threshold_hit_rate(preds_np, actuals_np, thresh_val)
                
                # Append metrics directly onto the dataframe structure
                df[f'hit_rate_{col_name}_{rp_year}yr_count'] = count_str
                df[f'hit_rate_{col_name}_{rp_year}yr_score'] = score

    # Enforce chronological sorting to ensure clean hydrograph continuity
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Commit report to disk
    csv_filename = os.path.join(output_dir, f"visual_report_basin_{basin}.csv")
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    print(f"Evaluation report compiled and exported to: {csv_filename}")


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

    # Step 3: Core loop executing sequential analysis basin-by-basin
    test_basins = test_dataset.basins
    print(f"[INFO] Identified {len(test_basins)} test basins. Starting execution loop...\n")

    for basin in test_basins:
        print(f"Processing Basin Context: {basin}")
        
        # Sequence inference processing loop
        basin_data = evaluate_basin_sequences(basin, test_dataset, model, device, config)
        
        if basin_data is None:
            print(f"  [WARNING] No continuous test windows found for basin {basin}. Skipping.")
            continue
            
        timestamps, actual_flows, pred_leads_dict = basin_data
        
        # Compile final consolidated data report
        build_and_export_report(basin, output_dir, timestamps, actual_flows, pred_leads_dict, config)

    print(f"\n[INFO] Testing loop completed successfully.")
    print(f"[INFO] All inference ready CSVs compiled inside: {output_dir}")


if __name__ == "__main__":
    main()