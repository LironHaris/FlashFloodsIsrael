"""
Module: quick_test.py
Description: Lightweight Evaluation Pipeline for an arbitrary checkpoint.
             Like test.py, but loads weights from config['checkpoint_path'] instead of
             run_dir/experiment_name/best_model.pt, and only produces the NSE empirical
             CDF (per lead time) and flood-event hydrographs - no summary tables and no
             full-test-period hydrographs.
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from model import EALSTMModel
from dataset import IsraelBasinsDataset
from test import evaluate_basin_sequences, build_and_export_report
import plot_hydrographs as ph
import find_flood_events as ffe


def load_config(yaml_path):
    """Load the YAML configuration file safely."""
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def setup_evaluation_from_checkpoint(config):
    """
    Handles model reconstruction and loads weights from config['checkpoint_path'],
    instead of test.py's fixed run_dir/experiment_name/best_model.pt location.
    Inference is forced on CPU to maintain simplicity and memory stability.
    """
    device = torch.device('cpu')
    print(f"[INFO] Evaluation environment locked on hardware target: {device}")

    print("[INFO] Reconstructing EA-LSTM architecture dynamically...")
    model = EALSTMModel(config).to(device)

    checkpoint_path = config.get('checkpoint_path')
    if not checkpoint_path:
        raise ValueError("config['checkpoint_path'] must be set to use quick_test.py.")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint_path not found: {checkpoint_path}")

    print(f"[INFO] Loading trained weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Lock dropout and state statistics
    print(f"[INFO] Model successfully deployed from checkpoint epoch {checkpoint.get('epoch', 'N/A')}")

    run_dir = config.get('run_dir', './runs/')
    exp_dir = os.path.join(run_dir, config['experiment_name'])
    return model, device, exp_dir


def compute_nse_per_lead(actual_leads_dict, pred_leads_dict, forecast_lead_times):
    """Same formula as test.py's generate_basin_summary_table, computed directly
    from evaluate_basin_sequences' output without building a summary table."""
    nse_per_lead = {}
    for lead in forecast_lead_times:
        actuals_np = np.array(actual_leads_dict[f"actual_lead_{lead}h"])
        preds_np = np.array(pred_leads_dict[f"pred_lead_{lead}h"])
        ss_tot = float(np.sum((actuals_np - actuals_np.mean()) ** 2))
        if ss_tot == 0:
            continue
        ss_res = float(np.sum((actuals_np - preds_np) ** 2))
        nse_per_lead[lead] = 1.0 - ss_res / ss_tot
    return nse_per_lead


def main():
    # Step 1: Config ingestion and model setup from the configured checkpoint
    config = load_config("configs/config.yml")
    model, device, exp_dir = setup_evaluation_from_checkpoint(config)

    # Step 2: Initialize sterile test split tracking arrays
    print("[INFO] Constructing test datasets and extracting sequential metadata...")
    test_dataset = IsraelBasinsDataset(split_type='test', config=config, use_basin_splits=False)

    # Set up dedicated output folder inside run directory (same convention as test.py)
    output_dir = os.path.join(exp_dir, "visualization_reports")
    os.makedirs(output_dir, exist_ok=True)

    use_wandb = config.get('use_wandb', False) and WANDB_AVAILABLE
    if use_wandb:
        api_key = config.get('wandb_api_key')
        if api_key:
            wandb.login(key=api_key)
        wandb.init(
            project=config.get('wandb_project', 'flash-floods-israel'),
            name=f"{config['experiment_name']}_quicktest",
            config=config,
        )

    # Step 3: Core loop executing sequential analysis basin-by-basin
    test_basins = test_dataset.basins
    print(f"[INFO] Identified {len(test_basins)} test basins. Starting execution loop...\n")

    nse_accumulator = {lead: [] for lead in config.get('forecast_lead_times', [0, 1, 2, 3])}

    for basin in test_basins:
        print(f"Processing Basin Context: {basin}")

        # Sequence inference processing loop
        basin_data = evaluate_basin_sequences(basin, test_dataset, model, device, config)

        if basin_data is None:
            print(f"  [WARNING] No continuous test windows found for basin {basin}. Skipping.")
            continue

        timestamps, actual_leads_dict, pred_leads_dict = basin_data

        # Compile report (needed by find_flood_events.py and plot_basin_storm_event,
        # both of which read this exact CSV from disk)
        build_and_export_report(basin, output_dir, timestamps, actual_leads_dict, pred_leads_dict, config)

        # Collect per-lead NSE for cross-basin CDF plots, computed directly - no summary table
        for lead, nse_val in compute_nse_per_lead(actual_leads_dict, pred_leads_dict, config['forecast_lead_times']).items():
            nse_accumulator[lead].append(nse_val)

    if use_wandb:
        # Scan for flood events and plot each one
        print("\n[INFO] Scanning for flood events across all basins...")
        ffe.main(config=config, basin_ids=test_basins)

        events_path = config['find_flood_events_output']
        if os.path.exists(events_path):
            events_df = pd.read_csv(events_path)
            for _, event in events_df.iterrows():
                basin = str(event['basin_id'])
                fig = ph.plot_basin_storm_event(basin, event['core_start'], event['core_end'], config)
                if fig is not None:
                    rp = int(event['return_period_years'])
                    idx = int(event['event_idx'])
                    key = f"quicktest/flood_events/{basin}/rp{rp}yr_event{idx}"
                    wandb.log({key: wandb.Image(fig)})
                    plt.close(fig)

        # Generate and upload NSE empirical CDF charts (one per lead time)
        print("\n[INFO] Generating NSE CDF charts per lead time...")
        for lead, nse_vals in nse_accumulator.items():
            cdf_fig = ph.plot_nse_cdf(lead, nse_vals, config)
            if cdf_fig:
                wandb.log({f"quicktest/nse_cdf/lead_{lead}h": wandb.Image(cdf_fig)})
                plt.close(cdf_fig)

        wandb.finish()

    print(f"\n[INFO] Quick-test loop completed successfully.")
    print(f"[INFO] All inference ready CSVs compiled inside: {output_dir}")


if __name__ == "__main__":
    main()
