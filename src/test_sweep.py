"""
Module: test_sweep.py
Description: Evaluates the top N sweep runs on the test set and produces
             overlaid NSE empirical CDF plots, one per lead time.

For each top run the script:
  1. Fetches run config and best_val_loss from the W&B sweep via the API.
  2. Loads the saved best_model.pt checkpoint from the local run directory.
  3. Reconstructs the config by merging the run's hyperparameters over the base config.
  4. Builds the test dataset and runs basin-by-basin inference.
  5. Computes per-basin NSE for every lead time.
  6. Plots all runs as overlaid CDF curves (one figure per lead time).

Usage:
    python src/test_sweep.py <SWEEP_ID>
    python src/test_sweep.py <SWEEP_ID> --top 3
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from model import EALSTMModel
from dataset import IsraelBasinsDataset
from test import evaluate_basin_sequences


# One distinct color per rank slot
RANK_COLORS = ['#2b8cbe', '#cb181d', '#8856a7', '#41ab5d', '#f16913']


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ==============================================================================
# 1. W&B API — Fetch top runs
# ==============================================================================
def fetch_top_runs(project, sweep_id, n):
    """Return the top N runs from a sweep ranked by ascending best_val_loss."""
    api = wandb.Api()
    sweep = api.sweep(f"{project}/{sweep_id}")
    valid = [r for r in sweep.runs if 'best_val_loss' in r.summary]
    if not valid:
        raise RuntimeError("No completed runs with 'best_val_loss' found in this sweep.")
    return sorted(valid, key=lambda r: r.summary['best_val_loss'])[:n]


# ==============================================================================
# 2. Config reconstruction
# ==============================================================================
def apply_run_config(base_config, run):
    """Overlay the sweep run's hyperparameters onto the base config."""
    config = base_config.copy()
    # Skip private wandb keys (prefixed with _)
    config.update({k: v for k, v in run.config.items() if not k.startswith('_')})
    # initial_lr (scalar) overrides epoch-0 entry in the milestone dict
    if 'initial_lr' in config:
        lr_schedule = dict(config['learning_rate'])
        lr_schedule[0] = float(config['initial_lr'])
        config['learning_rate'] = lr_schedule
    return config


# ==============================================================================
# 3. Model loading
# ==============================================================================
def load_model(config, run_id, device):
    run_dir = config.get('run_dir', './runs/')
    ckpt_path = os.path.join(run_dir, config['experiment_name'], run_id, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    model = EALSTMModel(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


# ==============================================================================
# 4. Per-basin NSE computation
# ==============================================================================
def compute_nse_per_basin(basin, test_dataset, model, device, config):
    """
    Runs inference for one basin and returns a {lead: nse} dict.
    Returns None if the basin has no valid windows or zero variance.
    """
    result = evaluate_basin_sequences(basin, test_dataset, model, device, config)
    if result is None:
        return None

    timestamps, actual_flows, pred_leads_dict = result
    actuals = np.array(actual_flows)
    ss_tot = float(np.sum((actuals - actuals.mean()) ** 2))
    if ss_tot == 0:
        return None

    nse_by_lead = {}
    for lead in config['forecast_lead_times']:
        preds = np.array(pred_leads_dict[f'pred_lead_{lead}h'])
        ss_res = float(np.sum((actuals - preds) ** 2))
        nse_by_lead[lead] = 1.0 - ss_res / ss_tot
    return nse_by_lead


# ==============================================================================
# 5. CDF comparison plot
# ==============================================================================
def build_run_label(rank, run, config):
    h    = run.config.get('hidden_size', config.get('hidden_size', '?'))
    loss = run.config.get('loss', config.get('loss', '?'))
    seq  = run.config.get('seq_length', config.get('seq_length', '?'))
    val  = run.summary.get('best_val_loss', float('nan'))
    return f"#{rank}  h={h}  {loss}  seq={seq}  val={val:.4f}"


def plot_cdf_comparison(lead, run_nse_list, output_dir):
    """
    Draws overlaid NSE CDF curves for all runs on a single figure.

    Parameters
    ----------
    run_nse_list : list of (label: str, nse_values: list[float])
    """
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150, facecolor='#fafafa')
    ax.set_facecolor('#ffffff')

    plotted = 0
    for i, (label, nse_values) in enumerate(run_nse_list):
        values = sorted([v for v in nse_values if v is not None and not np.isnan(v)])
        if not values:
            continue
        n = len(values)
        cdf_y = [(j + 1) / n for j in range(n)]
        median_nse = float(np.median(values))
        color = RANK_COLORS[i % len(RANK_COLORS)]
        ax.plot(values, cdf_y, color=color, linewidth=2,
                label=f"{label}  [med={median_nse:.3f}  n={n}]")
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_title(f'NSE Empirical CDF — Lead +{lead}h  (Top Sweep Models)',
                 fontsize=12, fontweight='bold', pad=15, color='#2c3e50')
    ax.set_xlabel('NSE', fontsize=10.5, labelpad=8)
    ax.set_ylabel('Fraction of Basins', fontsize=10.5, labelpad=8)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=':', alpha=0.5, color='#b0b0b0')
    ax.legend(loc='lower right', frameon=True, facecolor='#ffffff',
              edgecolor='#e2e2e2', fontsize=8.5)
    fig.tight_layout()

    out_path = os.path.join(output_dir, f'sweep_nse_cdf_lead_{lead}h.png')
    fig.savefig(out_path, dpi=150, facecolor='#fafafa')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ==============================================================================
# 6. Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Evaluate top W&B sweep runs on the test set and plot NSE CDFs.'
    )
    parser.add_argument('sweep_id', help='W&B sweep ID')
    parser.add_argument('--top', type=int, default=5,
                        help='Number of top runs to compare (default: 5)')
    args = parser.parse_args()

    base_config = load_config('configs/config.yml')

    api_key = base_config.get('wandb_api_key')
    if api_key:
        wandb.login(key=api_key)

    project = base_config.get('wandb_project', 'flash-floods-israel')
    print(f"[INFO] Fetching top {args.top} runs from sweep {project}/{args.sweep_id} ...")
    top_runs = fetch_top_runs(project, args.sweep_id, args.top)
    print(f"[INFO] Found {len(top_runs)} valid completed runs.\n")

    device = torch.device('cpu')
    output_dir = os.path.join(
        base_config.get('run_dir', './runs/'),
        base_config['experiment_name'],
        f'sweep_comparison_{args.sweep_id}'
    )
    os.makedirs(output_dir, exist_ok=True)

    lead_times = base_config.get('forecast_lead_times', [0, 1, 2, 3])
    # {lead: [(label, [nse_values]), ...]}
    nse_per_lead = {lead: [] for lead in lead_times}

    for rank, run in enumerate(top_runs, start=1):
        val = run.summary.get('best_val_loss', float('nan'))
        print(f"=== Run #{rank}: {run.id}  (best_val_loss={val:.4f}) ===")

        try:
            config = apply_run_config(base_config, run)
            model = load_model(config, run.id, device)
        except FileNotFoundError as exc:
            print(f"  [SKIP] {exc}\n")
            continue

        print("  Building test dataset ...")
        test_dataset = IsraelBasinsDataset(split_type='test', config=config, use_basin_splits=False)
        print(f"  Test basins: {len(test_dataset.basins)}")

        nse_accumulator = {lead: [] for lead in lead_times}

        for basin in tqdm(test_dataset.basins, desc=f"  Run #{rank} basins", leave=False):
            nse_by_lead = compute_nse_per_basin(basin, test_dataset, model, device, config)
            if nse_by_lead is None:
                continue
            for lead in lead_times:
                if lead in nse_by_lead:
                    nse_accumulator[lead].append(nse_by_lead[lead])

        label = build_run_label(rank, run, config)
        for lead in lead_times:
            nse_per_lead[lead].append((label, nse_accumulator[lead]))

        medians = {
            lead: round(float(np.median(v)), 4) if v else None
            for lead, v in nse_accumulator.items()
        }
        print(f"  Median NSE per lead: {medians}\n")

    print("[INFO] Generating CDF comparison plots ...")
    for lead in lead_times:
        if any(vals for _, vals in nse_per_lead[lead]):
            plot_cdf_comparison(lead, nse_per_lead[lead], output_dir)
        else:
            print(f"  [SKIP] Lead +{lead}h — no NSE values collected.")

    print(f"\n[INFO] All plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
