"""
Module: MSE_analysis.py
Description: Shared engine behind MSE_analysis_train.py / MSE_analysis_test.py -
             the train/test counterpart of train_sanity.py's validation loss
             breakdown + val_usage_audit.py, run against an already-trained
             model (config['checkpoint_path'], or
             run_dir/experiment_name/best_model.pt if unset) instead of during
             training. One forward pass over the split attributes the loss to
             basins and hydrological years (CSVs + MSE pie charts, top-N
             basins), then the model-independent usage audit (basin-hour pairs
             used / non-zero / threshold-crossed) runs on the same split, with
             the top-N basins ranked by that split's own loss.
             For the train split, every train period is clipped so nothing
             starts before hydrological year 2000 (see TRAIN_START_CUTOFF).
"""

import copy
import os

import torch
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from torch.utils.data import DataLoader

from train import load_config, get_loss_criterion
from dataset import IsraelBasinsDataset, get_dataloader, _resolve_num_workers
from model import EALSTMModel
from train_sanity import validate_epoch_with_breakdown, to_value_and_pct, plot_breakdown_pie
from val_usage_audit import (
    compute_val_usage_breakdown, top_n_basin_ids_by_value, write_usage_csvs, log_usage_to_wandb,
)

# Start of hydrological year 2000 (get_hydrological_year labels Oct 1999-Sep 2000 as 2000).
TRAIN_START_CUTOFF = '1999-10-01 08:00:00'


def clip_train_periods(train_periods, cutoff=TRAIN_START_CUTOFF):
    """
    Returns a copy of config['train_periods'] where no period starts before
    cutoff: an empty start_date (= each basin's earliest record) or an
    earlier one becomes cutoff, so each basin still starts as early as its
    own data allows but never before cutoff. Periods ending before cutoff
    are dropped entirely.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    clipped = []
    for period in train_periods:
        if pd.Timestamp(period['end_date']) < cutoff_ts:
            continue
        start = period.get('start_date')
        if not start or pd.Timestamp(start) < cutoff_ts:
            start = cutoff
        clipped.append({'start_date': start, 'end_date': period['end_date']})
    if not clipped:
        raise ValueError(f"No train period ends after {cutoff} - nothing to analyze.")
    return clipped


def build_split_loader(split_type, config, use_basin_splits):
    """
    Unshuffled, drop_last=False loader over an IsraelBasinsDataset, as
    validate_epoch_with_breakdown / compute_val_usage_breakdown's running
    sample_offset alignment requires. The regular train loader shuffles, so
    the train split gets its own DataLoader (over the clipped train periods).
    """
    if split_type == 'test':
        return get_dataloader(split_type='test', config=config, use_basin_splits=use_basin_splits)

    train_config = copy.deepcopy(config)
    train_config['train_periods'] = clip_train_periods(config['train_periods'])
    print(f"[INFO] Train periods clipped to start no earlier than {TRAIN_START_CUTOFF}: "
          f"{[(p['start_date'], p['end_date']) for p in train_config['train_periods']]}")
    train_dataset = IsraelBasinsDataset('train', train_config, use_basin_splits=use_basin_splits)
    return DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=_resolve_num_workers(config),
        drop_last=False,
    )


def load_trained_model(config, exp_dir):
    """Rebuilds the EA-LSTM and loads config['checkpoint_path'] if set (so the
    analysis can write into its own experiment dir while evaluating another
    run's weights), else exp_dir/best_model.pt (same location as test.py).
    Runs on config['device'] with train_sanity.py's CPU fallback."""
    device_str = config.get('device', 'cpu')
    device = torch.device(device_str if torch.cuda.is_available() or device_str == 'cpu' else 'cpu')
    print(f"[INFO] Execution target hardware configured to: {device}")

    best_checkpoint_path = config.get('checkpoint_path') or os.path.join(exp_dir, "best_model.pt")
    if not os.path.exists(best_checkpoint_path):
        raise FileNotFoundError(f"Missing trained weights at {best_checkpoint_path}. Train the model first.")

    model = EALSTMModel(config).to(device)
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"[INFO] Loaded {best_checkpoint_path} (epoch {checkpoint.get('epoch', 'N/A')})")
    return model, device


def write_loss_csv(sq_err_sum, values, pcts, total_elements, group_col, csv_path):
    rows = [
        {group_col: group, 'sq_err_sum': sq_sum, 'total_elements': total_elements,
         'value': values[group], 'percentage': pcts[group]}
        for group, sq_sum in sorted(sq_err_sum.items(), key=lambda kv: kv[1], reverse=True)
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def run_mse_analysis(config_path, split_type):
    if split_type not in ('train', 'test'):
        raise ValueError("split_type must be either 'train' or 'test'")
    config = load_config(config_path)

    cv_config = config.get('cross_validation', {}) or {}
    if cv_config.get('enabled', False):
        raise ValueError(
            f"MSE_analysis_{split_type}.py does not support cross_validation-enabled configs: it "
            "relies on IsraelBasinsDataset's sample_basin_mappings/sample_date_mappings, not present "
            "on the CV fold path. Use a config with cross_validation.enabled: false."
        )

    use_spatial = config.get('use_basin_splits', True)
    hydro_year_start_month = config.get('hydro_year_start_month', 10)
    top_n_basins = config.get('sanity_top_n_basins', 10)
    split_label = split_type.capitalize()

    run_dir = config.get('run_dir', './runs/')
    exp_dir = os.path.join(run_dir, config['experiment_name'])
    sanity_dir = os.path.join(exp_dir, "sanity_analysis")
    pies_dir = os.path.join(sanity_dir, "pies")
    os.makedirs(pies_dir, exist_ok=True)

    loader = build_split_loader(split_type, config, use_spatial)
    model, device = load_trained_model(config, exp_dir)
    criterion = get_loss_criterion(config.get('loss', 'MSE'), config)

    print(f"[INFO] Computing {split_type} loss breakdown by basin / hydrological year...")
    split_loss, basin_sq_err_sum, year_sq_err_sum, total_elements = validate_epoch_with_breakdown(
        model, loader, criterion, device, hydro_year_start_month)
    print(f"[INFO] {split_label} loss: {split_loss:.5f}")

    basin_values, basin_pcts = to_value_and_pct(basin_sq_err_sum, total_elements)
    year_values, year_pcts = to_value_and_pct(year_sq_err_sum, total_elements)
    write_loss_csv(basin_sq_err_sum, basin_values, basin_pcts, total_elements, 'basin_id',
                   os.path.join(sanity_dir, f"{split_type}_loss_by_basin.csv"))
    write_loss_csv(year_sq_err_sum, year_values, year_pcts, total_elements, 'year',
                   os.path.join(sanity_dir, f"{split_type}_loss_by_year.csv"))

    year_fig = plot_breakdown_pie(
        {k: (year_values[k], year_pcts[k]) for k in year_values},
        f"{split_label} Loss by Hydrological Year", os.path.join(pies_dir, f"{split_type}_year_pie.png"),
        top_n=None)
    basin_fig = plot_breakdown_pie(
        {k: (basin_values[k], basin_pcts[k]) for k in basin_values},
        f"{split_label} Loss by Basin", os.path.join(pies_dir, f"{split_type}_basin_pie.png"),
        top_n=top_n_basins)

    print(f"[INFO] Computing {split_type}-set usage audit (basin-hour pairs, non-zero, threshold-crossed)...")
    top_basin_ids = top_n_basin_ids_by_value(basin_values, top_n_basins)
    year_counts, basin_counts = compute_val_usage_breakdown(loader, config, hydro_year_start_month)
    year_df, basin_df = write_usage_csvs(year_counts, basin_counts, top_basin_ids, sanity_dir,
                                         split_name=split_type)
    print(f"[INFO] {split_label} loss/usage CSVs and pies written inside: {sanity_dir}")

    if config.get('use_wandb', False):
        api_key = config.get('wandb_api_key')
        if api_key:
            wandb.login(key=api_key)
        wandb.init(
            project=config.get('wandb_project', 'flash-floods-israel'),
            name=config['experiment_name'],
        )
        wandb.log({
            f'{split_type}_loss': split_loss,
            f'{split_type}_sanity/year_pie': wandb.Image(year_fig),
            f'{split_type}_sanity/basin_pie': wandb.Image(basin_fig),
        })
        log_usage_to_wandb(year_df, basin_df, key_prefix=f'{split_type}_usage_audit')
        wandb.finish()

    plt.close(year_fig)
    plt.close(basin_fig)
