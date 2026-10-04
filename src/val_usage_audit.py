"""
Module: val_usage_audit.py
Description: Standalone, model-independent audit of a validation split: for
             each hydrological year, and for the top-N basins (by
             contribution to total validation loss, matching
             train_sanity.py's basin loss pie), counts how many basin-hour
             pairs actually survive dataset filtering and get used, how many
             have a non-zero observed flow (at any forecast lead), and how
             many cross at least one of config['prediction_threshold']'s
             resolved per-basin flow thresholds (at any forecast lead).
             Depends only on the validation split + config, not on any
             trained model - can be run standalone against a config whose
             training already finished (reads that run's existing
             val_loss_by_basin.csv for basin ranking), or imported by
             train_sanity.py to produce the same CSVs from a live run's
             in-memory accumulators.
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

from train import load_config
from dataset import get_dataloader
from flow_quality_check import get_hydrological_year
import find_flood_events as ffe


def compute_val_usage_breakdown(val_loader, config, hydro_year_start_month):
    """
    Iterates val_loader once (no model forward pass) and, for every sample,
    buckets it by hydrological year and by basin id. Relies on the same
    shuffle=False/drop_last=False + running sample_offset alignment to
    dataset.sample_basin_mappings/sample_date_mappings documented in
    train_sanity.validate_epoch_with_breakdown.

    A sample counts as 'nonzero'/'threshold_crossed' if ANY of its forecast
    leads (batch['target_raw'], shape (B, num_leads)) is non-zero / meets or
    exceeds any of that basin's resolved prediction_threshold values.

    Returns (year_counts, basin_counts), each {group: {'used', 'nonzero',
    'threshold_crossed'}}.
    """
    dataset = val_loader.dataset

    threshold_specs = ffe.normalize_threshold_specs(config.get('prediction_threshold', []))
    area_map = (
        ffe.load_basin_area_map(config)
        if any(spec['type'] == 'specific_discharge' for spec in threshold_specs)
        else None
    )
    basin_thresholds = {}
    for basin_id in dataset.basins:
        resolved = [
            ffe.resolve_threshold_value(basin_id, spec, config, area_map)
            for spec in threshold_specs
        ]
        basin_thresholds[basin_id] = np.array(
            [v for v in resolved if v is not None and v > 0], dtype=np.float32)

    year_counts = defaultdict(lambda: {'used': 0, 'nonzero': 0, 'threshold_crossed': 0})
    basin_counts = defaultdict(lambda: {'used': 0, 'nonzero': 0, 'threshold_crossed': 0})

    sample_offset = 0
    for batch in tqdm(val_loader, desc="  Usage Audit Batches", leave=False):
        targets_raw = batch['target_raw'].numpy()  # (B, num_leads)
        batch_size = targets_raw.shape[0]

        for i in range(batch_size):
            idx = sample_offset + i
            basin_id = dataset.sample_basin_mappings[idx]
            date = dataset.sample_date_mappings[idx]
            year = int(get_hydrological_year(pd.DatetimeIndex([date]), start_month=hydro_year_start_month)[0])

            leads = targets_raw[i]
            is_nonzero = bool(np.any(leads != 0))
            thresholds = basin_thresholds[basin_id]
            is_crossed = thresholds.size > 0 and bool(np.any(leads[:, None] >= thresholds[None, :]))

            for counts in (year_counts[year], basin_counts[basin_id]):
                counts['used'] += 1
                counts['nonzero'] += int(is_nonzero)
                counts['threshold_crossed'] += int(is_crossed)

        sample_offset += batch_size

    return year_counts, basin_counts


def top_n_basin_ids_by_value(value_dict, top_n):
    """
    Sorts {basin_id: value} descending by value and returns the first top_n
    keys - the same selection plot_breakdown_pie (train_sanity.py) performs
    internally to pick the basin pie's non-'Other' slices, factored out so
    both a live run and this module's standalone script agree on which
    basins are 'top N'.
    """
    ordered = sorted(value_dict.items(), key=lambda kv: kv[1], reverse=True)
    return [basin_id for basin_id, _ in ordered[:top_n]]


def load_total_basin_values(basin_loss_csv_path):
    """
    Reads an existing val_loss_by_basin.csv (written by train_sanity.py) and
    returns {basin_id: value} from its 'TOTAL' row - the whole-run
    accumulated validation-loss contribution per basin, used to rank basins
    when no live in-memory accumulators exist (the standalone-script case).
    """
    if not os.path.exists(basin_loss_csv_path):
        raise FileNotFoundError(
            f"{basin_loss_csv_path} not found - run train_sanity.py for this config first "
            "(it writes the per-basin validation-loss breakdown this audit ranks basins by).")

    df = pd.read_csv(basin_loss_csv_path)
    df['epoch'] = df['epoch'].astype(str)
    total_rows = df[df['epoch'] == 'TOTAL']
    if total_rows.empty:
        raise ValueError(
            f"{basin_loss_csv_path} has no 'TOTAL' row - the train_sanity.py run for this "
            "config may not have finished. Let it complete before running this audit.")

    return dict(zip(total_rows['basin_id'], total_rows['value']))


def write_usage_csvs(year_counts, basin_counts, top_basin_ids, sanity_dir, split_name='val'):
    """
    Writes sanity_dir/{split_name}_usage_by_year.csv (one row per year,
    sorted ascending) and sanity_dir/{split_name}_usage_by_basin.csv (one
    row per basin in top_basin_ids, in that order). Returns (year_df,
    basin_df) so callers can log them (e.g. to wandb) without re-reading the
    CSVs back off disk.
    """
    year_rows = [
        {'year': year, 'used_pairs': c['used'], 'nonzero_pairs': c['nonzero'],
         'threshold_crossed_pairs': c['threshold_crossed']}
        for year, c in sorted(year_counts.items())
    ]
    year_df = pd.DataFrame(year_rows)
    year_df.to_csv(os.path.join(sanity_dir, f"{split_name}_usage_by_year.csv"), index=False)

    basin_rows = [
        {'basin_id': basin_id, 'used_pairs': basin_counts[basin_id]['used'],
         'nonzero_pairs': basin_counts[basin_id]['nonzero'],
         'threshold_crossed_pairs': basin_counts[basin_id]['threshold_crossed']}
        for basin_id in top_basin_ids
    ]
    basin_df = pd.DataFrame(basin_rows)
    basin_df.to_csv(os.path.join(sanity_dir, f"{split_name}_usage_by_basin.csv"), index=False)

    return year_df, basin_df


def log_usage_to_wandb(year_df, basin_df, key_prefix='usage_audit'):
    """Logs both usage-audit tables to the currently active wandb run. Callers
    own the use_wandb check and the wandb.init/wandb.finish lifecycle."""
    wandb.log({
        f'{key_prefix}/by_year': wandb.Table(dataframe=year_df),
        f'{key_prefix}/by_basin': wandb.Table(dataframe=basin_df),
    })


def run_usage_audit(config_path):
    config = load_config(config_path)

    cv_config = config.get('cross_validation', {}) or {}
    if cv_config.get('enabled', False):
        raise ValueError(
            "val_usage_audit.py does not support cross_validation-enabled configs: it relies "
            "on dataset.get_dataloader(split_type='val')'s IsraelBasinsDataset "
            "sample_basin_mappings/sample_date_mappings, not present on the CV fold path. "
            "Use a config with cross_validation.enabled: false."
        )

    use_spatial = config.get('use_basin_splits', True)
    val_loader = get_dataloader(split_type='val', config=config, use_basin_splits=use_spatial)

    hydro_year_start_month = config.get('hydro_year_start_month', 10)
    top_n_basins = config.get('sanity_top_n_basins', 10)

    run_dir = config.get('run_dir', './runs/')
    exp_dir = os.path.join(run_dir, config['experiment_name'])
    sanity_dir = os.path.join(exp_dir, "sanity_analysis")
    os.makedirs(sanity_dir, exist_ok=True)

    basin_loss_csv_path = os.path.join(sanity_dir, "val_loss_by_basin.csv")
    total_basin_values = load_total_basin_values(basin_loss_csv_path)
    top_basin_ids = top_n_basin_ids_by_value(total_basin_values, top_n_basins)

    print("[INFO] Computing validation-set usage audit (basin-hour pairs, non-zero, threshold-crossed)...")
    year_counts, basin_counts = compute_val_usage_breakdown(val_loader, config, hydro_year_start_month)
    year_df, basin_df = write_usage_csvs(year_counts, basin_counts, top_basin_ids, sanity_dir)
    print(f"[INFO] Usage audit CSVs written inside: {sanity_dir}")

    use_wandb = config.get('use_wandb', False)
    if use_wandb:
        api_key = config.get('wandb_api_key')
        if api_key:
            wandb.login(key=api_key)
        wandb.init(
            project=config.get('wandb_project', 'flash-floods-israel'),
            name=config['experiment_name'],
        )
        log_usage_to_wandb(year_df, basin_df)
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model-independent audit of a validation split: per-hydrological-year and "
                     "per-top-N-basin counts of basin-hour pairs used, non-zero, and "
                     "prediction-threshold-crossed. Can be run standalone against a config whose "
                     "train_sanity.py run already finished (reuses its val_loss_by_basin.csv for "
                     "basin ranking) - no model weights are touched.")
    parser.add_argument("--config", type=str, default="configs/config.yml",
                         help="Path to the YAML config file for this run.")
    args = parser.parse_args()
    run_usage_audit(args.config)
