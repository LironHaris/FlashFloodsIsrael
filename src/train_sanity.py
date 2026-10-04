"""
Module: train_sanity.py
Description: Sanity-check variant of train.py - trains identically (same
             model, optimizer, loss, checkpointing, wandb conventions) but
             additionally attributes each epoch's validation loss to the
             basins and hydrological years it came from. Writes two running
             CSVs (val_loss_by_year.csv, val_loss_by_basin.csv) plus two
             pie-chart PNGs every epoch, and a final accumulated-total row +
             two final pie charts once training finishes. Also writes a
             one-time, model-independent validation-set usage audit
             (val_usage_by_year.csv, val_usage_by_basin.csv - see
             val_usage_audit.py) once training completes. Fixed
             train/validation splits only - cross_validation-enabled configs
             are rejected (see validate_epoch_with_breakdown's docstring).
"""

import argparse
import os
import shutil
from collections import defaultdict

import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

from train import (
    load_config, get_tracked_hparams, get_optimizer, set_seed,
    train_epoch, validate_epoch, get_loss_criterion, plot_training_curves,
)
from dataset import get_dataloader
from model import EALSTMModel
from flow_quality_check import get_hydrological_year
from val_usage_audit import (
    compute_val_usage_breakdown, top_n_basin_ids_by_value, write_usage_csvs, log_usage_to_wandb,
)


def validate_epoch_with_breakdown(model, dataloader, criterion, device, hydro_year_start_month):
    """
    Like train.py:validate_epoch (same criterion call, same
    running_val_loss/len(dataloader) return value, so the reported val loss
    stays directly comparable to a plain train.py run) but additionally
    accumulates each sample's pre-reduction squared error into per-basin and
    per-hydrological-year buckets.

    Requires dataloader.dataset to be an IsraelBasinsDataset (exposes
    sample_basin_mappings/sample_date_mappings, index-aligned to
    __getitem__) - true for the fixed val/test split path
    (dataset.get_dataloader), NOT true for the cross-validation fold path
    (dataset.get_cv_fold_dataloaders wraps a bare ConcatDataset of
    SingleBasinDataset objects with no such mapping arrays). Callers must
    reject cross_validation-enabled configs before calling this.

    Relies on the validation DataLoader using shuffle=False with no custom
    sampler and drop_last=False (confirmed in dataset.py) - batch b's samples
    are exactly dataset indices [b*batch_size : b*batch_size+len(batch)], so
    a running sample_offset reconstructs each sample's basin/date without
    any change to __getitem__.

    Returns (val_loss, basin_sq_err_sum, year_sq_err_sum, total_elements) -
    the latter three describe THIS epoch only; the caller accumulates them
    into whole-run totals.
    """
    model.eval()
    running_val_loss = 0.0
    basin_sq_err_sum = defaultdict(float)
    year_sq_err_sum = defaultdict(float)
    total_elements = 0

    dataset = dataloader.dataset
    sample_offset = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Validation Batches", leave=False):
            x_dynamic = batch['dynamic'].to(device)
            x_static = batch['static'].to(device)
            targets = batch['target'].to(device)

            predictions = model(x_dynamic, x_static)
            loss = criterion(predictions, targets, batch)
            running_val_loss += loss.item()

            sq_err = (predictions - targets) ** 2  # (B, num_leads), pre-reduction
            batch_size = sq_err.shape[0]
            for i in range(batch_size):
                idx = sample_offset + i
                basin_id = dataset.sample_basin_mappings[idx]
                date = dataset.sample_date_mappings[idx]
                year = int(get_hydrological_year(pd.DatetimeIndex([date]), start_month=hydro_year_start_month)[0])
                sample_sum = sq_err[i].sum().item()
                basin_sq_err_sum[basin_id] += sample_sum
                year_sq_err_sum[year] += sample_sum
                total_elements += sq_err[i].numel()
            sample_offset += batch_size

    return running_val_loss / len(dataloader), basin_sq_err_sum, year_sq_err_sum, total_elements


def to_value_and_pct(sq_err_sum_dict, total_elements):
    """
    {group: sq_err_sum} -> ({group: value}, {group: percentage}), where
    value = sq_err_sum / total_elements (so every group's value sums exactly
    to the epoch's - or, for the final totals, the whole run's - overall
    validation MSE) and percentage = value / sum(all values) * 100.
    """
    values = {k: v / total_elements for k, v in sq_err_sum_dict.items()} if total_elements else {}
    total_value = sum(values.values())
    pcts = {k: (v / total_value * 100 if total_value > 0 else 0.0) for k, v in values.items()}
    return values, pcts


def plot_breakdown_pie(value_pct_dict, title, output_path, top_n=None):
    """
    Pie chart of {group: (value, percentage)}. top_n=None (years): one slice
    per group. top_n=N (basins): keep the top N by value, fold the rest into
    one 'Other (k basins)' slice, and annotate the figure with the top-N
    combined percentage / 100 - a quick read on whether loss is concentrated
    in a few bad basins or spread across many mediocre ones. Saves a PNG and
    returns the Figure (caller closes it / logs it to wandb).
    """
    items = sorted(value_pct_dict.items(), key=lambda kv: kv[1][0], reverse=True)

    annotation = None
    if top_n is not None and len(items) > top_n:
        top_items = items[:top_n]
        rest_items = items[top_n:]
        rest_value = sum(v for _, (v, _) in rest_items)
        rest_pct = sum(p for _, (_, p) in rest_items)
        plot_items = top_items + [(f'Other ({len(rest_items)} basins)', (rest_value, rest_pct))]
        top_pct_sum = sum(p for _, (_, p) in top_items)
        annotation = f'Top {top_n} basins = {top_pct_sum / 100:.2f} of total loss'
    else:
        plot_items = items

    labels = [f'{key}: {value:.4g} ({pct:.1f}%)' for key, (value, pct) in plot_items]
    sizes = [pct for _, (_, pct) in plot_items]
    colors = plt.get_cmap('tab20').colors

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150, facecolor="#fafafa")
    if sizes:
        ax.pie(sizes, labels=labels, startangle=90,
               colors=[colors[i % len(colors)] for i in range(len(plot_items))])
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15, color='#2c3e50')
    if annotation:
        fig.text(0.5, 0.02, annotation, ha='center', fontsize=10, style='italic', color='#2c3e50')
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    return fig


def _load_existing_breakdown(csv_path, group_col):
    """
    On resume, reloads a previous run's breakdown CSV so historical
    per-epoch rows survive verbatim (never recomputed or dropped) and the
    whole-run accumulators continue from where the prior run left off,
    rather than a naive rewrite silently truncating the file down to only
    the newly-run epochs. Drops any existing 'TOTAL' row (recomputed fresh
    once the full run finishes). Returns (rows_as_dicts,
    {group: total_sq_err_sum}, total_elements_all_epochs) - empty/zero if
    the file doesn't exist yet (a fresh, non-resumed run).
    """
    if not os.path.exists(csv_path):
        return [], defaultdict(float), 0

    df = pd.read_csv(csv_path)
    df['epoch'] = df['epoch'].astype(str)
    df = df[df['epoch'] != 'TOTAL']

    rows = df.to_dict('records')
    sq_err_sum_totals = defaultdict(float)
    for row in rows:
        sq_err_sum_totals[row[group_col]] += row['sq_err_sum']

    # total_elements is repeated across every row within an epoch - dedupe by
    # epoch before summing, or every group would multiply-count it.
    total_elements_all_epochs = int(df.drop_duplicates(subset='epoch')['total_elements'].sum())
    return rows, sq_err_sum_totals, total_elements_all_epochs


def main(config_path="configs/config.yml"):
    config = load_config(config_path)

    cv_config = config.get('cross_validation', {}) or {}
    if cv_config.get('enabled', False):
        raise ValueError(
            "train_sanity.py does not support cross_validation-enabled configs: the CV "
            "fold validation dataloader (dataset.get_cv_fold_dataloaders) wraps a bare "
            "ConcatDataset with no sample_basin_mappings/sample_date_mappings, which this "
            "script's per-basin/per-year loss breakdown depends on. Use a config with "
            "cross_validation.enabled: false."
        )

    set_seed(config.get('seed', 42))

    device_str = config.get('device', 'cpu')
    device = torch.device(device_str if torch.cuda.is_available() or device_str == 'cpu' else 'cpu')
    print(f"[INFO] Execution target hardware configured to: {device}")

    use_spatial = config.get('use_basin_splits', True)
    if not use_spatial:
        print("[INFO] Spatial basin splits disabled via config. Using strict temporal configuration.")
    else:
        print("[INFO] Spatial basin splits enabled via config. Loading specific basin split files.")

    print("[INFO] Constructing dataset pipelines and dataloaders...")
    epochs = config.get('epochs', 30)
    train_loader = get_dataloader(split_type='train', config=config, use_basin_splits=use_spatial)
    val_loader = get_dataloader(split_type='val', config=config, use_basin_splits=use_spatial)

    hydro_year_start_month = config.get('hydro_year_start_month', 10)
    top_n_basins = config.get('sanity_top_n_basins', 10)

    print("[INFO] Instantiating EA-LSTM model architecture dynamically...")
    model = EALSTMModel(config).to(device)

    loss_setting = config.get('loss', 'MSE')
    criterion = get_loss_criterion(loss_setting, config)
    print(f"[INFO] Optimization criterion set to: {loss_setting}")

    initial_lr = float(config['learning_rate'])
    optimizer = get_optimizer(config, model, initial_lr)

    start_epoch = 0
    checkpoint_path = config.get('checkpoint_path')
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint_path is set but not found: {checkpoint_path}")
        print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"[INFO] Resumed after epoch {start_epoch}. Re-validating to establish best_val_loss baseline...")

    run_dir = config.get('run_dir', './runs/')
    exp_dir = os.path.join(run_dir, config['experiment_name'])
    os.makedirs(exp_dir, exist_ok=True)

    sanity_dir = os.path.join(exp_dir, "sanity_analysis")
    pies_dir = os.path.join(sanity_dir, "pies")
    os.makedirs(pies_dir, exist_ok=True)
    year_csv_path = os.path.join(sanity_dir, "val_loss_by_year.csv")
    basin_csv_path = os.path.join(sanity_dir, "val_loss_by_basin.csv")

    saved_config_path = os.path.join(exp_dir, "config.yml")
    if os.path.abspath(config_path) != os.path.abspath(saved_config_path):
        shutil.copy(config_path, saved_config_path)

    use_wandb = config.get('use_wandb', False)
    if use_wandb:
        api_key = config.get('wandb_api_key')
        if api_key:
            wandb.login(key=api_key)
        wandb.init(
            project=config.get('wandb_project', 'flash-floods-israel'),
            name=config['experiment_name'],
            config=get_tracked_hparams(config),
        )

    # Resume: reload historical breakdown rows/accumulators - never recomputed, only appended to.
    year_rows, total_year_sq_err_sum, total_elements_all_epochs = _load_existing_breakdown(year_csv_path, 'year')
    basin_rows, total_basin_sq_err_sum, _ = _load_existing_breakdown(basin_csv_path, 'basin_id')

    # The baseline re-validation on resume isn't a real training epoch (no
    # epoch number of its own) - use the plain validate_epoch so it can't
    # invent a bogus CSV row, accumulator update, or pie chart.
    if start_epoch > 0:
        best_val_loss = validate_epoch(model, val_loader, criterion, device, config)
    else:
        best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []

    if start_epoch >= epochs:
        print(f"[INFO] Checkpoint already at epoch {start_epoch} >= target epochs {epochs}. Nothing to train.")
    else:
        print(f"[INFO] Initiating optimization loop for epochs {start_epoch + 1}-{epochs}.\n")

    for epoch in range(start_epoch, epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config)
        val_loss, basin_sq_err_sum, year_sq_err_sum, total_elements = validate_epoch_with_breakdown(
            model, val_loader, criterion, device, hydro_year_start_month)

        if loss_setting.upper() == 'RMSE':
            train_loss = torch.sqrt(torch.tensor(train_loss)).item()
            val_loss_report = torch.sqrt(torch.tensor(val_loss)).item()
            metric_label = "RMSE"
        else:
            val_loss_report = val_loss
            metric_label = "Loss"

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss_report)

        print(f"Epoch [{epoch + 1}/{epochs}] Completed:")
        print(f"  -> Train {metric_label}: {train_loss:.5f}")
        print(f"  -> Val {metric_label}:   {val_loss_report:.5f}")

        for k, v in basin_sq_err_sum.items():
            total_basin_sq_err_sum[k] += v
        for k, v in year_sq_err_sum.items():
            total_year_sq_err_sum[k] += v
        total_elements_all_epochs += total_elements

        basin_values, basin_pcts = to_value_and_pct(basin_sq_err_sum, total_elements)
        year_values, year_pcts = to_value_and_pct(year_sq_err_sum, total_elements)

        for basin_id, sq_sum in basin_sq_err_sum.items():
            basin_rows.append({
                'epoch': epoch + 1, 'basin_id': basin_id,
                'sq_err_sum': sq_sum, 'total_elements': total_elements,
                'value': basin_values[basin_id], 'percentage': basin_pcts[basin_id],
            })
        for year, sq_sum in year_sq_err_sum.items():
            year_rows.append({
                'epoch': epoch + 1, 'year': year,
                'sq_err_sum': sq_sum, 'total_elements': total_elements,
                'value': year_values[year], 'percentage': year_pcts[year],
            })

        pd.DataFrame(basin_rows).to_csv(basin_csv_path, index=False)
        pd.DataFrame(year_rows).to_csv(year_csv_path, index=False)

        year_pie_path = os.path.join(pies_dir, f"year_pie_epoch_{epoch + 1}.png")
        basin_pie_path = os.path.join(pies_dir, f"basin_pie_epoch_{epoch + 1}.png")
        year_fig = plot_breakdown_pie(
            {k: (year_values[k], year_pcts[k]) for k in year_values},
            f"Val Loss by Hydrological Year — Epoch {epoch + 1}", year_pie_path, top_n=None)
        basin_fig = plot_breakdown_pie(
            {k: (basin_values[k], basin_pcts[k]) for k in basin_values},
            f"Val Loss by Basin — Epoch {epoch + 1}", basin_pie_path, top_n=top_n_basins)

        if use_wandb:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss_report,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'sanity/year_pie': wandb.Image(year_fig),
                'sanity/basin_pie': wandb.Image(basin_fig),
            }, step=epoch + 1)
        plt.close(year_fig)
        plt.close(basin_fig)

        if val_loss_report < best_val_loss:
            best_val_loss = val_loss_report
            best_checkpoint_path = os.path.join(exp_dir, "best_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss_report,
            }, best_checkpoint_path)
            print(f"Validation improvement detected. Saved as best_model.pt")
            if use_wandb:
                wandb.run.summary['best_val_loss'] = val_loss_report
                wandb.run.summary['best_epoch'] = epoch + 1

        if (epoch + 1) % config.get('save_weights_every', 1) == 0:
            periodic_checkpoint_path = os.path.join(exp_dir, f"ealstm_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, periodic_checkpoint_path)

    # Final accumulated-total row + final pies, reflecting every epoch this
    # experiment has ever run (pre-resume history included).
    basin_rows = [r for r in basin_rows if str(r['epoch']) != 'TOTAL']
    year_rows = [r for r in year_rows if str(r['epoch']) != 'TOTAL']

    final_basin_values, final_basin_pcts = to_value_and_pct(total_basin_sq_err_sum, total_elements_all_epochs)
    final_year_values, final_year_pcts = to_value_and_pct(total_year_sq_err_sum, total_elements_all_epochs)

    for basin_id, sq_sum in total_basin_sq_err_sum.items():
        basin_rows.append({
            'epoch': 'TOTAL', 'basin_id': basin_id,
            'sq_err_sum': sq_sum, 'total_elements': total_elements_all_epochs,
            'value': final_basin_values[basin_id], 'percentage': final_basin_pcts[basin_id],
        })
    for year, sq_sum in total_year_sq_err_sum.items():
        year_rows.append({
            'epoch': 'TOTAL', 'year': year,
            'sq_err_sum': sq_sum, 'total_elements': total_elements_all_epochs,
            'value': final_year_values[year], 'percentage': final_year_pcts[year],
        })

    pd.DataFrame(basin_rows).to_csv(basin_csv_path, index=False)
    pd.DataFrame(year_rows).to_csv(year_csv_path, index=False)

    final_year_fig = plot_breakdown_pie(
        {k: (final_year_values[k], final_year_pcts[k]) for k in final_year_values},
        "Val Loss by Hydrological Year — TOTAL", os.path.join(pies_dir, "year_pie_TOTAL.png"), top_n=None)
    final_basin_fig = plot_breakdown_pie(
        {k: (final_basin_values[k], final_basin_pcts[k]) for k in final_basin_values},
        "Val Loss by Basin — TOTAL", os.path.join(pies_dir, "basin_pie_TOTAL.png"), top_n=top_n_basins)

    if use_wandb:
        wandb.log({
            'sanity/year_pie_final': wandb.Image(final_year_fig),
            'sanity/basin_pie_final': wandb.Image(final_basin_fig),
        })
    plt.close(final_year_fig)
    plt.close(final_basin_fig)

    # One-time, model-independent audit of the validation set itself (basin-hour
    # pairs used/non-zero/threshold-crossed) - see val_usage_audit.py.
    top_basin_ids = top_n_basin_ids_by_value(final_basin_values, top_n_basins)
    year_usage_counts, basin_usage_counts = compute_val_usage_breakdown(
        val_loader, config, hydro_year_start_month)
    year_usage_df, basin_usage_df = write_usage_csvs(
        year_usage_counts, basin_usage_counts, top_basin_ids, sanity_dir)
    if use_wandb:
        log_usage_to_wandb(year_usage_df, basin_usage_df)

    plot_training_curves(train_loss_history, val_loss_history, loss_setting, exp_dir)

    if use_wandb:
        wandb.finish()

    print(f"\n[INFO] Optimization sequence finished. Best Validation {loss_setting}: {best_val_loss:.5f}")
    print(f"[INFO] All outputs and checkpoints archived inside: {exp_dir}")
    print(f"[INFO] Validation loss breakdown CSVs/pies archived inside: {sanity_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the EA-LSTM flash flood model exactly like train.py, plus a "
                     "per-basin/per-year validation loss breakdown (CSVs + pie charts each "
                     "epoch, and a final accumulated total). Fixed train/validation splits "
                     "only - cross_validation-enabled configs are rejected.")
    parser.add_argument("--config", type=str, default="configs/config.yml",
                         help="Path to the YAML config file for this run.")
    args = parser.parse_args()
    main(args.config)
