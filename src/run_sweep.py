"""
Module: run_sweep.py
Description: W&B sweep agent entry point for Bayesian hyperparameter tuning.

Each call by the sweep agent runs one full training trial:
  1. wandb.init() picks up the trial's hyperparameters from the agent.
  2. Base config.yml is loaded and overridden with the swept values.
  3. Training and validation loops run using shared functions from train.py.
  4. Metrics are logged to the active W&B run; best checkpoint is saved.

Usage:
  wandb sweep configs/sweep.yaml          # register sweep, prints SWEEP_ID
  wandb agent <SWEEP_ID>                  # run agent (loops until budget exhausted)
"""

import os
import sys
import yaml
import torch
import wandb

# Ensure src/ is on the path when called directly by the W&B agent
sys.path.insert(0, os.path.dirname(__file__))

from model import EALSTMModel
from dataset import get_dataloader
from train import set_seed, train_epoch, validate_epoch, get_loss_criterion, get_tracked_hparams, get_optimizer


def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def is_below_top_k(project, sweep_id, current_run_id, best_val_loss_so_far, top_k):
    """True if this run's best_val_loss isn't competitive with the top_k
    best *finished* runs in the sweep so far. False if too few exist to compare."""
    api = wandb.Api()
    sweep = api.sweep(f"{project}/{sweep_id}")
    finished_losses = sorted(
        r.summary['best_val_loss']
        for r in sweep.runs
        if r.id != current_run_id and r.state == 'finished' and 'best_val_loss' in r.summary
    )
    if len(finished_losses) < top_k:
        return False
    return best_val_loss_so_far > finished_losses[top_k - 1]


def run_trial():
    # wandb agent spawns each trial as a fresh `python src/run_sweep.py` process with
    # no extra CLI args (hyperparameters arrive via wandb.config, not argv), so the base
    # config to sweep can't be passed as a normal --config flag - it's set once via
    # FLASHFLOODS_CONFIG before starting `wandb agent` (see cluster/run_sweep.sh).
    base_config_path = os.environ.get('FLASHFLOODS_CONFIG', 'configs/config.yml')
    base_config = load_config(base_config_path)
    sweep_config = load_config("configs/sweep.yaml")

    api_key = base_config.get('wandb_api_key')
    if api_key:
        wandb.login(key=api_key)

    project = sweep_config.get('project')
    wandb.init(project=project)

    early_drop_enabled = sweep_config.get('early_drop_enabled', False)
    early_drop_top_k = sweep_config.get('early_drop_top_k')
    early_drop_epoch = sweep_config.get('early_drop_epoch')

    # Load base config and apply swept hyperparameters
    config = base_config
    config.update(dict(wandb.config))

    # Push non-swept relevant hparams (epochs, forecast_lead_times, etc.) into the W&B run
    wandb.config.update(get_tracked_hparams(config))

    set_seed(config.get('seed', 42))

    device_str = config.get('device', 'cpu')
    device = torch.device(device_str if torch.cuda.is_available() or device_str == 'cpu' else 'cpu')

    # Unique output directory per trial — prevents checkpoint collisions across parallel runs
    run_dir = config.get('run_dir', './runs/')
    exp_dir = os.path.join(run_dir, config['experiment_name'], wandb.run.id)
    os.makedirs(exp_dir, exist_ok=True)

    use_spatial = config.get('use_basin_splits', True)
    train_loader = get_dataloader(split_type='train', config=config, use_basin_splits=use_spatial)
    val_loader = get_dataloader(split_type='val', config=config, use_basin_splits=use_spatial)

    model = EALSTMModel(config).to(device)

    loss_setting = config.get('loss', 'MSE')
    criterion = get_loss_criterion(loss_setting, config)

    sweep_lr = float(config['learning_rate'])
    optimizer = get_optimizer(config, model, sweep_lr)

    best_val_loss = float('inf')
    epochs = config.get('epochs', 30)
    patience = config.get('early_stop_patience', 10)
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config)
        val_loss = validate_epoch(model, val_loader, criterion, device, config)

        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
        }, step=epoch + 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(exp_dir, "best_model.pt"))
            wandb.run.summary['best_val_loss'] = val_loss
            wandb.run.summary['best_epoch'] = epoch + 1
        else:
            epochs_no_improve += 1
            if patience is not None and epochs_no_improve >= patience:
                print(f"[Early Stop] No val_loss improvement for {patience} epochs. Stopping at epoch {epoch + 1}.")
                break

        if early_drop_enabled and (epoch + 1) == early_drop_epoch:
            if is_below_top_k(project, wandb.run.sweep_id, wandb.run.id, best_val_loss, early_drop_top_k):
                print(f"[Early Drop] best_val_loss={best_val_loss:.4f} not in top {early_drop_top_k} "
                      f"at epoch {epoch + 1}. Abandoning this configuration.")
                wandb.run.summary['early_dropped'] = True
                break

    wandb.finish()


if __name__ == "__main__":
    run_trial()
