"""
Module: run_sweep.py
Description: W&B sweep agent entry point for Bayesian hyperparameter tuning.

Each call by the sweep agent runs one full training trial:
  1. wandb.init() picks up the trial's hyperparameters from the agent.
  2. Base config.yml is loaded and overridden with the swept values.
  3. Training and validation loops run using shared functions from train.py.
  4. Metrics are logged to the active W&B run; best checkpoint is saved.

If the base config has cross_validation.enabled: true, each trial mirrors
train.py's CV branch: total epochs = num_groups * training_rep, and every
epoch rebuilds its train/val loaders for that epoch's held-out fold via
dataset.py's build_cv_group_datasets/get_cv_fold_dataloaders - same fold
rotation, same "no special-casing" checkpoint/early-stop logic as train.py.

Usage:
  wandb sweep configs/sweep.yaml          # register sweep, prints SWEEP_ID
  wandb agent <SWEEP_ID>                  # run agent (loops until budget exhausted)

  # To register/run a differently-configured sweep (e.g. one that also
  # searches weight_decay), point FLASHFLOODS_SWEEP_CONFIG at that file -
  # it must match whichever YAML SWEEP_ID was actually registered from:
  wandb sweep configs/sweep_adamw.yaml
  FLASHFLOODS_SWEEP_CONFIG=configs/sweep_adamw.yaml wandb agent <SWEEP_ID>
"""

import os
import sys
import yaml
import torch
import wandb

# Ensure src/ is on the path when called directly by the W&B agent
sys.path.insert(0, os.path.dirname(__file__))

from model import EALSTMModel
from dataset import get_dataloader, build_cv_group_datasets, get_cv_fold_dataloaders
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
    sweep_config_path = os.environ.get('FLASHFLOODS_SWEEP_CONFIG', 'configs/sweep.yaml')
    sweep_config = load_config(sweep_config_path)

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

    cv_config = config.get('cross_validation', {}) or {}
    cv_enabled = cv_config.get('enabled', False)

    if cv_enabled:
        num_folds = len(cv_config['groups'])
        training_rep = cv_config['training_rep']
        if num_folds < 2:
            raise ValueError(f"cross_validation.groups must contain at least 2 groups, got {num_folds}.")
        if training_rep < 1:
            raise ValueError(f"cross_validation.training_rep must be >= 1, got {training_rep}.")
        epochs = num_folds * training_rep
        fixed_train_datasets, group_datasets = build_cv_group_datasets(config, use_basin_splits=use_spatial)
        train_loader, val_loader = None, None
    else:
        epochs = config.get('epochs', 30)
        train_loader = get_dataloader(split_type='train', config=config, use_basin_splits=use_spatial)
        val_loader = get_dataloader(split_type='val', config=config, use_basin_splits=use_spatial)

    model = EALSTMModel(config).to(device)

    loss_setting = config.get('loss', 'MSE')
    criterion = get_loss_criterion(loss_setting, config)

    sweep_lr = float(config['learning_rate'])
    optimizer = get_optimizer(config, model, sweep_lr)

    best_val_loss = float('inf')
    patience = config.get('early_stop_patience', 10)
    epochs_no_improve = 0

    for epoch in range(epochs):
        if cv_enabled:
            fold = epoch % num_folds
            train_loader, val_loader = get_cv_fold_dataloaders(fixed_train_datasets, group_datasets, fold, config)

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config)
        val_loss = validate_epoch(model, val_loader, criterion, device, config)

        log_dict = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
        }
        if cv_enabled:
            log_dict['cv/held_out_group'] = fold + 1
        wandb.log(log_dict, step=epoch + 1)

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
