"""
Module: train.py
Description: Main Training, Validation, and Optimization Pipeline for Dynamic Multi-Horizon EA-LSTM.
             run_training(config) trains one model on the configured train/validation
             split and returns the best validation loss (the objective for hyperparameter
             search - see run_sweep.py / cross_validation_sweep.py). With
             cross_validation.enabled it is the final training instead: from scratch on
             all train_periods, no validation, for --epochs (the CV sweep's avg_best_epoch). Dynamic inputs are normalized on the fly with
             train-period-only statistics (normalization.py), which are saved to
             exp_dir/normalization_stats.json and inside every checkpoint.
"""

import argparse
import os
import shutil
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import wandb

from model import EALSTMModel
from dataset import get_dataloader
from normalization import prepare_normalization
from loss import BatchAwareLossWrapper


def load_config(yaml_path):
    """Load the YAML configuration file safely."""
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def get_tracked_hparams(config):
    """Return the subset of config keys that genuinely affect model behavior."""
    keys = [
        'hidden_size', 'output_dropout', 'initial_forget_bias',
        'loss', 'nse_epsilon', 'learning_rate',
        'batch_size', 'epochs', 'seed',
        'target_noise_std', 'clip_gradient_norm',
        'seq_length', 'forecast_lead_times', 'use_basin_splits',
        'optimizer', 'weight_decay',
    ]
    tracked = {k: config[k] for k in keys if k in config}
    tracked['num_static_attributes'] = len(config.get('static_attributes', []))
    tracked['num_dynamic_inputs']    = len(config.get('dynamic_inputs', []))
    tracked['statics_embedding_type'] = config.get('statics_embedding', {}).get('type')
    return tracked


def get_optimizer(config, model, lr):
    """
    Construct the optimizer specified by config['optimizer'] (default 'Adam').
    weight_decay (if set) is passed identically to both - only their
    weight-decay update rule differs: Adam folds it into the gradient
    (L2 regularization), AdamW decouples it from the gradient-based update
    (Loshchilov & Hutter, 2019).
    """
    optimizer_name = config.get('optimizer', 'Adam')
    weight_decay = config.get('weight_decay') or 0.0
    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Supported: Adam, AdamW.")


def set_seed(seed):
    """Enforce deterministic behavior across runs for scientific reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)


def train_epoch(model, dataloader, optimizer, criterion, device, config):
    """Runs a single training epoch over all batched multi-basin sequences."""
    model.train()
    running_loss = 0.0
    log_interval = config.get('log_interval', 5)
    noise_std = config.get('target_noise_std', 0.0)
    clip_norm = config.get('clip_gradient_norm', None)

    progress_bar = tqdm(dataloader, desc="  Training Batches", leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        x_dynamic = batch['dynamic'].to(device)
        x_static = batch['static'].to(device)
        targets = batch['target'].to(device)

        # Regularization: Target Noise Injection (Train phase only).
        # Targets are per-basin z-scored, so raw zero-flow corresponds to -basin_mean/basin_std,
        # not 0.0 — clamp to that basin-specific floor instead of a literal zero.
        if model.training and noise_std > 0:
            basin_mean = batch['basin_mean'].to(device)
            basin_std = batch['basin_std'].to(device)
            noise = torch.randn_like(targets) * noise_std
            zero_flow_z = (-basin_mean / basin_std).unsqueeze(1)
            targets = torch.maximum(targets + noise, zero_flow_z)

        # Optimization Step
        optimizer.zero_grad()
        predictions = model(x_dynamic, x_static)
        
        loss = criterion(predictions, targets, batch)
        loss.backward()
        
        # Gradient Stabilization via Norm Clipping
        if clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % log_interval == 0:
            progress_bar.set_postfix({'Batch Loss': f'{loss.item():.5f}'})
            
    return running_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device, config):
    """Evaluates performance on validation basins to monitor overfitting boundaries."""
    model.eval() 
    running_val_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Validation Batches", leave=False):
            x_dynamic = batch['dynamic'].to(device)
            x_static = batch['static'].to(device)
            targets = batch['target'].to(device)
            
            predictions = model(x_dynamic, x_static)
            loss = criterion(predictions, targets, batch)

            running_val_loss += loss.item()
            
    return running_val_loss / len(dataloader)


def get_loss_criterion(loss_name: str, config: dict) -> BatchAwareLossWrapper:
    """Dynamically instantiates the target loss function based on config string."""
    name = loss_name.upper()
    if name in ('MSE', 'RMSE'):
        return BatchAwareLossWrapper(nn.MSELoss(), uses_basin_std=False)
    elif name == 'NSE':
        print("[ERROR] Loss 'NSE' divides by basin_std, which double-normalizes now that "
              "flow targets are already per-basin z-scored in dataset.py.")
        raise ValueError(
            "Loss 'NSE' is incompatible with normalized flow targets. Use 'MSE' or 'RMSE' "
            "instead (config.yml currently has loss: NSE — update it)."
        )
    else:
        raise ValueError(f"Unsupported loss function specified in config: {loss_name}")


def plot_training_curves(train_losses, val_losses, loss_setting, exp_dir):
    """
    Generates and saves a formal evaluation plot for the training and validation history.
    Dynamically includes the loss function type from the configuration in the title.
    """
    epochs_range = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 5.5), facecolor="#fafafa")
    ax = plt.axes()
    ax.set_facecolor("#ffffff")
    
    # Plot curves
    plt.plot(epochs_range, train_losses, color="#2b8cbe", linewidth=2, label="Training Loss")
    if val_losses:  # empty in final cross-validation training (no validation set)
        plt.plot(range(1, len(val_losses) + 1), val_losses, color="#cb181d", linewidth=2,
                 linestyle="--", label="Validation Loss")
    
    # Title and Labels using dynamic configuration values
    plt.title(f"EA-LSTM Training Optimization History\nOptimization Metric: {loss_setting.upper()}", 
              fontsize=12, fontweight='bold', pad=15, color="#2c3e50")
    plt.xlabel("Epochs", fontsize=10.5, labelpad=8)
    plt.ylabel(f"Loss Magnitude ({loss_setting.upper()})", fontsize=10.5, labelpad=8)
    
    plt.grid(True, linestyle=":", alpha=0.6, color="#b0b0b0")
    plt.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e2e2e2", fontsize=10)
    plt.tight_layout()
    
    # Save chart to the specific experiment directory
    output_path = os.path.join(exp_dir, "loss_training_curves.png")
    plt.savefig(output_path, dpi=300, facecolor="#fafafa")
    plt.close()
    print(f"\n[OK] Training loss curves chart successfully exported to: {output_path}")


def _save_checkpoint(path, epoch, model, optimizer, normalization_stats, **extra):
    """Every checkpoint carries the normalization statistics it was trained with,
    so the model can always be evaluated with exactly the same input scaling."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'normalization_stats': normalization_stats,
        **extra,
    }, path)


def _write_run_config(config, config_path, exp_dir):
    """exp_dir/config.yml: a copy of the source file when there is one, else the
    in-memory config (e.g. a sweep trial's base config + swept values)."""
    saved_config_path = os.path.join(exp_dir, "config.yml")
    if config_path:
        if os.path.abspath(config_path) != os.path.abspath(saved_config_path):
            shutil.copy(config_path, saved_config_path)
        return
    dumpable = {k: v for k, v in config.items() if k != 'normalization_stats'}
    with open(saved_config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(dumpable, f, allow_unicode=True, sort_keys=False)


def run_training(config, config_path=None, exp_dir=None, use_wandb=None, init_wandb=True,
                 epoch_callback=None, early_stop_patience=None, save_periodic=True,
                 wandb_prefix='', write_summary=True):
    """
    Trains ONE model and returns its objective.

    Two modes:
    - Regular (cross_validation absent / enabled: false): train on the train
      split, validate every epoch on the validation split, keep the best
      epoch's weights as best_model.pt.
    - Final cross-validation training (cross_validation.enabled: true): the
      hyperparameters were already selected by cross_validation_sweep.py, so
      the model is trained from scratch on ALL of train_periods (every
      non-test year, CV groups included) with no validation set, for exactly
      config['epochs'] epochs (set it - or pass train.py --epochs - to the
      winning trial's avg_best_epoch). The last epoch's weights are saved as
      best_model.pt so the test scripts work unchanged.

    - config: full config dict (already merged with any hyperparameter
      overrides, e.g. a sweep trial's values). Mutated: config gets
      'normalization_stats' attached.
    - exp_dir: output directory (default run_dir/experiment_name).
    - use_wandb: default config['use_wandb']. init_wandb=False means the
      caller already owns an active wandb run (sweep agent) - this function
      then only logs to it and never calls wandb.init/finish.
    - epoch_callback(epoch, val_loss, best_val_loss) -> bool: called after
      every epoch (1-based epoch); returning True stops training (pruning).
    - early_stop_patience: stop after this many epochs without improvement
      (default config['early_stop_patience'], None = disabled).
    - save_periodic: write ealstm_epoch_N.pt every save_weights_every epochs.
    - wandb_prefix: when set (e.g. 'fold_2/'), metrics are logged as
      '{prefix}train_loss' / '{prefix}val_loss' / '{prefix}epoch' without a
      global step, so several models (CV folds) can log into one run.
    - write_summary: when False, the run summary (best_val_loss/best_epoch)
      is left to the caller.

    Returns {'best_val_loss', 'best_epoch', 'exp_dir', 'train_loss_history',
    'val_loss_history'} (best_val_loss is None in final CV training).
    """
    final_cv_training = bool((config.get('cross_validation') or {}).get('enabled', False))

    set_seed(config.get('seed', 42))

    device_str = config.get('device', 'cpu')
    device = torch.device(device_str if torch.cuda.is_available() or device_str == 'cpu' else 'cpu')
    print(f"[INFO] Execution target hardware configured to: {device}")

    if exp_dir is None:
        exp_dir = os.path.join(config.get('run_dir', './runs/'), config['experiment_name'])
    os.makedirs(exp_dir, exist_ok=True)
    # Preserve an exact copy of the config that produced this run, so it's never
    # ambiguous later which settings a given checkpoint came from.
    _write_run_config(config, config_path, exp_dir)

    # Step 1: Normalization statistics - computed from train_periods ONLY, for
    # every train/val/test basin, attached to config (so every loader below
    # z-scores with them) and persisted to exp_dir/normalization_stats.json.
    normalization_stats = prepare_normalization(config, exp_dir)

    # Read the spatial split flag directly from the configuration file (default to True if missing)
    use_spatial = config.get('use_basin_splits', True)
    if not use_spatial:
        print("[INFO] Spatial basin splits disabled via config. Using strict temporal configuration.")
    else:
        print("[INFO] Spatial basin splits enabled via config. Loading specific basin split files.")

    # Step 2: Initialize Data Pipeline Loaders
    print("[INFO] Constructing dataset pipelines and dataloaders...")
    epochs = config.get('epochs', 30)
    train_loader = get_dataloader(split_type='train', config=config, use_basin_splits=use_spatial)
    if final_cv_training:
        # Final CV model: all of train_periods, no validation (see docstring)
        print(f"[INFO] cross_validation.enabled: final training from scratch on all train_periods "
              f"for {epochs} epochs, no validation set.")
        val_loader = None
    else:
        val_loader = get_dataloader(split_type='val', config=config, use_basin_splits=use_spatial)

    # Step 3: Construct Architecture and Optimization Engines
    print("[INFO] Instantiating EA-LSTM model architecture dynamically...")
    model = EALSTMModel(config).to(device)

    loss_setting = config.get('loss', 'MSE')
    criterion = get_loss_criterion(loss_setting, config)
    print(f"[INFO] Optimization criterion set to: {loss_setting}")

    initial_lr = float(config['learning_rate'])
    optimizer = get_optimizer(config, model, initial_lr)

    # Resume from a checkpoint if one is configured (model + optimizer state both restored)
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
        print(f"[INFO] Resumed after epoch {start_epoch}.")

    if use_wandb is None:
        use_wandb = config.get('use_wandb', False)
    if use_wandb and init_wandb:
        api_key = config.get('wandb_api_key')
        if api_key:
            wandb.login(key=api_key)
        wandb.init(
            project=config.get('wandb_project', 'flash-floods-israel'),
            name=config['experiment_name'],
            config=get_tracked_hparams(config),
        )

    if early_stop_patience is None:
        early_stop_patience = config.get('early_stop_patience')

    # Trackers for saving checkpoints and plotting history
    if start_epoch > 0 and val_loader is not None:
        best_val_loss = validate_epoch(model, val_loader, criterion, device, config)
        best_epoch = start_epoch
    else:
        best_val_loss = float('inf')
        best_epoch = None
    epochs_no_improve = 0
    train_loss_history = []
    val_loss_history = []

    # Step 4: Core Training and Validation Loop Execution
    if start_epoch >= epochs:
        print(f"[INFO] Checkpoint already at epoch {start_epoch} >= target epochs {epochs}. Nothing to train.")
    else:
        print(f"[INFO] Initiating optimization loop for epochs {start_epoch + 1}-{epochs}.\n")

    for epoch in range(start_epoch, epochs):
        # Part A: Execute Training Cycle
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config)

        # Part B: Execute Validation Cycle (none in final CV training)
        val_loss = validate_epoch(model, val_loader, criterion, device, config) if val_loader is not None else None

        # If RMSE is selected, calculate the physical square root error
        if loss_setting.upper() == 'RMSE':
            train_loss = torch.sqrt(torch.tensor(train_loss)).item()
            if val_loss is not None:
                val_loss = torch.sqrt(torch.tensor(val_loss)).item()
            metric_label = "RMSE"
        else:
            metric_label = "Loss"

        # Append to history trackers for downstream plotting
        train_loss_history.append(train_loss)
        if val_loss is not None:
            val_loss_history.append(val_loss)

        print(f"{wandb_prefix}Epoch [{epoch+1}/{epochs}] Completed:")
        print(f"  -> Train {metric_label}: {train_loss:.5f}")
        if val_loss is not None:
            print(f"  -> Val {metric_label}:   {val_loss:.5f}")

        if use_wandb:
            metrics = {'train_loss': train_loss, 'learning_rate': optimizer.param_groups[0]['lr']}
            if val_loss is not None:
                metrics['val_loss'] = val_loss
            if wandb_prefix:
                # Several models (CV folds) share one run: no global step, each
                # fold's curves are plotted against its own '{prefix}epoch'.
                metrics['epoch'] = epoch + 1
                wandb.log({f'{wandb_prefix}{k}': v for k, v in metrics.items()})
            else:
                wandb.log(metrics, step=epoch + 1)

        # Part C: Strategic Model Selection (Save Best Weights)
        if val_loss is None:
            # Final CV training: no validation - keep the latest weights
            best_epoch = epoch + 1
            _save_checkpoint(os.path.join(exp_dir, "best_model.pt"), epoch + 1, model, optimizer,
                             normalization_stats, val_loss=None)
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            _save_checkpoint(os.path.join(exp_dir, "best_model.pt"), epoch + 1, model, optimizer,
                             normalization_stats, val_loss=val_loss)
            print(f"Validation improvement detected. Saved as best_model.pt")
            if use_wandb and write_summary:
                wandb.run.summary['best_val_loss'] = val_loss
                wandb.run.summary['best_epoch'] = epoch + 1
        else:
            epochs_no_improve += 1

        # Part D: Standard Periodic Backup
        if save_periodic and (epoch + 1) % config.get('save_weights_every', 1) == 0:
            _save_checkpoint(os.path.join(exp_dir, f"ealstm_epoch_{epoch+1}.pt"), epoch + 1, model, optimizer,
                             normalization_stats, loss=train_loss)

        # Part E: Stopping rules (early stop / external pruning, e.g. a sweep's early drop)
        if val_loss is None:
            continue
        if early_stop_patience is not None and epochs_no_improve >= early_stop_patience:
            print(f"[Early Stop] No val_loss improvement for {early_stop_patience} epochs. "
                  f"Stopping at epoch {epoch + 1}.")
            break
        if epoch_callback is not None and epoch_callback(epoch + 1, val_loss, best_val_loss):
            print(f"[INFO] Training stopped by epoch_callback at epoch {epoch + 1}.")
            break

    # Step 5: Generate and export the history curves chart
    plot_training_curves(train_loss_history, val_loss_history, loss_setting, exp_dir)

    if use_wandb and init_wandb:
        wandb.finish()

    if final_cv_training:
        best_val_loss = None
        print(f"\n[INFO] Final cross-validation training finished after {best_epoch} epochs "
              f"(last epoch saved as best_model.pt).")
    else:
        print(f"\n[INFO] Optimization sequence finished. Best Validation {loss_setting}: {best_val_loss:.5f}")
    print(f"[INFO] All outputs and checkpoints archived inside: {exp_dir}")
    return {
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'exp_dir': exp_dir,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
    }


def main(config_path="configs/config.yml", epochs=None):
    config = load_config(config_path)
    if epochs is not None:
        config['epochs'] = epochs
    return run_training(config, config_path=config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the EA-LSTM flash flood model.")
    parser.add_argument("--config", type=str, default="configs/config.yml",
                         help="Path to the YAML config file for this run.")
    parser.add_argument("--epochs", type=int, default=None,
                         help="Override config['epochs']. For a cross_validation config this is the final "
                              "training length - use the winning sweep trial's avg_best_epoch.")
    args = parser.parse_args()
    main(args.config, epochs=args.epochs)
