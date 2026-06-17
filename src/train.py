"""
Module: train.py
Description: Main Training, Validation, and Optimization Pipeline for Dynamic Multi-Horizon EA-LSTM.
"""

import os
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
from loss import NSELoss, BatchAwareLossWrapper


def load_config(yaml_path):
    """Load the YAML configuration file safely."""
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def get_tracked_hparams(config):
    """Return the subset of config keys that genuinely affect model behavior."""
    keys = [
        'hidden_size', 'output_dropout', 'initial_forget_bias',
        'loss', 'nse_epsilon', 'learning_rate', 'initial_lr',
        'batch_size', 'epochs', 'seed',
        'target_noise_std', 'clip_gradient_norm',
        'seq_length', 'forecast_lead_times', 'use_basin_splits',
    ]
    tracked = {k: config[k] for k in keys if k in config}
    tracked['num_static_attributes'] = len(config.get('static_attributes', []))
    tracked['num_dynamic_inputs']    = len(config.get('dynamic_inputs', []))
    tracked['statics_embedding_type'] = config.get('statics_embedding', {}).get('type')
    return tracked


def set_seed(seed):
    """Enforce deterministic behavior across runs for scientific reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)


def update_learning_rate(optimizer, epoch, lr_schedule):
    """
    Scans the configuration schedule and updates the optimizer learning rate 
    if the current epoch matches a milestone boundary.
    """
    milestones = {int(k): float(v) for k, v in lr_schedule.items()}
    
    if epoch in milestones:
        new_lr = milestones[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"\n[INFO] Learning rate updated to {new_lr} for Epoch {epoch}.")


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
        
        # Regularization: Target Noise Injection (Train phase only)
        if model.training and noise_std > 0:
            noise = torch.randn_like(targets) * noise_std
            targets = targets + noise

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
        return BatchAwareLossWrapper(NSELoss(epsilon=config.get('nse_epsilon', 0.1)), uses_basin_std=True)
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
    plt.plot(epochs_range, val_losses, color="#cb181d", linewidth=2, linestyle="--", label="Validation Loss")
    
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


def main():
    # Step 1: Ingest setup assets and configuration parameters
    #CONFIG_PATH = "configs/config.yml"
    #config = load_config(CONFIG_PATH)
    if os.path.exists("configs/config_temp.yml"):
        CONFIG_PATH = "configs/config_temp.yml"
    else:
        CONFIG_PATH = "configs/config.yml"
        
    config = load_config(CONFIG_PATH)
    
    if os.path.exists("configs/config_temp.yml"):
        try:
            os.remove("configs/config_temp.yml")
        except:
            pass
    
    set_seed(config.get('seed', 42))
    
    device_str = config.get('device', 'cpu')
    device = torch.device(device_str if torch.cuda.is_available() or device_str == 'cpu' else 'cpu')
    print(f"[INFO] Execution target hardware configured to: {device}")

    # Read the spatial split flag directly from the configuration file (default to True if missing)
    use_spatial = config.get('use_basin_splits', True)
    if not use_spatial:
        print("[INFO] Spatial basin splits disabled via config. Using strict temporal configuration.")
    else:
        print("[INFO] Spatial basin splits enabled via config. Loading specific basin split files.")

    # Step 2: Initialize Data Pipeline Loaders
    print("[INFO] Constructing dataset pipelines and dataloaders...")
    train_loader = get_dataloader(split_type='train', config=config, use_basin_splits=use_spatial)
    val_loader = get_dataloader(split_type='val', config=config, use_basin_splits=use_spatial)

    # Step 3: Construct Architecture and Optimization Engines
    print("[INFO] Instantiating EA-LSTM model architecture dynamically...")
    model = EALSTMModel(config).to(device)
    
    loss_setting = config.get('loss', 'MSE')
    criterion = get_loss_criterion(loss_setting, config)
    print(f"[INFO] Optimization criterion set to: {loss_setting}")
    
    lr_schedule = config['learning_rate']
    initial_lr = float(lr_schedule.get(0, 1e-3))
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    
    run_dir = config.get('run_dir', './runs/')
    exp_dir = os.path.join(run_dir, config['experiment_name'])
    os.makedirs(exp_dir, exist_ok=True)

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

    # Trackers for saving checkpoints and plotting history
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []

    # Step 4: Core Training and Validation Loop Execution
    epochs = config.get('epochs', 30)
    print(f"[INFO] Initiating optimization loop for {epochs} epochs.\n")
    
    for epoch in range(epochs):
        update_learning_rate(optimizer, epoch, lr_schedule)
        
        # Part A: Execute Training Cycle
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config)
        
        # Part B: Execute Validation Cycle
        val_loss = validate_epoch(model, val_loader, criterion, device, config)
        
        # If RMSE is selected, calculate the physical square root error 
        if loss_setting.upper() == 'RMSE':
            train_loss = torch.sqrt(torch.tensor(train_loss)).item()
            val_loss = torch.sqrt(torch.tensor(val_loss)).item()
            metric_label = "RMSE"
        else:
            metric_label = "Loss"
            
        # Append to history trackers for downstream plotting
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] Completed:")
        print(f"  -> Train {metric_label}: {train_loss:.5f}")
        print(f"  -> Val {metric_label}:   {val_loss:.5f}")

        if use_wandb:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
            }, step=epoch + 1)
        
        # Part C: Strategic Model Selection (Save Best Weights)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(exp_dir, "best_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_checkpoint_path)
            print(f"Validation improvement detected. Saved as best_model.pt")
            if use_wandb:
                wandb.run.summary['best_val_loss'] = val_loss
                wandb.run.summary['best_epoch'] = epoch + 1
        
        # Part D: Standard Periodic Backup
        if (epoch + 1) % config.get('save_weights_every', 1) == 0:
            checkpoint_path = os.path.join(exp_dir, f"ealstm_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)

    # Step 5: Generate and export the history curves chart
    plot_training_curves(train_loss_history, val_loss_history, loss_setting, exp_dir)

    if use_wandb:
        wandb.finish()

    print(f"\n[INFO] Optimization sequence finished. Best Validation {loss_setting}: {best_val_loss:.5f}")
    print(f"[INFO] All outputs and checkpoints archived inside: {exp_dir}")


if __name__ == "__main__":
    main()