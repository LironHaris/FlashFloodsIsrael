"""
Module: train.py
Description: Main Training, Validation, and Optimization Pipeline for Dynamic Multi-Horizon EA-LSTM.

This script coordinates the training and validation workflow, reading hyperparameters directly from 
the configuration file, instantiating the models and loaders, and optimizing weights.

Operational Processes:
1. Learning Rate Scheduling: Manually updates the optimizer parameters based on epoch-bounded 
   thresholds defined inside the 'learning_rate' configuration dictionary.
2. Target Regularization (Noise Injection): Adds standard Gaussian noise to target labels 
   during training phases based on 'target_noise_std' to boost model generalization.
3. Gradient Stabilization: Constraints gradient explosions using norm clipping bounded by 
   'clip_gradient_norm'.
4. Validation & Model Selection: Executes an evaluation pass at the end of each epoch without 
   gradient tracking to monitor generalization performance and isolate the 'best_model.pt'.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from model import EALSTMModel
from dataset import get_dataloader


def load_config(yaml_path):
    """Load the YAML configuration file safely."""
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


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
        # Move data tensors to chosen compute device
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
        
        loss = criterion(predictions, targets)
        loss.backward()
        
        # Gradient Stabilization via Norm Clipping
        if clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            
        optimizer.step()
        
        # Metrics accounting
        running_loss += loss.item()
        
        if batch_idx % log_interval == 0:
            progress_bar.set_postfix({'Batch Loss': f'{loss.item():.5f}'})
            
    return running_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device, config):
    """Evaluates performance on validation basins to monitor overfitting boundaries."""
    model.eval() # Deactivate regularization layers (Dropout, Noise injection)
    running_val_loss = 0.0
    
    # Context manager disabling gradient tracking to conserve execution memory
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Validation Batches", leave=False):
            x_dynamic = batch['dynamic'].to(device)
            x_static = batch['static'].to(device)
            targets = batch['target'].to(device)
            
            predictions = model(x_dynamic, x_static)
            loss = criterion(predictions, targets)
            
            running_val_loss += loss.item()
            
    return running_val_loss / len(dataloader)


def get_loss_criterion(loss_name: str):
    """
    Dynamically instantiates the target loss function based on the configuration string.
    Supports native PyTorch losses and prepares slots for custom hydrological losses like NSE.
    """
    name = loss_name.upper()
    
    if name == 'MSE' or name == 'RMSE':
        return nn.MSELoss()
    elif name == 'NSE':
        print("[WARNING] NSE selected. Falling back to MSE until custom NSE class is integrated.")
        return nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function specified in config: {loss_name}")


def main():
    # Step 1: Ingest setup assets and configuration parameters
    CONFIG_PATH = "configs/config.yml"
    config = load_config(CONFIG_PATH)
    
    set_seed(config.get('seed', 42))
    
    # Establish device routing (CPU / CUDA / MPS)
    device_str = config.get('device', 'cpu')
    device = torch.device(device_str if torch.cuda.is_available() or device_str == 'cpu' else 'cpu')
    print(f"[INFO] Execution target hardware configured to: {device}")

    # Step 2: Initialize Data Pipeline Loaders
    print("[INFO] Constructing dataset pipelines and dataloaders...")
    train_loader = get_dataloader(split_type='train', config=config)
    val_loader = get_dataloader(split_type='val', config=config)

    # Step 3: Construct Architecture and Optimization Engines
    print("[INFO] Instantiating EA-LSTM model architecture dynamically...")
    model = EALSTMModel(config).to(device)
    
    # Dynamically extract and instantiate the loss function from configuration
    loss_setting = config.get('loss', 'MSE')
    criterion = get_loss_criterion(loss_setting)
    print(f"[INFO] Optimization criterion set to: {loss_setting}")
    
    # Initialize optimizer with the base rate (Epoch 0 value)
    lr_schedule = config['learning_rate']
    initial_lr = float(lr_schedule.get(0, 1e-3))
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    
    # Directory mapping for model checkpoints
    run_dir = config.get('run_dir', './runs/')
    exp_dir = os.path.join(run_dir, config['experiment_name'])
    os.makedirs(exp_dir, exist_ok=True)

    # Model Selection Tracker variables
    best_val_loss = float('inf')

    # Step 4: Core Training and Validation Loop Execution
    epochs = config.get('epochs', 30)
    print(f"[INFO] Initiating optimization loop for {epochs} epochs.\n")
    
    for epoch in range(epochs):
        # Trigger learning rate adjustments dynamically based on config milestones
        update_learning_rate(optimizer, epoch, lr_schedule)
        
        # Part A: Execute Training Cycle
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config)
        
        # Part B: Execute Validation Cycle
        val_loss = validate_epoch(model, val_loader, criterion, device, config)
        
        # If RMSE is selected, log the physical square root error for better metric clarity
        if loss_setting.upper() == 'RMSE':
            train_loss = torch.sqrt(torch.tensor(train_loss)).item()
            val_loss = torch.sqrt(torch.tensor(val_loss)).item()
            metric_label = "RMSE"
        else:
            metric_label = "Loss"
            
        print(f"Epoch [{epoch+1}/{epochs}] Completed:")
        print(f"  -> Train {metric_label}: {train_loss:.5f}")
        print(f"  -> Val {metric_label}:   {val_loss:.5f}")
        
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
        
        # Part D: Standard Periodic Backup
        if (epoch + 1) % config.get('save_weights_every', 1) == 0:
            checkpoint_path = os.path.join(exp_dir, f"ealstm_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)

    print(f"\n[INFO] Optimization sequence finished. Best Validation {loss_setting}: {best_val_loss:.5f}")
    print(f"[INFO] All outputs and checkpoints archived inside: {exp_dir}")


if __name__ == "__main__":
    main()