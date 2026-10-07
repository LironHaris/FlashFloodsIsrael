"""
Module: cross_validation_sweep.py
Description: W&B sweep agent entry point for Bayesian hyperparameter tuning with
             k-fold cross-validation over hydrological-year groups.

Each call by the sweep agent runs one trial = one hyperparameter set:
  1. wandb.init() picks up the trial's hyperparameters; the base config
     (FLASHFLOODS_CONFIG, must have cross_validation.enabled: true) is
     overridden with them.
  2. dataset.build_fold_configs splits it into k folds - fold j holds out
     cross_validation.groups[j]'s years as validation and trains on the rest of
     train_periods (pool years in no group are always trained on).
  3. The k folds are trained sequentially, each as a fully independent model
     (train.run_training): its own normalization statistics from its own train
     years (fold_j/normalization_stats.json - the held-out years never leak in),
     its own early stopping (early_stop_patience), its own best epoch/weights.
  4. Only after ALL k folds reached their own best weights, the objective is
     computed: val_loss = mean of the folds' best val_loss (the metric the
     Bayesian search minimizes), and avg_best_epoch = int(mean of the folds'
     best epochs) - the epoch count for the final training on all data:
       python src/train.py --config <cv config> --epochs <avg_best_epoch>

Early drop (top-k), per fold: at early_drop_epoch of fold j, if this fold's
best val_loss so far isn't within the top-k of the SAME fold's best val_loss
(summary 'fold_{j}_best_val_loss') among the sweep's finished trials, the trial
is abandoned (remaining folds skipped, early_dropped=True, no val_loss logged
so it can't mislead the search). With fewer than k finished trials nothing is
dropped - the first k trials always run in full.

W&B logging: each fold logs 'fold_{j}/train_loss', 'fold_{j}/val_loss' and
'fold_{j}/epoch'; the summary holds fold_{j}_best_val_loss, fold_{j}_best_epoch,
val_loss / best_val_loss (the CV mean), avg_best_epoch. Outputs per trial in
run_dir/experiment_name/<run_id>/: fold_{j}/ (best_model.pt, normalization
stats, config) and cv_summary.json.

Usage:
  wandb sweep configs/sweep_crossval.yaml                       # register sweep, prints SWEEP_ID
  FLASHFLOODS_CONFIG=configs/cross_val_0_3.yml \\
  FLASHFLOODS_SWEEP_CONFIG=configs/sweep_crossval.yaml wandb agent <SWEEP_ID>
  (on the cluster: sbatch cluster/run_cv_sweep.sh <SWEEP_ID> <config> configs/sweep_crossval.yaml)
"""

import json
import os
import sys

import numpy as np
import yaml
import wandb

# Ensure src/ is on the path when called directly by the W&B agent
sys.path.insert(0, os.path.dirname(__file__))

from dataset import build_fold_configs
from run_sweep import is_below_top_k
from train import get_tracked_hparams, run_training


def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_cv_trial():
    # Same agent conventions as run_sweep.py: hyperparameters arrive via
    # wandb.config, the base config / sweep YAML paths via environment variables.
    base_config_path = os.environ.get('FLASHFLOODS_CONFIG', 'configs/config.yml')
    base_config = load_config(base_config_path)
    sweep_config_path = os.environ.get('FLASHFLOODS_SWEEP_CONFIG', 'configs/sweep_crossval.yaml')
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
    if not (config.get('cross_validation') or {}).get('enabled', False):
        raise ValueError(f"{base_config_path} has no cross_validation.enabled: true - "
                         "use run_sweep.py for fixed train/validation configs.")
    wandb.config.update(get_tracked_hparams(config))

    fold_configs = build_fold_configs(config)
    num_folds = len(fold_configs)
    patience = config.get('early_stop_patience', 10)

    run_dir = config.get('run_dir', './runs/')
    trial_dir = os.path.join(run_dir, config['experiment_name'], wandb.run.id)
    os.makedirs(trial_dir, exist_ok=True)
    print(f"[INFO] Cross-validation trial {wandb.run.id}: {num_folds} folds, "
          f"early_stop_patience={patience}, outputs in {trial_dir}")

    fold_scores, fold_best_epochs = [], []
    dropped_at_fold = None

    for j, fold_config in enumerate(fold_configs):
        print(f"\n===== Fold {j + 1}/{num_folds}: validating on years "
              f"{config['cross_validation']['groups'][j]} =====")

        drop_flag = {'dropped': False}

        def fold_early_drop(epoch, val_loss, best_val_loss, j=j, drop_flag=drop_flag):
            """Compare this fold only with the same fold of the sweep's finished trials."""
            if not early_drop_enabled or epoch != early_drop_epoch:
                return False
            if is_below_top_k(project, wandb.run.sweep_id, wandb.run.id, best_val_loss,
                              early_drop_top_k, metric_key=f'fold_{j}_best_val_loss'):
                print(f"[Early Drop] fold {j}: best_val_loss={best_val_loss:.4f} not in the top "
                      f"{early_drop_top_k} of this fold at epoch {epoch}. Abandoning this configuration.")
                drop_flag['dropped'] = True
                return True
            return False

        result = run_training(
            fold_config,
            exp_dir=os.path.join(trial_dir, f'fold_{j}'),
            use_wandb=True,
            init_wandb=False,              # this agent owns the wandb run
            epoch_callback=fold_early_drop,
            early_stop_patience=patience,
            save_periodic=False,           # only each fold's best_model.pt
            wandb_prefix=f'fold_{j}/',
            write_summary=False,
        )

        if drop_flag['dropped']:
            dropped_at_fold = j
            break

        fold_scores.append(result['best_val_loss'])
        fold_best_epochs.append(result['best_epoch'])
        wandb.run.summary[f'fold_{j}_best_val_loss'] = result['best_val_loss']
        wandb.run.summary[f'fold_{j}_best_epoch'] = result['best_epoch']

    cv_summary = {
        'run_id': wandb.run.id,
        'hyperparameters': get_tracked_hparams(config),
        'groups': config['cross_validation']['groups'],
        'fold_scores': fold_scores,
        'fold_best_epochs': fold_best_epochs,
    }

    if dropped_at_fold is not None:
        # No val_loss: an incomplete trial must not look like a finished score to the search.
        wandb.run.summary['early_dropped'] = True
        wandb.run.summary['early_dropped_fold'] = dropped_at_fold
        cv_summary.update(early_dropped=True, early_dropped_fold=dropped_at_fold)
        print(f"[INFO] Trial dropped at fold {dropped_at_fold}.")
    else:
        mean_val_loss = float(np.mean(fold_scores))
        avg_best_epoch = int(np.mean(fold_best_epochs))
        wandb.log({'val_loss': mean_val_loss, 'avg_best_epoch': avg_best_epoch})
        wandb.run.summary['val_loss'] = mean_val_loss
        wandb.run.summary['best_val_loss'] = mean_val_loss
        wandb.run.summary['avg_best_epoch'] = avg_best_epoch
        wandb.run.summary['early_dropped'] = False
        cv_summary.update(val_loss=mean_val_loss, avg_best_epoch=avg_best_epoch, early_dropped=False)
        print(f"\n[INFO] CV mean best val_loss over {num_folds} folds: {mean_val_loss:.5f} | "
              f"fold best epochs {fold_best_epochs} -> avg_best_epoch {avg_best_epoch}")

    with open(os.path.join(trial_dir, 'cv_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(cv_summary, f, indent=1)

    wandb.finish()


if __name__ == "__main__":
    run_cv_trial()
