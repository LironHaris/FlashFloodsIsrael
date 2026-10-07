"""
Module: run_sweep.py
Description: W&B sweep agent entry point for Bayesian hyperparameter tuning.

Each call by the sweep agent runs one full training trial:
  1. wandb.init() picks up the trial's hyperparameters from the agent.
  2. Base config.yml is loaded and overridden with the swept values.
  3. train.run_training trains one model on the configured train/validation
     split (normalization statistics computed fresh from train_periods for
     this trial - no state shared between trials) and logs to this run.
  4. The objective is 'val_loss' (logged every epoch; best value in the run
     summary as 'best_val_loss'); best checkpoint is saved per trial.

Trials stop early on early_stop_patience (default 10) or, when early drop is
enabled in the sweep YAML, at early_drop_epoch if the trial's best_val_loss
is not in the top-k of the sweep's finished runs.

For k-fold cross-validation configs (cross_validation.enabled: true) use
cross_validation_sweep.py instead - this script rejects them.

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
import wandb

# Ensure src/ is on the path when called directly by the W&B agent
sys.path.insert(0, os.path.dirname(__file__))

from train import get_tracked_hparams, run_training


def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def is_below_top_k(project, sweep_id, current_run_id, best_val_loss_so_far, top_k,
                   metric_key='best_val_loss'):
    """True if this run's best_val_loss isn't competitive with the top_k
    best *finished* runs in the sweep so far, compared on summary[metric_key]
    (cross_validation_sweep.py compares a fold with the same fold of other
    trials via metric_key='fold_{j}_best_val_loss'). False if too few exist to compare."""
    api = wandb.Api()
    sweep = api.sweep(f"{project}/{sweep_id}")
    finished_losses = sorted(
        r.summary[metric_key]
        for r in sweep.runs
        if r.id != current_run_id and r.state == 'finished' and r.summary.get(metric_key) is not None
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
    if (config.get('cross_validation') or {}).get('enabled', False):
        raise ValueError("This base config has cross_validation.enabled - sweep it with "
                         "cross_validation_sweep.py (configs/sweep_crossval.yaml), not run_sweep.py.")

    # Push non-swept relevant hparams (epochs, forecast_lead_times, etc.) into the W&B run
    wandb.config.update(get_tracked_hparams(config))

    # Unique output directory per trial — prevents checkpoint collisions across parallel runs
    run_dir = config.get('run_dir', './runs/')
    exp_dir = os.path.join(run_dir, config['experiment_name'], wandb.run.id)

    def early_drop(epoch, val_loss, best_val_loss):
        """Prune this trial at early_drop_epoch if it isn't in the sweep's top-k so far."""
        if not early_drop_enabled or epoch != early_drop_epoch:
            return False
        if is_below_top_k(project, wandb.run.sweep_id, wandb.run.id, best_val_loss, early_drop_top_k):
            print(f"[Early Drop] best_val_loss={best_val_loss:.4f} not in top {early_drop_top_k} "
                  f"at epoch {epoch}. Abandoning this configuration.")
            wandb.run.summary['early_dropped'] = True
            return True
        return False

    run_training(
        config,
        exp_dir=exp_dir,
        use_wandb=True,
        init_wandb=False,           # this agent owns the wandb run
        epoch_callback=early_drop,
        early_stop_patience=config.get('early_stop_patience', 10),
        save_periodic=False,        # only best_model.pt per trial
    )

    wandb.finish()


if __name__ == "__main__":
    run_trial()
