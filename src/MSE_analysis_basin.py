"""
Module: MSE_analysis_basin.py
Description: Runs an already-trained model (config['checkpoint_path'], or
             run_dir/experiment_name/best_model.pt if unset) over one or more
             selected basins and, for each basin, draws pies of its MSE by
             hydrological year (top-N years + 'Other'), one pie per scope:
               - train: train_periods clipped to hydro-year >= 2000 (as MSE_analysis.py)
               - val / test: the split's own periods
               - full: the basin's whole timeseries record
             Each scope is evaluated on that basin alone, so the per-year
             breakdown of validate_epoch_with_breakdown is directly that basin's
             own - percentages are relative to the basin's MSE in that scope.
             A basin is evaluated on every scope's periods even if it isn't in
             that split's basin list (the pie title notes it). Independent of
             MSE_analysis_train.py / MSE_analysis_test.py outputs.
             Writes run_dir/experiment_name/sanity_analysis/basin_pies/{basin}/
             {scope}_year_pie.png plus {basin}_loss_by_year.csv.
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')  # file-only backend: never needs an X display (ssh -X / cluster nodes)
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from torch.utils.data import DataLoader

from train import load_config, get_loss_criterion
from dataset import IsraelBasinsDataset, _get_split_bounds_and_config, _resolve_num_workers
from train_sanity import validate_epoch_with_breakdown, to_value_and_pct, plot_breakdown_pie
from MSE_analysis import clip_train_periods, load_trained_model


def build_scopes(config, use_basin_splits):
    """[(scope_name, split_type, periods)] - split_type picks the NaN tolerance."""
    train_periods = [(p['start_date'], p['end_date']) for p in clip_train_periods(config['train_periods'])]
    val_periods = _get_split_bounds_and_config('val', config, use_basin_splits)[1]
    test_periods = _get_split_bounds_and_config('test', config, use_basin_splits)[1]
    return [
        ('train', 'train', train_periods),
        ('val', 'val', val_periods),
        ('test', 'test', test_periods),
        ('full', 'test', [(None, None)]),
    ]


def load_split_basin_lists(config):
    """{split_type: set of basin ids} from the split basin list files."""
    lists = {}
    for split_type, key in (('train', 'train_basin_file'), ('val', 'validation_basin_file'),
                            ('test', 'test_basin_file')):
        with open(config[key], 'r') as f:
            lists[split_type] = {line.strip() for line in f if line.strip()}
    return lists


def build_basin_loader(basin_id, split_type, periods, config, use_basin_splits):
    """Unshuffled, drop_last=False single-basin loader (see MSE_analysis.build_split_loader)."""
    dataset = IsraelBasinsDataset(split_type, config, use_basin_splits=use_basin_splits,
                                  basin_ids=[basin_id], periods=periods)
    return DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=_resolve_num_workers(config),
        drop_last=False,
    )


def run_basin_mse_analysis(config_path, basin_ids, top_n, use_wandb_override=None):
    config = load_config(config_path)

    cv_config = config.get('cross_validation', {}) or {}
    if cv_config.get('enabled', False):
        raise ValueError(
            "MSE_analysis_basin.py does not support cross_validation-enabled configs: it relies "
            "on IsraelBasinsDataset's sample_basin_mappings/sample_date_mappings, not present on "
            "the CV fold path. Use a config with cross_validation.enabled: false.")

    use_spatial = config.get('use_basin_splits', True)
    hydro_year_start_month = config.get('hydro_year_start_month', 10)

    run_dir = config.get('run_dir', './runs/')
    exp_dir = os.path.join(run_dir, config['experiment_name'])
    out_root = os.path.join(exp_dir, "sanity_analysis", "basin_pies")

    model, device = load_trained_model(config, exp_dir)
    criterion = get_loss_criterion(config.get('loss', 'MSE'), config)
    scopes = build_scopes(config, use_spatial)
    split_basin_lists = load_split_basin_lists(config) if use_spatial else {}

    use_wandb = config.get('use_wandb', False) if use_wandb_override is None else use_wandb_override
    if use_wandb:
        api_key = config.get('wandb_api_key')
        if api_key:
            wandb.login(key=api_key)
        wandb.init(
            project=config.get('wandb_project', 'flash-floods-israel'),
            name=f"{config['experiment_name']}_basin_pies",
            job_type='basin_mse_analysis',
            config={
                'experiment_name': config['experiment_name'],
                'checkpoint': config.get('checkpoint_path') or os.path.join(exp_dir, "best_model.pt"),
                'basins': list(basin_ids),
                'top_n': top_n,
                'scopes': {name: [list(p) for p in periods] for name, _, periods in scopes},
                'loss': config.get('loss', 'MSE'),
                'seq_length': config.get('seq_length'),
            },
        )
    mse_summary_rows = []  # one row per basin x scope, logged as a wandb table at the end

    for basin_id in basin_ids:
        basin_dir = os.path.join(out_root, basin_id)  # created by plot_breakdown_pie's savefig
        rows = []

        for scope_name, split_type, periods in scopes:
            print(f"[INFO] {basin_id} / {scope_name}: evaluating periods {periods}...")
            try:
                loader = build_basin_loader(basin_id, split_type, periods, config, use_spatial)
            except RuntimeError as e:
                print(f"  [Warning] {basin_id} / {scope_name}: no valid samples ({e}). Skipping.")
                continue

            _, _, year_sq_err_sum, total_elements = validate_epoch_with_breakdown(
                model, loader, criterion, device, hydro_year_start_month)
            values, pcts = to_value_and_pct(year_sq_err_sum, total_elements)
            basin_mse = sum(values.values())

            for year in sorted(year_sq_err_sum):
                rows.append({'scope': scope_name, 'year': year, 'sq_err_sum': year_sq_err_sum[year],
                             'total_elements': total_elements, 'value': values[year],
                             'percentage': pcts[year]})

            title = f"{scope_name.capitalize()} MSE of {basin_id} by Hydrological Year (MSE {basin_mse:.4g})"
            if scope_name != 'full' and use_spatial and basin_id not in split_basin_lists[split_type]:
                title += f"\n(not in {split_type} basin list)"
            fig = plot_breakdown_pie(
                {k: (values[k], pcts[k]) for k in values}, title,
                os.path.join(basin_dir, f"{scope_name}_year_pie.png"),
                top_n=top_n, group_label='years')
            print(f"  {basin_id} / {scope_name}: MSE {basin_mse:.5f} over {len(values)} years")
            mse_summary_rows.append({'basin_id': basin_id, 'scope': scope_name, 'mse': basin_mse,
                                     'n_years': len(values), 'total_elements': total_elements})

            if use_wandb:
                wandb.log({f'basin_pies/{basin_id}/{scope_name}_year_pie': wandb.Image(fig)})
                wandb.run.summary[f'{basin_id}/{scope_name}_mse'] = basin_mse
            plt.close(fig)

        if not rows:
            print(f"[Warning] {basin_id}: no scope had valid samples - nothing written.")
            continue
        csv_path = os.path.join(basin_dir, f"{basin_id}_loss_by_year.csv")
        basin_df = pd.DataFrame(rows, columns=['scope', 'year', 'sq_err_sum', 'total_elements', 'value',
                                               'percentage'])
        basin_df.to_csv(csv_path, index=False)
        print(f"[INFO] {basin_id}: pies and {csv_path} written inside {basin_dir}")
        if use_wandb:
            wandb.log({f'basin_pies/{basin_id}/loss_by_year': wandb.Table(dataframe=basin_df)})

    if use_wandb:
        if mse_summary_rows:
            wandb.log({'basin_pies/mse_by_basin_scope': wandb.Table(dataframe=pd.DataFrame(mse_summary_rows))})
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-basin MSE of a trained model by hydrological year (top-N years pie) "
                     "for the train/val/test periods and the basin's full record.")
    parser.add_argument("--config", type=str, default="configs/config.yml",
                         help="Path to the YAML config file for this run.")
    parser.add_argument("--basins", type=str, nargs='+', required=True,
                         help="One or more basin ids, e.g. il_12130 il_8146.")
    parser.add_argument("--top_n", type=int, default=5,
                         help="Number of years shown as their own slice; the rest fold into 'Other'.")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None,
                         help="Force W&B logging on (--wandb) or off (--no-wandb). "
                              "Defaults to the config's use_wandb.")
    args = parser.parse_args()
    run_basin_mse_analysis(args.config, args.basins, args.top_n, use_wandb_override=args.wandb)
