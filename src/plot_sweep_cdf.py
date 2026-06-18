"""
Module: plot_sweep_cdf.py
Description: Regenerates the overlaid NSE CDF comparison plots from a
             nse_cache_<sweep_id>.json file produced by test_sweep.py,
             without re-running model inference. Optionally logs the
             regenerated figures to W&B.

Usage:
    python src/plot_sweep_cdf.py <path/to/nse_cache_SWEEPID.json>
    python src/plot_sweep_cdf.py <path/to/nse_cache_SWEEPID.json> --no-wandb
"""

import os
import re
import sys
import json
import argparse
import matplotlib.pyplot as plt
import wandb

sys.path.insert(0, os.path.dirname(__file__))

from test_sweep import plot_cdf_comparison, load_config


def load_nse_cache(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return {int(lead): [(label, vals) for label, vals in runs] for lead, runs in raw.items()}


def extract_sweep_id(cache_path):
    name = os.path.basename(cache_path)
    match = re.match(r'nse_cache_(.+)\.json$', name)
    if not match:
        raise ValueError(f"Could not parse sweep_id from cache filename: {name}")
    return match.group(1)


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate NSE CDF comparison plots from a cached nse_cache_*.json file.'
    )
    parser.add_argument('cache_path', help='Path to nse_cache_<sweep_id>.json')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Skip W&B logging, only regenerate local PNGs')
    args = parser.parse_args()

    sweep_id = extract_sweep_id(args.cache_path)
    nse_per_lead = load_nse_cache(args.cache_path)
    output_dir = os.path.dirname(os.path.abspath(args.cache_path))

    wandb_run = None
    if not args.no_wandb:
        base_config = load_config('configs/config.yml')
        sweep_config = load_config('configs/sweep.yaml')

        api_key = base_config.get('wandb_api_key')
        if api_key:
            wandb.login(key=api_key)

        project = sweep_config.get('project', base_config.get('wandb_project', 'flash-floods-israel'))
        wandb_run = wandb.init(
            project=project,
            name=f"sweep_replot_{sweep_id}",
            job_type="test",
            config={"sweep_id": sweep_id, "source_cache": args.cache_path},
        )

    for lead, runs in nse_per_lead.items():
        if any(vals for _, vals in runs):
            fig = plot_cdf_comparison(lead, runs, output_dir)
            if fig is not None:
                if wandb_run is not None:
                    wandb.log({f"test/nse_cdf/lead_{lead}h": wandb.Image(fig)})
                plt.close(fig)
        else:
            print(f"  [SKIP] Lead +{lead}h — no NSE values in cache.")

    print(f"\n[INFO] All plots re-saved to: {output_dir}")

    if wandb_run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
