"""
Module: model_compare_events_by_year.py
Description: Full model_compare_test.py + model_compare_eval_test.py output
             suite (NSE empirical CDF, per-basin hydrographs, N-way flood-event
             TP/FN/FP classification, peak timing/magnitude analysis, and the
             final per-model precision/recall/F1 summary), but evaluated over
             arbitrary (possibly non-contiguous) hydrological year(s) - inside
             or outside each model's configured test split - instead of being
             pinned to it, for any number of models. Evaluates via
             model_compare_test.evaluate_model_over_years (same mechanism
             plot_events_by_year.py uses), then reuses
             model_compare_test.run_event_and_peaks_analysis's period-aware
             mode and model_compare_eval_test.build_eval_summary unchanged.
             Writes into its own {comparison_name}_years_{years_tag} output
             directory and evaluates under a years-tagged experiment_name, so
             running this for the same model against different year sets
             never collides on the same report path (unlike
             plot_events_by_year.py's plain {experiment_name}_custom_period,
             which is only ever invoked with one year-set per output folder).
"""

import argparse
import os

import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import dataset
from model_compare_test import (
    load_config, validate_comparable, evaluate_model_over_years,
    compute_model_nse, plot_nse_cdf_comparison, run_event_and_peaks_analysis,
)
from model_compare_eval_test import build_eval_summary


def main(comparison_config_path, years):
    comparison_config = load_config(comparison_config_path)
    model_config_paths = comparison_config['model_configs']
    model_labels_cfg = comparison_config.get('model_labels')

    print("=" * 75)
    print("      Multi-Model Comparison — Year-Scoped")
    print("=" * 75)
    print(f"[INFO] Comparing {len(model_config_paths)} models: {model_config_paths}")
    print(f"[INFO] Hydrological year(s): {years}")

    model_configs, model_labels, shared_basins, model_leads = validate_comparable(
        model_config_paths, model_labels_cfg
    )
    print(f"[INFO] Validation passed. {len(shared_basins)} shared test basins. Leads: {model_leads}")

    hydro_year_start_month = model_configs[0].get('hydro_year_start_month', 10)
    periods = dataset.build_year_periods(years, hydro_year_start_month)
    print(f"[INFO] Evaluation period(s): {periods}")

    years_tag = "_".join(str(y) for y in sorted(set(years)))

    # Evaluate under a years-tagged experiment_name so this never overwrites
    # the model's real (configured-test-split) visual_report_basin_*.csv, and
    # so a different --years run of this same script against the same model
    # never collides on the same report path either (checkpoint_path is
    # inherited unchanged - only the *output* location changes).
    eval_configs = [{**config, 'experiment_name': f"{config['experiment_name']}_custom_period_{years_tag}"}
                     for config in model_configs]

    # Namespaced at the directory level (distinct from model_compare_test.py's
    # own output_dir for this comparison_name) so filenames inside can stay
    # identical to model_compare_test.py's - which is what lets
    # model_compare_eval_test.build_eval_summary work unmodified below.
    output_dir = os.path.join(comparison_config.get('run_dir', './runs/model_comparisons/'),
                               f"{comparison_config['comparison_name']}_years_{years_tag}")
    os.makedirs(output_dir, exist_ok=True)

    use_wandb = comparison_config.get('use_wandb', False) and WANDB_AVAILABLE
    if use_wandb:
        api_key = comparison_config.get('wandb_api_key')
        if api_key:
            wandb.login(key=api_key)
        wandb.init(
            project=comparison_config.get('wandb_project', 'flash-floods-israel'),
            name=f"{comparison_config['comparison_name']}_events_by_year_{years_tag}",
            config={**comparison_config, 'years': years},
        )

    # Step 1: load each checkpoint and run inference over exactly the
    # requested period(s) - not the configured test split. Always recomputes
    # (no skip-if-exists caching, to avoid stale results).
    print("\n[INFO] Evaluating model(s) over the requested period(s)...")
    for label, config in zip(model_labels, eval_configs):
        print(f"  Evaluating '{label}' ({config['experiment_name']})...")
        evaluate_model_over_years(config, shared_basins, periods)

    # Step 2: comparison NSE CDF - one curve per model, same figure.
    model_nse = compute_model_nse(eval_configs, model_labels, model_leads, shared_basins)
    cdf_fig = plot_nse_cdf_comparison(model_nse, model_leads, comparison_config, output_dir)
    if cdf_fig is not None:
        if use_wandb:
            wandb.log({"compare/nse_cdf": wandb.Image(cdf_fig)})
        plt.close(cdf_fig)

    # Step 3: per-basin target-time merge, N-way flood-event
    # union/classification, hydrographs, and peaks - scanned period-aware so
    # events from disjoint requested years are never spuriously merged.
    run_event_and_peaks_analysis(eval_configs, model_labels, model_leads, shared_basins,
                                  output_dir, use_wandb, periods=periods, years=years)

    # Step 4: combine into the final per-model confusion-matrix/precision/
    # recall/F1 + peaks summary (same as running model_compare_eval_test.py
    # separately, done here so one invocation produces everything).
    combined_df = build_eval_summary(output_dir)
    if combined_df is not None:
        csv_path = os.path.join(output_dir, "model_comparison_eval.csv")
        combined_df.to_csv(csv_path, index=False)
        print("\nModel comparison evaluation summary:")
        print(combined_df.to_string(index=False))
        print(f"[INFO] Saved to: {csv_path}")

        if use_wandb:
            for _, row in combined_df.iterrows():
                label = row['model_label']
                wandb.log({
                    f"compare_eval/{label}/precision": row['precision'],
                    f"compare_eval/{label}/recall": row['recall'],
                    f"compare_eval/{label}/f1": row['f1'],
                    f"compare_eval/{label}/time_distance_h_TP_mean": row.get('time_distance_h_TP_mean'),
                    f"compare_eval/{label}/magnitude_diff_norm_TP_mean": row.get('magnitude_diff_norm_TP_mean'),
                })

    if use_wandb:
        wandb.finish()

    print(f"\n[INFO] Model comparison (year-scoped) completed. Outputs in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare any number of models (NSE CDF, hydrographs, flood-event "
                     "TP/FN/FP classification, peaks analysis, and the final precision/"
                     "recall/F1 summary) over arbitrary hydrological year(s) - inside or "
                     "outside each model's configured test split - instead of being pinned "
                     "to it.")
    parser.add_argument("--config", type=str, required=True,
                         help="Path to a comparison config YAML (same format as "
                              "model_compare_test.py; a single-entry model_configs list works too).")
    parser.add_argument("--years", type=int, nargs='+', required=True,
                         help="One or more hydrological years (Oct->Sep) to evaluate and compare over.")
    args = parser.parse_args()
    main(args.config, args.years)
