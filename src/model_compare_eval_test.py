"""
Module: model_compare_eval_test.py
Description: Combines model_compare_test.py's outputs into one per-model
             evaluation summary: event-level confusion matrix (TP/FP/FN)
             and precision/recall/F1 from flood_event_comparison.csv, plus
             the peak timing/magnitude summary stats from
             peaks_analysis_comparison_summary.csv. Reads only - does not
             re-run any model or recompute events; run model_compare_test.py
             first for the same comparison config.
"""

import argparse
import os
import yaml
import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

SCOPES = ('TP', 'TP_FN')


def load_config(yaml_path):
    """Load the YAML configuration file safely."""
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def _get_comparison_output_dir(comparison_config):
    run_dir = comparison_config.get('run_dir', './runs/model_comparisons/')
    return os.path.join(run_dir, comparison_config['comparison_name'])


def compute_confusion_matrix(events_df):
    """
    Per-model TP/FP/FN counts and precision/recall/F1 from
    flood_event_comparison.csv's 'label' column. F1 is computed directly
    from the counts (2*tp / (2*tp+fp+fn)) rather than from precision/recall,
    so it stays well-defined even when one of those is NaN.
    """
    rows = []
    for model_label, group in events_df.groupby('model_label'):
        counts = group['label'].value_counts()
        tp = int(counts.get('TP', 0))
        fp = int(counts.get('FP', 0))
        fn = int(counts.get('FN', 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float('nan')

        rows.append({
            'model_label': model_label, 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1,
        })

    return pd.DataFrame(rows)


def reshape_peaks_summary(peaks_summary_df):
    """
    Flattens peaks_analysis_comparison_summary.csv (one row per model x
    scope x {model_mean, model_variance}) into one row per model, with
    scope/mean-or-variance baked into the column name, e.g.
    time_distance_h_TP_mean, magnitude_diff_norm_TP_FN_var.
    """
    rows = []
    for model_label, model_df in peaks_summary_df.groupby('model_label'):
        row = {'model_label': model_label}

        lead_values = model_df['lead'].dropna()
        row['lead'] = lead_values.iloc[0] if not lead_values.empty else None

        for scope in SCOPES:
            scope_df = model_df[model_df['scope'] == scope]
            mean_row = scope_df[scope_df['basin_id'] == 'model_mean']
            var_row = scope_df[scope_df['basin_id'] == 'model_variance']

            row[f'n_events_{scope}'] = int(mean_row['n_events'].iloc[0]) if not mean_row.empty else 0
            for metric in ('time_distance_h', 'magnitude_diff_norm'):
                row[f'{metric}_{scope}_mean'] = mean_row[metric].iloc[0] if not mean_row.empty else float('nan')
                row[f'{metric}_{scope}_var'] = var_row[metric].iloc[0] if not var_row.empty else float('nan')

        rows.append(row)

    return pd.DataFrame(rows)


def build_eval_summary(output_dir):
    """
    Reads flood_event_comparison.csv and peaks_analysis_comparison_summary.csv
    from output_dir (both written by model_compare_test.py), and combines
    them into one per-model row. Returns None if either file is missing.
    """
    events_path = os.path.join(output_dir, "flood_event_comparison.csv")
    peaks_summary_path = os.path.join(output_dir, "peaks_analysis_comparison_summary.csv")

    if not os.path.exists(events_path) or not os.path.exists(peaks_summary_path):
        print(f"[Warning] Missing '{events_path}' and/or '{peaks_summary_path}'. "
              f"Run model_compare_test.py for this comparison config first.")
        return None

    events_df = pd.read_csv(events_path)
    peaks_summary_df = pd.read_csv(peaks_summary_path)

    confusion_df = compute_confusion_matrix(events_df)
    peaks_wide_df = reshape_peaks_summary(peaks_summary_df)

    return pd.merge(confusion_df, peaks_wide_df, on='model_label', how='outer')


def main(comparison_config_path="configs/compare_model_0_leads.yml"):
    comparison_config = load_config(comparison_config_path)
    output_dir = _get_comparison_output_dir(comparison_config)

    combined_df = build_eval_summary(output_dir)
    if combined_df is None:
        return None

    csv_path = os.path.join(output_dir, "model_comparison_eval.csv")
    combined_df.to_csv(csv_path, index=False)

    print("\nModel comparison evaluation summary:")
    print(combined_df.to_string(index=False))
    print(f"\n[INFO] Saved to: {csv_path}\n")

    use_wandb = comparison_config.get('use_wandb', False) and WANDB_AVAILABLE
    if use_wandb:
        api_key = comparison_config.get('wandb_api_key')
        if api_key:
            wandb.login(key=api_key)
        wandb.init(
            project=comparison_config.get('wandb_project', 'flash-floods-israel'),
            name=f"{comparison_config['comparison_name']}_eval",
            config=comparison_config,
        )
        for _, row in combined_df.iterrows():
            label = row['model_label']
            wandb.log({
                f"compare_eval/{label}/precision": row['precision'],
                f"compare_eval/{label}/recall": row['recall'],
                f"compare_eval/{label}/f1": row['f1'],
                f"compare_eval/{label}/time_distance_h_TP_mean": row['time_distance_h_TP_mean'],
                f"compare_eval/{label}/magnitude_diff_norm_TP_mean": row['magnitude_diff_norm_TP_mean'],
            })
        wandb.finish()

    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine model_compare_test.py's flood_event_comparison.csv and "
                     "peaks_analysis_comparison_summary.csv into one per-model confusion "
                     "matrix (TP/FP/FN), precision/recall/F1, and peak timing/magnitude summary CSV.")
    parser.add_argument("--config", type=str, default="configs/compare_model_0_leads.yml",
                         help="Path to the comparison config YAML (same one passed to model_compare_test.py).")
    args = parser.parse_args()

    main(args.config)
