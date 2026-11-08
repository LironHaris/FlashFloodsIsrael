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

import find_flood_events as ffe

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


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float('nan')
    return precision, recall, f1


def compute_cumulative_confusion(events_df, label_leads):
    """
    Cumulative-lead TP/FP/FN and precision/recall/F1 from flood_event_comparison.csv,
    one row per model label in label_leads ({model_label: lead}; other labels' rows
    are ignored, e.g. another family in the same comparison). For a model with lead L,
    an event counts as predicted when ANY label with lead >= L predicted it (label TP
    or FP), and as missed only when none did - so the longest lead is unchanged and
    each shorter lead adds its own detections to those of the longer leads. An event
    is real when its row is TP or FN (same for every model of that event). Events
    neither real nor predicted by the set (TN) are not counted.
    """
    columns = ['model_label', 'lead', 'cum_leads', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1']
    df = events_df[events_df['model_label'].isin(label_leads)]
    if df.empty:
        return pd.DataFrame(columns=columns)
    df = df.assign(real=df['label'].isin(['TP', 'FN']), pred=df['label'].isin(['TP', 'FP']),
                   lead=df['model_label'].map(label_leads))
    keys = ['basin_id', 'event_idx']
    real = df.groupby(keys)['real'].any()

    rows = []
    for label, lead in sorted(label_leads.items(), key=lambda kv: -kv[1]):
        cum_leads = sorted({l for l in label_leads.values() if l >= lead}, reverse=True)
        pred = df[df['lead'] >= lead].groupby(keys)['pred'].any().reindex(real.index, fill_value=False)
        tp = int((real & pred).sum())
        fn = int((real & ~pred).sum())
        fp = int((~real & pred).sum())
        precision, recall, f1 = _prf(tp, fp, fn)
        rows.append({'model_label': label, 'lead': lead, 'cum_leads': ';'.join(str(l) for l in cum_leads),
                     'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1})
    return pd.DataFrame(rows, columns=columns)


def infer_cumulative_groups(label_leads, explicit_groups=None):
    """
    {group: {label: lead}} - explicit_groups ({group: [labels]}) if given, else each
    label with its own '_{lead}' suffix stripped (best_crossval_1_3_12 -> best_crossval_1_3).
    Groups holding two labels with the same lead are dropped with a warning.
    """
    groups = {}
    if explicit_groups:
        for name, labels in explicit_groups.items():
            groups[name] = {l: label_leads[l] for l in labels if l in label_leads}
    else:
        for label, lead in label_leads.items():
            suffix = f"_{lead}"
            name = label[:-len(suffix)] if label.endswith(suffix) else label
            groups.setdefault(name, {})[label] = lead
    valid = {}
    for name, members in groups.items():
        if len(set(members.values())) != len(members):
            print(f"[Warning] Cumulative group '{name}' has duplicate leads {members} - skipped. "
                  f"Set cumulative_groups in the comparison config.")
            continue
        valid[name] = members
    return valid


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


def build_eval_summary(output_dir, label):
    """
    Reads flood_event_comparison_{label}.csv and
    peaks_analysis_comparison_summary_{label}.csv from output_dir (both
    written by model_compare_test.py/model_compare_events_by_year.py for that
    one prediction-threshold spec), and combines them into one per-model row.
    Returns None if either file is missing.
    """
    events_path = os.path.join(output_dir, f"flood_event_comparison_{label}.csv")
    peaks_summary_path = os.path.join(output_dir, f"peaks_analysis_comparison_summary_{label}.csv")

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
    """
    Runs build_eval_summary SEPARATELY for every prediction_threshold spec
    configured on the comparison's first model config, writing one
    model_comparison_eval_{label}.csv per threshold. Returns {label: DataFrame}.
    """
    comparison_config = load_config(comparison_config_path)
    output_dir = _get_comparison_output_dir(comparison_config)

    model_configs = [load_config(p) for p in comparison_config['model_configs']]
    first_model_config = model_configs[0]
    prediction_specs = ffe.normalize_threshold_specs(first_model_config['prediction_threshold'])
    model_labels = comparison_config.get('model_labels') or [c['experiment_name'] for c in model_configs]
    label_leads = {label: c['forecast_lead_times'][0] for label, c in zip(model_labels, model_configs)}
    cumulative_groups = infer_cumulative_groups(label_leads, comparison_config.get('cumulative_groups'))

    use_wandb = comparison_config.get('use_wandb', False) and WANDB_AVAILABLE
    if use_wandb:
        api_key = comparison_config.get('wandb_api_key')
        if api_key:
            wandb.login(key=api_key)
        wandb.init(
            project=comparison_config.get('wandb_project', 'flash-floods-israel'),
            name=f"{comparison_config['comparison_name']}_eval",
            config={k: v for k, v in comparison_config.items() if k != 'wandb_api_key'},
        )

    combined_dfs = {}
    for spec in prediction_specs:
        label = ffe.threshold_label(spec)
        combined_df = build_eval_summary(output_dir, label)
        if combined_df is None:
            continue
        combined_dfs[label] = combined_df

        csv_path = os.path.join(output_dir, f"model_comparison_eval_{label}.csv")
        combined_df.to_csv(csv_path, index=False)

        print(f"\nModel comparison evaluation summary [{label}]:")
        print(combined_df.to_string(index=False))
        print(f"\n[INFO] Saved to: {csv_path}\n")

        if use_wandb:
            for _, row in combined_df.iterrows():
                model_label = row['model_label']
                wandb.log({
                    f"compare_eval/{label}/{model_label}/precision": row['precision'],
                    f"compare_eval/{label}/{model_label}/recall": row['recall'],
                    f"compare_eval/{label}/{model_label}/f1": row['f1'],
                    f"compare_eval/{label}/{model_label}/time_distance_h_TP_mean": row['time_distance_h_TP_mean'],
                    f"compare_eval/{label}/{model_label}/magnitude_diff_norm_TP_mean": row['magnitude_diff_norm_TP_mean'],
                })

        # Cumulative-lead statistics, per model within its family group
        events_df = pd.read_csv(os.path.join(output_dir, f"flood_event_comparison_{label}.csv"))
        cum_frames = []
        for group, members in cumulative_groups.items():
            cum_df = compute_cumulative_confusion(events_df, members)
            if not cum_df.empty:
                cum_frames.append(cum_df.assign(group=group))
        if cum_frames:
            cum_df = pd.concat(cum_frames, ignore_index=True)
            cum_df = cum_df[['group'] + [c for c in cum_df.columns if c != 'group']]
            cum_path = os.path.join(output_dir, f"model_comparison_eval_cumulative_{label}.csv")
            cum_df.to_csv(cum_path, index=False)
            print(f"\nCumulative-lead evaluation summary [{label}]:")
            print(cum_df.to_string(index=False))
            print(f"\n[INFO] Saved to: {cum_path}\n")
            if use_wandb:
                wandb.log({f"compare_eval_cumulative/{label}/table": wandb.Table(dataframe=cum_df)})
                for _, row in cum_df.iterrows():
                    wandb.log({f"compare_eval_cumulative/{label}/{row['model_label']}/{m}": row[m]
                               for m in ('precision', 'recall', 'f1')})

    if use_wandb:
        wandb.finish()

    return combined_dfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine model_compare_test.py's flood_event_comparison.csv and "
                     "peaks_analysis_comparison_summary.csv into one per-model confusion "
                     "matrix (TP/FP/FN), precision/recall/F1, and peak timing/magnitude summary CSV.")
    parser.add_argument("--config", type=str, default="configs/compare_model_0_leads.yml",
                         help="Path to the comparison config YAML (same one passed to model_compare_test.py).")
    args = parser.parse_args()

    main(args.config)
