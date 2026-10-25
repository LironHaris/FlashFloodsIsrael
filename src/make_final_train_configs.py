"""
Module: make_final_train_configs.py
Description: Generates the final-training configs configs/train_<model>_<test years>.yml from a
             CV-epochs config (configs/cv_epochs_<model>.yml) and its avg_best_epoch: same
             hyperparameters, trained from scratch on ALL non-test years (cross_validation
             final mode) for avg_best_epoch epochs, once per test period in TEST_PERIODS.

             Text-level edits of the cv_epochs file (every comment survives); each edit's anchor
             must match exactly once, otherwise the script fails instead of writing a bad config.
             Per test period it rewrites: the header comment, experiment_name, basin lists,
             train/test periods, the cross_validation block, epochs, and the per-test-set
             processed data / availability report / flood events paths.

Usage:
  python src/make_final_train_configs.py --all
  python src/make_final_train_configs.py --cv-config configs/cv_epochs_flow_0_6.yml --epochs 18
  python src/make_final_train_configs.py --check   # regenerate committed configs, diff vs. disk
"""

import argparse
import os
import re

import yaml

CONFIG_DIR = 'configs'
RECORD_END = '2023-09-30 07:00:00'
TEST_PERIODS = [(2009, 2010, 2011), (2012, 2013, 2014), (2015, 2016, 2020), (2017, 2018, 2019),
                (2021, 2022, 2023)]

# avg_best_epoch of each finished cv_epochs_<model>.yml run
EPOCHS = {
    'flow_0_6': 18,
    'flow_normal_0_0_top1': 8, 'flow_normal_0_0_top2': 3, 'flow_normal_0_0_top3': 14,
    'flow_normal_0_3_top1': 10, 'flow_normal_0_3_top2': 17, 'flow_normal_0_3_top3': 8,
    'flow_normal_0_6_top1': 4, 'flow_normal_0_6_top2': 4, 'flow_normal_0_6_top3': 3,
    'flow_normal_0_12_top1': 32, 'flow_normal_0_12_top2': 22, 'flow_normal_0_12_top3': 15,
    'flow_normal_1_3_0_top1': 19, 'flow_normal_1_3_0_top2': 27, 'flow_normal_1_3_0_top3': 24,
    'flow_normal_1_3_3_top1': 40, 'flow_normal_1_3_3_top2': 34, 'flow_normal_1_3_3_top3': 21,
    'flow_normal_1_3_6_top1': 25, 'flow_normal_1_3_6_top2': 5, 'flow_normal_1_3_6_top3': 20,
    'flow_normal_1_3_12_top1': 13, 'flow_normal_1_3_12_top2': 26, 'flow_normal_1_3_12_top3': 13,
}

# already-committed final configs, used by --check to prove the transformation
CHECK_MODELS = ['flow_0_0', 'flow_0_3', 'flow_0_12', 'flow_1_3_0', 'flow_1_3_3', 'flow_1_3_6', 'flow_1_3_12']

MASK_WORDING = {'flow': ("flow-masked", "only on hours labeled 'flow'"),
                'flow_normal': ("flow+normal-masked", "only on hours labeled 'flow' or 'normal'")}


def hydro_start(year):
    return f"'{year - 1}-10-01 08:00:00'"


def hydro_end(year):
    return f"'{year}-09-30 07:00:00'"


def runs(years):
    """Consecutive runs of sorted years, e.g. (2015, 2016, 2020) -> [(2015, 2016), (2020, 2020)]."""
    out = []
    for y in sorted(years):
        if out and y == out[-1][1] + 1:
            out[-1] = (out[-1][0], y)
        else:
            out.append((y, y))
    return out


def periods_block(years):
    years_txt = ', '.join(str(y) for y in years)
    test_runs = runs(years)
    lines = ["# Time periods (format = 'YYYY-MM-DD HH:MM:SS'). No validation period.",
             f"# Test = hydro-years {years_txt} (hydro-year N = Oct N-1 .. Sep N); train_periods = every",
             "# other year of the record. A period's empty start_date = each basin's own earliest record.",
             "train_periods:"]
    start = None  # empty = each basin's earliest record
    for first, last in test_runs:
        lines += [f"  - start_date:{' ' + start if start else ''}", f"    end_date: {hydro_end(first - 1)}"]
        start = hydro_start(last + 1)
    if test_runs[-1][1] < int(RECORD_END[:4]):
        lines += [f"  - start_date: {start}", f"    end_date: '{RECORD_END}'"]
    lines.append("")
    if len(test_runs) == 1:
        lines += [f"test_start_date: {hydro_start(years[0])}", f"test_end_date: {hydro_end(years[-1])}"]
    else:
        lines += ["# Non-consecutive test years: explicit test_periods instead of test_start_date/test_end_date.",
                  "test_periods:"]
        lines += [f"  - [{hydro_start(first)}, {hydro_end(last)}]" for first, last in test_runs]
    return '\n'.join(lines) + '\n'


CV_BLOCK = ("# enabled: true puts train.py in final mode: from scratch on ALL train_periods, no validation set,\n"
            "# last epoch saved as best_model.pt. The CV groups (used only by the sweep) are intentionally omitted.\n"
            "cross_validation:\n"
            "  enabled: true\n")


def sub_once(pattern, repl, text):
    new_text, n = re.subn(pattern, lambda _: repl, text, count=0, flags=re.M | re.S)
    assert n == 1, f"pattern matched {n} times (expected 1): {pattern}"
    return new_text


def make_config(cv_path, epochs, years):
    model = os.path.basename(cv_path)[len('cv_epochs_'):-len('.yml')]
    tag = '_'.join(str(y) for y in years)
    name = f"train_{model}_{tag}"
    with open(cv_path, 'r', encoding='utf-8') as f:
        text = f.read().replace('\r\n', '\n')
    cfg = yaml.safe_load(text)
    masked, labeled = MASK_WORDING[cfg['train_flow_type_mask']]
    lead = cfg['forecast_lead_times'][0]
    base = re.sub(r'^flow(_normal)?_', '', model)
    top = re.search(r'_top(\d)$', model)
    hyper = (f"CV-sweep hyperparameter set #{top.group(1)}" if top else
             f"winning {'flow' if cfg['train_flow_type_mask'] == 'flow' else 'flow+normal'}-mask CV-sweep hyperparameters")
    model_label = base.replace(f'_top{top.group(1)}', f' top{top.group(1)}') if top else base
    header = (f"# Final {masked} model {model_label} (lead {lead}h): {hyper} of {os.path.basename(cv_path)},\n"
              f"# trained from scratch on ALL non-test years - {labeled} (train_flow_type_mask) - for\n"
              f"# epochs = avg_best_epoch ({epochs}) of the 50-epoch CV run. Test hydro-years: "
              f"{', '.join(str(y) for y in years)} (test is not masked).\n"
              f"# Run: python src/train.py --config configs/{name}.yml\n"
              f"# Data: preprocessed per test set with configs/preprocess_test_{tag}.yml (run it first).\n")

    text = sub_once(r'\A# CV epoch-estimation config[^\n]*\n', header, text)
    text = sub_once(r'^experiment_name: .*?$', f"experiment_name: {name}", text)
    text = sub_once(r'^train_basin_file: .*?$', f"train_basin_file: data/basin_lists/test_{tag}/israel_train.txt", text)
    text = sub_once(r'^test_basin_file: .*?$', f"test_basin_file: data/basin_lists/test_{tag}/israel_test.txt", text)
    text = sub_once(r"^# Time periods .*?^test_end_date: [^\n]*\n", periods_block(years), text)
    text = sub_once(r"^# k-fold cross-validation .*?^early_stop_patience:[^\n]*\n", CV_BLOCK, text)
    text = sub_once(r'^epochs: 50$', f"epochs: {epochs}", text)
    text = sub_once(r'^processed_timeseries_dir: ./data/processed/timeseries$',
                    f"processed_timeseries_dir: ./data/processed/timeseries_test_{tag}", text)
    text = sub_once(r'^availability_report_file: .*?$',
                    f"availability_report_file: ./data/processed/data quality reports/test_{tag}/data_availability_report.csv",
                    text)
    text = sub_once(r'^find_flood_events_output: .*?$',
                    f"find_flood_events_output: ./data/processed/flood_events/flood_events_test_{tag}.csv", text)

    out_cfg = yaml.safe_load(text)
    assert out_cfg['epochs'] == epochs and out_cfg['experiment_name'] == name
    return os.path.join(CONFIG_DIR, f"{name}.yml"), text


def write_all(epochs_by_model):
    for model, epochs in epochs_by_model.items():
        for years in TEST_PERIODS:
            path, text = make_config(os.path.join(CONFIG_DIR, f"cv_epochs_{model}.yml"), epochs, years)
            with open(path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(text)
            print(f"  wrote {path}")


def check():
    """Regenerate the committed train_flow_* configs (epochs read from them) and compare to disk."""
    mismatches = 0
    for model in CHECK_MODELS:
        for years in TEST_PERIODS:
            path = os.path.join(CONFIG_DIR, f"train_{model}_{'_'.join(str(y) for y in years)}.yml")
            with open(path, 'r', encoding='utf-8') as f:
                on_disk = f.read().replace('\r\n', '\n')
            _, text = make_config(os.path.join(CONFIG_DIR, f"cv_epochs_{model}.yml"),
                                  yaml.safe_load(on_disk)['epochs'], years)
            if text.rstrip('\n') != on_disk.rstrip('\n'):
                mismatches += 1
                print(f"  [MISMATCH] {path}")
    print(f"[CHECK] {mismatches} mismatches")
    return mismatches == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate final train configs for every test period.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="All models in EPOCHS.")
    group.add_argument("--cv-config", type=str, help="One cv_epochs config (needs --epochs).")
    group.add_argument("--check", action="store_true", help="Regenerate committed configs and diff.")
    parser.add_argument("--epochs", type=int, help="avg_best_epoch for --cv-config.")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(0 if check() else 1)
    if args.all:
        write_all(EPOCHS)
    else:
        assert args.epochs, "--cv-config needs --epochs"
        model = os.path.basename(args.cv_config)[len('cv_epochs_'):-len('.yml')]
        write_all({model: args.epochs})
