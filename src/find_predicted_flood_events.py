"""
Module: find_predicted_flood_events.py
Description: Scans model predictions (not actual flow) for periods where a
             forecast crossed a return-period threshold, per lead time.
"""

import os
import pandas as pd

import find_flood_events as ffe


def main(config=None, basin_ids=None):
    if config is None:
        config = ffe.load_config("configs/config.yml")

    buffer_days    = config.get('visual_buffer_days', 4)
    return_periods = config.get('return_periods_years', [2, 5, 10, 15, 20, 30])
    lead_times     = config.get('forecast_lead_times', [0, 1, 2, 3])

    run_dir = config.get('run_dir', './runs/')
    exp_dir = os.path.join(run_dir, config['experiment_name'])
    output_path = os.path.join(exp_dir, "predicted_flood_events.csv")

    if basin_ids is None:
        with open(config['test_basin_file']) as f:
            basins = [line.strip() for line in f if line.strip()]
    else:
        basins = list(basin_ids)

    print("=" * 75)
    print("      Automated Flash Flood Event Scanner — Predicted (Model) Exceedances")
    print("=" * 75)
    print(f"[INFO] Basins: {len(basins)} | Return periods: {return_periods} | "
          f"Lead times: {lead_times}h | Merge gap: {config.get('event_merge_gap_hours', 0)}h")
    print("-" * 75 + "\n")

    rows = []
    for basin in basins:
        print(f"Scanning Basin: {basin}...")
        for lead in lead_times:
            value_column = f'pred_lead_{lead}h'
            for rp in return_periods:
                events = ffe.scan_for_events(basin, rp, buffer_days, config, value_column=value_column)
                for idx, ev in enumerate(events):
                    rows.append({
                        "basin_id":            basin,
                        "lead_time_h":         lead,
                        "return_period_years": rp,
                        "event_idx":           idx + 1,
                        "core_start":          ev["core_start"],
                        "core_end":            ev["core_end"],
                        "peak_flow":           ev["peak_flow"],
                        "plot_ready_start":    ev["plot_ready_start"],
                        "plot_ready_end":      ev["plot_ready_end"],
                    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\n[INFO] Wrote {len(rows)} predicted flood events to {output_path}")


if __name__ == "__main__":
    main()
