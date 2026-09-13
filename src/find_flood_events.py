"""
Module: find_flood_events.py
Description: Automated Flash Flood Event Finder for Selected Return Periods.
             Scans test outputs to locate timestamps where actual discharge 
             crossed critical benchmarks and appends visual padding buffers.
"""

import os
import yaml
import pandas as pd


def load_config(yaml_path):
    """Load the YAML configuration file safely."""
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def load_basin_threshold(basin_id, target_rp, config):
    """Loads the cross-checked streamflow threshold for a given return period from
    the combined return-period file (see new_return_periods.py)."""
    combined_path = config['hourly_flow_return_periods_combined']
    if not os.path.exists(combined_path):
        print(f"  [Warning] Combined return-period file missing at {combined_path}.")
        return None

    combined_df = pd.read_csv(combined_path, dtype={'basin_id': str})
    # ':g' normalizes 2/2.0/2.00 all to 'RP2' - target_rp may arrive as a
    # plain int (legacy direct callers) or a float (normalize_threshold_specs
    # coerces every spec value to float), and the combined file's columns are
    # always integer-styled ('RP2', not 'RP2.0').
    col = f'RP{target_rp:g}'
    if col not in combined_df.columns:
        available_rps = [c[2:] for c in combined_df.columns if c.startswith('RP')]
        print(f"  [Warning] Return Period {target_rp}yr not available in combined file. Available: {available_rps}")
        return None

    row = combined_df[combined_df['basin_id'] == str(basin_id)]
    if row.empty:
        print(f"  [Warning] Basin {basin_id} not found in combined return-period file.")
        return None

    value = row.iloc[0][col]
    if pd.isna(value) or value <= 0:
        return None
    return float(value)


def normalize_threshold_specs(raw):
    """
    Normalizes any of today's/tomorrow's prediction-threshold config shapes into
    a list of {'type': 'return_period'|'specific_discharge', 'value': float} dicts:
      - None/missing -> []
      - a bare number (today's 'prediction_threshold: 2') -> [{'type': 'return_period', 'value': 2}]
      - a list of bare numbers (today's 'return_periods_years: [2, 5]') -> one
        return_period spec per number, in order
      - a list of {'type', 'value'} dicts (the new explicit form) -> returned as-is
        (values coerced to float)
      - a list mixing bare numbers and dicts -> each element normalized individually
    This is the single parsing rule for return_periods_years, prediction_threshold,
    and shown_threshold_bars, so a legacy bare-scalar/bare-list config produces
    identical behavior (and, per threshold_label, identical column names) to before
    this was generalized to support other threshold types.
    """
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    specs = []
    for item in items:
        if isinstance(item, dict):
            specs.append({'type': item['type'], 'value': float(item['value'])})
        else:
            specs.append({'type': 'return_period', 'value': float(item)})
    return specs


def threshold_label(spec):
    """
    Filename/column-safe label for a threshold spec. return_period reproduces
    today's exact f'{rp_year}yr' string (e.g. '2yr'), so existing single-RP
    configs get byte-identical report columns after this refactor.
    """
    value = spec['value']
    value_str = f'{value:g}'
    if spec['type'] == 'return_period':
        return f'{value_str}yr'
    if spec['type'] == 'specific_discharge':
        return f'sq{value_str}'
    raise ValueError(f"Unknown threshold spec type: {spec['type']!r}")


def threshold_column_name(spec):
    """
    The visual_report_basin_*.csv column name storing a threshold spec's
    resolved flow value. return_period keeps today's exact 'threshold_{label}_rp'
    form (e.g. 'threshold_2yr_rp'); specific_discharge drops the '_rp' suffix
    (it isn't a return period) -> 'threshold_{label}' (e.g. 'threshold_sq0.1').
    """
    label = threshold_label(spec)
    return f'threshold_{label}_rp' if spec['type'] == 'return_period' else f'threshold_{label}'


def merge_threshold_specs(*spec_lists):
    """Flattens and dedupes threshold specs by label (first occurrence wins) -
    used to build the union set of threshold columns a report needs to compute."""
    merged = {}
    for specs in spec_lists:
        for spec in normalize_threshold_specs(specs):
            label = threshold_label(spec)
            if label not in merged:
                merged[label] = spec
    return list(merged.values())


def report_threshold_specs(config):
    """
    The full set of threshold specs any part of the pipeline might need for
    this config: the union (deduped by label) of return_periods_years,
    prediction_threshold, and shown_threshold_bars. Used both for which
    threshold columns test.py:build_and_export_report stores in
    visual_report_basin_*.csv, and for which thresholds this module's own
    main()/find_predicted_flood_events.py scan real/predicted events at -
    the latter matters so a prediction_threshold entry (e.g. a
    specific_discharge spec) that isn't also in return_periods_years still
    gets scanned upstream, since compare_flood_events.py/peaks_analyze.py
    filter down to prediction_threshold specs afterward and would otherwise
    find nothing for a spec nobody scanned.
    """
    return merge_threshold_specs(
        config.get('return_periods_years', [2, 5, 10]),
        config.get('prediction_threshold', []),
        config.get('shown_threshold_bars', []),
    )


def load_basin_area_map(config):
    """{basin_id: area_km2} from config['static_attributes_file'] (the raw, not
    normalized/z-scored, static attributes file - normalized area wouldn't be in
    physical km2). Same one-CSV-read, cached-dict pattern as
    peaks_analyze.load_flow_std_map."""
    static_df = pd.read_csv(config['static_attributes_file'], dtype={'gauge_id': str})
    return dict(zip(static_df['gauge_id'], static_df['area']))


def resolve_threshold_value(basin_id, spec, config, area_map=None):
    """
    Resolves a threshold spec to a physical flow value (m3/s) for one basin.
    return_period delegates to load_basin_threshold unchanged. specific_discharge
    multiplies spec['value'] by the basin's area (km2) from area_map (built via
    load_basin_area_map(config) if not supplied - callers scanning many basins
    should build it once and pass it through to avoid re-reading the CSV).
    Returns None if the basin/threshold is unresolvable.
    """
    if spec['type'] == 'return_period':
        return load_basin_threshold(basin_id, spec['value'], config)
    if spec['type'] == 'specific_discharge':
        if area_map is None:
            area_map = load_basin_area_map(config)
        area = area_map.get(basin_id)
        if area is None or pd.isna(area) or area <= 0:
            print(f"  [Warning] Basin {basin_id} has no usable area for specific-discharge threshold.")
            return None
        return float(spec['value']) * float(area)
    raise ValueError(f"Unknown threshold spec type: {spec['type']!r}")


def _scan_series_for_events(df, value_column, threshold_value, buffer_days, merge_gap_hours):
    """
    Core exceedance-block detection + merge-gap + padding logic, operating on
    an already-loaded, already-timestamp-parsed DataFrame (must have a
    'timestamp' column). Extracted out of scan_for_events so callers with an
    in-memory frame that isn't a single model's own report CSV (e.g.
    model_compare_test.py's coalesced multi-model frame) can reuse this exact
    logic without needing a report file on disk. Returns the same list-of-dicts
    shape as scan_for_events (empty list if never exceeded).
    """
    # Locate indices where the value column crossed the threshold
    exceedance_mask = df[value_column] >= threshold_value
    if not exceedance_mask.any():
        return []

    df = df.copy()

    # Step 1 — find initial exceedance blocks
    df['is_exceedance'] = exceedance_mask.astype(int)
    df['event_id'] = (df['is_exceedance'] != df['is_exceedance'].shift()).cumsum()

    # Step 2 — collect raw core events (no padding yet)
    raw_events = []
    for _, chunk in df[df['is_exceedance'] == 1].groupby('event_id'):
        raw_events.append({
            'core_start': chunk['timestamp'].min(),
            'core_end':   chunk['timestamp'].max(),
            'peak_flow':  chunk[value_column].max(),
        })

    # Step 3 — merge events whose inter-event gap is within the configured tolerance
    if merge_gap_hours > 0 and len(raw_events) > 1:
        merge_td = pd.Timedelta(hours=merge_gap_hours)
        merged = [raw_events[0].copy()]
        for ev in raw_events[1:]:
            gap = ev['core_start'] - merged[-1]['core_end']
            if gap <= merge_td:
                merged[-1]['core_end']  = ev['core_end']
                merged[-1]['peak_flow'] = max(merged[-1]['peak_flow'], ev['peak_flow'])
            else:
                merged.append(ev.copy())
        raw_events = merged

    # Step 4 — apply padding and format output
    discovered_events = []
    for ev in raw_events:
        padded_start = max(ev['core_start'] - pd.Timedelta(days=buffer_days), df['timestamp'].min())
        padded_end   = min(ev['core_end']   + pd.Timedelta(days=buffer_days), df['timestamp'].max())

        discovered_events.append({
            'core_start':       ev['core_start'].strftime('%Y-%m-%d %H:%M:%S'),
            'core_end':         ev['core_end'].strftime('%Y-%m-%d %H:%M:%S'),
            'peak_flow':        float(ev['peak_flow']),
            'plot_ready_start': padded_start.strftime('%Y-%m-%d %H:%M:%S'),
            'plot_ready_end':   padded_end.strftime('%Y-%m-%d %H:%M:%S'),
        })

    return discovered_events


def cluster_events(tagged_events):
    """
    Merges overlapping (unpadded core_start/core_end) events from ANY source
    (real flow, any model's predictions, or - for merge_events_across_thresholds
    below - any threshold's own scan) into unified clusters via the standard
    merge-overlapping-intervals sweep (sort by start, extend while the next
    event's start <= the cluster's current end). Each cluster keeps its member
    tagged events for the caller (classify_clusters, or the threshold-dedup
    logic below) to inspect. Relocated here (was model_compare_test.py) so
    both the single-model (test.py/quick_test.py) and multi-model
    (model_compare_test.py/plot_events_by_year.py) pipelines can share it
    without a circular import; re-exported from model_compare_test for
    existing callers.
    """
    if not tagged_events:
        return []
    events_sorted = sorted(tagged_events, key=lambda e: e['core_start'])
    clusters = [{'core_start': events_sorted[0]['core_start'],
                 'core_end': events_sorted[0]['core_end'],
                 'members': [events_sorted[0]]}]
    for ev in events_sorted[1:]:
        if ev['core_start'] <= clusters[-1]['core_end']:
            clusters[-1]['core_end'] = max(clusters[-1]['core_end'], ev['core_end'])
            clusters[-1]['members'].append(ev)
        else:
            clusters.append({'core_start': ev['core_start'], 'core_end': ev['core_end'], 'members': [ev]})
    return clusters


def merge_events_across_thresholds(events_by_label, value_by_label):
    """
    Dedup helper: takes each prediction-threshold's own already-computed
    events/clusters for one basin (events_by_label: {label: [{'core_start',
    'core_end', **payload}, ...]}) and merges overlapping windows across
    thresholds into one plotting window per real overlapping time span - "if
    the window is flagged by more than one threshold, plot it once."

    For each merged window, the winning contributor is whichever threshold
    that overlaps it has the LOWEST resolved flow value in value_by_label
    ({label: float}) - per-basin resolved values, since a return_period and a
    specific_discharge threshold are only comparable once both are resolved
    to an actual flow number for this basin. Returns a list of
    {'core_start', 'core_end', 'winning_label', 'contributing_labels': [...],
     **winning_payload} (payload = whatever extra keys the winning event dict
    carried, e.g. 'rows'/'label'/'event_idx').
    """
    tagged = []
    for label, events in events_by_label.items():
        for ev in events:
            tagged.append({**ev, 'source': label})

    clusters = cluster_events(tagged)

    merged_windows = []
    for cluster in clusters:
        contributing_labels = sorted({m['source'] for m in cluster['members']})
        winning_label = min(contributing_labels, key=lambda l: value_by_label.get(l, float('inf')))
        winning_member = next(m for m in cluster['members'] if m['source'] == winning_label)
        payload = {k: v for k, v in winning_member.items() if k not in ('source', 'core_start', 'core_end')}
        merged_windows.append({
            'core_start': cluster['core_start'],
            'core_end': cluster['core_end'],
            'winning_label': winning_label,
            'contributing_labels': contributing_labels,
            **payload,
        })
    return merged_windows


def scan_for_events(basin_id, spec, buffer_days, config, value_column='actual_flow', area_map=None):
    """
    Scans the basin's test report to isolate continuous blocks where the given
    value column (actual streamflow by default, or a pred_lead_*h column for
    predicted-exceedance scans) crossed the designated threshold spec (a
    {'type', 'value'} dict - see normalize_threshold_specs/resolve_threshold_value),
    returning bound parameters.
    """
    label = threshold_label(spec)
    run_dir = config.get('run_dir', './runs/')
    exp_dir = os.path.join(run_dir, config['experiment_name'])
    report_path = os.path.join(exp_dir, "visualization_reports", f"visual_report_basin_{basin_id}.csv")

    if not os.path.exists(report_path):
        print(f"  [Error] Evaluation report missing for basin {basin_id}. Run test.py first.")
        return []

    # Load test results
    df = pd.read_csv(report_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Fetch the critical physical threshold value
    threshold_value = resolve_threshold_value(basin_id, spec, config, area_map)
    if threshold_value is None or threshold_value <= 0:
        print(f"  [Info] Threshold value for {label} is unstable (<=0). Skipping basin scan.")
        return []

    discovered_events = _scan_series_for_events(
        df, value_column, threshold_value, buffer_days, config.get('event_merge_gap_hours', 0)
    )

    if not discovered_events:
        print(f"  [-] Basin {basin_id} never exceeded the {label} threshold "
              f"in '{value_column}' during the test split.")
        return []

    print(f"  [★] Found {len(discovered_events)} distinct exceedance event(s) for {label}:")
    for ev in discovered_events:
        print(f"      • Event Peak: {ev['peak_flow']:.2f} m3/s | Core Duration: [{ev['core_start']} -> {ev['core_end']}]")
        print(f"        Padded Window for Plotting: '{ev['plot_ready_start']}' TO '{ev['plot_ready_end']}'")

    return discovered_events


def main(config=None, basin_ids=None):
    if config is None:
        config = load_config("configs/config.yml")

    buffer_days   = config.get('visual_buffer_days', 4)
    threshold_specs = report_threshold_specs(config)
    area_map      = load_basin_area_map(config) if any(s['type'] == 'specific_discharge' for s in threshold_specs) else None
    output_path   = config['find_flood_events_output']

    if basin_ids is None:
        with open(config['test_basin_file']) as f:
            basins = [line.strip() for line in f if line.strip()]
    else:
        basins = list(basin_ids)

    labels = [threshold_label(s) for s in threshold_specs]
    print("=" * 75)
    print("      Automated Flash Flood Event Scanner — Test Dataset Evaluation")
    print("=" * 75)
    print(f"[INFO] Basins: {len(basins)} | Thresholds: {labels} | Merge gap: {config.get('event_merge_gap_hours', 0)}h")
    print("-" * 75 + "\n")

    rows = []
    for basin in basins:
        print(f"Scanning Basin: {basin}...")
        for spec in threshold_specs:
            events = scan_for_events(basin, spec, buffer_days, config, area_map=area_map)
            for idx, ev in enumerate(events):
                rows.append({
                    "basin_id":       basin,
                    "threshold_label": threshold_label(spec),
                    "event_idx":      idx + 1,
                    "core_start":     ev["core_start"],
                    "core_end":       ev["core_end"],
                    "peak_flow":      ev["peak_flow"],
                    "plot_ready_start": ev["plot_ready_start"],
                    "plot_ready_end":   ev["plot_ready_end"],
                })

    columns = ["basin_id", "threshold_label", "event_idx", "core_start", "core_end",
               "peak_flow", "plot_ready_start", "plot_ready_end"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)
    print(f"\n[INFO] Wrote {len(rows)} flood events to {output_path}")


if __name__ == "__main__":
    main()