# Radar → Flash-Flood LSTM (Ariel's part): handover to Liron

Oct 7, 2026 · @ariel

## TL;DR

The IMS short-range (SR) radar is validated, extracted for 53 basins (2012-23), and 8 radar-only EA-LSTM models are trained on your current code. Their test NSE is ~0 (median) vs ~0.2 for your gauge models, but the gap is mostly a **data-coverage problem, not radar quality**: our radar quality rules removed most stormy windows, so the models were trained and tested on only ~24% of the test floods.

- **Done:** radar product validated (orientation, meaning of the hourly field, clock incl. DST, year quality), basin mapping + selection, extraction to the cluster, model-ready inputs, 8 trained models (2 input variants x 4 lead times), your comparison tools running on them.
- **Main finding:** 76% of the 156 RP2 floods in 2016-10 to 2019-09 get no radar prediction at all. Causes: radar files missing on rainy days (unfixable for a radar-only model) and a basin-coverage rule (>= 90%) that was too strict during storms.
- **Recommended next step:** re-extract with a 50% coverage rule + a looser gap rule (keeps the last 6 h clean). That raises flood coverage from 19% to 55% of flood hours. Then retrain the 8 models (~2.5 h) and run your `model_compare_test.py` against your gauge models on the same flood hours.

## Where everything is

Two storage places share a name but are **not** the same disk: the Windows share `\\vscifs.cc.huji.ac.il\hydrolab1\sci\labs\efratmorin\ariel.monzon` is invisible to the cluster, and the cluster's `/sci/labs/efratmorin/ariel.monzon` is invisible from Windows. Files reach the cluster by WinSCP (the cluster also cannot reach GitHub).

| What | Where |
| --- | --- |
| Radar source (exact times) | `hydrolab1\hydrolab\ShareData\radarDatabase\IMS\IMS_Adjusted\daily_SR\<year>\YYYYMMDD.mat` (Windows share) |
| Extraction v1 (53 basins, 2012-23) | share: `ariel.monzon\SR_radar\IMS_SR_adjusted_basin_hourly_v1\` (basin CSVs, hourly grid cube, basin masks, README) |
| Same data on the cluster | `/sci/labs/efratmorin/ariel.monzon/SR_radar/IMS_SR_adjusted_basin_hourly_v1/basin_hourly/` + `model_inputs_v1/` |
| Extended statics (your 90 + 4) | cluster: `/sci/labs/efratmorin/ariel.monzon/SR_radar/static/static_attributes_normalized_plus4.csv` |
| Trained runs, logs | cluster: `/sci/labs/efratmorin/ariel.monzon/SR_radar/runs/` and `.../SR_radar/logs/` |
| Code on the cluster | `/sci/labs/efratmorin/ariel.monzon/FlashFloodsIsrael_radar/` (the branch `Ariel`, older) |
| Code, current | GitHub `LironHaris/FlashFloodsIsrael`, branch `Ariel-radar-v2` (this branch) = your cluster code + radar additions |
| Validation scripts + outputs | handover zip `radar_extraction_and_validation/` (originally Ariel's PC: `Documents\פרויקט שטפונות\basin_mapping\SR_basin_mapping\` (`validation\`, `prepare\`, `extract\`)) |
| Full technical report | `RADAR_INTEGRATION_REPORT.md` in the project folder; copy on the share in `ariel.monzon\SR_radar\` |
| W&B runs | your project `flash-floods-israel`, runs named `SRradar_B_*` |

## What was validated about the radar

The stored hourly field is usable as-is once the hour labels are shifted; years before 2012-13 are not.

| Check | Result |
| --- | --- |
| Product | IMS_Adjusted SR, 640 x 640 cells of 500 m (ITM). Use the `.mat` files: the `.nc` copies hold the same values but their `time` is float32 (~90 min resolution) |
| Orientation | `.mat` arrays (via h5py) equal the netCDF exactly, no flip |
| `hourly_rain_mm` | Reproduced exactly: hour labelled L = mean of the scans in (L-1h, L], values < 0.05 mm set to 0. The label is the hour **end**; missing scans are already compensated (do not rescale by `nvols`) |
| Time alignment vs gauges/flow | Radar hour-end label L -> pipeline hour start **L - 3 h in winter, L - 4 h in Israeli DST** (1 h label + 2 h / 3 h clock offset). Lag analysis: winter 341 basin-events, DST 365, peak at lag 0 after the shift |
| Year quality | Producer and our own check agree: daily r ~0 up to 2010-11, ~0.6 from 2012-13. Use **2012-13 onward** |
| Missing days | No file on ~45% of days. 97% of them were dry by the gauges (skipped by the producer). Real outages: Jan-Feb 2019, Oct-Nov 2020 |

Details, numbers and the scripts for each check are in `RADAR_INTEGRATION_REPORT.md`.

## Decisions and why

| Decision | Why |
| --- | --- |
| **52 basins** for training | Radar sees 56 of your basins (effective coverage >= 0.39); 53 are in the project list; il_17123 dropped: on 2013-01-08 the official peak was 99 m3/s but the hourly series shows 0.0 |
| 4 basins added to your statics: il_14115, il_17110, il_17117, il_18131 | Missing from your `static_attributes_normalized.csv` (it has 90 basins). Their rows are computed with your `feature_statistics.csv`; the script checks it reproduces an existing basin of yours to 1e-14 before writing |
| Comparison basins = **48 shared** | Your models have no predictions for the 4 added basins |
| Test 2016-10 to 2019-09, validation 2020-21, train 2012-13 to 2015-16 + 2019-20 + 2021-23 | Test = your test period, so your trained gauge models are the baseline without retraining. 2020-21 (outage year) for validation costs the least training data. Option: swap with 2019-20 |
| `seq_length` 72, one model per lead (L0-L3) | Same as your setup |
| Your hyperparameters unchanged | Clean comparison; tuning later, on validation only |
| Inputs: 1a = radar basin mean + mask; 1b = mean, max, p90, wet fraction, spread, rain volume (m3/h, sum of depth x pixel area over the covered basin; added for v2) + mask | 1b tests whether in-basin rain structure helps. No weekly cumulative rain yet |
| Missing radar day -> 0 mm if the basin gauge was dry that day (< 1 mm), else a gap | The producer skipped dry days. Gauge values never enter the radar series |
| Gaps -> 0 mm + mask, window kept if each gap <= 2 h and <= 6 h total | v1 rule. **Too strict**, see below |
| Normalization: per-basin z-score from training years, real radar hours only | Same family as your preprocessing |
| Dry periods kept in training | Same as your pipeline; the model needs no-rain examples |

## What was trained and the results so far

Eight models trained on Moriah (salmon, L40S, ~2.5 h each, 50 epochs, best validation epoch kept): median test NSE between -0.11 and 0.01, against ~0.2 for your gauge models. Hit rates at the 2-year threshold are ~0: the models almost never predict above it.

| Run | Median NSE (52 basins) | Basins with NSE > 0 | RP2 hit rate (hourly) |
| --- | --- | --- | --- |
| SRradar_B_mean_mask_L0 | 0.013 | 44% | 0.00 |
| SRradar_B_mean_mask_L1 | -0.051 | 37% | 0.00 |
| SRradar_B_mean_mask_L2 | -0.004 | 40% | 0.00 |
| SRradar_B_mean_mask_L3 | -0.040 | 37% | 0.00 |
| SRradar_B_allstats_mask_L0 | -0.109 | 33% | 0.00 |
| SRradar_B_allstats_mask_L1 | -0.006 | 40% | 0.12 |
| SRradar_B_allstats_mask_L2 | -0.079 | 37% | 0.00 |
| SRradar_B_allstats_mask_L3 | -0.030 | 38% | 0.00 |

These came from the older `test.py` (hourly metrics). They are **not comparable to your numbers yet**: different test hours, and no event-level metrics. The mean NSE is meaningless here (basins with almost no flow in the test years blow it up). Your `model_compare_test.py` + `model_compare_eval_test.py` now run on these checkpoints (smoke-tested), but the real comparison has not been run.

## Why the results look weak: the radar models barely saw floods

Of the 156 RP2 floods in the test years (41 basins, your thresholds), **76% get no radar prediction** and only 19-24% of flood hours are covered. The same rule filtered training, so the models rarely learned from floods.

In the 72 h before each flood until its end, hours were: 49% real radar, 11% skipped dry days, **21% missing radar day on a wet day** (no file exists), **18% basin coverage under 90%** (median 47% of the basin had data), 1% too few scans. The 21% cannot be fixed for a radar-only model; the 18% came from our 90% coverage rule, too strict during storms.

| Rule | Flood hours covered | Floods with any prediction |
| --- | --- | --- |
| v1: coverage >= 90%, gaps <= 2 h in a row and <= 6 h per window | 19% | 24% |
| Coverage >= 50% only | 34% | 37% |
| **Recommended v2: coverage >= 50%, gaps <= 12 h in a row and <= 24 h per window, last 6 h always real** | **55%** | **54%** |
| Most lenient tested (coverage >= 30%, gaps <= 24 / 36 h, last 6 h real) | 60% | 61% |

About 60% is the ceiling for radar-only (the rest falls on days with no radar file). v2 keeps at least 48 of 72 hours real and never fills the hours that matter most for 0-3 h leads. Your own plots made this visible: in a December 2018 event window only ~4 hours had predictions, so the flood almost vanished. Data: `prepare/outputs/flood_coverage_events.csv` and `coverage_scenarios.csv`.

## Next steps, in order

1. **Decide the v2 rule** (recommended above) and put it in config: `min_valid_area_frac: 0.5` in `basin_mapping/SR_basin_mapping/config.yml`; `gap_max_run: 12`, `gap_max_total: 24` and a new "last 6 h clean" option in `configs/make_radar_configs.py` + `src/prepare_radar_inputs.py` (window_ok).
2. **Re-extract to a new folder** `IMS_SR_adjusted_basin_hourly_v2` (`extract/extract_sr_basin_hourly.py`, ~25 min on a Windows PC with access to the hydrolab share; v1 keeps no rain values for hours under 90% coverage, so it cannot be patched; v2 also adds the `radar_volume_m3` column). Then point `EXTRACTION` in `configs/make_radar_configs.py` at v2: the configs already list the volume input, so they fail on v1 data.
3. **Pack and upload**: `extract/pack_for_cluster.py` builds one ~30 MB zip; WinSCP it to the cluster and unzip into `SR_radar/IMS_SR_adjusted_basin_hourly_v2/`. Upload this branch's code folder too.
4. **Retrain the 8 models** (prepare job, then 8 training jobs; see the next section). ~2.5 h.
5. **Run your comparison**: one comparison config per lead with your gauge model and the two radar models (eval configs in `configs/eval/`), on the 48 shared basins. Then `model_compare_eval_test.py` for TP/FN/FP, precision/recall/F1, peak timing and magnitude.
6. **Score both on the same flood hours**: restrict your gauge model's evaluation to hours where the radar model has a prediction, otherwise the gauge model is credited for floods radar could not see. Not built yet.
7. **Gauge-control run** (if radar still trails): same code, years and samples, input = gauge rain (already in the prepared files as `hourly_precipitation`). It separates "radar is worse" from "our setup is worse" (fewer years, no weekly rain total, the validation year).

Later ideas already discussed: model 2 (radar + gauge rain), model 3 (CNN on the saved hourly grid cube), your k-fold validation instead of one validation year, weekly cumulative rain from radar.

## How to run it on Moriah

From the code folder, one CPU job prepares the inputs and statics, then the 8 training jobs start after it (each trains, then tests):

```bash
cd /sci/labs/efratmorin/ariel.monzon/FlashFloodsIsrael_radar
PREP=$(sbatch --parsable jobs/radar_prepare.sbatch)
for v in SRradar_B_mean_mask SRradar_B_allstats_mask; do for k in 0 1 2 3; do
  sbatch --dependency=afterok:$PREP -J ${v}_L$k --export=ALL,RUN=${v}_L$k jobs/radar_modelB.sbatch
done; done
```

- Test only, on a saved model: add `,TEST_ONLY=1` to `--export`.
- Configs are generated: edit `configs/make_radar_configs.py`, run it, never hand-edit the `.yml` files. Training configs leave `checkpoint_path` empty (your `train.py` resumes from it); `configs/eval/*.yml` carry it for `test.py`, `quick_test.py` and `model_compare_*`.
- Jobs use your conda env `flashfloods` (read-only), partition `salmon`, 1 GPU, 4 CPU, 32G, 24 h; logs go to `SR_radar/logs/`.
- Resubmitting after a failed prepare job: its dependent jobs disappear from the queue and must be submitted again.

## Loose ends and cautions

- **W&B login:** the jobs read `/sci/labs/efratmorin/ariel.monzon/.netrc` (compute nodes do not see `/cs/usr/...`). It is a file from Sep 10 that logs into your team project. Before Ariel leaves: point the jobs at your own credentials and delete that file; Ariel revokes their key.
- **Your code was changed only on the branch:** `dataset.py` got three opt-in options (`sample_filter_column`, `require_contiguous_windows`, `exclude_basins`) and a vectorized window search with the same result; `plot_hydrographs.py` got `rain_overlay_column`. Your behaviour is unchanged when the options are absent.
- **Your current code is not on GitHub:** the branch imports it from your cluster folder (2026-10-04). Commit yours to avoid two diverging copies.
- **`prediction_threshold: 2`** in the radar configs is your scripts' default; check it against your configs.
- **Why the 5 basins are missing from your statics** is unknown. Worth checking before adding them for good.
- **Southern basins** (il_23103, il_23105, il_55165, il_25191, il_23134) showed radar rain with zero gauge rain and zero flow on a few autumn 2021 days: possibly false echoes. Not resolved.
- **Open checks:** half-cell (250 m) georeference ambiguity in the radar grid (accepted); which clock carries DST (the 2 h / 3 h rule works either way); sub-hourly intensity not extracted.

## The handover bundle (zip on the drive)

`radar_handover_2026-10-07.zip` holds everything that is not already in this branch or on the cluster:

| Folder in the zip | What it is |
| --- | --- |
| `README_RADAR.md`, `Radar_handover.pdf` | this document (Markdown and PDF) |
| `RADAR_INTEGRATION_REPORT.md` | the full technical report: every analysis, number and decision |
| `radar_extraction_and_validation/` | the radar side that lives outside this repo: `extract/` (extraction, packing), `validation/` (steps 1-7 + outputs), `prepare/` (sample counts, coverage, flow QC + CSVs), `config.yml` (all radar rules), `data/processed/` (basin x radar-cell mapping), `outputs/reports/` (basin selection lists) |
| `basin_mapping_LR_and_map/` | the first (LR) mapping code and the interactive basin map (`outputs/basin_map.html`) |
| `code_branch_Ariel-radar-v2/` | snapshot of this branch, in case GitHub is not at hand |
| `data_for_cluster/` | `basin_hourly.zip` (extraction v1, 53 basins, 31 MB) and `static_attributes_nh_all95.csv` (raw statics incl. the 4 added basins) |

Python environment for the radar scripts: `requirements.txt` in `radar_extraction_and_validation/` (h5py, netCDF4, geopandas, pandas, scipy, pyarrow, tzdata, matplotlib, pyyaml).
