import os
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RETURN_PERIODS_DIR = r"C:\Users\user\Documents\פרויקט שטפונות\floods\return_periods"
ANNUAL_MAX_CSV     = r"C:\Users\user\Downloads\AnnualMaxDischarge_UTF8.csv"
TIMESERIES_DIR     = r"C:\Users\user\Documents\פרויקט שטפונות\floods\data\timeseries"
OUT_DIR            = RETURN_PERIODS_DIR

# ── 1. Load AnnualMaxDischarge ─────────────────────────────────────────────────
am = pd.read_csv(ANNUAL_MAX_CSV, encoding="utf-8")
am.columns = am.columns.str.strip()

# Rename to stable keys
col_map = {
    am.columns[0]: "station_id",
    am.columns[3]: "hydro_year_str",
    am.columns[5]: "momentary_max",
    am.columns[6]: "am_date",
    am.columns[7]: "am_time",
}
am = am.rename(columns=col_map)[["station_id", "hydro_year_str", "momentary_max", "am_date", "am_time"]]
am["am_peak_datetime"] = pd.to_datetime(
    am["am_date"].astype(str) + " " + am["am_time"].astype(str),
    format="%d/%m/%Y %H:%M:%S", errors="coerce"
)
am = am.drop(columns=["am_date", "am_time"])
am["station_id"]   = pd.to_numeric(am["station_id"],   errors="coerce")
am["momentary_max"] = pd.to_numeric(am["momentary_max"], errors="coerce")
am = am.dropna(subset=["station_id", "hydro_year_str", "momentary_max"])
am = am[am["momentary_max"] > 0]
am["station_id"] = am["station_id"].astype(int)

# Extract end-year from "YYYY/YYYY"
am["hydro_year"] = am["hydro_year_str"].str.split("/").str[1].astype(int)
am = am.drop(columns=["hydro_year_str"])

# ── 2. Enumerate Hourly Empirical Files ───────────────────────────────────────
basin_dirs = sorted(
    d for d in os.listdir(RETURN_PERIODS_DIR)
    if os.path.isdir(os.path.join(RETURN_PERIODS_DIR, d)) and d.startswith("il_")
)

enum_rows = []
hourly_frames = []

for idx, bdir in enumerate(basin_dirs, start=1):
    basin_id = int(bdir.replace("il_", ""))
    fpath = os.path.join(RETURN_PERIODS_DIR, bdir, "Hourly_Flow_empirical_AM.csv")
    if not os.path.isfile(fpath):
        enum_rows.append({
            "index": idx, "basin_id": basin_id, "dir": bdir,
            "has_file": False, "n_years": None,
            "year_min": None, "year_max": None,
            "value_min": None, "value_max": None,
        })
        continue
    df = pd.read_csv(fpath)
    df["basin_id"] = basin_id
    hourly_frames.append(df)
    enum_rows.append({
        "index": idx, "basin_id": basin_id, "dir": bdir,
        "has_file": True, "n_years": len(df),
        "year_min": int(df["Hydro_Year"].min()),
        "year_max": int(df["Hydro_Year"].max()),
        "value_min": float(df["Value"].min()),
        "value_max": float(df["Value"].max()),
    })

enumerated = pd.DataFrame(enum_rows)
enumerated.to_csv(os.path.join(OUT_DIR, "hourly_empirical_enumerated.csv"),
                  index=False, encoding="utf-8-sig")
print(f"Enumerated {len(enumerated)} basin dirs, {enumerated['has_file'].sum()} with flow files.")

# ── 3. Join ────────────────────────────────────────────────────────────────────
hourly_all = pd.concat(hourly_frames, ignore_index=True)
hourly_all = hourly_all.rename(columns={
    "Hydro_Year":                 "hydro_year",
    "Value":                      "hourly_value",
    "Rank":                       "hourly_rank",
    "Exceedance_Probability":     "hourly_exceedance_prob",
    "Empirical_Return_Period_Years": "hourly_empirical_rp",
})
hourly_all["basin_id"] = hourly_all["basin_id"].astype(int)

joined = am.merge(hourly_all, left_on=["station_id", "hydro_year"],
                              right_on=["basin_id", "hydro_year"], how="inner")
joined = joined.drop(columns=["basin_id"])
print(f"Matched pairs (station x year): {len(joined)} across {joined['station_id'].nunique()} basins.")

# ── 3b. Add raw hourly peak datetime from timeseries ─────────────────────────
print("Loading raw timeseries to find hourly peak timestamps...")
needed = joined.groupby("station_id")["hydro_year"].apply(list).to_dict()

raw_peak_map = {}
for sid, years in needed.items():
    ts_path = os.path.join(TIMESERIES_DIR, f"il_{sid}.csv")
    if not os.path.isfile(ts_path):
        continue
    ts = pd.read_csv(ts_path, parse_dates=["date"])
    for yr in years:
        start = pd.Timestamp(yr - 1, 10, 1)
        end   = pd.Timestamp(yr, 9, 30, 23, 59, 59)
        sub   = ts[(ts["date"] >= start) & (ts["date"] <= end) & ts["Flow_m3_sec"].notna()]
        if sub.empty:
            continue
        raw_peak_map[(sid, yr)] = ts.loc[sub["Flow_m3_sec"].idxmax(), "date"]

joined["raw_hourly_peak_datetime"] = joined.apply(
    lambda r: raw_peak_map.get((r["station_id"], r["hydro_year"])), axis=1
)
joined["am_peak_hour"] = joined["am_peak_datetime"].dt.floor("h")
joined["same_hour"] = joined["am_peak_hour"] == joined["raw_hourly_peak_datetime"]
joined["hours_apart"] = (
    (joined["am_peak_datetime"] - joined["raw_hourly_peak_datetime"])
    .abs().dt.total_seconds() / 3600
)

has_both = joined["raw_hourly_peak_datetime"].notna() & joined["am_peak_datetime"].notna()
n_both = has_both.sum()
ha = joined.loc[has_both, "hours_apart"]
print(f"  Pairs with both timestamps : {n_both}")
print(f"  Same hour (exact match)    : {(ha == 0).sum()} ({(ha == 0).sum()/n_both*100:.1f}%)")
print(f"  0h < gap <= 1h             : {((ha > 0) & (ha <= 1)).sum()} ({((ha > 0) & (ha <= 1)).sum()/n_both*100:.1f}%)")
print(f"  1h < gap <= 6h             : {((ha > 1) & (ha <= 6)).sum()} ({((ha > 1) & (ha <= 6)).sum()/n_both*100:.1f}%)")
print(f"  6h < gap <= 24h            : {((ha > 6) & (ha <= 24)).sum()} ({((ha > 6) & (ha <= 24)).sum()/n_both*100:.1f}%)")
print(f"  gap > 24h                  : {(ha > 24).sum()} ({(ha > 24).sum()/n_both*100:.1f}%)")
print(f"  No raw timeseries match    : {len(joined) - n_both}")

# ── 4. Within-Scope Analysis ──────────────────────────────────────────────────
# Per-basin range from the full empirical record
basin_range = (
    hourly_all.groupby("basin_id")["hourly_value"]
    .agg(emp_min="min", emp_max="max")
    .reset_index()
    .rename(columns={"basin_id": "station_id"})
)
joined = joined.merge(basin_range, on="station_id", how="left")

joined["within_scope"] = (joined["momentary_max"] >= joined["emp_min"]) & \
                          (joined["momentary_max"] <= joined["emp_max"])
joined["above_scope"]  = joined["momentary_max"] > joined["emp_max"]
joined["below_scope"]  = joined["momentary_max"] < joined["emp_min"]

joined.to_csv(os.path.join(OUT_DIR, "hourly_vs_momentary_joined.csv"),
              index=False, encoding="utf-8-sig")

total = len(joined)
n_within = joined["within_scope"].sum()
n_above  = joined["above_scope"].sum()
n_below  = joined["below_scope"].sum()
print(f"\nGlobal scope (N={total}):")
print(f"  Within scope : {n_within} ({n_within/total*100:.1f}%)")
print(f"  Above scope  : {n_above}  ({n_above/total*100:.1f}%)")
print(f"  Below scope  : {n_below}  ({n_below/total*100:.1f}%)")

# ── 5. R² Correlation + Time-frame comparison ────────────────────────────────
# Pre-compute per-station year sets from both datasets
am_years_by_station   = am.groupby("station_id")["hydro_year"].apply(set)
hourly_years_by_basin = hourly_all.groupby("basin_id")["hydro_year"].apply(set)

summary_rows = []
for sid, grp in joined.groupby("station_id"):
    grp = grp.dropna(subset=["momentary_max", "hourly_value"])
    n   = len(grp)
    r2  = slope = intercept = None
    if n >= 3:
        res       = linregress(grp["hourly_value"], grp["momentary_max"])
        r2        = res.rvalue ** 2
        slope     = res.slope
        intercept = res.intercept
    n_w = grp["within_scope"].sum()

    # Time-frame comparison
    am_yrs     = am_years_by_station.get(sid, set())
    hourly_yrs = hourly_years_by_basin.get(sid, set())
    overlap    = am_yrs & hourly_yrs
    am_only    = am_yrs - hourly_yrs
    hourly_only = hourly_yrs - am_yrs

    summary_rows.append({
        "station_id":           sid,
        "n_pairs":              n,
        # time-frame
        "am_year_min":          int(min(am_yrs))     if am_yrs     else None,
        "am_year_max":          int(max(am_yrs))     if am_yrs     else None,
        "am_n_years":           len(am_yrs),
        "hourly_year_min":      int(min(hourly_yrs)) if hourly_yrs else None,
        "hourly_year_max":      int(max(hourly_yrs)) if hourly_yrs else None,
        "hourly_n_years":       len(hourly_yrs),
        "overlap_n_years":      len(overlap),
        "overlap_year_min":     int(min(overlap))    if overlap    else None,
        "overlap_year_max":     int(max(overlap))    if overlap    else None,
        "am_only_n_years":      len(am_only),
        "hourly_only_n_years":  len(hourly_only),
        "same_timeframe":       (len(am_only) == 0 and len(hourly_only) == 0),
        # correlation
        "R2":                   round(r2, 4)        if r2        is not None else None,
        "slope":                round(slope, 4)     if slope     is not None else None,
        "intercept":            round(intercept, 4) if intercept is not None else None,
        "n_within_scope":       int(n_w),
        "pct_within_scope":     round(n_w / n * 100, 1) if n > 0 else None,
    })

summary = pd.DataFrame(summary_rows).sort_values("station_id")
summary.to_csv(os.path.join(OUT_DIR, "hourly_vs_momentary_summary.csv"),
               index=False, encoding="utf-8-sig")

# Overall R²
valid      = joined.dropna(subset=["momentary_max", "hourly_value"])
overall    = linregress(valid["hourly_value"], valid["momentary_max"])
overall_r2 = overall.rvalue ** 2
print(f"\nOverall R2 (all basins pooled, N={len(valid)}): {overall_r2:.4f}")
print(f"  slope={overall.slope:.4f}, intercept={overall.intercept:.4f}")
print(f"\nPer-basin R2 (basins with >=3 pairs):")
r2_vals = summary.dropna(subset=["R2"])
print(f"  Basins with R2 : {len(r2_vals)}")
print(f"  Median R2      : {r2_vals['R2'].median():.4f}")
print(f"  Mean R2        : {r2_vals['R2'].mean():.4f}")
print(f"  Min R2         : {r2_vals['R2'].min():.4f}  (basin {r2_vals.loc[r2_vals['R2'].idxmin(),'station_id']})")
print(f"  Max R2         : {r2_vals['R2'].max():.4f}  (basin {r2_vals.loc[r2_vals['R2'].idxmax(),'station_id']})")

n_same = summary["same_timeframe"].sum()
print(f"\nTime-frame alignment ({len(summary)} basins):")
print(f"  Identical time frames : {n_same}")
print(f"  Partial overlap       : {len(summary) - n_same}")
print(f"  Median overlap years  : {summary['overlap_n_years'].median():.0f}")

# ── 6. Scatter plot ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))

ax.scatter(valid["hourly_value"], valid["momentary_max"],
           alpha=0.35, s=18, color="steelblue", edgecolors="none",
           label=f"Basin-year pairs (N={len(valid)})")

# Regression line
x_line = np.linspace(valid["hourly_value"].min(), valid["hourly_value"].max(), 300)
y_line = overall.slope * x_line + overall.intercept
ax.plot(x_line, y_line, color="crimson", linewidth=1.8,
        label=f"Regression (slope={overall.slope:.3f}, intercept={overall.intercept:.2f})")

# 1:1 reference line
xy_max = max(valid["hourly_value"].max(), valid["momentary_max"].max())
ax.plot([0, xy_max], [0, xy_max], "--", color="gray", linewidth=1.2, label="1:1 line")

ax.set_xlabel("Hourly empirical annual max (m³/s)", fontsize=12)
ax.set_ylabel("Momentary annual max (m³/s)", fontsize=12)
ax.set_title("Momentary vs Hourly Empirical Annual Maximum Flow\n(all basins pooled)", fontsize=13)
ax.legend(fontsize=10)
ax.text(0.05, 0.92, f"$R^2$ = {overall_r2:.4f}", transform=ax.transAxes,
        fontsize=12, color="crimson",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="crimson", alpha=0.8))
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.tight_layout()

plot_path = os.path.join(OUT_DIR, "scatter_momentary_vs_hourly.png")
fig.savefig(plot_path, dpi=150)
plt.close()
print("Scatter plot (best-fit) saved.")

# ── 7. Forced slope=1, intercept=0 plot (Nash-Sutcliffe) ─────────────────────
obs  = valid["momentary_max"].values
sim  = valid["hourly_value"].values

residuals   = obs - sim                               # how much is "lost"
ss_res      = np.sum(residuals ** 2)
ss_tot      = np.sum((obs - obs.mean()) ** 2)
nse         = 1.0 - ss_res / ss_tot                  # Nash-Sutcliffe Efficiency
mean_bias   = np.mean(residuals)                      # positive → momentary > hourly
rmse        = np.sqrt(np.mean(residuals ** 2))
rel_bias    = np.mean(residuals / np.where(obs > 0, obs, np.nan)) * 100  # %

print(f"\nForced y=x (slope=1, intercept=0) metrics:")
print(f"  NSE (Nash-Sutcliffe) : {nse:.4f}")
print(f"  Mean bias (mom-hrly) : {mean_bias:.3f} m3/s")
print(f"  RMSE                 : {rmse:.3f} m3/s")
print(f"  Mean relative bias   : {rel_bias:.1f}%")

fig2, ax2 = plt.subplots(figsize=(8, 7))

# colour points by deviation from 1:1
deviation = obs - sim
vmax = np.percentile(np.abs(deviation), 95)
sc = ax2.scatter(sim, obs, c=deviation, cmap="RdBu_r",
                 vmin=-vmax, vmax=vmax,
                 alpha=0.55, s=20, edgecolors="none")
cb = fig2.colorbar(sc, ax=ax2, pad=0.02)
cb.set_label("Momentary − Hourly (m³/s)", fontsize=10)

# forced 1:1 line
xy_max2 = max(sim.max(), obs.max())
ax2.plot([0, xy_max2], [0, xy_max2], "-", color="black", linewidth=2.0,
         label="Forced y = x  (slope=1, intercept=0)")

# best-fit regression for reference (dashed)
x_line2 = np.linspace(sim.min(), sim.max(), 300)
y_line2 = overall.slope * x_line2 + overall.intercept
ax2.plot(x_line2, y_line2, "--", color="crimson", linewidth=1.5,
         label=f"Best-fit regression ($R^2$={overall_r2:.3f})")

ax2.set_xlabel("Hourly empirical annual max (m³/s)", fontsize=12)
ax2.set_ylabel("Momentary annual max (m³/s)", fontsize=12)
ax2.set_title("Information loss: Momentary vs Hourly Annual Max\n(forced slope=1, intercept=0)", fontsize=13)
ax2.legend(fontsize=10)

stats_text = (
    f"NSE = {nse:.4f}\n"
    f"Mean bias = {mean_bias:+.2f} m³/s\n"
    f"RMSE = {rmse:.2f} m³/s\n"
    f"Rel. bias = {rel_bias:+.1f}%"
)
ax2.text(0.04, 0.97, stats_text, transform=ax2.transAxes,
         fontsize=10, verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", alpha=0.85))
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)
plt.tight_layout()

plot2_path = os.path.join(OUT_DIR, "scatter_forced_y_equals_x.png")
fig2.savefig(plot2_path, dpi=150)
plt.close()
print("Scatter plot (forced y=x) saved.")
print("\nOutputs written to return_periods dir.")
