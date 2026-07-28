import os
import glob
import math
import pandas as pd
import numpy as np

BASE_DIR = r"C:\Users\user\Downloads\floods\return_periods"
OUT_JOINED = os.path.join(BASE_DIR, "empirical_vs_new_RP_joined.csv")
OUT_DIAG   = os.path.join(BASE_DIR, "empirical_vs_new_RP_diagnostics.csv")

RP_LEVELS = [2, 5, 10, 20, 50, 100]
TARGET_YEARS = [2023, 2024]

# ── locate and load the new RP file ───────────────────────────────────────────
rp_candidates = glob.glob(os.path.join(BASE_DIR, "hydro_st_RP*.xlsx"))
rp_candidates = [f for f in rp_candidates if not os.path.basename(f).startswith("~$")]
if not rp_candidates:
    raise FileNotFoundError("No hydro_st_RP*.xlsx file found in " + BASE_DIR)
rp_path = rp_candidates[0]

rp = pd.read_excel(rp_path, sheet_name=0, header=1)
rp = rp[["basin_id", "basin_name", "RP2", "RP5", "RP10", "RP20", "RP50", "RP100"]].copy()
rp["basin_id"] = pd.to_numeric(rp["basin_id"], errors="coerce")
rp = rp[rp["basin_id"].notna()].copy()
rp["basin_id"] = rp["basin_id"].astype(int)
rp = rp.drop_duplicates(subset="basin_id", keep="first")
rp_by_id = rp.set_index("basin_id")

# ── enumerate basin directories ────────────────────────────────────────────────
basin_dirs = sorted(
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d)) and d.startswith("il_")
)
dir_basin_ids = {int(d.replace("il_", "")): d for d in basin_dirs}

all_basin_ids = sorted(set(rp_by_id.index.tolist()) | set(dir_basin_ids.keys()))


def load_am(path):
    """Load an empirical annual-maxima CSV, or None if missing/empty."""
    if path is None or not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    if df.empty or "Hydro_Year" not in df.columns:
        return None
    return df


def year_row(df, year):
    """Return (value, rp) for a given Hydro_Year, or (None, None) if absent."""
    if df is None:
        return None, None
    match = df[df["Hydro_Year"] == year]
    if match.empty:
        return None, None
    row = match.iloc[0]
    return float(row["Value"]), float(row["Empirical_Return_Period_Years"])


def alltime_max_row(df):
    """Return (year, value, rp) for the all-time maximum Value, or (None, None, None)."""
    if df is None:
        return None, None, None
    row = df.loc[df["Value"].idxmax()]
    return int(row["Hydro_Year"]), float(row["Value"]), float(row["Empirical_Return_Period_Years"])


def nearest_rp_level(rp_emp):
    """Snap an empirical return period to the closest of RP_LEVELS, in log-space."""
    if rp_emp is None or rp_emp <= 0:
        return None
    return min(RP_LEVELS, key=lambda k: abs(math.log(rp_emp) - math.log(k)))


def grade_point(value, rp_emp, rp_row):
    """
    Snap rp_emp to the nearest RP bucket k, scale `value` by the same factor
    (k / rp_emp) that would turn rp_emp into k, then compare the scaled value
    against the new RP file's official value for bucket k.
    Returns a dict of the intermediate numbers plus a % difference "grade".
    """
    out = {
        "nearest_RP": None, "scale_factor": None, "scaled_value": None,
        "official_RP_value": None, "grade_pct_diff": None,
    }
    if value is None or rp_emp is None:
        return out
    k = nearest_rp_level(rp_emp)
    factor = k / rp_emp
    scaled_value = value * factor
    official_value = None
    if rp_row is not None:
        col = f"RP{k}"
        if pd.notna(rp_row[col]):
            official_value = float(rp_row[col])
    grade = None
    if official_value not in (None, 0):
        grade = (scaled_value - official_value) / official_value * 100.0
    out.update({
        "nearest_RP": k, "scale_factor": factor, "scaled_value": scaled_value,
        "official_RP_value": official_value, "grade_pct_diff": grade,
    })
    return out


def series_block(prefix, am_df, rp_row):
    """Build all columns for one series (hourly/daily): 2023, 2024, and all-time fallback."""
    block = {}
    present = {}
    for year in TARGET_YEARS:
        val, rp_emp = year_row(am_df, year)
        present[year] = val is not None
        g = grade_point(val, rp_emp, rp_row)
        block[f"{prefix}_{year}_present"]          = present[year]
        block[f"{prefix}_{year}_value"]             = val
        block[f"{prefix}_{year}_empirical_rp"]       = rp_emp
        block[f"{prefix}_{year}_nearest_RP"]         = g["nearest_RP"]
        block[f"{prefix}_{year}_scale_factor"]       = g["scale_factor"]
        block[f"{prefix}_{year}_scaled_value"]       = g["scaled_value"]
        block[f"{prefix}_{year}_official_RP_value"]  = g["official_RP_value"]
        block[f"{prefix}_{year}_grade_pct_diff"]     = g["grade_pct_diff"]

    has_data = am_df is not None
    used_fallback = has_data and not present[TARGET_YEARS[0]] and not present[TARGET_YEARS[1]]
    block[f"{prefix}_has_data"] = has_data
    block[f"{prefix}_used_alltime_fallback"] = used_fallback

    fb_year = fb_val = fb_rp = None
    if used_fallback:
        fb_year, fb_val, fb_rp = alltime_max_row(am_df)
    g = grade_point(fb_val, fb_rp, rp_row)
    block[f"{prefix}_fallback_year"]             = fb_year
    block[f"{prefix}_fallback_value"]            = fb_val
    block[f"{prefix}_fallback_empirical_rp"]      = fb_rp
    block[f"{prefix}_fallback_nearest_RP"]        = g["nearest_RP"]
    block[f"{prefix}_fallback_scale_factor"]      = g["scale_factor"]
    block[f"{prefix}_fallback_scaled_value"]      = g["scaled_value"]
    block[f"{prefix}_fallback_official_RP_value"] = g["official_RP_value"]
    block[f"{prefix}_fallback_grade_pct_diff"]    = g["grade_pct_diff"]
    return block


joined_rows = []
diag_rows = []

for bid in all_basin_ids:
    rp_row = rp_by_id.loc[bid] if bid in rp_by_id.index else None
    basin_dir = dir_basin_ids.get(bid)
    basin_name = rp_row["basin_name"] if rp_row is not None else None

    hourly_path = os.path.join(BASE_DIR, basin_dir, "Hourly_Flow_empirical_AM.csv") if basin_dir else None
    daily_path  = os.path.join(BASE_DIR, basin_dir, "Daily_Flow_empirical_AM.csv")  if basin_dir else None

    hourly_df = load_am(hourly_path)
    daily_df  = load_am(daily_path)

    def rp_val(col):
        return float(rp_row[col]) if rp_row is not None and pd.notna(rp_row[col]) else None

    row = {
        "basin_id": bid,
        "basin_name": basin_name,
        "RP2": rp_val("RP2"), "RP5": rp_val("RP5"), "RP10": rp_val("RP10"),
        "RP20": rp_val("RP20"), "RP50": rp_val("RP50"), "RP100": rp_val("RP100"),
    }
    row.update(series_block("hourly", hourly_df, rp_row))
    row.update(series_block("daily", daily_df, rp_row))
    joined_rows.append(row)

    diag_rows.append({
        "basin_id": bid,
        "basin_name": basin_name,
        "has_rp_row": rp_row is not None,
        "has_basin_dir": basin_dir is not None,
        "missing_hourly_file": hourly_path is None or not os.path.isfile(hourly_path),
        "missing_daily_file":  daily_path is None or not os.path.isfile(daily_path),
        "hourly_has_data": row["hourly_has_data"],
        "hourly_present_2023": row["hourly_2023_present"],
        "hourly_present_2024": row["hourly_2024_present"],
        "hourly_used_alltime_fallback": row["hourly_used_alltime_fallback"],
        "hourly_fallback_year": row["hourly_fallback_year"],
        "daily_has_data": row["daily_has_data"],
        "daily_present_2023": row["daily_2023_present"],
        "daily_present_2024": row["daily_2024_present"],
        "daily_used_alltime_fallback": row["daily_used_alltime_fallback"],
        "daily_fallback_year": row["daily_fallback_year"],
    })

joined = pd.DataFrame(joined_rows)
diag   = pd.DataFrame(diag_rows)

try:
    joined.to_csv(OUT_JOINED, index=False, encoding="utf-8-sig")
    diag.to_csv(OUT_DIAG, index=False, encoding="utf-8-sig")
except PermissionError:
    raise SystemExit(
        "Could not write output - the CSV is probably still open in Excel. "
        "Close it and re-run."
    )

print("RP file used:", rp_path)
print("Total basins (union of RP rows and directories):", len(all_basin_ids))
print("  with RP row:        ", diag["has_rp_row"].sum())
print("  with basin dir:     ", diag["has_basin_dir"].sum())
print("  missing hourly file:", diag["missing_hourly_file"].sum())
print("  missing daily file: ", diag["missing_daily_file"].sum())
print("  hourly: has data", diag["hourly_has_data"].sum(), "| of those, used fallback:", diag["hourly_used_alltime_fallback"].sum())
print("  daily:  has data", diag["daily_has_data"].sum(), "| of those, used fallback: ", diag["daily_used_alltime_fallback"].sum())
print("Joined CSV ->", OUT_JOINED)
print("Diagnostics CSV ->", OUT_DIAG)
