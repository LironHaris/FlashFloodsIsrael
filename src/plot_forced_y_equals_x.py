import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

OUT_DIR   = r"C:\Users\user\Documents\פרויקט שטפונות\floods\return_periods"
JOINED    = os.path.join(OUT_DIR, "hourly_vs_momentary_joined.csv")

df   = pd.read_csv(JOINED, encoding="utf-8-sig")
valid = df.dropna(subset=["momentary_max", "hourly_value"])
obs  = valid["momentary_max"].values
sim  = valid["hourly_value"].values

# ── best-fit regression (for reference line) ──────────────────────────────────
overall    = linregress(sim, obs)
overall_r2 = overall.rvalue ** 2

# ── forced y=x metrics (Nash-Sutcliffe) ───────────────────────────────────────
residuals = obs - sim
ss_res    = np.sum(residuals ** 2)
ss_tot    = np.sum((obs - obs.mean()) ** 2)
nse       = 1.0 - ss_res / ss_tot
mean_bias = float(np.mean(residuals))
rmse      = float(np.sqrt(np.mean(residuals ** 2)))
rel_bias  = float(np.nanmean(residuals / np.where(obs > 0, obs, np.nan)) * 100)

print(f"Forced y=x (slope=1, intercept=0) metrics (N={len(valid)}):")
print(f"  NSE (Nash-Sutcliffe) : {nse:.4f}")
print(f"  Mean bias (mom-hrly) : {mean_bias:+.3f} m3/s")
print(f"  RMSE                 : {rmse:.3f} m3/s")
print(f"  Mean relative bias   : {rel_bias:+.1f}%")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))

vmax = np.percentile(np.abs(residuals), 95)
sc = ax.scatter(sim, obs, c=residuals, cmap="RdBu_r",
                vmin=-vmax, vmax=vmax, alpha=0.55, s=20, edgecolors="none")
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("Momentary − Hourly (m³/s)", fontsize=10)

xy_max = max(sim.max(), obs.max())
ax.plot([0, xy_max], [0, xy_max], "-", color="black", linewidth=2.0,
        label="Forced y = x  (slope=1, intercept=0)")

x_ref = np.linspace(sim.min(), sim.max(), 300)
ax.plot(x_ref, overall.slope * x_ref + overall.intercept,
        "--", color="crimson", linewidth=1.5,
        label=f"Best-fit regression ($R^2$={overall_r2:.3f})")

ax.set_xlabel("Hourly empirical annual max (m³/s)", fontsize=12)
ax.set_ylabel("Momentary annual max (m³/s)", fontsize=12)
ax.set_title("Information loss: Momentary vs Hourly Annual Max\n(forced slope=1, intercept=0)", fontsize=13)
ax.legend(fontsize=10)

stats_text = (
    f"NSE = {nse:.4f}\n"
    f"Mean bias = {mean_bias:+.2f} m³/s\n"
    f"RMSE = {rmse:.2f} m³/s\n"
    f"Rel. bias = {rel_bias:+.1f}%"
)
ax.text(0.04, 0.97, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", alpha=0.85))
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "scatter_forced_y_equals_x.png")
fig.savefig(out_path, dpi=150)
plt.close()
print("Saved: scatter_forced_y_equals_x.png")
