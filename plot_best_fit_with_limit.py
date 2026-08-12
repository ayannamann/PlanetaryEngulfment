import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_dusty_sed_overlay import read_diff_json, uJy_to_lamFlam
from fit_dusty_grid import load_model_sed, fit_epoch
from atlas_reader import read_atlas_forcedphot_file  # adjust import to wherever this lives


# --- WISE detections ---
f1_dets, f2_dets = read_diff_json("ZTF23abomysh_diff.json")[:2]
lam_w1, lam_w2 = 33526 / 1e4, 46028 / 1e4

obs_lam = np.array([lam_w1, lam_w2])
obs_flux = np.array([
    uJy_to_lamFlam(f1_dets["flux_uJy"][0], lam_w1),
    uJy_to_lamFlam(f2_dets["flux_uJy"][0], lam_w2),
])
obs_unc = np.array([
    uJy_to_lamFlam(f1_dets["fluxunc_uJy"][0], lam_w1),
    uJy_to_lamFlam(f2_dets["fluxunc_uJy"][0], lam_w2),
])

# --- tightest nearby ATLAS-o non-detection ---
snr_thresh = 5
atlas_path = "/Users/ayannamann/Downloads/ATLAS.txt" 
time_window_days = 30

odets, cdets, olims, clims = read_atlas_forcedphot_file(atlas_path, snr_thresh=snr_thresh)

wise_mjd = f1_dets["mjd"][0]
nearby_lims = olims[(olims["mjd"] - wise_mjd).abs() <= time_window_days]
if len(nearby_lims) == 0:
    raise ValueError(f"No ATLAS-o non-detections within {time_window_days} days of MJD={wise_mjd:.2f}")

tightest_idx = nearby_lims["duJy"].idxmin()
closest_lim = nearby_lims.loc[tightest_idx]

lam_atlas_o = 0.6786
lim_flux_uJy = snr_thresh * closest_lim["duJy"]
lim_unc_uJy = closest_lim["duJy"]

lim_lam = np.array([lam_atlas_o])
lim_flux = np.array([uJy_to_lamFlam(lim_flux_uJy, lam_atlas_o)])
lim_unc = np.array([uJy_to_lamFlam(lim_unc_uJy, lam_atlas_o)])

# --- run the fit and get the best model ---
results_df = fit_epoch("grid_summary.csv", obs_lam, obs_flux, obs_unc,
                        lim_lam=lim_lam, lim_flux=lim_flux, lim_unc=lim_unc,
                        out_csv="grid_fit_wise_epoch.csv")

best = results_df.iloc[0]
grid_summary = pd.read_csv("grid_summary.csv")
best_row = grid_summary[
    (grid_summary["tstar"] == best["tstar"]) &
    (grid_summary["tdust"] == best["tdust"]) &
    (grid_summary["tau"] == best["tau"]) &
    (grid_summary["dust_type"] == best["dust_type"])
].iloc[0]
lam_model, flx_model = load_model_sed(best_row["sed_path"])
scale = best["scale"]

# --- plot ---
fig, ax = plt.subplots(figsize=(8, 6))

mask = flx_model > 0
ax.plot(lam_model[mask], flx_model[mask] * scale, color="0.3", lw=1.5,
        label=f"Best fit ({best['dust_type']}, "
              f"$T_\\star$={best['tstar']:.0f}K, $T_d$={best['tdust']:.0f}K, "
              f"$\\tau$={best['tau']:.3g})")

ax.errorbar(lam_w1, obs_flux[0], yerr=obs_unc[0], fmt="s", ms=9, color="black",
            markeredgecolor="k", capsize=3, label="WISE W1 detection")
ax.errorbar(lam_w2, obs_flux[1], yerr=obs_unc[1], fmt="s", ms=9, color="orange",
            markeredgecolor="k", capsize=3, label="WISE W2 detection")

# ATLAS-o non-detection: hollow pink hexagon, distinct from the WISE squares,
# with a downward arrow (standard convention for an upper limit)
ax.errorbar(lim_lam[0], lim_flux[0], yerr=lim_flux[0] * 0.3, uplims=True,
            fmt="h", ms=12, mfc="none", mec="deeppink", mew=1.8,
            ecolor="deeppink", capsize=3, label="ATLAS-o non-detection (5$\\sigma$)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.3, 20)
ax.set_xlabel(r"$\lambda$ [$\mu$m]")
ax.set_ylabel(r"$\lambda F_\lambda$ [erg s$^{-1}$ cm$^{-2}$]")
ax.set_title("Best-fit DUSTY model vs WISE detections + ATLAS-o limit")
ax.legend(loc="best", frameon=False)
ax.grid(alpha=0.2, which="both")

plt.tight_layout()
plt.savefig("dusty_best_fit_with_atlas_limit.png", dpi=200)
plt.show()
