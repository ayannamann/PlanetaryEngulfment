import numpy as np
import pandas as pd


def load_model_sed(sed_path):
    """Load a DUSTY sed.dat file written by run_dusty_grid.py."""
    data = np.loadtxt(sed_path, delimiter=",", skiprows=2)
    lam_um = data[:, 0]
    flx = data[:, 1]
    return lam_um, flx


def _model_at(lam_model, flx_model, lam_query):
    """Log-log interpolate the model SED at given wavelength(s)."""
    mask = flx_model > 0
    log_model = np.interp(
        np.log10(lam_query), np.log10(lam_model[mask]), np.log10(flx_model[mask])
    )
    return 10 ** log_model


def chi2_for_model(lam_model, flx_model, obs_lam, obs_flux, obs_unc,
                    lim_lam=None, lim_flux=None, lim_unc=None, scale_free=True):
    """
    Chi-square between one model SED and observed points, with optional
    upper limits (non-detections).

    Detections (obs_lam/obs_flux/obs_unc) get the usual two-sided penalty:
    the model is wrong whether it over- or under-predicts.

    Limits (lim_lam/lim_flux/lim_unc) get a one-sided penalty: a model
    is only penalized if it predicts MORE flux than the limit allows.
    A model predicting less flux than the limit is fully consistent with
    a non-detection and contributes zero to chi2 -- it isn't "rewarded"
    for being faint, it's just not constrained by that point.

    scale_free=True fits a single log-space normalization using only the
    detections (not the limit -- an upper limit can't anchor a scale,
    only rule out models once a scale from detections is applied), then
    applies that same scale to check the limit.
    """
    model_at_obs = _model_at(lam_model, flx_model, obs_lam)

    if scale_free:
        scale = 10 ** np.mean(np.log10(obs_flux) - np.log10(model_at_obs))
    else:
        scale = 1.0
    model_at_obs = model_at_obs * scale

    chi2 = np.sum(((obs_flux - model_at_obs) / obs_unc) ** 2)

    if lim_lam is not None:
        lim_lam = np.atleast_1d(lim_lam)
        lim_flux = np.atleast_1d(lim_flux)
        lim_unc = np.atleast_1d(lim_unc)

        model_at_lim = _model_at(lam_model, flx_model, lim_lam) * scale

        # one-sided: only penalize where model exceeds the limit
        over = np.clip(model_at_lim - lim_flux, a_min=0, a_max=None)
        chi2 += np.sum((over / lim_unc) ** 2)

    return chi2, scale


def fit_epoch(grid_csv, obs_lam, obs_flux, obs_unc,
              lim_lam=None, lim_flux=None, lim_unc=None,
              out_csv="grid_fit_results.csv"):
    """
    obs_lam, obs_flux, obs_unc: detections for one epoch -- wavelength [um],
    lambda*F_lambda [erg/s/cm2], and its uncertainty.

    lim_lam, lim_flux, lim_unc: optional non-detections (upper limits) for
    the same epoch, same units. lim_flux should be the limiting flux (e.g.
    an N-sigma non-detection threshold), lim_unc its associated uncertainty
    for the one-sided penalty scale.
    """
    grid = pd.read_csv(grid_csv)
    grid = grid[grid["ierror"] == 0]

    results = []
    for _, row in grid.iterrows():
        lam_model, flx_model = load_model_sed(row["sed_path"])
        chi2, scale = chi2_for_model(lam_model, flx_model, obs_lam, obs_flux, obs_unc,
                                      lim_lam=lim_lam, lim_flux=lim_flux, lim_unc=lim_unc)
        results.append({
            "tstar": row["tstar"], "tdust": row["tdust"], "tau": row["tau"],
            "dust_type": row["dust_type"], "shell_thickness": row["shell_thickness"],
            "chi2": chi2, "scale": scale,
        })

    results_df = pd.DataFrame(results).sort_values("chi2")
    results_df.to_csv(out_csv, index=False)

    best = results_df.iloc[0]
    n_points = len(obs_lam) + (0 if lim_lam is None else len(np.atleast_1d(lim_lam)))
    n_dof = n_points - 4  # tstar, tdust, tau, scale
    print(f"Best fit ({out_csv}):")
    print(f"  tstar = {best['tstar']:.0f} K")
    print(f"  tdust = {best['tdust']:.0f} K")
    print(f"  tau   = {best['tau']:.4g}")
    print(f"  dust_type = {best['dust_type']}")
    print(f"  chi2  = {best['chi2']:.3f}  (dof = {n_dof}, {n_points} points incl. limits)")

    # note: dof/chi2 interpretation with an upper limit included is
    # approximate -- a non-detection contributes 0 or a real penalty
    # depending on the model, it isn't a fixed-information point the way
    # a detection is, so treat this chi2 as a ranking refinement more than
    # a rigorous statistic, same caveat as with 2-3 detections alone
    return results_df


if __name__ == "__main__":
    from plot_dusty_sed_overlay import read_diff_json, uJy_to_lamFlam

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


# ATLAS SECTION- adding nearby ATLAS-o non-detection to the WISE epoch fit as a limit to further constrain the curve of the fit. Using the closest non detection with the lowest error margin.
    from atlas_reader import read_atlas_forcedphot_file  # adjust import to wherever this lives

    snr_thresh = 5  # keep consistent with whatever read_atlas_forcedphot_file used
    atlas_path = "/Users/ayannamann/Downloads/ATLAS.txt"  # <-- set to your actual file path

    odets, cdets, olims, clims = read_atlas_forcedphot_file(atlas_path, snr_thresh=snr_thresh)

    # among ATLAS-o non-detections within a time window of the WISE epoch,
    # pick the tightest (smallest duJy) rather than merely the nearest in
    # time -- a tighter limit a bit further away is more constraining than
    # a loose one right next to the epoch
    time_window_days = 30  # adjust: how far from the WISE epoch counts as "nearby"

    wise_mjd = f1_dets["mjd"][0]
    nearby_lims = olims[(olims["mjd"] - wise_mjd).abs() <= time_window_days]

    if len(nearby_lims) == 0:
        raise ValueError(
            f"No ATLAS-o non-detections within {time_window_days} days of "
            f"WISE epoch MJD={wise_mjd:.2f}. Widen time_window_days or check atlas_path."
        )

    tightest_idx = nearby_lims["duJy"].idxmin()
    closest_lim = nearby_lims.loc[tightest_idx]

    print(f"Tightest ATLAS-o non-detection within {time_window_days} days: "
          f"MJD={closest_lim['mjd']:.2f} (dt={closest_lim['mjd'] - wise_mjd:+.2f} days), "
          f"duJy={closest_lim['duJy']:.3g}  "
          f"[{len(nearby_lims)} candidates in window]")

    lam_atlas_o = 0.6786  # ATLAS o-band pivot wavelength, microns

    lim_flux_uJy = snr_thresh * closest_lim["duJy"]  # N-sigma non-detection threshold
    lim_unc_uJy = closest_lim["duJy"]                # 1-sigma, sets penalty steepness

    lim_lam = np.array([lam_atlas_o])
    lim_flux = np.array([uJy_to_lamFlam(lim_flux_uJy, lam_atlas_o)])
    lim_unc = np.array([uJy_to_lamFlam(lim_unc_uJy, lam_atlas_o)])

    fit_epoch("grid_summary.csv", obs_lam, obs_flux, obs_unc,
              lim_lam=lim_lam, lim_flux=lim_flux, lim_unc=lim_unc,
              out_csv="grid_fit_wise_epoch.csv")

    # for the peak epoch (MJD 60258.40, ATLAS_c + ZTF_g + ZTF_r), build
    # obs_lam_peak, obs_flux_peak, obs_unc_peak the same way and call:
    #   fit_epoch("grid_summary.csv", obs_lam_peak, obs_flux_peak,
    #             obs_unc_peak, out_csv="grid_fit_peak_epoch.csv")
    # then compare tau/tdust between the two grid_fit_*.csv results to see
    # how the dust column evolved between the two epochs
