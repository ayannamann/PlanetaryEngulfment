import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
from pathlib import Path
import itertools as it
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from pydusty.dusty import Dusty, DustyParameters
from pydusty.parameters import Parameter
from pydusty.utils import getLogger


def make_leaf_and_key(tstar, tdust, tau, dust_type, shell_thickness, ndigits=6):
    tstar_i = int(round(float(tstar)))
    tdust_i = int(round(float(tdust)))
    tau_f = round(float(tau), ndigits)
    thick_f = round(float(shell_thickness), ndigits)

    leaf = (f"Tstar_{tstar_i}_Tdust_{tdust_i}_tau_{tau_f:.{ndigits}g}_"
            f"{dust_type}_thick_{thick_f:.{ndigits}g}").replace('.', '_')
    key = (tstar_i, tdust_i, tau_f, dust_type, thick_f)
    return leaf, key


def run_single_model(job):
    """Run one DUSTY model in its own subdirectory."""
    (tstar_val, tdust_val, tau_val, dust_type_val, shell_thick_val,
     tau_wav_micron, dusty_file_dir, base_workdir, ndigits) = job

    leaf, key = make_leaf_and_key(tstar_val, tdust_val, tau_val,
                                   dust_type_val, shell_thick_val, ndigits=ndigits)
    run_dir = (Path(base_workdir) / leaf).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    sed_path = run_dir / "sed.dat"

    tstar = Parameter(name='tstar', value=tstar_val, is_variable=False)
    tdust = Parameter(name='tdust', value=tdust_val, is_variable=True)
    tau = Parameter(name='tau', value=tau_val, is_variable=False)
    blackbody = Parameter(name='blackbody', value=True)
    shell_thickness = Parameter(name='shell_thickness', value=shell_thick_val)
    dust_type = Parameter(name='dust_type', value=dust_type_val)
    tstarmin = Parameter(name='tstarmin', value=3500)
    tstarmax = Parameter(name='tstarmax', value=48999)
    custom_grain_distribution = Parameter(name='custom_grain_distribution', value=False)
    tau_wav = Parameter(name='tau_wav', value=tau_wav_micron, is_variable=False)

    dusty_parameters = DustyParameters(
        tstar=tstar,
        tdust=tdust,
        tau=tau,
        blackbody=blackbody,
        shell_thickness=shell_thickness,
        dust_type=dust_type,
        tstarmin=tstarmin,
        tstarmax=tstarmax,
        custom_grain_distribution=custom_grain_distribution,
        tau_wavelength_microns=tau_wav,
    )

    prev_cwd = os.getcwd()
    try:
        os.chdir(dusty_file_dir) #to ensure DUSTY can find its code files, which are in the same directory as dusty.py
        dusty_runner = Dusty(
            parameters=dusty_parameters,
            dusty_working_directory=str(run_dir),
            dusty_file_directory=dusty_file_dir,
        )

        os.chdir(str(run_dir))
        dusty_runner.generate_input()
        dusty_runner.run()
        lam, flx, npt, r1, ierror = dusty_runner.get_results()
        os.chdir(prev_cwd)

        if ierror != 0:
            return dict(tstar=tstar_val, tdust=tdust_val, tau=tau_val,
                        dust_type=dust_type_val, shell_thickness=shell_thick_val,
                        sed_path=None, r1=np.nan, ierror=ierror,
                        error="DUSTY returned nonzero ierror")

        # trim to actual output length, in case DUSTY preallocates larger arrays
        lam = np.asarray(lam)[:npt]
        flx = np.asarray(flx)[:npt]

        with sed_path.open('w') as f:
            f.write(f"# {r1}\n")
            f.write("lam, flux\n")
            for lam_val, flux_val in zip(lam, flx):
                f.write(f"{lam_val}, {flux_val}\n")

        return dict(tstar=tstar_val, tdust=tdust_val, tau=tau_val,
                    dust_type=dust_type_val, shell_thickness=shell_thick_val,
                    sed_path=str(sed_path), r1=r1, ierror=0, error=None)

    except Exception as e:
        os.chdir(prev_cwd)
        return dict(tstar=tstar_val, tdust=tdust_val, tau=tau_val,
                    dust_type=dust_type_val, shell_thickness=shell_thick_val,
                    sed_path=None, r1=np.nan, ierror=1, error=str(e))


def _parse_list(s):
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run a DUSTY grid over tstar, tdust, tau, dust_type")
    parser.add_argument("workdir", type=str, help="directory to store per-model DUSTY runs")
    parser.add_argument("--dusty_file_dir", type=str, required=True,
                         help="directory with DUSTY code files")
    parser.add_argument("--tau_wav_micron", type=float, default=0.55,
                         help="wavelength (microns) at which tau is specified; 0.55 = V band")
    parser.add_argument("--shell_thickness", type=float, default=2.0)
    parser.add_argument("--dust_types", type=str, choices=['graphite', 'silicate',
                                            'amorphous_carbon', 'silicate_carbide'],
                                default="silicate,amorphous_carbon,graphite",
                         help="comma-separated dust types to run")
    parser.add_argument("--tstar_list", type=str, default=None,
                         help="override default tstar grid, comma-separated")
    parser.add_argument("--tdust_list", type=str, default=None,
                         help="override default tdust grid, comma-separated")
    parser.add_argument("--tau_list", type=str, default=None,
                         help="override default tau grid, comma-separated")
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--loglevel", type=str, default="INFO")
    parser.add_argument("--logfile", type=str, default=None)
    args = parser.parse_args()

    logger = getLogger(args.loglevel, args.logfile)

    # stellar temperature (K)
    tstar_values = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000,
                     6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]
    # inner dust temperature (K)
    tdust_values = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    # for optical depth at tau_wav_micron= 100 micron
    # tau_values = list(np.r_[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]) #(.1-30)
    # using optical depth at tau_wav_micron= 0.55 micron (V band), therefore:
    tau_values = list(np.r_[1e-1, 5e-1, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]) #(.1-30)

    if args.tstar_list:
        tstar_values = _parse_list(args.tstar_list)
    if args.tdust_list:
        tdust_values = _parse_list(args.tdust_list)
    if args.tau_list:
        tau_values = _parse_list(args.tau_list)

    dust_types = [s.strip() for s in args.dust_types.split(",") if s.strip()]

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    dusty_file_dir = str(Path(args.dusty_file_dir).resolve())

    out_csv = Path(args.out_csv).resolve() if args.out_csv else workdir / "grid_summary.csv"

    # resume support: skip combos with a completed sed.dat from a prior run
    completed = set()
    if not args.force_rerun and out_csv.exists():
        prev = pd.read_csv(out_csv)
        prev_ok = prev[(prev["ierror"] == 0) & (prev["sed_path"].notna())]
        for _, r in prev_ok.iterrows():
            _, k = make_leaf_and_key(r["tstar"], r["tdust"], r["tau"], r["dust_type"], r["shell_thickness"])
            completed.add(k)
        logger.info(f"Found {len(completed)} completed models to skip.")

    jobs = []
    skipped = 0
    for t, d, tv, dtype in it.product(tstar_values, tdust_values, tau_values, dust_types):
        _, key = make_leaf_and_key(t, d, tv, dtype, args.shell_thickness)
        if key in completed:
            skipped += 1
            continue
        jobs.append((t, d, tv, dtype, args.shell_thickness, args.tau_wav_micron,
                     dusty_file_dir, str(workdir), 6))

    logger.info(f"Skipping {skipped} cached models, running {len(jobs)} new ones.")
    if not jobs:
        logger.info("Nothing to run.")
        return

    max_workers = min(os.cpu_count() or 2, args.n_workers)
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_single_model, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            status = "OK" if res["ierror"] == 0 else "ERROR"
            logger.info(f"[{i}/{len(jobs)}] {status} T*={res['tstar']} Td={res['tdust']} "
                        f"tau={res['tau']} dust={res['dust_type']}")

    new_df = pd.DataFrame(results)
    if not args.force_rerun and out_csv.exists():
        existing = pd.read_csv(out_csv)
        df = pd.concat([existing, new_df], ignore_index=True)
    else:
        df = new_df
    df = df.sort_values(["dust_type", "tstar", "tdust", "tau"])
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
