# A Photometric Blueprint for Planetary Engulfment

Analysis code for **"A Photometric Blueprint for Planetary Engulfment: Multiwavelength Analysis of a Second Candidate"** (Mann, De, & Karambelkar, in prep.).

This repository contains the code used to identify and characterize **ZTF SLRN-2023**, the second candidate planetary engulfment event, and to compare it with the first known event, ZTF SLRN-2020 ([De et al. 2023](https://www.nature.com/articles/s41586-023-05842-x)).

---

## Overview

When a star swells and engulfs a close-in planet, the merger can produce a faint, dusty optical outburst followed by a long-lived infrared brightening. This project uses archival multiwavelength photometry to:

1. **Find and vet candidates.** Search archival optical and infrared surveys, then analyze light curves and color–color / color–magnitude diagrams to separate engulfment candidates from other faint transients.
2. **Model the dust.** Fit the infrared spectral energy distribution with the [DUSTY](https://github.com/ivezic/dusty) radiative transfer code to constrain the luminosity, temperature, distance, extinction, and dust mass. The dust mass serves as a proxy for the mass of the engulfed planet.
3. **Measure the outburst energetics.** Compute blackbody luminosities, the total radiated energy, the time to radiate 90% of that energy (*t*₉₀), the luminosity over that time (*L*₉₀), and the plateau duration.
4. **Estimate ejecta properties.** Apply the analytic merger models of Matsumoto & Metzger (2022) to estimate ejecta mass and velocity.
5. **Place the event in context.** Compare it with ZTF SLRN-2020 and luminous red novae in the *L*₉₀–duration plane, following Fig. 4 of De et al. (2023).

This repository currently contains the code for the light curves, color evolution, blackbody fitting, and DUSTY modeling (steps 1–2 and the blackbody luminosities). The energetics, ejecta, and comparison analyses (steps 3–5) will be added later.

---

## Repository contents

### Notebooks

| File | Description |
|------|-------------|
| `Light_Curve_Figure.ipynb` | Builds the multiwavelength (optical + infrared) light curve figure. |
| `Color_Evolution_Figure-2.ipynb` | Tracks the color evolution of the transient over time (color–color / color–magnitude diagrams). |
| `BBfit_extinction-2-3.ipynb` | Fits blackbodies to the photometry with extinction correction to derive luminosity and temperature evolution. |

### DUSTY scripts

| File | Description |
|------|-------------|
| `run_dusty_grid.py` | Generates and runs a grid of DUSTY radiative transfer models across dust and stellar parameters. |
| `fit_dusty_grid_true.py` | Fits the DUSTY model grid to the observed SED to find the best-fitting parameters. |
| `plot_best_fit_with_limit.py` | Plots the best-fit DUSTY model against the observed SED, including upper limits. |

---

## Requirements

- Python ≥ 3.9
- `numpy`, `scipy`, `matplotlib`, `astropy`, `pandas`
- `jupyter` (for the notebooks)
- [DUSTY](https://github.com/ivezic/dusty) (Fortran; compile separately and point the scripts to the executable)

Install the Python dependencies:

```bash
pip install numpy scipy matplotlib astropy pandas jupyter
```

---

## Usage

**Photometric analysis:** run the notebooks in Jupyter:

1. `Light_Curve_Figure.ipynb`
2. `Color_Evolution_Figure-2.ipynb`
3. `BBfit_extinction-2-3.ipynb`

**DUSTY modeling:** after compiling DUSTY and setting the path to its executable, run:

```bash
# 1. Generate the DUSTY model grid
python run_dusty_grid.py

# 2. Fit the grid to the observed SED
python fit_dusty_grid_true.py

# 3. Plot the best-fit model
python plot_best_fit_with_limit.py
```

---

## Data

This work uses public data from:

- **Zwicky Transient Facility (ZTF)**: optical light curves
- **Asteroid Terrestrial-impact Last Alert System (ATLAS)**: optical light curves
- **Wide-field Infrared Survey Explorer (WISE/NEOWISE)**: mid-infrared light curves

> Add any other surveys or follow-up data used, and note which data files are not included in the repo (e.g., because of size or proprietary status).

---

## Citation

If you use this code, please cite:

```
Mann, A.T., De, K., & Karambelkar, V. "A Photometric Blueprint for Planetary
Engulfment: Multiwavelength Analysis of a Second Candidate." In prep.
```

Please also cite the tools and methods this work builds on:

- De, K., et al. 2023, *Nature*, 617, 55. "An infrared transient from a star engulfing a planet"
- Matsumoto, T., & Metzger, B. D. 2022, *ApJ*, 938, 5. "Light-curve Model for Luminous Red Novae and Inferences about the Ejecta of Stellar Mergers"
- Ivezić, Ž., & Elitzur, M. 1997, *MNRAS*, 287, 799. DUSTY radiative transfer code

---

## Contact

A.T. Mann — atm2184@columbia.edu | https://www.linkedin.com/in/ayanna-mann-274a421b3/
