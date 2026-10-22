# Drogon seismic-assisted history-matching experiments

Reproduction code for the realistic example (Section 5.1) of *Subsurface Data
Assimilation: Theory and Applications*. The original working directory was
`Jupiter2:~/bookchapter/DROGON_w_seis/`.

The experiments run OPM Flow on the Drogon model, the open-petro-elastic PEM
through SimulatorWrap, PET's ES-MDA and multilevel hybrid ES-MDA, and an
experiment-local SMLES scheme. Observation data are frozen inputs; no
observation-generation step is part of setup or the run scripts.

## Experiments

| Method | Scheme / analysis | Ensemble allocation | Deck |
|---|---|---:|---|
| ES-MDA | `esmda` / `approx`, 4 equal-weight steps | 100 | `DROGON.mako` |
| ES-MDA + adaptive localization | `esmda` / `approx`, 4 steps | 100 | `DROGON.mako` |
| MLHES-MDA | `esmda` / `hybrid`, 4 steps | `[1973, 191, 50, 30]` | `DROGON_ML.mako` |
| SMLES | local `smles` / `approx`, sequential levels | `[4933, 953, 502, 100]` | `DROGON_ML.mako` |

The four fidelity levels have 300, 9,000, 17,500 and 104,098 cells (300,
8,999, 17,487 and 73,496 active in the grid files). The data are the oil and
water rates of producers A1–A4 and three 4D acoustic-impedance vintages.

## Requirements

- Linux x86-64 with Python 3.12
- OPM Flow **2025.10** (`flow`) first on `PATH`, the version of the original
  runs; `run_experiment.py` stops if it finds another version. The Ubuntu
  packages of 2025.10 are no longer in the OPM PPA; the official container
  image `docker.io/openporousmedia/opmreleases:2025.10` contains it. Run on
  the same Drogon deck, that build reproduces the summary of an original
  member exactly.
- An MPI runtime. The number of MPI ranks changes Flow's time stepping and
  hence the forecasts (by up to about 1% in the well rates), so the configs
  keep the original settings: ES-MDA runs Flow without MPI, the other
  methods use `mpirun --bind-to none -np 2`.
- Memory and disk for the priors, which are float64 like the originals:
  0.4 GB (`prior_100`), 9.3 GB (`prior_2244`) and 27 GB (`prior_6488`)
- A C/C++ toolchain if `resdata` has to build from source

`pyproject.toml` pins PET to the public commit this package was written
against and SimulatorWrap to `df94874` (branch `resdata_reader`), the
forward-model version of the original multilevel run (see below).

## Setup and run

```bash
./setup.sh
```

Setup creates `.venv`, installs the package, builds `Levels/`, generates the
seeded priors in `priors/`, and runs ES-MDA (set `EXPERIMENT=MLHES-MDA` or
another method name to choose the final run). Then:

```bash
source .venv/bin/activate
python scripts/run_experiment.py ES-MDA-loc
./run_all.sh            # all methods, skipping finished ones, then plots
```

`DROGON_PARALLEL` sets the number of concurrent members and `DROGON_MPI` the
MPI command passed to Flow. Results are written to `<method>/Results/` and
marked with the forward-model version; results of other versions are not
overwritten or mixed.

## Fidelity levels

`scripts/build_levels.py` assembles `Levels/` (generated, not tracked):

- **Fine level**: the public Drogon grid, faults, transmissibility edits,
  schedule and region files from `include/`.
- **Coarse levels**: the coarse geometries, region numbers, faults,
  transmissibility edits, well schedules and volume-average maps
  (`TransformMatMean.npz`) are outputs of NORCE's legacy upscaler, which is
  not public. They are shipped unchanged in `data/levels/`.
- **Facies probabilities** (`probfaction.npz`) of all levels are derived
  from the reference SATNUM: 0.8 for a cell's own SATNUM, 0.05 for each other
  SATNUM of its truncation group, and -1 in the 33,126 cells with zero
  reference porosity; coarse levels take the volume average.

All files in `Levels/` are byte-identical to the levels of the original runs:

```bash
.venv/bin/python scripts/validate_levels.py --original /path/to/DROGON_w_seis/Levels
```

### Why not upscaling_v2?

The chapter describes the coarse grids as linear interpolations of the fine
geometry. The legacy tool distributed the coarse pillars uniformly at
*fractional* fine-pillar positions (every 4.6 fine pillars for the 300-cell
level) and placed faults with heuristics. Its volume map counted a fine cell
only if its *centre* lies inside the coarse cell's bounding box, weighted by
the intersection volume of the two cells' convex hulls, so inactive and
collapsed fine cells also enter. upscaling_v2's fault-aware hybrid grids use
integer fine-cell boundaries and exact tetrahedral overlaps; they give
different grids and averaging matrices and hence different forecasts.

`drogon_exp.levels.legacy_volume_map` reimplements the legacy map
(`build_levels.py --recompute`, needs the `recompute` extra). Computed from
the shipped grids, it matches the original maps to a median row difference of
1e-3, with up to 0.26 in about 0.1% of coarse cells: the original maps were
computed from the unrounded coarse geometry, which was written to
`Grid.grdecl` with two decimals, and a perturbation of that size reproduces
the differences. The book runs used the original maps, so they are shipped.
The legacy region upscaling cannot be recomputed either (about 1% of coarse
cells differ); only EQLNUM and PVTNUM affect the flow, since the MULTREGT
multipliers are 1.

## Forward model

The forward model follows the original multilevel (MLHES) run, the one
original run whose code can be pinned:

- **Decks.** The single-level runs used `DROGON.mako`, the multilevel runs
  `DROGON_ML.mako`, both copied from the original runs. The multilevel deck
  upscales the state with `TransformMatMean`. Its facies list
  `[1, 2, 3, 4, 7, …, 12]` is combined with `index + 1`, so facies 7–12 are
  written as SATNUM 5–10 on every level; the single-level deck keeps the
  reference numbering. Both are kept as run.
- **4D seismic.** SimulatorWrap `df94874` computes the change in acoustic
  impedance (vintage minus baseline) on the full grid of the simulated level,
  sets inactive cells to 0, and scales each vintage linearly to [-1, 1].
  NaN PEM values are replaced by the mean.
- **Coarse to fine.** Coarse fields are mapped to the fine grid with the
  transpose of the level's volume-average map, as the original PET option
  `interpolate`/`interpolate_map` did, and are not rescaled.
- **Bias correction.** PET shifts each coarse level's predicted mean to the
  fine level's (the original `ml_error_corr: bias_corr, ens`).
- **Observations.** `data/observations` holds the normalized observations of
  the original multilevel run: relative AI resized to the grid by first-order
  spline interpolation, scaled to [-1, 1] and reversed along J. Predictions
  are not reversed.
- **Priors.** `make_prior.py` reproduces the original priors to < 2e-6 (FFT
  rounding; the fault multipliers are exact): `prior_100` draws one
  realization at a time as the single-level runs did, `prior_2244` and
  `prior_6488` draw all realizations of a field at once as the multilevel
  runs did. The two generators give different members for the same seed.

## Validation against the original runs

Checked on Jupiter2 against the original working directory (October 2026):

| Item | Result |
|---|---|
| `Levels/` (grids, maps, regions, faults, schedules, facies) | Byte-identical |
| Priors (`make_prior.py`) | ≤ 2e-6 from the original files; the original environment itself now gives the same values, so the stored priors came from an environment state that no longer exists |
| Flow 2025.10 container build vs. original Ubuntu package, same deck and ranks | Identical summary |
| Original MLHES prior members rerun through this forward model (two per level) | Well rates identical; AI change ≤ 1.3e-6 (fine) and ≤ 1e-15 (coarse, after the bias shift) |
| Original ES-MDA prior members 0 and 1, rendered with `DROGON.mako` | Member 0 identical; member 1 differs by up to 1.7% in water rates although reruns here are bit-identical |
| upscaling_v2 exact overlap on the archived 300-cell grid | 31% (relative Frobenius norm) from the original map; 51,245 of its 155,580 entries absent |

Full assimilation runs are therefore reproduced statistically, not bit for
bit: Flow's adaptive time stepping turns the 1e-7 prior differences into
small differences in the forecasts.

## Differences from the original runs

- **The original runs did not share one forward model.** The SimulatorWrap
  seismic code changed several times in November 2025. The surviving
  single-level ES-MDA run (2 Nov) used `b31a699`, which scaled predictions
  to [0, 1] while the observations are in [-1, 1]; a localized ES run
  (`ES-loc-SIM2SEIS`) used `db72f12`, whose swapped min/max made the predicted
  AI almost constant. The multilevel run used `df94874` (its fine-level
  predictions span [-1, 1] and correlate negatively with the observations, as
  vintage-minus-baseline does). `d8c4734`, which reverses the sign "to match
  data polarity", was committed after that run's prior forecast. This package
  uses `df94874` for all four methods.
- **Unknown original runs.** The run behind the chapter's localized ES-MDA
  figures and the SMLES run were not found in the original working
  directory. Localization uses the settings of the surviving localized run
  (cutoff 0.75, sigmoid taper).
- **PET.** The original runs used `KriFos1/PET` (`load_dataframe`,
  `d478e32`); this package runs current public PET with the same scheme
  settings.
- **Active cells.** The chapter reports 70,972 active fine cells (those with
  non-zero reference porosity); the grid's ACTNUM has 73,496.

## Plots

```bash
.venv/bin/python scripts/make_plots.py
```

Figures are written to `plots/`: prior/posterior well-rate traces, the
final-vintage AI residual mean and predictive standard deviation, and the
posterior porosity mean and standard deviation on the chapter's `i=18` slice.
As in the chapter, residuals in `[-0.4, 0.4]` and standard deviations `<=0.1`
are masked. The script only reads observations and saved results.

## Repository layout

- `data/observations/`: frozen observations and variances.
- `data/levels/`: legacy-upscaler outputs for the three coarse levels.
- `data/pem_config_drogon.yml`: open-petro-elastic configuration.
- `include/`: Drogon simulator inputs (Equinor, ODbL/DbCL; see the headers).
- `scripts/build_levels.py`, `scripts/validate_levels.py`: fidelity levels.
- `scripts/make_prior.py`: seeded prior ensembles.
- `scripts/run_experiment.py`, `run_all.sh`: experiments.
- `scripts/DROGON.mako`, `scripts/DROGON_ML.mako`: simulation decks.
- `src/drogon_exp/`: simulator adapter, level helpers, memory-bounded
  adaptive localization, and the SMLES scheme.
- `<method>/config.yaml`: PET configurations.
