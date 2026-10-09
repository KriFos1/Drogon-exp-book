# Drogon seismic-assisted history-matching experiments

Reproducibility package for Section 5.1 of *Subsurface Data Assimilation:
Theory and Applications*. The source experiment directory was
`Jupiter2:~/bookchapter/DROGON_w_seis/`.

The package uses the latest public PET multilevel ES-MDA implementation,
registers an experiment-local SMLES scheme, and runs the archived
open-petro-elastic forward model. The grids, volume-weighted upscaling maps,
facies probabilities, fault/transmissibility inputs, and well schedules in
`Levels/` are archived from the Jupiter2 experiment and are used directly by
all four methods. Setup does not regenerate the model levels. Observation data
are frozen inputs copied from the last Jupiter2 ML run.

## Experiments

| Method | Scheme / analysis | Ensemble allocation |
|---|---|---:|
| ES-MDA | `esmda` / `approx`, 4 equal-weight steps | 100 |
| ES-MDA + adaptive localization | `esmda` / `approx`, 4 steps | 100 |
| MLHES-MDA | `esmda` / `hybrid`, 4 steps | `[1973, 191, 50, 30]` |
| SMLES | local `smles` / `approx`, sequential levels | `[4933, 953, 502, 100]` |

The fidelity grids have 300, 9,000, 17,500, and 104,098 total cells. The
full-grid data vector includes the eight producer rate series and three
4D acoustic-impedance vintages. The public data in `data/observations/` uses the
latest Jupiter2 normalization of the seismic observations to `[-1, 1]` for all
four methods.

The localized 520k-parameter state cannot use PET's dense `numpy.corrcoef(X,
Y)` temporary, which would allocate a square state-space matrix. The
experiment-local `drogon_exp.localization` registers a blockwise implementation
of the same cross-correlation calculation and taper, keeping localization
memory-bounded while retaining PET's adaptive-localization behavior.

## Requirements

- Python 3.12
- OPM Flow (`flow`) installed separately as a system dependency and available
  on `PATH`. `setup.sh` and `pyproject.toml` do **not** install OPM Flow; install
  it using the supported system/package-manager process for your host, then
  verify it with `flow --version` before running the experiments.
- MPI runtime for the configured OPM Flow runs
- Enough memory and disk for large prior ensembles and result artifacts
- Roughly 32 GB RAM and 40 GB free disk for the full 6,488-member SMLES prior
  and saved forecasts
- `resdata` native dependencies if a wheel is unavailable

## Setup and run

```bash
./setup.sh
```

Setup creates `.venv`, installs the dependencies pinned in `pyproject.toml`,
verifies the archived level inputs, generates the 100-, 2,244-, and 6,488-member
prior ensembles, and runs ES-MDA by default.
Set `EXPERIMENT=MLHES-MDA` or another method name before running setup to choose
the final setup run.

The largest prior is memory- and disk-intensive; run the full setup on a
workstation/HPC node with the resources noted above. To check the archived
inputs without generating priors or running Flow, run
`.venv/bin/python scripts/check_archived_levels.py`.

Run one experiment:

```bash
source .venv/bin/activate
python scripts/run_experiment.py ES-MDA-loc
```

Run every experiment, skipping methods that already have a posterior state:

```bash
./run_all.sh
```

Completed results are marked with their level source. Runs made with other grids
cannot be reused: move their `Results/` directories aside before running the
archived models. Existing results are not overwritten or silently mixed.

On constrained machines set `DROGON_PARALLEL` to an appropriate number of
member processes. `DROGON_MPI` controls the MPI command passed to Flow; the
default is `mpirun --bind-to none -np 2`.

## Plots

```bash
.venv/bin/python scripts/make_plots.py
```

Figures are written to `plots/` and include prior/posterior well-rate traces,
final-vintage acoustic-impedance residual mean and predictive standard
deviation, and posterior porosity mean and standard deviation on the chapter's
`i=18` slice. Residual values in `[-0.4, 0.4]` and standard deviations `<=0.1`
are masked to match the chapter figures. The script only reads observations and
saved results.

## Data and model provenance

- `data/observations/` contains the frozen `data.pkl`, `var.pkl`, report dates,
  data-type list, and assimilation index from the latest Jupiter2 ML run.
- `include/` contains the fixed Drogon simulator inputs.
- `Levels/` contains the archived Jupiter2 grid, fault, transmissibility,
  schedule, facies, and normalized volume-overlap inputs. Each coarse level's
  `TransformMatMean.npz` is used for member-state upscaling and seismic
  downscaling; archived fault and transmissibility includes stay paired with
  their own grids. The fine level uses the existing region inputs in `include/`.
- `Levels/Level104098/Grid.grdecl` is the archived 46x73x31 fine-grid geometry.
- `data/pem_config_drogon.yml` configures the archived open-petro-elastic PEM.
- The SEG-Y source files and the observation-generation script are deliberately
  not part of setup; observation data is not regenerated.
- The Equinor model input files retain their ODBL/DBCL attribution headers;
  see those headers for their data licensing terms.

The book describes a fixed structural model, the simplified computational-grid
sim2seis workflow, and six scalar fault-transmissibility multipliers. The
Mako deck keeps the six `MULTFLT` parameters and clips state variables to the
published bounds.

The chapter reports 70,972 active fine cells. The archived Jupiter2
`Levels/Level104098/Grid.grdecl` contains 73,496 active cells in its `ACTNUM`
keyword; the archived inputs follow the recorded run files.
