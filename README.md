# Drogon seismic-assisted history-matching experiments

Reproduction code for the realistic example (Section 5.1) of *Subsurface Data
Assimilation: Theory and Applications*.

The four runs behind the chapter's Drogon figures were made on the NORCE HPC
in November 2025 (`/cluster/work/users/krfo/DROGON/`). Their well-rate
figures match the chapter pixel for pixel. This repository reruns them with
the same configuration, code versions, Python environment and simulator
build:

| Chapter figures | Method | Original run | Ensemble sizes | Flow MPI ranks |
|---|---|---|---|---:|
| 20–22 | `ES-MDA` | `ES-MDA-SIM2SEIS` | 100 | 2 |
| 23–25 | `ES-MDA-loc` | `ES-MDA-loc-SIM2SEIS` | 100 | 1 |
| 26–28 | `MLHES-MDA` | `ML_ES-MDA-SIM2SEIS` | `[1973, 191, 50, 30]` | 2 |
| 29–31 | `SMLES` | `ML-SEQ_ES-MDA-SIM2SEIS` | `[4933, 953, 502, 100]` | 1 |

Each method folder holds the run's original `drogon.toml` (only the PEM
config path changed, and unused restart/debug dumps switched off) and its
original plotting scripts. Observation data are frozen inputs; no
observation-generation step is part of setup or the run scripts.

## Requirements

- Linux x86-64 and **Python 3.10** (the runs used 3.10.4; for example
  `uv python install 3.10.4`).
- **OPM Flow 2024.10, the NORCE HPC build** (EasyBuild
  `opm-simulators/2024.10-foss-2022a`) with its OpenMPI 4.1.4, as `flow` and
  `mpirun` first on `PATH`. `run_experiment.py` refuses other Flow versions.
  Other 2024.10 builds (for example the Ubuntu package) reproduce the 300- and
  9,000-cell levels exactly but change 17,500-cell and fine-grid forecasts by
  up to about 2%. A relocatable copy of the HPC build (binary, libraries,
  OpenMPI, FlexiBLAS/OpenBLAS backend; `bin/flow` and `bin/mpirun` wrappers)
  runs on other Linux machines with a newer glibc; `scripts/flow/make_hpc_bundle.sh`
  creates it on the HPC. Put its `bin` first on `PATH`.
- Disk: 0.4 GB (`prior_100`), 9.3 GB (`prior_2244`) and 20 GB (`prior_4933`)
  for the priors; run outputs take about 4 GB each for ES-MDA and
  ES-MDA-loc, 26 GB for MLHES-MDA and 40 GB for SMLES.
- Memory: the original multilevel jobs reserved 400 GB.

`requirements.txt` pins the Python environment of the original runs:
PET (`KriFos1/PET` `d478e32`, installed separately by `setup.sh`),
SimulatorWrap `d8c4734`, numpy 1.26.4, scipy 1.14.1 and every other package.

## Setup and run

```bash
./setup.sh                           # PYTHON=/path/to/python3.10 if needed
source .venv/bin/activate
python scripts/run_experiment.py ES-MDA-loc
./run_all.sh                         # all methods, skipping finished ones, then figures
```

Setup creates `.venv`, installs the pinned environment, builds `Levels/`,
writes the seeded priors to `priors/`, and runs ES-MDA (set
`EXPERIMENT=SMLES` or another method to choose). `run_experiment.py` links the
observations, the prior, the facies probabilities and the deck into the method
folder and runs the statements of the original `run_script.py`. Results stay
in the method folder, as in the original runs. `DROGON_PARALLEL` overrides the
number of simultaneous Flow runs (the originals used 100–250).

`scripts/make_figures.py` runs each method's original `plot_data.py` and
`plot_parameters.py`: well-rate PNGs, and GRDECL fields of the posterior AI
residual/standard deviation and porosity/permeability mean/standard deviation,
from which the chapter's 3D panels were drawn in ResInsight.

## Bit-for-bit reproduction

Checked against the original runs on an Intel Xeon Gold 6254 workstation
(October 2026):

| Item | Result |
|---|---|
| `Levels/` (grids, maps, regions, faults, schedules, facies) | Byte-identical |
| Observations, include files, PEM config, decks | Byte-identical |
| `prior_100`, `prior_2244` from `make_prior.py` | Identical (`prior_100` to ≤ 2e-15) |
| `prior_4933` from `make_prior.py` | Differs by ~1e-7 (rounding level); its exact generation environment is unknown |
| Original prior members rerun with the HPC Flow build, two per level and method | Well rates and AI identical (≤ 7e-7 relative; mostly 0) |
| Figures and GRDECL fields from the original ES-MDA outputs via `make_figures.py` | Pixel- and byte-identical to the chapter and the run |

Numerical libraries choose CPU-specific kernels, and the last digits matter
for long assimilation chains. The runs were made on AMD EPYC (Zen 3) CPUs, so
`run_experiment.py` selects the same OpenBLAS kernels and disables numpy's
AVX-512 loops (`OPENBLAS_CORETYPE=ZEN`, `NPY_DISABLE_CPU_FEATURES`; harmless
on AMD), and `make_prior.py` uses for each prior the kernels of the machine
that generated the original.

Flow occasionally fails to start under heavy load; PET then replaces the
member by a randomly chosen successful one and logs "has been replaced" in
`pet_logger.log`. The original MLHES and SMLES runs replaced 15 and 25 prior
members this way, so full reruns match them statistically rather than member
for member.

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

`scripts/validate_levels.py --original <original Levels>` compares a build
with the original levels.

### Why not upscaling_v2?

The chapter describes the coarse grids as linear interpolations of the fine
geometry. The legacy tool distributed the coarse pillars uniformly at
*fractional* fine-pillar positions (every 4.6 fine pillars for the 300-cell
level) and placed faults with heuristics. Its volume map counted a fine cell
only if its *centre* lies inside the coarse cell's bounding box, weighted by
the intersection volume of the two cells' convex hulls, so inactive and
collapsed fine cells also enter. upscaling_v2's fault-aware hybrid grids use
integer fine-cell boundaries and exact tetrahedral overlaps; on the 300-cell
grid its map differs from the original by 31%.

`scripts/drogon_levels.py` reimplements the legacy map
(`build_levels.py --recompute`; needs `trimesh==4.8.3 manifold3d==3.2.1`).
Computed from the shipped grids it matches the original maps to a median row
difference of 1e-3, with up to 0.26 in about 0.1% of coarse cells: the
original maps were computed from the unrounded coarse geometry, which was
written to `Grid.grdecl` with two decimals. The legacy region upscaling cannot
be recomputed either (about 1% of coarse cells differ).

## Notes on the chapter text

- The 4D AI change is baseline minus vintage on the full grid of each level
  (inactive cells 0), scaled to [-1, 1]; coarse fields are mapped to the fine
  grid with the transpose of the averaging map (PET `interpolate`).
- The observations are resized to the grid, scaled to [-1, 1] and reversed
  along J; predictions are not reversed.
- MLHES corrects each coarse level's predicted mean to the fine level's
  (`ml_error_corr: bias_corr`); SMLES does not.
- SMLES (PET `esmda_seq`) updates level by level with 3 ES-MDA steps; after
  each update it keeps the members with the lowest pre-update data misfit and
  forecasts them on the next level. The chapter states equal covariance
  weights; the run used `cov_wgt = [1, 1, 1, 1]`.
- The multilevel deck combines the facies list `[1, 2, 3, 4, 7, …, 12]` with
  `index + 1`, so facies 7–12 are written as SATNUM 5–10 on every level; the
  single-level deck keeps the reference numbering. Both are kept as run.
- The chapter reports 70,972 active fine cells (non-zero reference porosity);
  the grid's ACTNUM has 73,496.

## Repository layout

- `<method>/`: `drogon.toml`, `plot_data.py`, `plot_parameters.py` of the
  original run; outputs are written here.
- `data/observations/`: frozen observations and variances of the runs.
- `data/levels/`: legacy-upscaler outputs for the three coarse levels.
- `data/pem_config_drogon.yml`: open-petro-elastic configuration.
- `include/`: Drogon simulator inputs (Equinor, ODbL/DbCL; see the headers).
- `scripts/`: level build and validation, prior generation, the two decks,
  the experiment runner and the figure script; `scripts/flow/` makes the
  relocatable copy of the HPC Flow build.
- `requirements.txt`, `setup.sh`, `run_all.sh`.
