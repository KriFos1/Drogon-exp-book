# Drogon seismic-assisted history-matching experiments

Reproduction code for the realistic example (Section 5.1) of *Subsurface Data
Assimilation: Theory and Applications*.

The four runs behind the chapter's Drogon figures were made on the NORCE HPC
in November 2025 (`/cluster/work/users/krfo/DROGON/`). Their well-rate
figures match the chapter pixel for pixel. This repository reruns them with
their configuration, simulator build and numerics on a current, public Python
stack:

| Chapter figures | Method | Original run | Ensemble sizes | Flow MPI ranks |
|---|---|---|---|---:|
| 20–22 | `ES-MDA` | `ES-MDA-SIM2SEIS` | 100 | 2 |
| 23–25 | `ES-MDA-loc` | `ES-MDA-loc-SIM2SEIS` | 100 | 1 |
| 26–28 | `MLHES-MDA` | `ML_ES-MDA-SIM2SEIS` | `[1973, 191, 50, 30]` | 2 |
| 29–31 | `SMLES` | `ML-SEQ_ES-MDA-SIM2SEIS` | `[4933, 953, 502, 100]` | 1 |

Each method folder holds the run's configuration (`config.yaml`, the original
`drogon.toml` in current PET syntax) and its original plotting scripts.
Observation data are frozen inputs; no observation-generation step is part of
setup or the run scripts.

## Software

- **PET**, public `Python-Ensemble-Toolbox/PET` `86d7893`. The book runs used
  a development branch (`KriFos1/PET` `d478e32`) that is not public.
  `src/drogon_exp/legacy.py` adds its analyses to current PET through PET's
  scheme registry (`analysis: legacy`): the approximate ES-MDA update with
  its singular-value truncation and adaptive localization, the multilevel
  hybrid update, and the sequential multilevel scheme (SMLES). Random numbers
  are drawn in the original order. The schemes write the original result
  files, so the original plotting scripts run unchanged.
- **SimulatorWrap** `d8c4734` (the commit of the book runs) with
  open-petro-elastic 1.4.1. `src/drogon_exp/simulator.py` selects the
  fidelity level and maps coarse acoustic-impedance changes to the fine grid,
  as the original runs did.
- **Python 3.12** with numpy 2.5.4, scipy 1.18.1, matplotlib 3.10.7 (the
  book runs' version, for pixel-identical PNGs) and every other package
  pinned in `requirements.txt`.
- **Priors** are generated with numpy 1.26.4 (`requirements-prior.txt`, in
  `.venv-prior`). The prior fields are Cholesky factors of nearly singular
  covariance matrices applied to random numbers; numpy 2.5.4's OpenBLAS
  changes them by up to 2e-6, which reaches the simulator decks.
- **OPM Flow 2024.10, the NORCE HPC build** (EasyBuild
  `opm-simulators/2024.10-foss-2022a`) with its OpenMPI 4.1.4, as `flow` and
  `mpirun` first on `PATH`. `run_experiment.py` refuses other Flow versions.
  Other 2024.10 builds (for example the Ubuntu package) changed 17,500-cell
  and fine-grid forecasts of test members by up to about 2%. A relocatable copy of the HPC build (binary, libraries,
  OpenMPI, FlexiBLAS/OpenBLAS backend; `bin/flow` and `bin/mpirun` wrappers)
  runs on other Linux machines with a newer glibc; `scripts/flow/make_hpc_bundle.sh`
  creates it on the HPC. Put its `bin` first on `PATH`. Flow's results also
  depend on the CPU: the runs used AMD EPYC 7713 (Zen 3) processors, and on
  an Intel Xeon the HPC build reproduces only about 70% of the members bit for
  bit (see below).

## Requirements

- Linux x86-64, Python 3.12 and Flow as above.
- Disk: 0.4 GB (`prior_100`), 9.3 GB (`prior_2244`) and 20 GB (`prior_4933`)
  for the priors. Run outputs take about 4 GB each for ES-MDA and
  ES-MDA-loc, 21 GB for MLHES-MDA and 40 GB for SMLES.
- Memory: the original multilevel jobs reserved 400 GB.

## Setup and run

```bash
./setup.sh                           # PYTHON=/path/to/python3.12 if needed
source .venv/bin/activate
python scripts/run_experiment.py ES-MDA-loc
./run_all.sh                         # all methods, skipping finished ones, then figures
```

Setup creates `.venv` and `.venv-prior`, installs the pinned environments,
builds `Levels/`, writes the seeded priors to `priors/`, and runs ES-MDA (set
`EXPERIMENT=SMLES` or another method to choose). `run_experiment.py` copies
the deck into the method folder and runs the assimilation there. The results
are written in the files and formats of the original run: `obs_var.npz`,
`prior_forecast.npz`, `debug_analysis_step_<k>.npz` (ES-MDA and ES-MDA-loc),
and the posterior state and forecast (`.npz` for the single-level runs,
pickles `.p` for the multilevel runs). The log is `ASSIM.log`.
`DROGON_PARALLEL` overrides the number of simultaneous Flow runs (the
originals used 100–250); it does not change the results.

`scripts/make_figures.py` runs each method's original `plot_data.py` and
`plot_parameters.py`: well-rate PNGs, and GRDECL fields of the posterior AI
residual/standard deviation and porosity/permeability mean/standard deviation,
from which the chapter's 3D panels were drawn in ResInsight.

## Reproduction status

Checked against the original runs on an Intel Xeon Gold 6254 workstation
(October 2026):

| Item | Result |
|---|---|
| `Levels/` (grids, maps, regions, faults, schedules, facies) | Byte-identical |
| Observations, include files, PEM config, decks | Byte-identical |
| Priors from `make_prior.py` | `prior_2244` identical, `prior_100` to 2e-15; `prior_4933` differs by up to 2e-6 (its generation environment is unknown) |
| Each method's assimilation replayed with Flow replaced by the original run's saved predictions | Updated ensembles within 1e-11 (ES-MDA), 9e-12 (ES-MDA-loc), 6e-14 (MLHES) and 7e-12 (SMLES) of the original at every step; data misfits within 4e-15 (relative); result files identical in structure |
| Figures from the original outputs and the SMLES replay via `make_figures.py` | PNGs pixel-identical; GRDECL fields byte-identical, except 13 cells of the SMLES posterior permeability fields (last printed digit, from the replay's 1e-12 differences) |
| Prior forecasts with the HPC Flow build on the Xeon (full ES-MDA and ES-MDA-loc runs, 100 members each) | 71% (2 MPI ranks) and 73% (1 rank) of the members identical; the others differ by up to 20% (oil rates) and 35% (water rates) |
| Full ES-MDA-loc rerun on the Xeon (4 h) | Posterior mean and standard deviation of porosity and permeability correlate 0.91–0.96 with the chapter's fields (standard deviation median ratio 0.99–1.00); posterior AI residual 0.91–0.95; final data misfit 2.7e11 (original 3.7e11) |
| Full ES-MDA rerun on the Xeon (3.5 h) | Posterior means correlate 0.63–0.66 with the chapter's fields, standard deviations 0.30–0.41; final data misfit 7.3e11 (original 1.6e12) |

The replay and comparison scripts used the original runs' files and are not
part of this repository. The analyses differ from the original only in the
last digits, from numpy 2.5.4: the book runs used numpy 1.26.4, which does not
support the rest of this stack.

The forward model is the limit. On the Xeon, Flow is deterministic and its
results do not depend on the glibc version (the HPC's glibc 2.28 gives the
same results), the number of OpenMP threads or the machine load. Yet about 30%
of the members differ from the HPC runs, with one or two MPI ranks: the
differences follow the CPU (AMD EPYC 7713 on the HPC). Tiny differences change
Flow's time stepping for these members, and the assimilation carries them
on. A full rerun on another CPU therefore reproduces the chapter's results
statistically. With localization the posterior is robust to this; plain
ES-MDA with 100 members is not, and its posterior differs markedly from the
chapter's. The 1e-11 differences of the analyses alter the printed simulator
input, so even on the HPC a rerun would not repeat the original runs bit for
bit after the first update.

Numerical libraries choose CPU-specific kernels, and the last digits matter
for long assimilation chains. The runs were made on AMD EPYC (Zen 3) CPUs, so
`run_experiment.py` selects the same OpenBLAS kernels and disables numpy's
AVX-512 loops (`OPENBLAS_CORETYPE=ZEN`, `NPY_DISABLE_CPU_FEATURES`; harmless
on AMD), and `make_prior.py` uses for each prior the kernels of the machine
that generated the original.

Flow occasionally fails to start under heavy load. The configurations retry
such a run once (`rerun: 1`, same result). The original runs had no retry:
PET replaced a failed member by a randomly chosen successful one and logged
"has been replaced". Their MLHES and SMLES runs replaced 15 and 25 prior
members this way, so full reruns of those match them statistically rather
than member for member.

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
- The original schemes analysed the prior as imported: members whose prior
  forecast failed were replaced by copies in the forecast state only.
- The multilevel deck combines the facies list `[1, 2, 3, 4, 7, …, 12]` with
  `index + 1`, so facies 7–12 are written as SATNUM 5–10 on every level; the
  single-level deck keeps the reference numbering. Both are kept as run.
- The chapter reports 70,972 active fine cells (non-zero reference porosity);
  the grid's ACTNUM has 73,496.

## Repository layout

- `<method>/`: `config.yaml`, `plot_data.py`, `plot_parameters.py` of the
  original run; outputs are written here.
- `src/drogon_exp/`: the original analyses and schemes on current PET
  (`legacy.py`) and the simulator adapter (`simulator.py`).
- `data/observations/`: frozen observations and variances of the runs.
- `data/levels/`: legacy-upscaler outputs for the three coarse levels.
- `data/pem_config_drogon.yml`: open-petro-elastic configuration.
- `include/`: Drogon simulator inputs (Equinor, ODbL/DbCL; see the headers).
- `scripts/`: level build and validation, prior generation, the two decks,
  the experiment runner and the figure script; `scripts/flow/` makes the
  relocatable copy of the HPC Flow build.
- `tests/`: checks of the level helpers and the original analyses.
- `pyproject.toml`, `requirements.txt`, `requirements-prior.txt`, `setup.sh`,
  `run_all.sh`.
