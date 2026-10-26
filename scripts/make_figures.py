"""Make the chapter figures with the original plotting scripts of each run.

In every finished method folder this runs ``plot_data.py`` (well rates; its
``plot_sim2seis`` writes the acoustic-impedance mean/residual/std fields) and
``plot_parameters.py`` (posterior porosity/permeability fields). PNGs and
GRDECL fields are written to ``<method>/Figures`` (SMLES writes its parameter
fields to the method folder, as the original script did). The chapter's 3D
panels were drawn from these GRDECL files in ResInsight.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import yaml

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
FINISHED = {"ES-MDA": "posterior_forecast.npz", "ES-MDA-loc": "posterior_forecast.npz",
            "MLHES-MDA": "posterior_forecast.p", "SMLES": "posterior_forecast.p"}


def link(target: Path, name: Path) -> None:
    if name.is_symlink() or name.exists():
        name.unlink()
    name.symlink_to(os.path.relpath(target, name.parent))


def stage(folder: Path) -> None:
    """Link the report dates, assimilation indices and prior, which the scripts read from the run folder."""
    for name in ("reportdates.csv", "assim_index.csv"):
        link(ROOT / "data" / "observations" / name, folder / name)
    prior = (folder / yaml.safe_load((folder / "config.yaml").read_text())["ensemble"]["importstate"]).resolve()
    link(prior, folder / prior.name)


def main() -> None:
    for name, marker in FINISHED.items():
        folder = ROOT / name
        if not (folder / marker).is_file():
            print(f"Skipping {name}: no {marker}")
            continue
        stage(folder)
        os.chdir(folder)
        namespace = runpy.run_path("plot_data.py")  # plots the well rates
        namespace["plot_sim2seis"]([31, 73, 46], 1, 27)
        runpy.run_path("plot_parameters.py")
        print(f"{name}: figures in {folder / 'Figures'}")


if __name__ == "__main__":
    main()
