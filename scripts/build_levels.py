"""Build the four Drogon fidelity levels in ``Levels/`` from shipped inputs.

Fine level (104,098 cells): the public Drogon grid, faults, transmissibility
edits, schedule and region files from ``include/``.

Coarse levels (300, 9,000 and 17,500 cells): the geometry, region numbers,
fault, transmissibility and schedule files written by NORCE's legacy upscaler
are shipped in ``data/levels``, together with the volume-average maps
(``TransformMatMean.npz``) of the book runs. This script derives the facies
probabilities of every level from the reference SATNUM.

``--recompute`` instead recomputes the maps with the legacy overlap rule
(``drogon_exp.levels.legacy_volume_map``). The shipped grids are rounded to
two decimals, so the recomputed maps differ slightly from those of the book
runs (see README). The overlap computation runs one single-threaded worker per
CPU (``--jobs`` limits this) and takes a few minutes on a many-core server.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from scipy import sparse

from drogon_exp.levels import (
    facies_probabilities,
    legacy_fine_geometry,
    legacy_volume_map,
    read_geometry,
    read_grdecl_keywords,
)

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ROOT / "include"
SHIPPED = ROOT / "data" / "levels"
FINE = 104098
# Total cells: (dimensions, active cells)
COARSE = {300: ((10, 15, 2), 300), 9000: ((20, 30, 15), 8999), 17500: ((25, 35, 20), 17487)}
FINE_FILES = {
    "Grid.grdecl": INCLUDE / "grid" / "drogon.grid",
    "FAULT.INC": INCLUDE / "grid" / "drogon.faults",
    "drogon_US.trans": INCLUDE / "edit" / "drogon.trans",
    "US_schdl.sch": INCLUDE / "schedule" / "drogon_hist.sch",
    "EQLNUM.grdecl": INCLUDE / "regions" / "drogon.eqlnum",
    "FIPNUM.grdecl": INCLUDE / "regions" / "drogon.fipnum",
    "FIPZON.grdecl": INCLUDE / "regions" / "drogon.fipzon",
    "PVTNUM.grdecl": INCLUDE / "regions" / "drogon.pvtnum",
    "MULTNUM.grdecl": INCLUDE / "grid" / "drogon.multnum",
}
COARSE_FILES = (
    "Grid.grdecl", "FAULT.INC", "drogon_US.trans", "US_schdl.sch",
    "EQLNUM.grdecl", "FIPNUM.grdecl", "FIPZON.grdecl", "MULTNUM.grdecl", "PVTNUM.grdecl",
)
# The legacy coarse files hold the probabilities of the ten SATNUM values in use.
COARSE_FACIES = tuple(f"p{value}" for value in (1, 2, 3, 4, 7, 8, 9, 10, 11, 12))


def grdecl_float32(values: np.ndarray) -> np.ndarray:
    """Round like the legacy tool chain: float32 written to GRDECL with eight
    significant digits, then read back as float32."""
    single = np.asarray(values, dtype=np.float32).astype(np.float64)
    return np.array([float(f"{value:.7E}") for value in single], dtype=np.float32)


def fine_facies() -> dict[str, np.ndarray]:
    satnum = read_grdecl_keywords(INCLUDE / "regions" / "drogon.satnum", ("SATNUM",))["SATNUM"]
    porosity = read_grdecl_keywords(INCLUDE / "grid" / "drogon.poro", ("PORO",))["PORO"]
    # PORO > 0 marks the 70,972 cells that the reference model keeps active.
    return facies_probabilities(satnum, porosity > 0)


def build_fine(output: Path, facies: dict[str, np.ndarray]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, source in FINE_FILES.items():
        shutil.copy2(source, output / name)
    np.savez(output / "probfaction.npz", **facies)
    sparse.save_npz(output / "TransformMatMean.npz", sparse.identity(FINE, format="csr"))


def build_coarse(size: int, output: Path, fine: dict | None, facies: dict[str, np.ndarray], jobs: int | None) -> None:
    dimensions, active = COARSE[size]
    output.mkdir(parents=True, exist_ok=True)
    for name in COARSE_FILES:
        shutil.copy2(SHIPPED / f"Level{size}" / name, output / name)
    coarse = read_geometry(output / "Grid.grdecl")
    if coarse["dims"] != dimensions:
        raise ValueError(f"Level{size} has dimensions {coarse['dims']}, expected {dimensions}")
    actnum = read_grdecl_keywords(output / "Grid.grdecl", ("ACTNUM",))["ACTNUM"]
    if int(np.count_nonzero(actnum)) != active:
        raise ValueError(f"Level{size} has {np.count_nonzero(actnum)} active cells, expected {active}")

    if fine is None:
        shutil.copy2(SHIPPED / f"Level{size}" / "TransformMatMean.npz", output / "TransformMatMean.npz")
        mean = sparse.load_npz(output / "TransformMatMean.npz").tocsr()
    else:
        print(f"Level{size}: computing legacy volume map", flush=True)
        volume, mean, coarse_volume = legacy_volume_map(fine, coarse, n_jobs=jobs)
        sparse.save_npz(output / "TransformMatVolume.npz", volume)
        sparse.save_npz(output / "TransformMatMean.npz", mean)
        np.savez(output / "CoarseCellVolume.npz", v_coarse=coarse_volume)
    if mean.shape != (size, FINE):
        raise ValueError(f"Level{size} map has shape {mean.shape}")
    np.savez(output / "probfaction.npz", **{name: grdecl_float32(mean @ facies[name]) for name in COARSE_FACIES})
    print(f"Level{size}: {mean.nnz} map entries", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "Levels")
    parser.add_argument("--jobs", type=int, default=None, help="worker processes (default: all CPUs)")
    parser.add_argument("--force", action="store_true", help="rebuild levels that already exist")
    parser.add_argument("--recompute", action="store_true",
                        help="recompute the volume maps instead of using those of the book runs")
    args = parser.parse_args()

    facies = fine_facies()
    fine_dir = args.output / f"Level{FINE}"
    if args.force or not (fine_dir / "probfaction.npz").is_file():
        build_fine(fine_dir, facies)

    fine = None
    for size in COARSE:
        folder = args.output / f"Level{size}"
        if not args.force and (folder / "probfaction.npz").is_file():
            print(f"Keeping existing {folder}")
            continue
        if args.recompute and fine is None:
            fine = legacy_fine_geometry(INCLUDE / "grid" / "drogon.grid")
        build_coarse(size, folder, fine, facies, args.jobs)
    print(f"Drogon levels ready in {args.output}")


if __name__ == "__main__":
    main()
