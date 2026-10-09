"""Verify the archived, mutually compatible Drogon level inputs before a run."""

from pathlib import Path

import numpy as np
from resdata.grid import Grid
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
LEVELS = {
    300: ((10, 15, 2), 300, 155580),
    9000: ((20, 30, 15), 8999, 441360),
    17500: ((25, 35, 20), 17487, 527592),
    104098: ((46, 73, 31), 73496, None),
}
REGIONS = ("MULTNUM", "EQLNUM", "FIPNUM", "FIPZON", "PVTNUM")
FACIES = tuple(f"p{number}" for number in (1, 2, 3, 4, 7, 8, 9, 10, 11, 12))


def check_levels() -> None:
    for size, (dimensions, active_count, overlap_count) in LEVELS.items():
        folder = ROOT / "Levels" / f"Level{size}"
        inputs = [folder / name for name in (
            "Grid.grdecl", "FAULT.INC", "drogon_US.trans", "US_schdl.sch", "probfaction.npz",
        )]
        if overlap_count is not None:
            inputs.extend(folder / f"{region}.grdecl" for region in REGIONS)
            inputs.append(folder / "TransformMatMean.npz")
        for path in inputs:
            if not path.is_file():
                raise FileNotFoundError(f"Missing archived Level{size} input: {path}")

        grid = Grid.load_from_grdecl(str(folder / "Grid.grdecl"))
        actual_dimensions = (grid.get_nx(), grid.get_ny(), grid.get_nz())
        if actual_dimensions != dimensions or grid.get_num_active() != active_count:
            raise ValueError(f"Level{size} grid dimensions or ACTNUM differ from the archived model")

        with np.load(folder / "probfaction.npz") as facies:
            for name in FACIES:
                if name not in facies.files or facies[name].size != size:
                    raise ValueError(f"Level{size} has an invalid facies field {name}")

        if overlap_count is not None:
            transform = sparse.load_npz(folder / "TransformMatMean.npz")
            if transform.shape != (size, 104098) or transform.nnz != overlap_count:
                raise ValueError(f"Level{size} has a non-archived upscaling map")

    for path in (ROOT / "include/grid/drogon.multnum", *(
        ROOT / "include/regions" / f"drogon.{region.lower()}" for region in REGIONS[1:]
    )):
        if not path.is_file():
            raise FileNotFoundError(f"Missing fine-grid region input: {path}")

    print("Archived Drogon grids, maps, and simulation inputs verified")


if __name__ == "__main__":
    check_levels()
