"""Compare the rebuilt ``Levels/`` with the archived levels of the book runs.

The archived directory is ``DROGON_w_seis/Levels`` from the original working
directory. Copied inputs must be identical. The recomputed volume maps cannot
match bit for bit: the legacy maps were computed from the unrounded coarse
geometry, which was written to ``Grid.grdecl`` with two decimals.
"""

from __future__ import annotations

import argparse
import filecmp
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
COPIED = {
    104098: ("Grid.grdecl", "FAULT.INC", "drogon_US.trans", "US_schdl.sch", "EQLNUM.grdecl",
             "FIPNUM.grdecl", "FIPZON.grdecl", "MULTNUM.grdecl", "PVTNUM.grdecl"),
    **{size: ("Grid.grdecl", "FAULT.INC", "drogon_US.trans", "US_schdl.sch", "EQLNUM.grdecl",
              "FIPNUM.grdecl", "FIPZON.grdecl", "MULTNUM.grdecl", "PVTNUM.grdecl")
       for size in (300, 9000, 17500)},
}


def compare_maps(built: Path, original: Path) -> str:
    new = sparse.load_npz(built / "TransformMatMean.npz").tocsr()
    old = sparse.load_npz(original / "TransformMatMean.npz").tocsr()
    rows = np.asarray(abs(new - old).sum(axis=1)).ravel()
    pattern_old, pattern_new = set(zip(*old.nonzero())), set(zip(*new.nonzero()))
    quantiles = np.quantile(rows, [0.5, 0.99, 1.0])
    return (f"mean map row L1 difference median/p99/max {quantiles[0]:.1e}/{quantiles[1]:.1e}/"
            f"{quantiles[2]:.1e}; entries only archived/only rebuilt "
            f"{len(pattern_old - pattern_new)}/{len(pattern_new - pattern_old)} of {old.nnz}")


def compare_facies(built: Path, original: Path) -> str:
    with np.load(built / "probfaction.npz") as new, np.load(original / "probfaction.npz") as old:
        if sorted(new.files) != sorted(old.files):
            return f"facies fields differ: {sorted(new.files)} vs {sorted(old.files)}"
        diff = max(float(np.max(np.abs(new[name].astype(float) - old[name].astype(float)))) for name in old.files)
    return f"facies probabilities max difference {diff:.1e}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True, help="archived Levels directory")
    parser.add_argument("--levels", type=Path, default=ROOT / "Levels")
    args = parser.parse_args()

    for size, names in COPIED.items():
        built, original = args.levels / f"Level{size}", args.original / f"Level{size}"
        differing = [name for name in names if not filecmp.cmp(built / name, original / name, shallow=False)]
        print(f"Level{size}: copied inputs {'identical' if not differing else 'DIFFER: ' + ', '.join(differing)}")
        print(f"Level{size}: {compare_facies(built, original)}")
        if size != 104098:
            print(f"Level{size}: {compare_maps(built, original)}")


if __name__ == "__main__":
    main()
