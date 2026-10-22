"""Generate the seeded Drogon prior ensembles of the book runs.

Two generators were used, and they give different members for the same seed:
the single-level runs drew one realization at a time (``prior_100.npz``),
the multilevel runs drew all realizations of a field at once
(``prior_2244.npz`` and ``prior_6488.npz``). Both are reproduced here in
float64; regenerated fields differ from the original files by FFT rounding
(< 1e-6) only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from geostat.gaussian_sim import fast_gaussian

from get_ecl_key_val import read_file

ROOT = Path(__file__).resolve().parents[1]
GRID_DIMS = np.array([46, 73, 31])
FIELD_SIZE = int(np.prod(GRID_DIMS))
PERM_STD = 0.3
PORO_STD = 0.065
CORR = np.array([5, 10, 4])
SATNUM_CORR = np.array([5, 10, 1])
# Ensemble size -> generator used by the runs that import it.
SIZES = {100: "member", 2244: "batch", 6488: "batch"}


def gaussian(std: float, corr: np.ndarray, members: int, mode: str) -> np.ndarray:
    if mode == "member":
        return np.column_stack([np.squeeze(fast_gaussian(GRID_DIMS, np.array([std]), corr)) for _ in range(members)])
    return np.squeeze(fast_gaussian(GRID_DIMS, np.array([std]), corr, num_samples=members))


def generate(members: int, mode: str, output_dir: Path) -> Path:
    np.random.seed(0)
    means = {
        "permx": read_file("PERMX", str(ROOT / "include/grid/drogon.perm")),
        "permy": read_file("PERMY", str(ROOT / "include/grid/drogon.perm")),
        "permz": read_file("PERMZ", str(ROOT / "include/grid/drogon.perm")),
        "poro": read_file("PORO", str(ROOT / "include/grid/drogon.poro")),
    }
    if any(np.asarray(values).size != FIELD_SIZE for values in means.values()):
        raise ValueError("Reference property arrays do not match the 46x73x31 grid")

    prior: dict[str, np.ndarray] = {}
    for name in ("permx", "permy", "permz", "poro"):
        std = PORO_STD if name == "poro" else PERM_STD
        mean = means[name] if name == "poro" else np.log(np.maximum(means[name], 1e-3)) - std**2 / 2
        prior[name] = np.atleast_2d(mean).T + gaussian(std, CORR, members, mode)
    prior["satnum"] = gaussian(1.0, SATNUM_CORR, members, mode)
    for number in range(1, 7):
        prior[f"f{number}"] = np.random.normal(np.array([0]), np.array([1]), size=(1, members))

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"prior_{members}.npz"
    print(f"Writing {members:,} members ({sum(x.nbytes for x in prior.values()) / 1e9:.1f} GB) to {destination}", flush=True)
    np.savez(destination, **prior)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=int, nargs="*", choices=SIZES, default=list(SIZES))
    parser.add_argument("--output", type=Path, default=ROOT / "priors")
    args = parser.parse_args()
    for size in args.members:
        destination = args.output / f"prior_{size}.npz"
        if destination.exists():
            print(f"Keeping existing {destination}")
            continue
        generate(size, SIZES[size], args.output)


if __name__ == "__main__":
    main()
