"""Generate seeded Drogon state ensembles from the published prior model."""

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
PERM_CORR = np.array([5, 10, 4])
PORO_STD = 0.065
SATNUM_CORR = np.array([5, 10, 1])


def generate(members: int, output_dir: Path, dtype=np.float32) -> Path:
    if members < 2:
        raise ValueError("Ensemble size must be at least two")
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
    for name in ("permx", "permy", "permz"):
        log_mean = np.log(np.maximum(means[name], 1e-3)) - PERM_STD**2 / 2
        values = fast_gaussian(GRID_DIMS, np.array([PERM_STD]), PERM_CORR, num_samples=members)
        prior[name] = (values + log_mean[:, None]).astype(dtype)

    porosity = fast_gaussian(GRID_DIMS, np.array([PORO_STD]), PERM_CORR, num_samples=members)
    prior["poro"] = (porosity + means["poro"][:, None]).astype(dtype)
    prior["satnum"] = fast_gaussian(
        GRID_DIMS, np.array([1.0]), SATNUM_CORR, num_samples=members
    ).astype(dtype)
    for number in range(1, 7):
        prior[f"f{number}"] = np.random.normal(0.0, 1.0, size=(1, members)).astype(dtype)

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"prior_{members}.npz"
    print(f"Writing {members:,} members ({sum(x.nbytes for x in prior.values()) / 1e9:.1f} GB) to {destination}", flush=True)
    np.savez_compressed(destination, **prior)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=int, nargs="*", default=[100, 2244, 6488])
    parser.add_argument("--output", type=Path, default=ROOT / "priors")
    args = parser.parse_args()
    for size in args.members:
        destination = args.output / f"prior_{size}.npz"
        if destination.exists():
            print(f"Keeping existing {destination}")
            continue
        generate(size, args.output)


if __name__ == "__main__":
    main()
