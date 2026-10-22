"""Run one Drogon chapter experiment as the original run did.

The method folder holds the original ``drogon.toml``. The shared inputs are
linked into it under the names that file uses, and the assimilation runs with
the statements of the original ``run_script.py``.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ("ES-MDA", "ES-MDA-loc", "MLHES-MDA", "SMLES")
MULTILEVEL = {"MLHES-MDA", "SMLES"}
OBSERVATIONS = ("data.pkl", "var.pkl", "assim_index.csv", "datatyp.csv", "reportdates.csv", "true_data_index.csv")
FLOW_VERSION = "flow 2024.10"
# The runs were made on AMD EPYC (Zen 3) CPUs. Numerical libraries pick CPU
# specific kernels, so use the same ones (OpenBLAS Zen kernels, no AVX-512
# numpy loops) on other x86-64 CPUs. Set before numpy is imported; Flow and
# the PEM workers inherit them.
CPU_ENV = {
    "OPENBLAS_CORETYPE": "ZEN",
    "NPY_DISABLE_CPU_FEATURES": "AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL "
                                "AVX512_KNL AVX512_KNM AVX512VNNI",
}


def link(target: Path, name: Path) -> None:
    if name.is_symlink() or name.exists():
        name.unlink()
    name.symlink_to(os.path.relpath(target, name.parent))


def stage(folder: Path) -> None:
    """Link data, prior, facies probabilities and the deck into the method folder."""
    for name in OBSERVATIONS:
        link(ROOT / "data" / "observations" / name, folder / name)
    prior = re.search(r'^importstaticvar\s*=\s*"([^"]+)"', (folder / "drogon.toml").read_text(), re.M).group(1)
    if not (ROOT / "priors" / prior).is_file():
        raise SystemExit(f"Missing priors/{prior}; run scripts/make_prior.py first")
    link(ROOT / "priors" / prior, folder / prior)
    # The single-level deck reads the fine facies probabilities from the run folder.
    link(ROOT / "Levels" / "Level104098" / "probfaction.npz", folder / "satnum_probability_fields.npz")
    template = "DROGON_ML.mako" if folder.name in MULTILEVEL else "DROGON.mako"
    shutil.copy2(ROOT / "scripts" / template, folder / "DROGON.mako")


def run(name: str) -> None:
    version = subprocess.run(["flow", "--version"], capture_output=True, text=True).stdout
    if FLOW_VERSION not in version:
        raise SystemExit(f"The book runs used OPM Flow 2024.10; found {version.strip()!r}")
    folder = ROOT / name
    stage(folder)
    os.chdir(folder)

    # The original run_script.py:
    from pipt.loop.assimilation import Assimilate
    from subsurface.multphaseflow.flow_rock import flow_equinor_sim2seis
    from input_output import read_config
    from pipt import pipt_init
    import numpy as np

    np.random.seed(10)
    kd, kf, _ = read_config.read_toml("drogon.toml")
    if os.environ.get("DROGON_PARALLEL"):
        kf["parallel"] = int(os.environ["DROGON_PARALLEL"])
    sim = flow_equinor_sim2seis(kf)
    analysis = pipt_init.init_da(kd, kd, sim)
    assimilation = Assimilate(analysis)
    assimilation.run()


def main() -> None:
    for key, value in CPU_ENV.items():
        os.environ.setdefault(key, value)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    run(parser.parse_args().experiment)


if __name__ == "__main__":
    main()
