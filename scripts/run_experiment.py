"""Run one Drogon chapter experiment.

The method folder holds the run's configuration (``config.yaml``: the original
``drogon.toml`` in current PET syntax) and its original plotting scripts. The
deck is copied in, PET runs the original analyses (``drogon_exp.legacy``), and
the results are written to the method folder in the files and formats of the
original run.
"""

from __future__ import annotations

import os

# The runs were made on AMD EPYC (Zen 3) CPUs. Numerical libraries pick CPU
# specific kernels, so use the same ones (OpenBLAS Zen kernels, no AVX-512
# numpy loops) on other x86-64 CPUs. Set before numpy is imported; Flow and
# the PEM workers inherit them.
CPU_ENV = {
    "OPENBLAS_CORETYPE": "ZEN",
    "NPY_DISABLE_CPU_FEATURES": "AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL "
                                "AVX512_KNL AVX512_KNM AVX512VNNI",
}
for _key, _value in CPU_ENV.items():
    os.environ.setdefault(_key, _value)

import argparse  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ("ES-MDA", "ES-MDA-loc", "MLHES-MDA", "SMLES")
MULTILEVEL = {"MLHES-MDA", "SMLES"}
LEVELS = ("Level300", "Level9000", "Level17500", "Level104098")
FLOW_VERSION = "flow 2024.10"


def link(target: Path, name: Path) -> None:
    if name.is_symlink() or name.exists():
        name.unlink()
    name.symlink_to(os.path.relpath(target, name.parent))


def stage(folder: Path) -> None:
    """Copy the deck into the method folder, with the facies probabilities the single-level deck reads."""
    if folder.name in MULTILEVEL:
        shutil.copy2(ROOT / "scripts" / "DROGON_ML.mako", folder / "DROGON.mako")
    else:
        shutil.copy2(ROOT / "scripts" / "DROGON.mako", folder / "DROGON.mako")
        link(ROOT / "Levels" / "Level104098" / "probfaction.npz", folder / "satnum_probability_fields.npz")


def simulator(name: str, keys_sim: dict):
    """The forward model of the run: Flow and the PEM, with coarse AI mapped to the fine grid."""
    from drogon_exp.simulator import DrogonFlowSim2Seis

    if name in MULTILEVEL:
        transforms = [str(ROOT / "Levels" / level / "TransformMatMean.npz") for level in LEVELS]
        return DrogonFlowSim2Seis(keys_sim, level_transforms=transforms, multilevel=True)
    return DrogonFlowSim2Seis(keys_sim)


def run(name: str) -> None:
    version = subprocess.run(["flow", "--version"], capture_output=True, text=True).stdout
    if FLOW_VERSION not in version:
        raise SystemExit(f"The book runs used OPM Flow 2024.10; found {version.strip()!r}")
    folder = ROOT / name
    os.chdir(folder)

    import drogon_exp.legacy  # noqa: F401  (registers the original schemes and analyses)
    from input_output import read_config
    from pipt import pipt_init

    keys_da, keys_sim, keys_en = read_config.read("config.yaml")
    if not Path(keys_en["importstate"]).is_file():
        raise SystemExit(f"Missing {keys_en['importstate']}; run scripts/make_prior.py first")
    stage(folder)
    if os.environ.get("DROGON_PARALLEL"):
        keys_sim["parallel"] = int(os.environ["DROGON_PARALLEL"])
    scheme = pipt_init.init_da(keys_da, keys_en, simulator(name, keys_sim))
    result = scheme.run_assimilation()
    print(f"{name}: data misfit {result.prior_data_misfit:.6g} -> {result.data_misfit:.6g}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    run(parser.parse_args().experiment)


if __name__ == "__main__":
    main()
