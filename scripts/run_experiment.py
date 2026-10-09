"""Run one Drogon chapter experiment with latest PET and SimulatorWrap."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

import numpy as np
from input_output import read_config
from pipt import pipt_init
from drogon_exp.simulator import DrogonFlowSim2Seis

ROOT = Path(__file__).resolve().parents[1]
LEVELS = (300, 9000, 17500, 104098)
EXPERIMENTS = ("ES-MDA", "ES-MDA-loc", "MLHES-MDA", "SMLES")
LEVEL_SOURCE = "archived-jupiter2-levels"


def run(name: str) -> None:
    if name not in EXPERIMENTS:
        raise SystemExit(f"Unknown experiment {name!r}; choose from {', '.join(EXPERIMENTS)}")

    if name == "SMLES":
        import drogon_exp.smles  # register the public-PET-compatible sequential scheme

    folder = ROOT / name
    result_dir = folder / "Results"
    source_marker = result_dir / "level_inputs.txt"
    if (result_dir / "posterior_state_estimate.npz").exists() and (
        not source_marker.exists() or source_marker.read_text().strip() != LEVEL_SOURCE
    ):
        raise SystemExit(f"{name} has results from a different level set; move Results aside before rerunning")
    shutil.copy2(ROOT / "scripts" / "DROGON.mako", folder / "DROGON.mako")
    old_cwd = Path.cwd()
    os.chdir(folder)
    try:
        keys_da, keys_sim, keys_en = read_config.read("config.yaml")
        if os.environ.get("DROGON_PARALLEL"):
            keys_sim["parallel"] = int(os.environ["DROGON_PARALLEL"])
        if os.environ.get("DROGON_MPI"):
            keys_sim["simoptions"] = [["mpi", os.environ["DROGON_MPI"]]]

        transform_paths = [str(ROOT / "Levels" / f"Level{count}" / "TransformMatMean.npz")
                           for count in LEVELS]
        sim = DrogonFlowSim2Seis(
            keys_sim,
            level_transforms=transform_paths,
            multilevel=name in {"MLHES-MDA", "SMLES"},
        )
        scheme = pipt_init.init_da(keys_da, keys_en, sim)
        result_dir.mkdir(exist_ok=True)

        if name == "MLHES-MDA":
            original_after_prior = scheme.after_prior_forecast

            def save_multilevel_prior():
                original_after_prior()
                frames = scheme.ensemble.sim_data
                fine = frames[-1] if isinstance(frames, list) else frames
                fine.to_pickle(result_dir / "prior_forecast.pkl")

            scheme.after_prior_forecast = save_multilevel_prior
        elif name == "SMLES":
            # SMLES starts its recursion on the coarsest prior, so make the
            # fine-grid prior forecast separately for the time-series figures.
            fine_prior = scheme.ensemble.enX[-1]
            scheme._forecast_level(fine_prior, len(LEVELS) - 1)
            scheme.ensemble.sim_data.to_pickle(result_dir / "prior_forecast.pkl")

        result = scheme.run_assimilation()

        if name == "MLHES-MDA":
            # PET's ordinary result writer takes one ensemble matrix; save the
            # finest-fidelity posterior and forecast in its standard artifact
            # names for plotting.
            fine_state = scheme.ensemble.enX[-1]
            np.savez(result_dir / "posterior_state_estimate.npz",
                     **scheme.ensemble.state_layout.to_dict(fine_state))
            forecast = scheme.ensemble.sim_data
            if isinstance(forecast, list):
                forecast = forecast[-1]
            forecast.to_pickle(result_dir / "posterior_forecast.pkl")

        source_marker.write_text(LEVEL_SOURCE + "\n")
        print(f"{name}: {result.prior_data_misfit:.6g} -> {result.data_misfit:.6g}")
    finally:
        os.chdir(old_cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    run(parser.parse_args().experiment)


if __name__ == "__main__":
    main()
