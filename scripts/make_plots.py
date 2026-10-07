"""Render the Drogon chapter's well, seismic and porosity plots.

This script reads the frozen observations and PET's saved forecasts/states.
It never invokes Flow or regenerates seismic observations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from get_ecl_key_val import read_file

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "observations"
PLOTS = ROOT / "plots"
METHODS = ("ES-MDA", "ES-MDA-loc", "MLHES-MDA", "SMLES")
GRID_SHAPE = (31, 73, 46)  # K, J, I; Eclipse ordering is I-fastest.
FINAL_VINTAGE_DATE = pd.Timestamp("2020-07-01")
FINAL_LAYER = 26  # Chapter figures use layer 27 (one-based).
LEVEL_DIMS = {300: (2, 15, 10), 9000: (15, 30, 20),
              17500: (20, 35, 25), 104098: GRID_SHAPE}


def load_observations():
    return pd.read_pickle(DATA / "data.pkl")


def read_forecast(method: str, kind: str):
    path = ROOT / method / "Results" / f"{kind}_forecast.pkl"
    if not path.exists():
        return None
    return pd.read_pickle(path)


def state_fields(method: str):
    path = ROOT / method / "Results" / "posterior_state_estimate.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as archive:
        state = {key: archive[key] for key in archive.files}
    return state


def as_ensemble(value) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object:
        array = np.asarray(array.tolist())
    return array


def vector_ensemble(value, spatial_size: int) -> np.ndarray:
    """Return ``(cells, members)`` from a saved PET dataframe cell."""
    array = as_ensemble(value)
    if array.ndim == 1:
        if array.size == spatial_size:
            return array.reshape(spatial_size, 1)
        return array.reshape(1, -1)
    if array.ndim == 2:
        if array.shape[0] == spatial_size:
            return array
        if array.shape[1] == spatial_size:
            return array.T
    raise ValueError(f"Expected a spatial ensemble with {spatial_size} cells, got {array.shape}")


def well_ensemble(frame: pd.DataFrame, key: str) -> tuple[np.ndarray, list[np.ndarray]]:
    dates = pd.to_datetime(frame.index)
    values = []
    for entry in frame[key].tolist():
        if entry is None:
            values.append(np.empty(0))
            continue
        array = as_ensemble(entry).ravel()
        values.append(array)
    return dates, values


def plot_well_history(method: str, title: str) -> None:
    observations = load_observations()
    prior = read_forecast(method, "prior")
    posterior = read_forecast(method, "posterior")
    if prior is None or posterior is None:
        print(f"Skipping {method} well histories: prior/posterior forecast is missing")
        return

    for key in ("WOPR:A1", "WWPR:A2"):
        dates, prior_values = well_ensemble(prior, key)
        _, posterior_values = well_ensemble(posterior, key)
        observed = observations[key].to_numpy()
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
        for ax, members, label in zip(axes, (prior_values, posterior_values), ("Prior", "Posterior")):
            med = np.asarray([np.mean(v) if v.size else np.nan for v in members])
            low = np.asarray([np.percentile(v, 10) if v.size else np.nan for v in members])
            high = np.asarray([np.percentile(v, 90) if v.size else np.nan for v in members])
            ax.fill_between(dates, low, high, color="0.7", alpha=0.7, label="10–90% ensemble")
            ax.plot(dates, med, color="tab:orange", label="Ensemble mean")
            ax.plot(dates, observed, ".", color="tab:red", label="Observed")
            ax.set_ylabel(key.replace(":", " "))
            ax.set_title(f"{title}: {label}")
            ax.grid(alpha=0.2)
            ax.legend(loc="best", fontsize=8)
        axes[-1].set_xlabel("Date")
        fig.savefig(PLOTS / f"{method}_{key.replace(':', '_')}.png", dpi=180)
        plt.close(fig)


def plot_acoustic_impedance(method: str, title: str) -> None:
    observations = load_observations()
    posterior = read_forecast(method, "posterior")
    if posterior is None:
        print(f"Skipping {method} seismic plot: posterior forecast is missing")
        return
    vintage_index = pd.DatetimeIndex(pd.to_datetime(observations.index)).get_loc(FINAL_VINTAGE_DATE)
    observed = np.asarray(observations.iloc[vintage_index]["sim2seis"], dtype=float)
    predicted = vector_ensemble(posterior.iloc[vintage_index]["sim2seis"], int(np.prod(GRID_SHAPE)))
    mean = predicted.mean(axis=1)
    std = predicted.std(axis=1, ddof=1)
    residual = mean - observed
    residual[np.abs(residual) <= 0.4] = np.nan
    std[std <= 0.1] = np.nan
    arrays = [observed, residual, std]
    titles = ["Observed normalized AI", "Posterior mean residual", "Posterior predictive standard deviation"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for axis, values, label in zip(axes, arrays, titles):
        image = axis.imshow(values.reshape(GRID_SHAPE)[FINAL_LAYER], origin="lower", cmap="seismic")
        axis.set_title(label)
        axis.set_xlabel("j-cell")
        axis.set_ylabel("k-cell")
        fig.colorbar(image, ax=axis, shrink=0.78)
    fig.suptitle(f"{title}: final seismic vintage, layer 27")
    fig.savefig(PLOTS / f"{method}_final_vintage_AI.png", dpi=180)
    plt.close(fig)


def plot_porosity(method: str, title: str) -> None:
    state = state_fields(method)
    if state is None or "poro" not in state:
        print(f"Skipping {method} porosity plot: posterior state is missing")
        return
    ensemble = np.asarray(state["poro"])
    if ensemble.ndim == 1:
        ensemble = ensemble[:, None]
    mean = ensemble.mean(axis=1).reshape(GRID_SHAPE)[:, :, 17]
    std = ensemble.std(axis=1, ddof=1).reshape(GRID_SHAPE)[:, :, 17]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    for axis, value, label in zip(axes, (mean, std), ("Mean", "Standard deviation")):
        image = axis.imshow(value, origin="lower", cmap="viridis")
        axis.set_title(label)
        axis.set_xlabel("j-cell")
        axis.set_ylabel("k-cell")
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.suptitle(f"{title}: posterior porosity, i-index 18")
    fig.savefig(PLOTS / f"{method}_posterior_porosity_i18.png", dpi=180)
    plt.close(fig)


def plot_model_and_regions() -> None:
    """Chapter Figures 17--19: model/observed AI, EQLNUM, and SATNUM."""
    observed = load_observations()
    final_index = pd.DatetimeIndex(pd.to_datetime(observed.index)).get_loc(FINAL_VINTAGE_DATE)
    ai = np.asarray(observed.iloc[final_index]["sim2seis"], dtype=float).reshape(GRID_SHAPE)
    active = np.asarray(read_file("ACTNUM", str(ROOT / "Levels" / "Level104098" / "Grid.grdecl")), dtype=bool)
    active_layer = active.reshape(GRID_SHAPE)[FINAL_LAYER]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].imshow(active_layer, origin="lower", cmap="gray_r", interpolation="nearest")
    axes[0].set_title("Drogon computational grid, layer 27")
    ai_layer = ai[FINAL_LAYER].copy()
    ai_layer[(ai_layer >= -0.3) & (ai_layer <= -0.2)] = np.nan
    ai_layer[~active_layer] = np.nan
    image = axes[1].imshow(ai_layer, origin="lower", cmap="seismic", vmin=-1, vmax=1)
    axes[1].set_title("Observed normalized AI, final vintage")
    for axis in axes:
        axis.set_xlabel("i-cell")
        axis.set_ylabel("j-cell")
    fig.colorbar(image, ax=axes[1], shrink=0.8)
    fig.savefig(PLOTS / "figure_17_model_and_observed_ai.png", dpi=180)
    plt.close(fig)

    eql_fields = []
    for cells, shape in LEVEL_DIMS.items():
        values = read_file("EQLNUM", str(ROOT / "LevelsV2" / f"Level{cells}" / "EQLNUM.grdecl"))
        k_index = min(shape[0] // 2, shape[0] - 1)
        eql_fields.append(np.asarray(values).reshape(shape)[k_index])
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    image = None
    for axis, (cells, values) in zip(axes, zip(LEVEL_DIMS, eql_fields)):
        image = axis.imshow(values, origin="lower", cmap="tab20", interpolation="nearest")
        axis.set_title(f"{cells:,} cells")
        axis.set_xlabel("i-cell")
        axis.set_ylabel("j-cell")
    fig.colorbar(image, ax=axes, shrink=0.8)
    fig.savefig(PLOTS / "figure_18_eqlnum_levels.png", dpi=180)
    plt.close(fig)

    prior_path = ROOT / "priors" / "prior_100.npz"
    if not prior_path.exists():
        return
    with np.load(prior_path) as prior, np.load(
        ROOT / "LevelsV2" / "Level104098" / "probfaction.npz"
    ) as probabilities:
        latent = [prior["satnum"][:, index] for index in (0, 1)]
        probability_names = ["p1", "p2", "p3", "p4", "p7", "p8", "p9", "p10", "p11", "p12"]
        probabilities_stack = np.stack([probabilities[name] for name in probability_names], axis=-1)
        reference = read_file("SATNUM", str(ROOT / "include" / "regions" / "drogon.satnum"))
        realizations = []
        for gaussian in latent:
            cumulative = np.cumsum(probabilities_stack, axis=-1)
            comparisons = norm.cdf(gaussian)[:, None] <= cumulative
            categories = np.argmax(comparisons, axis=-1) + 1
            categories[~np.any(comparisons, axis=-1)] = 12
            categories[probabilities["p1"] < 0] = 12
            realizations.append(categories)
    k_index = GRID_SHAPE[0] // 2
    fields = [np.asarray(reference).reshape(GRID_SHAPE)[k_index]] + [
        value.reshape(GRID_SHAPE)[k_index] for value in realizations
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, values, label in zip(axes, fields, ("Reference", "Realization 1", "Realization 2")):
        image = axis.imshow(values, origin="lower", cmap="tab20", vmin=1, vmax=12,
                            interpolation="nearest")
        axis.set_title(label)
        axis.set_xlabel("i-cell")
        axis.set_ylabel("j-cell")
    fig.colorbar(image, ax=axes, ticks=np.arange(1, 13), shrink=0.8)
    fig.savefig(PLOTS / "figure_19_satnum_regions.png", dpi=180)
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    plot_model_and_regions()
    for method, label in zip(METHODS, ("ES-MDA", "ES-MDA + localization", "MLHES-MDA", "SMLES")):
        plot_well_history(method, label)
        plot_acoustic_impedance(method, label)
        plot_porosity(method, label)
    print(f"Drogon figures written to {PLOTS}")


if __name__ == "__main__":
    main()
