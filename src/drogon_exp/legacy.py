"""Analyses and schemes of the PET version that produced the book's Drogon runs.

The four runs behind the chapter figures used ``KriFos1/PET`` at ``d478e32``.
Current PET computes the same updates, but not with the same floating-point
operations, and its adaptive localization and sequential multilevel scheme
differ. This module restores the original behaviour on top of current PET
through its public scheme registry:

* ``legacy_approx_update``: the original approximate (ES/ES-MDA) update with
  its singular-value truncation, operation order, and adaptive localization
  (correlation of the scaled anomalies with the gain applied to the
  innovations, fixed sigmoid taper).
* ``legacy_hybrid_update``: the original multilevel hybrid gain, weighted by
  the normalised covariance weights and applied in blocks of 1000 rows.
* ``SMLES``: the original sequential multilevel ES-MDA (``esmda_seq_h``).

Random numbers are drawn from the ensemble's seeded stream in the original
order, so a run reproduces the original one draw for draw. Like the original,
the schemes analyse the prior as imported even when failed prior members were
replaced: the original replaced them in the forecast state but analysed its
pre-forecast copy, so only their predictions were copies.

The schemes also write the original result files, in the original formats,
for the original plotting scripts: ``obs_var.npz``, ``prior_forecast.npz``,
``debug_analysis_step_<k>.npz`` (the variables named by ``savedata``) and the
posterior state and forecast (``.npz`` for single-level runs, pickles ``.p``
for multilevel runs).
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from copy import deepcopy

import numpy as np
from scipy.linalg import solve
from scipy.special import expit

from misc.sampling import gen_real
from pipt.update_schemes.analysis.base import AnalysisBase, AnalysisResult
from pipt.update_schemes.esmda import ESMDA
from pipt.update_schemes.multilevel import MultilevelEnsemble, esmda_hybrid
from pipt.update_schemes.registry import register_scheme

__all__ = ["legacy_approx_update", "legacy_hybrid_update", "LegacyESMDA", "MLHESMDA", "SMLES",
           "legacy_misfit"]


def _truncate(u_d, s_d, v_d, energy):
    """Keep the singular values whose cumulative energy is at most ``energy``."""
    if energy < 1:
        keep = (np.cumsum(s_d) / sum(s_d)) <= energy
        u_d, s_d, v_d = u_d[:, keep].copy(), s_d[keep].copy(), v_d[keep, :].copy()
    return u_d, s_d, v_d


def legacy_misfit(pert_obs, pred_data, cov_data):
    """Per-member data misfit as the original PET computed it."""
    residual = pred_data - pert_obs
    return np.diag(residual.T.dot(residual * (cov_data ** (-1))[:, None]))


def _stack_members(outputs):
    """One level's forecast as the original PET held it.

    Per report time, the members' values of each data type side by side
    (``None`` where no member has a value).
    """
    return [{key: np.concatenate([np.atleast_2d(member[time][key]).T for member in outputs], axis=1)
             if any(member[time][key] is not None for member in outputs) else None
             for key in outputs[0][0]}
            for time in range(len(outputs[0]))]


def _original_obs_var(data_df, var_df):
    """Observations and variances per report time, as the original ``obs_var.npz`` holds them."""
    def missing(value):
        return value is None or (np.ndim(value) == 0 and np.isnan(value))

    obs = [{key: None if missing(value) else np.atleast_1d(np.asarray(value, dtype=float))
            for key, value in data_df.loc[label].items()} for label in data_df.index]
    var = [{key: None if missing(value) else (float(value) if np.ndim(value) == 0 else np.asarray(value))
            for key, value in var_df.loc[label].items()} for label in var_df.index]
    return obs, var


def _bias_correct(forecast, obs):
    """Shift the coarse levels' observed predictions to the finest level's mean (``bias_corr``)."""
    for levels, observed in zip(forecast, obs, strict=True):
        for key in levels[-1]:
            if observed.get(key) is not None:
                reference = levels[-1][key].mean(axis=1)
                for level in levels[:-1]:
                    level[key] += reference[:, np.newaxis] - level[key].mean(axis=1)[:, np.newaxis]


def _localization_settings(scheme):
    """The ``localization`` block of the config when it asks for adaptive localization.

    PET removes ``name`` from the block when it builds its localization object,
    so the method is read from that object.
    """
    if getattr(getattr(scheme, "localization", None), "name", None) != "autoadaloc":
        return None
    settings = scheme.keys_da.get("localization")
    return settings if isinstance(settings, Mapping) else {}


class legacy_approx_update(AnalysisBase):
    """The original approximate update, with optional adaptive localization."""

    def update(self, enX, enY, enE, **kwargs):
        scheme = self.scheme
        ne = enX.shape[1]
        proj = scheme.proj
        scaling = scheme.state_scaling
        precision = scheme.scale_data ** (-1)

        pert_preddata = np.dot(np.expand_dims(precision, axis=1), np.ones((1, ne))) * np.dot(enY, proj)
        u_d, s_d, v_d = np.linalg.svd(pert_preddata, full_matrices=False)
        u_d, s_d, v_d = _truncate(u_d, s_d, v_d, scheme.trunc_energy)
        pert_state = (scaling ** (-1))[:, None] * np.dot(enX, proj)
        delta = np.dot(np.expand_dims(precision, axis=1), np.ones((1, ne))) * (enE - enY)

        settings = _localization_settings(scheme)
        if settings is not None:
            gain = np.dot(np.dot(v_d.T, np.diag(s_d)),
                          solve((scheme.lam + 1) * np.eye(len(s_d)) + np.diag(s_d ** 2), u_d.T))
            step = self._auto_ada_loc(scheme, settings, scaling[:, None] * pert_state, np.dot(gain, delta))
        else:
            x_1 = np.dot(u_d.T, delta)
            x_2 = solve((scheme.lam + 1) * np.eye(len(s_d)) + np.diag(s_d ** 2), x_1)
            x_3 = np.dot(np.dot(v_d.T, np.diag(s_d)), x_2)
            step = np.dot(scaling[:, None] * pert_state, x_3)
        return AnalysisResult(step=step)

    @staticmethod
    def _corr_matrix(pert_state, proj_pred_data):
        ne = pert_state.shape[1]
        std_model = np.std(pert_state, axis=1)
        std_model[std_model < 10 ** -6] = 10 ** -6
        std_data = np.std(proj_pred_data, axis=1)
        std_data[std_data < 10 ** -6] = 10 ** -6
        normalized_ensemble = np.divide(pert_state - np.outer(np.mean(pert_state, axis=1), np.ones(ne)),
                                        np.outer(std_model, np.ones(ne)))
        normalized_data = np.divide(proj_pred_data - np.outer(np.mean(proj_pred_data, axis=1), np.ones(ne)),
                                    np.outer(std_data, np.ones(ne)))
        corr = np.divide(np.dot(normalized_ensemble, np.transpose(normalized_data)), ne)
        corr[std_model < 10 ** -6, :] = 0
        corr[:, std_data < 10 ** -6] = 0
        return corr

    def _auto_ada_loc(self, scheme, settings, pert_state, proj_pred_data):
        """The original ``auto_ada_loc``: taper the anomalies, then apply the gain."""
        ne = pert_state.shape[1]
        shuffle = scheme.ensemble.rng.permutation(ne)  # drawn as in the original, used by noise-based tapers
        corr = self._corr_matrix(pert_state, proj_pred_data)
        corr_shuffled = None
        taper = np.ones(corr.shape)
        nstd = float(settings.get("cutoff", 1))
        for start, stop in scheme.state_layout.indices.values():
            rows = np.arange(start, stop)
            if settings.get("type") != "sigm" and settings.get("threshold") != "fixed" and corr_shuffled is None:
                corr_shuffled = self._corr_matrix(pert_state[:, shuffle], proj_pred_data)
            taper[rows, :] = self._taper(settings, nstd, corr[rows, :],
                                         None if corr_shuffled is None else corr_shuffled[rows, :])
        return np.dot(np.multiply(taper, pert_state), proj_pred_data)

    @staticmethod
    def _taper(settings, nstd, corr, corr_shuffled):
        out = np.zeros(corr.shape)
        for column in range(corr.shape[1]):
            current = corr[:, column]
            if settings.get("type") == "sigm":
                out[:, column] = expit((np.absolute(current) - (1 - nstd)) * 50).flatten()
                continue
            noise = np.median(np.absolute(corr_shuffled[:, column]), axis=0) / 0.6745
            if settings.get("threshold") == "universal":
                cutoff = np.sqrt(2 * np.log(np.prod(current.shape))) * noise
            elif settings.get("threshold") == "fixed":
                cutoff = nstd
            else:
                cutoff = nstd * noise
            if settings.get("type") == "soft":
                raise NotImplementedError("the soft taper is not used by the book runs")
            hard = np.zeros(current.shape)
            hard[np.absolute(current) > cutoff] = 1
            out[:, column] = hard.flatten()
        return out


class legacy_hybrid_update(AnalysisBase):
    """The original multilevel hybrid update (Fossum et al., 2020)."""

    def update(self, enX, enY, enE, **kwargs):
        scheme = self.scheme
        levels = scheme.tot_level
        scaling = scheme.state_scaling
        # PET normalises the weights as the original did (integer weights, exact sum).
        weights = list(scheme.multilevel["ml_weights"])

        x_3, pert_state = [], []
        for level in range(levels):
            ne = enX[level].shape[1]
            precision = scheme.scale_data[level] ** (-1)
            pert_preddata = (np.dot(np.expand_dims(precision, axis=1), np.ones((1, ne)))
                             * np.dot(enY[level], scheme.proj[level]))
            pert_state.append((scaling ** (-1))[:, None] * np.dot(enX[level], scheme.proj[level]))
            u_d, s_d, v_d = np.linalg.svd(pert_preddata, full_matrices=False)
            u_d, s_d, v_d = _truncate(u_d, s_d, v_d, scheme.trunc_energy)
            x_2 = solve((scheme.lam + 1) * np.eye(len(s_d)) + np.diag(s_d ** 2), u_d.T)
            x_3.append(np.dot(np.dot(v_d.T, np.diag(s_d)), x_2))

        step = [np.empty(pert_state[level].shape) for level in range(levels)]
        nrows = scaling.shape[0]
        size = min(1000, int(nrows / 2))
        blocks = [np.arange(start, start + size) for start in np.arange(0, nrows - size, size)]
        blocks.append(np.arange(blocks[-1][-1] + 1, nrows))
        for rows in blocks:
            gain = sum([weights[level] * np.dot(pert_state[level][rows, :], x_3[level]) for level in range(levels)])
            for level in range(levels):
                ne = enX[level].shape[1]
                precision = scheme.scale_data[level] ** (-1)
                step[level][rows, :] = np.dot(scaling[rows, None] * gain,
                                              np.dot(np.expand_dims(precision, axis=1), np.ones((1, ne)))
                                              * (enE[level] - enY[level]))
        return AnalysisResult(step=step)


class _OriginalPrior:
    """Analyse the imported prior, not its forecast copy with replaced members."""

    _prior_forecast_done = False

    def run_forecast(self, state):
        if self._prior_forecast_done:
            return super().run_forecast(state)
        self._prior_forecast_done = True
        self.ensemble.forecast(deepcopy(state))
        return self.after_forecast(state)


class _OriginalFiles:
    """Save the results in the files and formats of the original PET."""

    def _original_forecast(self):
        """The latest forecast as the original ``pred_data``."""
        raise NotImplementedError

    def _original_state(self):
        """The current ensemble as the original ``state``."""
        raise NotImplementedError

    def _save_prior_forecast(self):
        if not self._saving_enabled:
            return
        obs, var = _original_obs_var(self.data_df, self.data_var_df)
        np.savez(self._save_path("obs_var.npz"), obs=obs, var=var)
        np.savez(self._save_path("prior_forecast.npz"), pred_data=self._original_forecast())

    def _save_iteration_data(self):
        # The original saved after each analysis step, not after the prior forecast.
        if self.iteration == 0 or not self._saving_enabled:
            return
        values = {"pred_data": self._original_forecast, "state": self._original_state,
                  "data_misfit": lambda: self.data_misfit_mean,
                  "prev_data_misfit": lambda: self.prev_data_misfit_mean}
        np.savez(self._save_path(f"debug_analysis_step_{self.iteration}.npz"),
                 **{key: values[key]() for key in self._savedata_keys})

    def _save_posterior_results(self):
        state, forecast = self._original_state(), self._original_forecast()
        if isinstance(state, dict):
            np.savez(self._save_path("posterior_state_estimate.npz"), **state)
            np.savez(self._save_path("posterior_forecast.npz"), pred_data=forecast)
            return
        # The original fell back to pickles for the per-level lists.
        for name, value in (("posterior_state_estimate.p", state), ("posterior_forecast.p", forecast)):
            with open(self._save_path(name), "wb") as file:
                pickle.dump(value, file)


class LegacyESMDA(_OriginalFiles, _OriginalPrior, ESMDA):
    """ES-MDA with the original approximate update (and localization)."""

    COMPATIBLE_ANALYSES = {"legacy": legacy_approx_update}

    def _original_forecast(self):
        return _stack_members(self.ensemble.member_outputs[0])

    def _original_state(self):
        return self.state_layout.to_dict(self.enX)


class MLHESMDA(_OriginalFiles, _OriginalPrior, esmda_hybrid):
    """Multilevel hybrid ES-MDA with the original hybrid update.

    Each coarse level's predictions are shifted to the finest level's mean
    (``ml_error_corr: bias_corr``) by PET's multilevel forecast.
    """

    COMPATIBLE_ANALYSES = {"legacy": legacy_hybrid_update}

    def _original_forecast(self):
        # Per report time, one entry per level, bias corrected as the analysis sees it.
        levels = [_stack_members(outputs) for outputs in self.ensemble.member_outputs]
        forecast = [list(times) for times in zip(*levels)]
        _bias_correct(forecast, _original_obs_var(self.data_df, self.data_var_df)[0])
        return forecast

    def _original_state(self):
        return [self.state_layout.to_dict(level) for level in self.enX]


class _NoBiasCorrectionEnsemble(MultilevelEnsemble):
    """Multilevel ensemble without the coarse-to-fine mean shift."""

    def treat_modeling_error(self) -> None:
        return None


class SMLES(_OriginalFiles, ESMDA):
    """The original sequential multilevel ES-MDA (PET ``esmda_seq_h``).

    The prior is forecast on the coarsest level. Step ``k`` updates the
    members of level ``k - 1`` with their forecast; the updated members with
    the lowest misfit *before* the update continue to level ``k`` and are
    forecast there.
    """

    ENSEMBLE_CLASS = _NoBiasCorrectionEnsemble
    COMPATIBLE_ANALYSES = {"legacy": legacy_approx_update}

    def __init__(self, keys_da, keys_en, sim, analysis=None, ensemble=None):
        super().__init__(keys_da, keys_en, sim, analysis=analysis, ensemble=ensemble)
        self.ml_sizes = list(self.ensemble.ml_ne)
        # The forecast members of each level reached so far.
        self.level_states = []

    def _forecast_level(self, members, level):
        """Forecast ``members`` on ``level`` alone; returns them (failed members replaced) and their predictions."""
        multilevel = self.ensemble.multilevel
        sizes = list(multilevel["ml_ne"])
        multilevel["ml_ne"] = [members.shape[1] if index == level else 0 for index in range(len(sizes))]
        multilevel["ne"] = [range(count) for count in multilevel["ml_ne"]]
        state = [members if index == level else np.empty((members.shape[0], 0)) for index in range(len(sizes))]
        try:
            self.ensemble.forecast(state)
        finally:
            multilevel["ml_ne"] = sizes
            multilevel["ne"] = [range(count) for count in sizes]
        # PET keeps the predictions of the levels that ran, here only this one.
        (prediction,) = self.ensemble.pred_data
        return state[level], prediction.matrix

    def run_assimilation(self):
        observations = self.enObs_conv
        limits = {key: self.prior_info[key].get("limits", (None, None)) for key in self.idX}
        members = np.asarray(self.ensemble.enX[0])
        # Analyse the imported prior (see the module docstring); its forecast copy is saved.
        forecast, prediction = self._forecast_level(members.copy(), 0)
        self.level_states = [forecast]
        misfit = self._score_level(observations, prediction)
        self._save_prior_forecast()
        for step in range(len(self.ml_sizes) - 1):
            size = members.shape[1]
            self.ensemble.ne = size
            self.proj = (np.eye(size) - (1 / size) * np.ones((size, size))) / np.sqrt(size - 1)
            self.enObs, self.scale_data = gen_real(self.vecObs, self.alpha[step] * self.cov_data, size,
                                                   rng=self.ensemble.rng, return_chol=True)
            updated = members + self.analysis.update(members, prediction, self.enObs).step
            self.state_layout.clip(updated, limits)
            best = np.argsort(misfit)[:self.ml_sizes[step + 1]]
            observations = observations[:, best]
            members, prediction = self._forecast_level(updated[:, best], step + 1)
            self.level_states.append(members)
            self.iteration = step + 1
            misfit = self._score_level(observations, prediction)
            if self._savedata_keys:
                self._save_iteration_data()
        self.ensemble.enX = members
        self.ensemble.ne = members.shape[1]
        self.conv_msg = "All levels assimilated"
        self.after_loop(converged=False)
        return self._finalize(converged=False)

    def _score_level(self, observations, prediction):
        """Score a level's forecast and log it as the original did."""
        misfit = legacy_misfit(observations, prediction, self.cov_data)
        self.prev_data_misfit_mean = self.data_misfit_mean
        self.data_misfit_mean = float(np.mean(misfit))
        self.data_misfit_std = float(np.std(misfit))
        self.ensemble_misfit = misfit
        if self.iteration == 0:
            self.prior_data_misfit_mean = self.data_misfit_mean
            self.logger(f"Prior run complete with data misfit: {self.data_misfit_mean:0.1f}.")
        else:
            change = "reduced" if self.data_misfit_mean < self.prev_data_misfit_mean else "increased"
            self.logger(f"MDA iteration number {self.iteration}! Objective function {change} from "
                        f"{self.prev_data_misfit_mean:0.1f} to {self.data_misfit_mean:0.1f}.")
        return misfit

    def _original_forecast(self):
        # The forecast level's entry follows empty entries for the levels below it.
        below = [None] * (len(self.level_states) - 1)
        return [below + [times] for times in _stack_members(self.ensemble.member_outputs[0])]

    def _original_state(self):
        empty = np.empty((self.state_layout.nx, 0))
        return [self.state_layout.to_dict(self.level_states[level] if level < len(self.level_states) else empty)
                for level in range(len(self.ml_sizes))]


register_scheme("esmda", "legacy", LegacyESMDA)
register_scheme("mlhesmda", "legacy", MLHESMDA)
register_scheme("smles", "legacy", SMLES)
