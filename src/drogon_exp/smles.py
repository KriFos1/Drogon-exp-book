"""Sequential multilevel ES-MDA for the Drogon book experiment."""

from __future__ import annotations

import numpy as np
from misc.sampling import gen_real
from pipt.misc_tools import analysis_tools as at
from pipt.update_schemes.analysis.approx import approx_update
from pipt.update_schemes.esmda import ESMDA
from pipt.update_schemes.multilevel import MultilevelEnsemble
from pipt.update_schemes.registry import register_scheme


class SMLES(ESMDA):
    """Update on one fidelity at a time and thin by data misfit."""

    ENSEMBLE_CLASS = MultilevelEnsemble
    COMPATIBLE_ANALYSES = {"approx": approx_update}

    def __init__(self, keys_da, keys_en, sim, analysis=None, ensemble=None):
        super().__init__(keys_da, keys_en, sim, analysis=analysis, ensemble=ensemble)
        self.ml_sizes = list(self.ensemble.ml_ne)
        self.maxiter = max(len(self.ml_sizes) - 1, 1)
        configured = keys_da.get("mda", {}).get("inflation_param")
        if configured is None:
            self.alpha = np.full(self.maxiter, float(self.maxiter))
        else:
            self.alpha = np.asarray(configured, dtype=float)
            if len(self.alpha) != self.maxiter or not np.isclose(np.sum(1 / self.alpha), 1.0):
                raise ValueError("SMLES inflation factors must match update levels and sum reciprocals to one")
        self.analysis = approx_update(self)
        self.iteration = 0
        self.ensemble.iteration = 0

    def _state_for_level(self, members, level):
        state = [np.empty((members.shape[0], 0), dtype=members.dtype) for _ in self.ml_sizes]
        state[level] = members
        return state

    def _forecast_level(self, members, level):
        multi = self.ensemble.multilevel
        old_sizes = list(multi["ml_ne"])
        old_ne = self.ensemble.ne
        multi["ml_ne"] = [members.shape[1] if index == level else 0 for index in range(len(old_sizes))]
        multi["ne"] = [range(count) for count in multi["ml_ne"]]
        self.ensemble.ne = members.shape[1]
        state = self._state_for_level(members, level)
        try:
            self.ensemble.forecast(state)
            frame = self.ensemble.pred_data[0]
        finally:
            multi["ml_ne"] = old_sizes
            multi["ne"] = [range(count) for count in old_sizes]
            self.ensemble.ne = old_ne
        return state[level], frame

    def run_assimilation(self):
        members = np.asarray(self.ensemble.enX[0])
        observations = self.ensemble.obs_vector
        covariance = self.ensemble.obs_variance
        observation_ensemble = gen_real(observations, covariance, members.shape[1], rng=self.ensemble.rng)
        self.prior_data_misfit = None

        for step in range(self.maxiter):
            level = step
            members, forecast = self._forecast_level(members, level)
            prediction = forecast.matrix
            size = members.shape[1]
            self.ensemble.ne = size
            self.proj = (np.eye(size) - np.ones((size, size)) / size) / np.sqrt(max(size - 1, 1))
            self.vecObs = observations
            self.cov_data = covariance
            if self.prior_data_misfit is None:
                self.prior_data_misfit = float(np.mean(at.calc_objectivefun(observation_ensemble, prediction, covariance)))
                self.prior_data_misfit_mean = self.prior_data_misfit
                self.data_misfit_mean = self.prior_data_misfit

            perturbed, self.scale_data = gen_real(
                observations, self.alpha[step] * covariance, size,
                rng=self.ensemble.rng, return_chol=True,
            )
            update = self.analysis.update(members, prediction, perturbed)
            proposal = members + update.step
            limits = {key: self.prior_info[key].get("limits", (None, None)) for key in self.idX}
            self.ensemble.state_layout.clip(proposal, limits)

            updated, posterior_forecast = self._forecast_level(proposal, level)
            misfit = at.calc_objectivefun(observation_ensemble, posterior_forecast.matrix, covariance)
            selected = np.argsort(misfit)[:self.ml_sizes[level + 1]]
            members = updated[:, selected]
            observation_ensemble = observation_ensemble[:, selected]
            self.ensemble.enX = self._state_for_level(members, level + 1)
            self.iteration = step + 1
            self.ensemble.iteration = self.iteration
            self.prev_data_misfit_mean = self.data_misfit_mean
            self.data_misfit_mean = float(np.mean(misfit[selected]))
            self.data_misfit_std = float(np.std(misfit[selected]))
            self.ensemble_misfit = np.asarray(misfit)[selected]

        members, final_forecast = self._forecast_level(members, len(self.ml_sizes) - 1)
        self.ensemble.enX = members
        self.ensemble.ne = members.shape[1]
        self.ensemble.pred_data = [final_forecast]
        final_misfit = at.calc_objectivefun(observation_ensemble, final_forecast.matrix, covariance)
        self.data_misfit_mean = float(np.mean(final_misfit))
        self.data_misfit_std = float(np.std(final_misfit))
        self.ensemble_misfit = final_misfit
        self.after_loop(converged=False)
        return self._finalize(converged=False)


register_scheme("smles", "approx", SMLES)
