"""Checks of the original analyses, schemes and result formats in drogon_exp.legacy.

The analyses were validated against the book runs by replaying them step by
step; these tests guard their algebra and the original file formats.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from drogon_exp.legacy import (MLHESMDA, SMLES, LegacyESMDA, _bias_correct, _original_obs_var, _stack_members,
                               legacy_approx_update, legacy_hybrid_update)
from pipt.update_schemes.registry import get_scheme


def small_problem(ne=6, seed=0):
    rng = np.random.RandomState(seed)
    state, prediction, perturbed = rng.randn(4, ne), rng.randn(3, ne), rng.randn(3, ne)
    variance = np.array([0.5, 1.0, 2.0])
    proj = (np.eye(ne) - np.ones((ne, ne)) / ne) / np.sqrt(ne - 1)
    return state, prediction, perturbed, variance, proj


def test_config_names_resolve_to_the_original_schemes():
    assert get_scheme("esmda", "legacy") is LegacyESMDA
    assert get_scheme("mlhesmda", "legacy") is MLHESMDA
    assert get_scheme("smles", "legacy") is SMLES


def test_approximate_update_is_the_es_mda_update():
    state, prediction, perturbed, variance, proj = small_problem()
    scheme = SimpleNamespace(proj=proj, state_scaling=np.std(state, axis=1), scale_data=np.sqrt(variance),
                             trunc_energy=1.0, lam=0, localization=None)
    step = legacy_approx_update(scheme).update(state, prediction, perturbed).step
    a_x, a_y = state @ proj, prediction @ proj
    expected = a_x @ a_y.T @ np.linalg.solve(a_y @ a_y.T + np.diag(variance), perturbed - prediction)
    np.testing.assert_allclose(step, expected, rtol=1e-10, atol=1e-12)


def test_hybrid_update_with_one_level_is_the_approximate_update():
    state, prediction, perturbed, variance, proj = small_problem()
    common = dict(state_scaling=np.std(state, axis=1), trunc_energy=0.9, lam=0)
    single = SimpleNamespace(proj=proj, scale_data=np.sqrt(variance), localization=None, **common)
    hybrid = SimpleNamespace(proj=[proj], scale_data=[np.sqrt(variance)], tot_level=1,
                             multilevel={"ml_weights": [1.0]}, **common)
    expected = legacy_approx_update(single).update(state, prediction, perturbed).step
    step = legacy_hybrid_update(hybrid).update([state], [prediction], [perturbed]).step
    np.testing.assert_allclose(step[0], expected, rtol=1e-10, atol=1e-12)


def test_forecasts_are_stacked_as_the_original_pred_data():
    outputs = [[{"WOPR:A1": 1.0 + member, "sim2seis": None},
                {"WOPR:A1": 2.0 + member, "sim2seis": np.array([member, -member])}] for member in range(3)]
    forecast = _stack_members(outputs)
    assert forecast[0]["sim2seis"] is None
    np.testing.assert_array_equal(forecast[0]["WOPR:A1"], [[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(forecast[1]["sim2seis"], [[0, 1, 2], [0, -1, -2]])


def test_bias_correction_shifts_observed_coarse_predictions_to_the_fine_mean():
    coarse = {"WOPR:A1": np.array([[1.0, 3.0]]), "WWPR:A1": np.array([[5.0, 7.0]])}
    fine = {"WOPR:A1": np.array([[10.0, 20.0, 30.0]]), "WWPR:A1": np.array([[0.0, 1.0, 2.0]])}
    _bias_correct([[coarse, fine]], [{"WOPR:A1": np.array([1.0]), "WWPR:A1": None}])
    np.testing.assert_array_equal(coarse["WOPR:A1"], [[19.0, 21.0]])
    np.testing.assert_array_equal(coarse["WWPR:A1"], [[5.0, 7.0]])
    np.testing.assert_array_equal(fine["WOPR:A1"], [[10.0, 20.0, 30.0]])


def test_observations_and_variances_have_the_original_format():
    index = ["2018-01-05", "2018-07-01"]
    data = pd.DataFrame({"WOPR:A1": [100.0, 110.0], "sim2seis": [None, np.array([0.5, -0.5])]}, index=index)
    var = pd.DataFrame({"WOPR:A1": [4.0, 5.0], "sim2seis": [None, np.array([0.1, 0.1])]}, index=index)
    obs, variance = _original_obs_var(data, var)
    assert obs[0]["sim2seis"] is None and variance[0]["sim2seis"] is None
    np.testing.assert_array_equal(obs[1]["WOPR:A1"], [110.0])
    np.testing.assert_array_equal(obs[1]["sim2seis"], [0.5, -0.5])
    assert type(variance[1]["WOPR:A1"]) is float


def test_smles_reads_the_predictions_of_the_level_it_forecast():
    members = np.arange(28.0).reshape(4, 7)
    multilevel = {"ml_ne": [7, 3, 2], "ne": [range(7), range(3), range(2)]}

    def forecast(state):
        assert [level.shape[1] for level in state] == [0, 7, 0]
        assert multilevel["ml_ne"] == [0, 7, 0]
        # PET keeps the predictions of the levels that ran.
        ensemble.pred_data = [SimpleNamespace(matrix="level 1 forecast")]

    ensemble = SimpleNamespace(multilevel=multilevel, forecast=forecast)
    state, prediction = SMLES._forecast_level(SimpleNamespace(ensemble=ensemble), members, 1)
    assert state is members
    assert prediction == "level 1 forecast"
    assert multilevel["ml_ne"] == [7, 3, 2]
