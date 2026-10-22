import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy import sparse

from drogon_exp.simulator import DrogonFlowSim2Seis
from pipt.ensembles import AssimilationEnsemble
from subsurface.multphaseflow.flow_rock import flow_equinor_sim2seis


def simulator_at(level, transform=None):
    simulator = DrogonFlowSim2Seis.__new__(DrogonFlowSim2Seis)
    simulator.level = level
    simulator.level_transforms = ["level-transform"]
    simulator._transform_cache = {} if transform is None else {level: transform}
    return simulator


def predict(simulator, reports):
    with patch.object(flow_equinor_sim2seis, "run_fwd_sim", return_value=reports):
        return simulator.run_fwd_sim({}, 0)


class DrogonPredictionTest(unittest.TestCase):
    def test_coarse_ai_is_mapped_with_the_transposed_average_map(self):
        transform = sparse.csr_matrix([[0.75, 0.25, 0.0], [0.0, 0.25, 0.75]])
        coarse = np.array([1.0, -0.5])
        actual = predict(simulator_at(0, transform), [{"sim2seis": coarse.copy()}])[0]["sim2seis"]
        np.testing.assert_allclose(actual, [0.75, 0.125, -0.375])

    def test_fine_prediction_and_placeholders_pass_through(self):
        values = np.linspace(-1.0, 1.0, 5)
        reports = predict(simulator_at(3), [{"sim2seis": values}])
        np.testing.assert_array_equal(reports[0]["sim2seis"], values)

        transform = sparse.csr_matrix(np.ones((2, 3)) / 3)
        reports = predict(simulator_at(1, transform), [{"sim2seis": np.zeros(1), "WOPR A1": 3.0}])
        np.testing.assert_array_equal(reports[0]["sim2seis"], np.zeros(1))

    def test_pet_multilevel_forecast_matches_fine_mean_without_collapsing_spread(self):
        coarse = np.array([[0.1, 0.3, 0.5], [0.4, 0.6, 0.8]])
        fine = np.array([[-0.2, 0.2], [0.3, 0.7]])
        original_spread = np.ptp(coarse, axis=1)
        ensemble = SimpleNamespace(pred_data=[SimpleNamespace(matrix=coarse), SimpleNamespace(matrix=fine)])

        AssimilationEnsemble.treat_modeling_error(ensemble)

        np.testing.assert_allclose(coarse.mean(axis=1), fine.mean(axis=1), atol=1e-15)
        np.testing.assert_allclose(np.ptp(coarse, axis=1), original_spread)
        np.testing.assert_allclose(fine, [[-0.2, 0.2], [0.3, 0.7]])


if __name__ == "__main__":
    unittest.main()
