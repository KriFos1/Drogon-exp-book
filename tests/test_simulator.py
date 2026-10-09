import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from drogon_exp.simulator import DrogonFlowSim2Seis, downscale
from subsurface.multphaseflow.flow_rock import flow_equinor_sim2seis


class DrogonPredictionTest(unittest.TestCase):
    def test_missing_coarse_ai_is_filled_without_changing_valid_predictions(self):
        simulator = DrogonFlowSim2Seis.__new__(DrogonFlowSim2Seis)
        simulator.level = 0
        simulator.level_transforms = ["level-0-transform"]
        simulator._current_active_indices = np.arange(300)
        simulator._level_transform = lambda level: sparse.csr_matrix(
            (np.ones(300), (np.arange(300), np.arange(300))),
            shape=(300, 104098),
        )

        def predict(values):
            with patch.object(flow_equinor_sim2seis, "run_fwd_sim", return_value=[{"sim2seis": values}]):
                return simulator.run_fwd_sim({}, 0)[0]["sim2seis"]

        valid = np.linspace(-1.0, 1.0, 300)
        expected = predict(valid.copy())
        missing = valid.copy()
        missing[36] = np.nan
        missing[186] = np.inf
        actual = predict(missing)

        affected = np.zeros(104098, dtype=bool)
        affected[[36, 186]] = True
        affected = np.flip(affected.reshape((31, 73, 46)), axis=1).ravel()
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual[~affected], expected[~affected])

    def test_downscaling_preserves_overlapping_coarse_observations(self):
        transform = sparse.csr_matrix([[0.75, 0.25, 0.0], [0.0, 0.25, 0.75]])
        coarse_values = np.array([0.4, -0.2])

        fine_values = downscale(transform, coarse_values)

        np.testing.assert_allclose(transform @ fine_values, coarse_values, atol=1e-10)
        np.testing.assert_allclose(fine_values, np.linalg.pinv(transform.toarray()) @ coarse_values)


if __name__ == "__main__":
    unittest.main()
