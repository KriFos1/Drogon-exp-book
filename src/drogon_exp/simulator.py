"""SimulatorWrap adapter for the Drogon sim2seis forward model of the book runs."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from subsurface.multphaseflow.flow_rock import flow_equinor_sim2seis

FINE_LEVEL = 3


class DrogonFlowSim2Seis(flow_equinor_sim2seis):
    """Add PET level selection and the coarse-to-fine AI mapping.

    PET selects the fidelity with ``setup_fwd_run(level=i)``. SimulatorWrap
    (``d8c4734``) returns each 4D acoustic-impedance change, baseline minus
    vintage, on the full grid of the simulated level (inactive cells 0) and
    scaled to [-1, 1]. As in the original multilevel runs (PET option
    ``interpolate``), coarse fields are mapped to the fine observation grid
    with the transpose of the level's volume-average map and are not rescaled.
    """

    def __init__(self, input_dict, level_transforms=None, multilevel=False):
        super().__init__(input_dict)
        self.level_transforms = level_transforms or []
        self.fixed_fine_level = not multilevel
        self._transform_cache = {}

    def setup_fwd_run(self, level=-1, redund_sim=None, **kwargs):
        self.redund_sim = redund_sim
        super().setup_fwd_run(level=level, redund_sim=redund_sim, **kwargs)
        if self.fixed_fine_level:
            self.level = FINE_LEVEL

    def _level_transform(self, level):
        if level not in self._transform_cache:
            self._transform_cache[level] = sparse.load_npz(self.level_transforms[level]).tocsr()
        return self._transform_cache[level]

    def run_fwd_sim(self, state, member_i, del_folder=True):
        level = getattr(self, "level", -1)
        prediction = super().run_fwd_sim(state, member_i, del_folder=del_folder)
        if prediction is False or not 0 <= level < FINE_LEVEL:
            return prediction

        transform = self._level_transform(level)
        for report in prediction:
            values = report.get("sim2seis")
            # Reports without a vintage hold a placeholder that PET ignores.
            if values is not None and np.size(values) == transform.shape[0]:
                report["sim2seis"] = transform.T @ np.asarray(values, dtype=float).ravel()
        return prediction
