"""Latest-PET adapter for the archived Drogon sim2seis forward model."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr

from subsurface.multphaseflow.flow_rock import flow_equinor_sim2seis


def downscale(transform, values):
    return lsmr(transform, values, atol=1e-7, btol=1e-7)[0]


class DrogonFlowSim2Seis(flow_equinor_sim2seis):
    """Add PET level selection and coarse-to-fine AI prediction mapping.

    PET asks the simulator to switch fidelity with ``setup_fwd_run(level=i)``.
    For levels 0--2, the PEM returns acoustic impedance on the coarse grid;
    PET requires predictions in the fine observation space. The pseudo-inverse
    of the archived volume-average map performs that downscaling.
    """

    def __init__(self, input_dict, level_transforms=None, multilevel=False):
        super().__init__(input_dict)
        self.level_transforms = level_transforms or []
        self.multilevel_run = multilevel
        self.fixed_fine_level = not multilevel
        self._transform_cache = {}
        self._current_active_indices = None

    def setup_fwd_run(self, level=-1, redund_sim=None, **kwargs):
        self.redund_sim = redund_sim
        super().setup_fwd_run(level=level, redund_sim=redund_sim, **kwargs)
        if self.fixed_fine_level:
            self.level = 3

    def _level_transform(self, level):
        if level not in self._transform_cache:
            self._transform_cache[level] = sparse.load_npz(self.level_transforms[level]).tocsr()
        return self._transform_cache[level]

    def run_fwd_sim(self, state, member_i, del_folder=True):
        level = getattr(self, "level", -1)
        prediction = super().run_fwd_sim(state, member_i, del_folder=del_folder)
        if prediction is False or level < 0:
            return prediction

        transform = self._level_transform(level) if level < 3 and self.level_transforms else None
        active_indices = self._current_active_indices
        if active_indices is None:
            raise RuntimeError("No ACTNUM mapping was captured from this Flow member")
        coarse_size = len(active_indices)
        full_size = 104098
        for report in prediction:
            values = report.get("sim2seis")
            if values is not None:
                values = np.asarray(values, dtype=float).reshape(-1)
                # SimulatorWrap initializes non-vintage observations with a
                # 1x1 placeholder; PET ignores those rows because the data
                # frame has no acoustic-impedance observation there.
                if values.size == 1 and (300, 9000, 17500, full_size)[level] > 1:
                    continue
                if level < 3 and not np.isfinite(values).all():
                    # An undefined coarse-cell PEM result has no usable 4D AI
                    # change; do not spread it across fine-grid observations.
                    values = np.where(np.isfinite(values), values, 0.0)
                if values.size == len(active_indices):
                    expanded = np.zeros((300, 9000, 17500, full_size)[level], dtype=float)
                    expanded[active_indices] = values
                    values = expanded
                elif values.size != (300, 9000, 17500, full_size)[level]:
                    raise ValueError(
                        f"Level {level} sim2seis size {values.size} is neither active-cell "
                        f"count {coarse_size} nor full-grid size {(300, 9000, 17500, full_size)[level]}"
                    )
                if transform is not None:
                    values = downscale(transform, values)
                # The frozen observations use the chapter's [-1, 1] vintage
                # normalization. Normalize each predicted 4D field likewise.
                finite = np.isfinite(values)
                if finite.any():
                    low, high = float(np.min(values[finite])), float(np.max(values[finite]))
                    if high > low:
                        values[finite] = 2.0 * (values[finite] - low) / (high - low) - 1.0
                    else:
                        values[finite] = 0.0
                # The archived seismic preprocessing reverses the J axis
                # after interpolation onto the (K, J, I) grid.
                report["sim2seis"] = np.flip(values.reshape((31, 73, 46)), axis=1).reshape(-1)
        return prediction

    def extract_data(self, member):
        # MINPV may deactivate a few cells during initialization; use ACTNUM
        # from the per-member EGRID rather than a static GRDECL-derived mask.
        self._current_active_indices = np.flatnonzero(
            np.asarray(self.ecl_case.grid()["ACTNUM"], dtype=bool).ravel()
        )
        return super().extract_data(member)
