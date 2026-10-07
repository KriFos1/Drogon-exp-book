"""Memory-bounded implementation of PET's auto-adaptive correlation taper."""

from __future__ import annotations

import numpy as np

from pipt.localization.auto_ada_loc import AutoAdaptiveLocalization
from pipt.localization.factory import register_localization


class BlockwiseAutoAdaptiveLocalization(AutoAdaptiveLocalization):
    """Preserve PET's taper while avoiding a square covariance allocation.

    The upstream ``corr_matrix`` calls ``np.corrcoef(X, Y)``, which constructs
    an ``(n_state + n_projected_data)^2`` matrix. Drogon's state has over half
    a million entries, so that temporary is terabytes. Only the cross block
    ``corr(X, Y)`` is needed; compute it in row blocks instead.
    """

    @staticmethod
    def corr_matrix(X, Y, eps=1e-6, block_size=16_384):
        x = np.asarray(X, dtype=float)
        y = np.asarray(Y, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
            raise ValueError(f"Expected (n_state, n_members) and (n_data, n_members), got {x.shape}, {y.shape}")
        members = x.shape[1]
        if members < 1:
            return np.zeros((x.shape[0], y.shape[0]), dtype=float)

        y_centered = y - y.mean(axis=1, keepdims=True)
        y_std = np.sqrt(np.mean(y_centered * y_centered, axis=1))
        corr = np.empty((x.shape[0], y.shape[0]), dtype=float)
        for first in range(0, x.shape[0], block_size):
            last = min(first + block_size, x.shape[0])
            block = x[first:last]
            centered = block - block.mean(axis=1, keepdims=True)
            x_std = np.sqrt(np.mean(centered * centered, axis=1))
            denominator = x_std[:, None] * y_std[None, :]
            numerator = (centered @ y_centered.T) / members
            corr[first:last] = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator >= eps,
            )
        return np.nan_to_num(corr)


def _build_blockwise_autoadaloc(*, info, rng=None, **_):
    return BlockwiseAutoAdaptiveLocalization(info, rng=rng)


register_localization("autoadaloc", _build_blockwise_autoadaloc, overwrite=True)
