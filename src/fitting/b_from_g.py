from __future__ import annotations

import numpy as np


VALID_G_TYPES = {"g", "g_max", "g_lin_max", "g_thorsten"}


def b_from_g(
    g_mTm: np.ndarray,
    *,
    N: float,
    gamma: float,
    delta_ms: float,
    delta_app_ms: float,
    g_type: str,
) -> np.ndarray:
    """
    Match the notebook formula:
      if g_type == 'g_thorsten': g = sqrt(2)*|g|
      b = N * gamma^2 * delta^2 * delta_app * g^2 / 1e9

    Expected units:
      gamma: 1/(ms*mT)
      delta_ms, delta_app_ms: ms
      g_mTm: mT/m
      b: s/mm^2
    """
    g = np.asarray(g_mTm, dtype=float)
    if g_type not in VALID_G_TYPES:
        raise ValueError(f"b_from_g: unrecognized g_type {g_type!r}. Allowed values: {sorted(VALID_G_TYPES)}.")
    if g_type == "g_thorsten":
        g = np.sqrt(2.0) * np.abs(g)
    return N * (gamma**2) * (delta_ms**2) * delta_app_ms * (g**2) / 1e9
