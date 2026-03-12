from __future__ import annotations
import numpy as np


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
    Replica notebook:
      if g_type == 'gthorsten': g = sqrt(2)*|g|
      b = N * gamma^2 * delta^2 * delta_app * g^2 / 1e9

    Unidades esperadas:
      gamma: 1/(ms*mT)
      delta_ms, delta_app_ms: ms
      g_mTm: mT/m
      b: s/mm^2
    """
    g = np.asarray(g_mTm, dtype=float)
    if g_type == "gthorsten":
        g = np.sqrt(2.0) * np.abs(g)
    return N * (gamma**2) * (delta_ms**2) * delta_app_ms * (g**2) / 1e9
