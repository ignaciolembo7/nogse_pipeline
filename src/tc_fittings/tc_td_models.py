from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# tc vs Td model functions
#
# Each function takes Td as the first argument and returns tc values.
# These are the curves you fit to your (Td, tc) data.
#
# To add a new model: define the function here and register it in
# tc_td_registry.METHODS.
# ---------------------------------------------------------------------------

def tc_pseudohuber(Td: np.ndarray, c: float, delta: float, alpha_macro: float) -> np.ndarray:
    """
    Pseudo-Huber model for tc vs Td.

    Parameters:
      c           — tc at Td=0 (ms), related to unrestricted diffusion
      delta       — crossover time (ms): below delta, behaviour is quadratic;
                    above delta, it becomes linear
      alpha_macro — asymptotic slope (dimensionless), related to tortuosity

    Limits:
      Small Td (Td << delta): tc ≈ c + (alpha_macro / 2*delta) * Td^2
      Large Td (Td >> delta): tc ≈ (c - alpha_macro*delta) + alpha_macro * Td
    """
    Td = np.asarray(Td, dtype=float)
    return c + alpha_macro * delta * (np.sqrt(1.0 + (Td / delta) ** 2) - 1.0)


def tc_linear(Td: np.ndarray, c: float, slope: float) -> np.ndarray:
    """
    Linear model for tc vs Td.

    Parameters:
      c     — tc at Td=0 (ms)
      slope — rate of change of tc with Td (dimensionless)

    Useful as a sanity check or when only the large-Td regime is measured.
    """
    Td = np.asarray(Td, dtype=float)
    return c + slope * Td


# ---------------------------------------------------------------------------
# Derived quantities from pseudohuber parameters
# ---------------------------------------------------------------------------

def alpha_of_Td(Td: np.ndarray, delta: float, alpha_macro: float) -> np.ndarray:
    """Instantaneous slope d(tc)/d(Td) for the pseudohuber model."""
    Td = np.asarray(Td, dtype=float)
    return alpha_macro * Td / np.sqrt(delta ** 2 + Td ** 2)


def tc_quadratic_smallTd(Td: np.ndarray, c: float, delta: float, alpha_macro: float) -> np.ndarray:
    """Small-Td quadratic approximation of tc_pseudohuber."""
    Td = np.asarray(Td, dtype=float)
    return c + (alpha_macro / (2.0 * delta)) * Td ** 2


def tc_linear_largeTd(Td: np.ndarray, c: float, delta: float, alpha_macro: float) -> np.ndarray:
    """Large-Td linear approximation of tc_pseudohuber."""
    Td = np.asarray(Td, dtype=float)
    return (c - alpha_macro * delta) + alpha_macro * Td


def A_from_params(delta: float, alpha_macro: float) -> float:
    """A = alpha_macro / delta (slope coefficient in the quadratic regime)."""
    return float(alpha_macro / delta) if delta > 0 else float("nan")


def qquad_from_params(delta: float, alpha_macro: float) -> float:
    """q_quad = alpha_macro / (2*delta) (quadratic coefficient)."""
    return float(alpha_macro / (2.0 * delta)) if delta > 0 else float("nan")


def qquad_se(delta: float, alpha_macro: float, delta_se: float, alpha_se: float) -> float:
    """Propagated standard error for q_quad = alpha_macro / (2*delta)."""
    if not np.isfinite(delta) or delta <= 0:
        return float("nan")
    if not np.isfinite(delta_se) or not np.isfinite(alpha_se):
        return float("nan")
    dqda = 1.0 / (2.0 * delta)
    dqdd = -alpha_macro / (2.0 * delta ** 2)
    return float(np.sqrt((dqda * alpha_se) ** 2 + (dqdd * delta_se) ** 2))
