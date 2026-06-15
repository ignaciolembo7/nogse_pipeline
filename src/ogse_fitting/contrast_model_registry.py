from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from models.model_fitting import (
    M_ogse_mixed_offset,
    OGSE_contrast_vs_g_free,
    OGSE_contrast_vs_g_mixed,
    OGSE_contrast_vs_g_rest,
    OGSE_contrast_vs_g_rest_offset,
    OGSE_contrast_vs_g_tort,
)


# ---------------------------------------------------------------------------
# ContrastFitSpec — minimal spec for running the OGSE contrast fitting loop.
#
# To add a new OGSE contrast model:
#   1. Write an _eval_mymodel() function below following the same signature.
#   2. Add an entry to OGSE_CONTRAST_FIT_SPECS.
#   3. That's it — the fitting loop, the plot panels, and the experiment
#      validation in fitting/experiments.py will all pick it up automatically.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContrastFitSpec:
    """
    Specification for one OGSE contrast fitting model.

    Attributes
    ----------
    name      : short identifier used in --model flags and output filenames
    evaluator : function(td_ms, G1, G2, n_1, n_2, params) → contrast array
    maxfev    : maximum function evaluations passed to scipy curve_fit
    """
    name: str
    evaluator: Callable[[float, np.ndarray, np.ndarray, int, int, dict[str, Any]], np.ndarray]
    maxfev: int


# ---------------------------------------------------------------------------
# Evaluator functions — one per model
# Each takes (td_ms, G1, G2, n_1, n_2, params) and returns a contrast array.
# ---------------------------------------------------------------------------

def _eval_free(
    td_ms: float, G1: np.ndarray, G2: np.ndarray, n_1: int, n_2: int,
    params: dict[str, Any],
) -> np.ndarray:
    return OGSE_contrast_vs_g_free(td_ms, G1, G2, n_1, n_2, float(params["M0"]), float(params["D0_m2_ms"]))


def _eval_tort(
    td_ms: float, G1: np.ndarray, G2: np.ndarray, n_1: int, n_2: int,
    params: dict[str, Any],
) -> np.ndarray:
    return OGSE_contrast_vs_g_tort(
        td_ms, G1, G2, n_1, n_2,
        float(params["alpha"]), float(params["M0"]), float(params["D0_m2_ms"]),
    )


def _eval_rest(
    td_ms: float, G1: np.ndarray, G2: np.ndarray, n_1: int, n_2: int,
    params: dict[str, Any],
) -> np.ndarray:
    return OGSE_contrast_vs_g_rest(
        td_ms, G1, G2, n_1, n_2,
        float(params["tc_ms"]), float(params["M0"]), float(params["D0_m2_ms"]),
    )


def _eval_rest_offset(
    td_ms: float, G1: np.ndarray, G2: np.ndarray, n_1: int, n_2: int,
    params: dict[str, Any],
) -> np.ndarray:
    return OGSE_contrast_vs_g_rest_offset(
        td_ms, G1, G2, n_1, n_2,
        float(params["tc_ms"]), float(params["M0"]), float(params["D0_m2_ms"]), float(params["C"]),
    )


def _eval_mixed(
    td_ms: float, G1: np.ndarray, G2: np.ndarray, n_1: int, n_2: int,
    params: dict[str, Any],
) -> np.ndarray:
    return OGSE_contrast_vs_g_mixed(
        td_ms, G1, G2, n_1, n_2,
        float(params["tc_ms"]), float(params["alpha"]), float(params["M0"]), float(params["D0_m2_ms"]),
    )


def _eval_ogse_mixed_offset(
    td_ms: float, G1: np.ndarray, G2: np.ndarray, n_1: int, n_2: int,
    params: dict[str, Any],
) -> np.ndarray:
    x1 = float(td_ms) / float(n_1)
    x2 = float(td_ms) / float(n_2)
    return M_ogse_mixed_offset(
        td_ms, G1, n_1, x1,
        float(params["tc_ms"]), float(params["alpha"]),
        float(params["M0"]), float(params["D0_m2_ms"]),
        float(params.get("C", 0.0)), float(params.get("RN", 0.0)),
    ) - M_ogse_mixed_offset(
        td_ms, G2, n_2, x2,
        float(params["tc_ms"]), float(params["alpha"]),
        float(params["M0"]), float(params["D0_m2_ms"]),
        float(params.get("C", 0.0)), float(params.get("RN", 0.0)),
    )


# ---------------------------------------------------------------------------
# Registry — this is the single source of truth for OGSE contrast models.
# ---------------------------------------------------------------------------

OGSE_CONTRAST_FIT_SPECS: dict[str, ContrastFitSpec] = {
    "free":              ContrastFitSpec(name="free",              evaluator=_eval_free,             maxfev=400000),
    "tort":              ContrastFitSpec(name="tort",              evaluator=_eval_tort,             maxfev=600000),
    "rest":              ContrastFitSpec(name="rest",              evaluator=_eval_rest,             maxfev=600000),
    "rest_offset":       ContrastFitSpec(name="rest_offset",       evaluator=_eval_rest_offset,      maxfev=800000),
    "rest_offset_globC": ContrastFitSpec(name="rest_offset_globC", evaluator=_eval_rest_offset,      maxfev=800000),
    "mixed":             ContrastFitSpec(name="mixed",             evaluator=_eval_mixed,            maxfev=800000),
    "ogse_mixed_offset": ContrastFitSpec(name="ogse_mixed_offset", evaluator=_eval_ogse_mixed_offset, maxfev=800000),
}
