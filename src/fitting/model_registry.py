from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from models.model_fitting import (
    M_nogse_free,
    M_nogse_free_offset,
    M_nogse_mixed,
    M_nogse_mixto_offset,
    M_nogse_rest,
    M_nogse_rest_offset,
    M_ogse_free,
    M_ogse_mixed,
    M_ogse_mixed_offset,
    M_ogse_rest,
    M_ogse_rest_offset,
)


SignalEvaluator = Callable[[float, np.ndarray, float, float, dict[str, float]], np.ndarray]


@dataclass(frozen=True)
class SignalModelSpec:
    name: str
    family: str
    param_names: tuple[str, ...]
    evaluator: SignalEvaluator
    default_modes: dict[str, str]
    default_inits: dict[str, float]
    default_bounds: dict[str, tuple[float, float]]
    log_params: tuple[str, ...] = ()


DEFAULT_PARAMETER_INITS: dict[str, float] = {
    "tc_ms": 5.0,
    "alpha": 0.5,
    "RN": 0.0,
    "M0": np.nan,
    "C": 0.0,
    "D0_m2_ms": 2.3e-12,
}


DEFAULT_PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "tc_ms": (0.1, 1000.0),
    "alpha": (0.0, 1.0),
    "RN": (0.0, 1e9),
    "M0": (0.0, 1e9),
    "C": (-1e9, 1e9),
    "D0_m2_ms": (1e-16, 1e-10),
}


DEFAULT_PARAMETER_MODES: dict[str, str] = {
    "tc_ms": "global_td",
    "alpha": "global_td",
    "RN": "global_td",
    "M0": "global_contrast",
    "C": "global_contrast",
    "D0_m2_ms": "fixed",
}


LOG_PARAMETERS = ("tc_ms", "D0_m2_ms")


def _params(*names: str) -> tuple[str, ...]:
    return tuple(str(name) for name in names)


def _model_defaults(param_names: tuple[str, ...]) -> tuple[dict[str, str], dict[str, float], dict[str, tuple[float, float]]]:
    return (
        {name: DEFAULT_PARAMETER_MODES[name] for name in param_names},
        {name: DEFAULT_PARAMETER_INITS[name] for name in param_names},
        {name: DEFAULT_PARAMETER_BOUNDS[name] for name in param_names},
    )


def _spec(
    name: str,
    family: str,
    param_names: tuple[str, ...],
    evaluator: SignalEvaluator,
) -> SignalModelSpec:
    modes, inits, bounds = _model_defaults(param_names)
    return SignalModelSpec(
        name=name,
        family=family,
        param_names=param_names,
        evaluator=evaluator,
        default_modes=modes,
        default_inits=inits,
        default_bounds=bounds,
        log_params=tuple(name for name in param_names if name in LOG_PARAMETERS),
    )


def _ogse_free(td_ms: float, G: np.ndarray, N: float, x_ms: float, params: dict[str, float]) -> np.ndarray:
    return M_ogse_free(td_ms, G, N, x_ms, params["M0"], params["D0_m2_ms"])


def _ogse_rest(td_ms: float, G: np.ndarray, N: float, x_ms: float, params: dict[str, float]) -> np.ndarray:
    return M_ogse_rest(td_ms, G, N, x_ms, params["tc_ms"], params["M0"], params["D0_m2_ms"])


def _ogse_rest_offset(td_ms: float, G: np.ndarray, N: float, x_ms: float, params: dict[str, float]) -> np.ndarray:
    return M_ogse_rest_offset(td_ms, G, N, x_ms, params["tc_ms"], params["M0"], params["D0_m2_ms"], params["C"])


def _ogse_mixed(td_ms: float, G: np.ndarray, N: float, x_ms: float, params: dict[str, float]) -> np.ndarray:
    return M_ogse_mixed(td_ms, G, N, x_ms, params["tc_ms"], params["alpha"], params["M0"], params["D0_m2_ms"])


def _ogse_mixed_offset(td_ms: float, G: np.ndarray, N: float, x_ms: float, params: dict[str, float]) -> np.ndarray:
    return M_ogse_mixed_offset(
        td_ms,
        G,
        N,
        x_ms,
        params["tc_ms"],
        params["alpha"],
        params["M0"],
        params["D0_m2_ms"],
        params["C"],
        params["RN"],
    )


def _nogse_free(td_ms: float, G: np.ndarray, N: float, x_ms: float, params: dict[str, float]) -> np.ndarray:
    return M_nogse_free(td_ms, G, N, x_ms, params["M0"], params["D0_m2_ms"])


def _nogse_free_offset(td_ms: float, G: np.ndarray, N: float, x_ms: float, params: dict[str, float]) -> np.ndarray:
    return M_nogse_free_offset(td_ms, G, N, x_ms, params["M0"], params["D0_m2_ms"], params["C"])


def _nogse_rest(td_ms: float, G: np.ndarray, N: float, x_ms: float, params: dict[str, float]) -> np.ndarray:
    return M_nogse_rest(td_ms, G, N, x_ms, params["tc_ms"], params["M0"], params["D0_m2_ms"])


def _nogse_rest_offset(td_ms: float, G: np.ndarray, N: float, x_ms: float, params: dict[str, float]) -> np.ndarray:
    return M_nogse_rest_offset(td_ms, G, N, x_ms, params["tc_ms"], params["M0"], params["D0_m2_ms"], params["C"])


def _nogse_mixed(td_ms: float, G: np.ndarray, N: float, x_ms: float, params: dict[str, float]) -> np.ndarray:
    return M_nogse_mixed(td_ms, G, N, x_ms, params["tc_ms"], params["alpha"], params["M0"], params["D0_m2_ms"])


def _nogse_mixed_offset(td_ms: float, G: np.ndarray, N: float, x_ms: float, params: dict[str, float]) -> np.ndarray:
    return M_nogse_mixto_offset(
        td_ms,
        G,
        N,
        x_ms,
        params["tc_ms"],
        params["alpha"],
        params["M0"],
        params["D0_m2_ms"],
        params["C"],
    )


SIGNAL_MODEL_REGISTRY: dict[str, SignalModelSpec] = {
    "ogse_free": _spec("ogse_free", "ogse", _params("M0", "D0_m2_ms"), _ogse_free),
    "ogse_rest": _spec("ogse_rest", "ogse", _params("tc_ms", "M0", "D0_m2_ms"), _ogse_rest),
    "ogse_rest_offset": _spec("ogse_rest_offset", "ogse", _params("tc_ms", "M0", "D0_m2_ms", "C"), _ogse_rest_offset),
    "ogse_mixed": _spec("ogse_mixed", "ogse", _params("tc_ms", "alpha", "M0", "D0_m2_ms"), _ogse_mixed),
    "ogse_mixed_offset": _spec(
        "ogse_mixed_offset",
        "ogse",
        _params("tc_ms", "alpha", "M0", "D0_m2_ms", "C", "RN"),
        _ogse_mixed_offset,
    ),
    "nogse_free": _spec("nogse_free", "nogse", _params("M0", "D0_m2_ms"), _nogse_free),
    "nogse_free_offset": _spec("nogse_free_offset", "nogse", _params("M0", "D0_m2_ms", "C"), _nogse_free_offset),
    "nogse_rest": _spec("nogse_rest", "nogse", _params("tc_ms", "M0", "D0_m2_ms"), _nogse_rest),
    "nogse_rest_offset": _spec("nogse_rest_offset", "nogse", _params("tc_ms", "M0", "D0_m2_ms", "C"), _nogse_rest_offset),
    "nogse_mixed": _spec("nogse_mixed", "nogse", _params("tc_ms", "alpha", "M0", "D0_m2_ms"), _nogse_mixed),
    "nogse_mixed_offset": _spec(
        "nogse_mixed_offset",
        "nogse",
        _params("tc_ms", "alpha", "M0", "D0_m2_ms", "C"),
        _nogse_mixed_offset,
    ),
}


def get_signal_model(name: str) -> SignalModelSpec:
    key = str(name)
    try:
        return SIGNAL_MODEL_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported signal model {name!r}. Allowed values: {sorted(SIGNAL_MODEL_REGISTRY)}.") from exc


def signal_model_names(*, family: str | None = None) -> tuple[str, ...]:
    if family is None:
        return tuple(sorted(SIGNAL_MODEL_REGISTRY))
    family_key = str(family)
    return tuple(sorted(name for name, spec in SIGNAL_MODEL_REGISTRY.items() if spec.family == family_key))


def evaluate_signal_model(
    spec: SignalModelSpec,
    *,
    td_ms: float,
    G: np.ndarray,
    N: float,
    params: dict[str, float],
    x_ms: float | None = None,
) -> np.ndarray:
    resolved_x_ms = float(td_ms) / float(N) if x_ms is None else float(x_ms)
    return spec.evaluator(float(td_ms), np.asarray(G, dtype=float), float(N), float(resolved_x_ms), params)
