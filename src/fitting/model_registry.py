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
    NOGSE_contrast_vs_g_free,
    NOGSE_contrast_vs_g_free_grad_offset,
    NOGSE_contrast_vs_g_rest,
    NOGSE_contrast_vs_g_tort,
    OGSE_contrast_vs_g_free,
    OGSE_contrast_vs_g_mixed,
    OGSE_contrast_vs_g_rest,
    OGSE_contrast_vs_g_rest_offset,
    OGSE_contrast_vs_g_tort,
)


SignalEvaluator = Callable[[float, np.ndarray, float, float, dict[str, float]], np.ndarray]
OGSEContrastEvaluator = Callable[[float, np.ndarray, np.ndarray, int, int, dict[str, float]], np.ndarray]
NOGSEContrastEvaluator = Callable[[float, np.ndarray, int, dict[str, float]], np.ndarray]


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


@dataclass(frozen=True)
class ContrastModelSpec:
    name: str
    family: str
    param_names: tuple[str, ...]
    evaluator: OGSEContrastEvaluator | NOGSEContrastEvaluator
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
    "g0_mTm": 0.0,
}


DEFAULT_PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "tc_ms": (0.1, 1000.0),
    "alpha": (0.0, 1.0),
    "RN": (0.0, 1e9),
    "M0": (0.0, 1e9),
    "C": (-1e9, 1e9),
    "D0_m2_ms": (1e-16, 1e-10),
    "g0_mTm": (-20.0, 20.0),
}


DEFAULT_PARAMETER_MODES: dict[str, str] = {
    "tc_ms": "global_td",
    "alpha": "global_td",
    "RN": "global_td",
    "M0": "global_contrast",
    "C": "global_contrast",
    "D0_m2_ms": "fixed",
    "g0_mTm": "free",
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


def _contrast_spec(
    name: str,
    family: str,
    param_names: tuple[str, ...],
    evaluator: OGSEContrastEvaluator | NOGSEContrastEvaluator,
) -> ContrastModelSpec:
    modes, inits, bounds = _model_defaults(param_names)
    return ContrastModelSpec(
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


def _ogse_contrast_free(td_ms: float, G1: np.ndarray, G2: np.ndarray, n_1: int, n_2: int, params: dict[str, float]) -> np.ndarray:
    return OGSE_contrast_vs_g_free(td_ms, G1, G2, n_1, n_2, params["M0"], params["D0_m2_ms"])


def _ogse_contrast_tort(td_ms: float, G1: np.ndarray, G2: np.ndarray, n_1: int, n_2: int, params: dict[str, float]) -> np.ndarray:
    return OGSE_contrast_vs_g_tort(td_ms, G1, G2, n_1, n_2, params["alpha"], params["M0"], params["D0_m2_ms"])


def _ogse_contrast_rest(td_ms: float, G1: np.ndarray, G2: np.ndarray, n_1: int, n_2: int, params: dict[str, float]) -> np.ndarray:
    return OGSE_contrast_vs_g_rest(td_ms, G1, G2, n_1, n_2, params["tc_ms"], params["M0"], params["D0_m2_ms"])


def _ogse_contrast_rest_offset(td_ms: float, G1: np.ndarray, G2: np.ndarray, n_1: int, n_2: int, params: dict[str, float]) -> np.ndarray:
    return OGSE_contrast_vs_g_rest_offset(td_ms, G1, G2, n_1, n_2, params["tc_ms"], params["M0"], params["D0_m2_ms"], params["C"])


def _ogse_contrast_mixed(td_ms: float, G1: np.ndarray, G2: np.ndarray, n_1: int, n_2: int, params: dict[str, float]) -> np.ndarray:
    return OGSE_contrast_vs_g_mixed(td_ms, G1, G2, n_1, n_2, params["tc_ms"], params["alpha"], params["M0"], params["D0_m2_ms"])


def _nogse_contrast_free(td_ms: float, G: np.ndarray, n_value: int, params: dict[str, float]) -> np.ndarray:
    return NOGSE_contrast_vs_g_free(td_ms, G, n_value, params["M0"], params["D0_m2_ms"])


def _nogse_contrast_free_grad_offset(td_ms: float, G: np.ndarray, n_value: int, params: dict[str, float]) -> np.ndarray:
    return NOGSE_contrast_vs_g_free_grad_offset(td_ms, G, n_value, params["M0"], params["D0_m2_ms"], params["g0_mTm"])


def _nogse_contrast_tort(td_ms: float, G: np.ndarray, n_value: int, params: dict[str, float]) -> np.ndarray:
    return NOGSE_contrast_vs_g_tort(td_ms, G, n_value, params["alpha"], params["M0"], params["D0_m2_ms"])


def _nogse_contrast_rest(td_ms: float, G: np.ndarray, n_value: int, params: dict[str, float]) -> np.ndarray:
    return NOGSE_contrast_vs_g_rest(td_ms, G, n_value, params["tc_ms"], params["M0"], params["D0_m2_ms"])


CONTRAST_MODEL_REGISTRY: dict[str, ContrastModelSpec] = {
    "ogse_free": _contrast_spec("ogse_free", "ogse", _params("M0", "D0_m2_ms"), _ogse_contrast_free),
    "ogse_tort": _contrast_spec("ogse_tort", "ogse", _params("alpha", "M0", "D0_m2_ms"), _ogse_contrast_tort),
    "ogse_rest": _contrast_spec("ogse_rest", "ogse", _params("tc_ms", "M0", "D0_m2_ms"), _ogse_contrast_rest),
    "ogse_rest_offset": _contrast_spec("ogse_rest_offset", "ogse", _params("tc_ms", "M0", "D0_m2_ms", "C"), _ogse_contrast_rest_offset),
    "ogse_mixed": _contrast_spec("ogse_mixed", "ogse", _params("tc_ms", "alpha", "M0", "D0_m2_ms"), _ogse_contrast_mixed),
    "nogse_free": _contrast_spec("nogse_free", "nogse", _params("M0", "D0_m2_ms"), _nogse_contrast_free),
    "nogse_free_grad_offset": _contrast_spec(
        "nogse_free_grad_offset",
        "nogse",
        _params("M0", "D0_m2_ms", "g0_mTm"),
        _nogse_contrast_free_grad_offset,
    ),
    "nogse_tort": _contrast_spec("nogse_tort", "nogse", _params("alpha", "M0", "D0_m2_ms"), _nogse_contrast_tort),
    "nogse_rest": _contrast_spec("nogse_rest", "nogse", _params("tc_ms", "M0", "D0_m2_ms"), _nogse_contrast_rest),
}


SIGNAL_MODEL_ALIASES: dict[tuple[str, str], str] = {
    ("ogse", "free_ogse"): "ogse_free",
    ("ogse", "free"): "ogse_free",
    ("ogse", "rest"): "ogse_rest",
    ("ogse", "rest_offset"): "ogse_rest_offset",
    # NOGSE prefixed names (canonical)
    ("nogse", "nogse_free_cpmg"): "nogse_free",
    ("nogse", "nogse_free_hahn"): "nogse_free",
    ("nogse", "nogse_mixed_global"): "nogse_mixed",
    ("nogse", "free"): "nogse_free",
    ("nogse", "rest"): "nogse_rest",
    # backward-compat aliases
    ("nogse", "free_cpmg"): "nogse_free",
    ("nogse", "free_hahn"): "nogse_free",
    ("nogse", "mixed_global"): "nogse_mixed",
}


CONTRAST_MODEL_ALIASES: dict[tuple[str, str], str] = {
    ("ogse", "free"): "ogse_free",
    ("ogse", "tort"): "ogse_tort",
    ("ogse", "rest"): "ogse_rest",
    ("ogse", "rest_offset"): "ogse_rest_offset",
    ("ogse", "rest_offset_globC"): "ogse_rest_offset",
    ("ogse", "mixed"): "ogse_mixed",
    ("ogse", "ogse_mixed_global"): "ogse_mixed",
    ("nogse", "free"): "nogse_free",
    ("nogse", "tort"): "nogse_tort",
    ("nogse", "rest"): "nogse_rest",
    # backward-compat alias
    ("ogse", "mixed_global"): "ogse_mixed",
}


def canonical_signal_model_name(name: str, *, family: str | None = None) -> str:
    key = str(name)
    if key in SIGNAL_MODEL_REGISTRY:
        return key
    if family is not None and (str(family), key) in SIGNAL_MODEL_ALIASES:
        return SIGNAL_MODEL_ALIASES[(str(family), key)]
    matches = [target for (alias_family, alias), target in SIGNAL_MODEL_ALIASES.items() if alias == key and (family is None or alias_family == str(family))]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Unsupported signal model {name!r}. Allowed values: {sorted(SIGNAL_MODEL_REGISTRY)}.")


def canonical_contrast_model_name(name: str, *, family: str | None = None) -> str:
    key = str(name)
    if key in CONTRAST_MODEL_REGISTRY:
        return key
    if family is not None and (str(family), key) in CONTRAST_MODEL_ALIASES:
        return CONTRAST_MODEL_ALIASES[(str(family), key)]
    matches = [target for (alias_family, alias), target in CONTRAST_MODEL_ALIASES.items() if alias == key and (family is None or alias_family == str(family))]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Unsupported contrast model {name!r}. Allowed values: {sorted(CONTRAST_MODEL_REGISTRY)}.")


def get_signal_model(name: str) -> SignalModelSpec:
    key = canonical_signal_model_name(name)
    try:
        return SIGNAL_MODEL_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported signal model {name!r}. Allowed values: {sorted(SIGNAL_MODEL_REGISTRY)}.") from exc


def signal_model_names(*, family: str | None = None) -> tuple[str, ...]:
    if family is None:
        return tuple(sorted(SIGNAL_MODEL_REGISTRY))
    family_key = str(family)
    return tuple(sorted(name for name, spec in SIGNAL_MODEL_REGISTRY.items() if spec.family == family_key))


def get_contrast_model(name: str, *, family: str | None = None) -> ContrastModelSpec:
    key = canonical_contrast_model_name(name, family=family)
    try:
        return CONTRAST_MODEL_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported contrast model {name!r}. Allowed values: {sorted(CONTRAST_MODEL_REGISTRY)}.") from exc


def contrast_model_names(*, family: str | None = None) -> tuple[str, ...]:
    if family is None:
        return tuple(sorted(CONTRAST_MODEL_REGISTRY))
    family_key = str(family)
    return tuple(sorted(name for name, spec in CONTRAST_MODEL_REGISTRY.items() if spec.family == family_key))


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
