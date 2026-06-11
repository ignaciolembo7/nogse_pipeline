from __future__ import annotations

from typing import Sequence

import numpy as np


MODEL_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "free": ("M0", "D0_m2_ms"),
    "tort": ("alpha", "M0", "D0_m2_ms"),
    "rest": ("tc_ms", "M0", "D0_m2_ms"),
    "rest_offset": ("tc_ms", "M0", "D0_m2_ms", "C"),
    "rest_offset_globC": ("tc_ms", "M0", "D0_m2_ms", "C"),
    "mixed": ("tc_ms", "alpha", "M0", "D0_m2_ms"),
}


PARAM_ALIASES = {
    "D0": "D0_m2_ms",
    "D0_m2_ms": "D0_m2_ms",
    "D0_mm2_s": "D0_m2_ms",
    "M0": "M0",
    "tc": "tc_ms",
    "tc_ms": "tc_ms",
    "alpha": "alpha",
    "C": "C",
}


def normalize_global_contrast_params(model: str, values: Sequence[str] | None) -> list[str]:
    """Return the physical parameters shared across curves in a global contrast fit."""
    if not values:
        return []
    if len(values) == 1 and str(values[0]).upper() in {"NONE", "LOCAL"}:
        return []

    allowed = set(MODEL_PARAM_NAMES.get(str(model), ()))
    out: list[str] = []
    for item in values:
        for token in str(item).replace(",", " ").split():
            if not token:
                continue
            resolved = PARAM_ALIASES.get(token, PARAM_ALIASES.get(token.strip(), None))
            if resolved is None:
                raise ValueError(f"Unknown global parameter {token!r}. Allowed aliases: {sorted(PARAM_ALIASES)}.")
            if resolved not in allowed:
                raise ValueError(f"Parameter {resolved!r} is not available for model={model!r}. Allowed: {sorted(allowed)}.")
            if resolved not in out:
                out.append(resolved)
    return out


def seed_bounds_for_contrast_parameter(
    name: str,
    *,
    M0_value: float,
    D0_value: float,
    C_value: float,
    tc_value: float,
    tc_bounds: tuple[float, float] | None,
    m0_bounds: tuple[float, float] | None,
    d0_bounds: tuple[float, float] | None,
    c_bounds: tuple[float, float] | None,
) -> tuple[float, float, float]:
    """Return initial value and bounds for one physical contrast-fit parameter."""
    if name == "M0":
        lo, hi = (0.0, 5.0) if m0_bounds is None else (float(m0_bounds[0]), float(m0_bounds[1]))
        return float(np.clip(float(M0_value), lo, hi)), lo, hi
    if name == "D0_m2_ms":
        lo, hi = (float(D0_value / 100.0), float(D0_value * 100.0)) if d0_bounds is None else (
            float(d0_bounds[0]),
            float(d0_bounds[1]),
        )
        return float(np.clip(float(D0_value), lo, hi)), lo, hi
    if name == "tc_ms":
        lo, hi = (0.1, 1000.0) if tc_bounds is None else (float(tc_bounds[0]), float(tc_bounds[1]))
        return float(np.clip(float(tc_value), lo, hi)), lo, hi
    if name == "alpha":
        return 0.5, 0.0, 1.0
    if name == "C":
        lo, hi = (0.0, 1.0) if c_bounds is None else (float(c_bounds[0]), float(c_bounds[1]))
        return float(np.clip(float(C_value), lo, hi)), lo, hi
    raise ValueError(f"Unsupported fit parameter {name!r}.")


def contrast_parameter_vary_flags(*, M0_vary: bool, D0_vary: bool, C_vary: bool, tc_vary: bool) -> dict[str, bool]:
    """Return which local contrast parameters may vary curve by curve."""
    return {
        "M0": bool(M0_vary),
        "D0_m2_ms": bool(D0_vary),
        "tc_ms": bool(tc_vary),
        "C": bool(C_vary),
        "alpha": True,
    }


def fixed_contrast_parameter_values(*, M0_value: float, D0_value: float, C_value: float, tc_value: float) -> dict[str, float]:
    """Return fallback fixed values for contrast parameters that are not fitted."""
    return {
        "M0": float(M0_value),
        "D0_m2_ms": float(D0_value),
        "tc_ms": float(tc_value),
        "C": float(C_value),
        "alpha": 0.5,
    }
