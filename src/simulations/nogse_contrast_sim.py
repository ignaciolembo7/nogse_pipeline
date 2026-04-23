from __future__ import annotations

import importlib
import importlib.util
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


def _import_make_contrast():
    try:
        from fitting.contrast import make_contrast  # type: ignore
        return make_contrast
    except Exception as exc:
        raise ImportError("Could not import fitting.contrast.make_contrast.") from exc


def load_module(module_name: str, module_path: str | None = None):
    """
    Load a python module either by import name (module_name) or from a file path (module_path).

    Robust behavior for this repo:
      - If module_name is "nogse_model_fitting", try "nogse_models.nogse_model_fitting"
      - If import fails, try to locate "<name>.py" under repo_root/src/nogse_models/
    """
    if module_path:
        p = Path(module_path)
        if not p.exists():
            raise FileNotFoundError(f"Model module path does not exist: {p}")
        spec = importlib.util.spec_from_file_location(module_name, str(p))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from path: {p}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        # --- 1) Shorthand mapping for this repo
        candidates: list[str] = []
        if module_name == "nogse_model_fitting":
            candidates.append("nogse_models.nogse_model_fitting")
        elif "." not in module_name:
            candidates.append(f"nogse_models.{module_name}")

        for c in candidates:
            try:
                return importlib.import_module(c)
            except ModuleNotFoundError:
                pass

        # --- 2) Try file lookup inside the repo
        repo_root = Path(__file__).resolve().parents[2]  # .../repo/src/simulations/file.py -> parents[2] == repo
        base = module_name.split(".")[-1]

        file_candidates = [
            repo_root / f"{base}.py",
            repo_root / "src" / f"{base}.py",
            repo_root / "src" / "nogse_models" / f"{base}.py",
        ]

        for p in file_candidates:
            if p.exists():
                spec = importlib.util.spec_from_file_location(base, str(p))
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                return mod

        raise e



def list_models(mod) -> list[str]:
    """
    List candidate model functions in nogse_model_fitting.py.
    Default: show contrast models (NOGSE/OGSE/PGSE) but you can still call any function by name.
    """
    names = []
    for name, obj in vars(mod).items():
        if callable(obj) and not name.startswith("_"):
            if name.startswith(("NOGSE_contrast_", "OGSE_contrast_", "PGSE_")):
                names.append(name)
    return sorted(names)


def parse_grid(spec: str) -> np.ndarray:
    """
    Parse grid specification:
      - "0,5,10" -> array([0,5,10])
      - "0:80:41" -> linspace(start=0, stop=80, num=41)
    """
    s = spec.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad grid spec '{spec}'. Use 'a:b:n' or 'v1,v2,v3'.")
        a = float(parts[0])
        b = float(parts[1])
        n = int(float(parts[2]))
        if n < 2:
            raise ValueError("Grid num points must be >= 2 for a:b:n form.")
        return np.linspace(a, b, n)
    # CSV
    vals = [v.strip() for v in s.split(",") if v.strip() != ""]
    if not vals:
        raise ValueError(f"Empty grid spec: '{spec}'")
    return np.array([float(v) for v in vals], dtype=float)


def parse_kv(item: str) -> tuple[str, Any]:
    """
    Parse "key=value" with basic type inference (int/float/bool/str).
    """
    if "=" not in item:
        raise ValueError(f"Expected key=value, got '{item}'")
    k, v = item.split("=", 1)
    k = k.strip()
    v = v.strip()

    # bool
    if v.lower() in ("true", "false"):
        return k, v.lower() == "true"

    # int
    try:
        if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
            return k, int(v)
    except Exception:
        pass

    # float (supports scientific notation)
    try:
        return k, float(v)
    except Exception:
        return k, v


def _split_kwargs_side(model_kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Heuristic split of kwargs into:
      - shared: applied to both sides (param_<k>)
      - side1: keys ending with '1' (e.g., G1, N1) -> param_<base>
      - side2: keys ending with '2' (e.g., G2, N2) -> param_<base>
    """
    shared: dict[str, Any] = {}
    side1: dict[str, Any] = {}
    side2: dict[str, Any] = {}

    for k, v in model_kwargs.items():
        if k.endswith("1") and len(k) > 1:
            side1[k[:-1]] = v
        elif k.endswith("2") and len(k) > 1:
            side2[k[:-1]] = v
        else:
            shared[k] = v
    return shared, side1, side2


@dataclass(frozen=True)
class SimSpec:
    model_name: str
    model_module: str = "nogse_models.nogse_model_fitting"
    model_module_path: str | None = None

    # Grid injection (e.g. G, Lc, bvalue, G1/G2, etc.)
    grid_name: str = "G"
    grid_spec: str = "0:80:41"
    grid2_name: str | None = None
    grid2_spec: str | None = None

    # Table meta
    out_path: str = "OGSE_signal/contrast/simulated/sim.contrast.long.parquet"
    axes: tuple[str, ...] = ("long", "tra")
    rois: tuple[str, ...] = ("SIM_ROI",)
    stat: str = "mean"

    # Key columns used by make_contrast merge
    key_cols: tuple[str, ...] = ("stat", "roi", "direction", "b_step")

    # CLI-provided params
    model_kwargs: dict[str, Any] = None  # type: ignore[assignment]
    meta_params: dict[str, Any] = None   # type: ignore[assignment]


def simulate_contrast_long(spec: SimSpec) -> pd.DataFrame:
    make_contrast = _import_make_contrast()

    mod = load_module(spec.model_module, spec.model_module_path)
    if not hasattr(mod, spec.model_name):
        raise AttributeError(f"Model '{spec.model_name}' not found in module '{spec.model_module}'")

    fn: Callable[..., Any] = getattr(mod, spec.model_name)
    sig = inspect.signature(fn)

    grid = parse_grid(spec.grid_spec)

    # optional second grid (for OGSE where you might want G1 and G2)
    if spec.grid2_name and spec.grid2_spec:
        grid2 = parse_grid(spec.grid2_spec)
        if len(grid2) != len(grid):
            raise ValueError("grid2 must have same length as grid (same number of b_steps).")
    else:
        grid2 = None

    model_kwargs = dict(spec.model_kwargs or {})
    meta_params = dict(spec.meta_params or {})

    # Inject grid(s) into kwargs for model call
    model_kwargs[spec.grid_name] = grid
    if spec.grid2_name:
        model_kwargs[spec.grid2_name] = grid2 if grid2 is not None else grid  # default: copy grid

    # Validate required args (except those we inject)
    missing = []
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is p.empty and p.name not in model_kwargs:
            missing.append(p.name)
    if missing:
        raise TypeError(
            f"Missing required model kwargs for {spec.model_name}: {missing}\n"
            f"Provided: {sorted(model_kwargs.keys())}"
        )

    # Compute contrast curve (vectorized)
    contrast = fn(**model_kwargs)
    contrast = np.asarray(contrast, dtype=float)
    if contrast.shape[0] != grid.shape[0]:
        raise ValueError(
            f"Model output length {contrast.shape[0]} does not match grid length {grid.shape[0]}."
        )

    # Build "fake" ref/cmp signal tables so make_contrast produces the canonical output format.
    # ref has value=contrast, cmp has value=0 => contrast out is exactly the model curve.
    shared, side1, side2 = _split_kwargs_side({k: v for k, v in model_kwargs.items() if k not in (spec.grid_name, spec.grid2_name)})

    rows_ref = []
    rows_cmp = []

    for roi in spec.rois:
        for direction in spec.axes:
            for i, gval in enumerate(grid):
                base = {
                    "stat": spec.stat,
                    "roi": roi,
                    "direction": direction,
                    "b_step": int(i),
                }

                # gradient columns expected downstream (contrast.py will suffix _1/_2)
                ref = dict(base)
                cmp = dict(base)

                # side 1 gradient
                ref["g"] = float(gval)
                ref["g_lin_max"] = float(gval)
                ref["g_thorsten"] = float(gval)

                # side 2 gradient (if there is a second grid name, use that)
                g2val = float(grid2[i]) if (grid2 is not None) else float(gval)
                cmp["g"] = g2val
                cmp["g_lin_max"] = g2val
                cmp["g_thorsten"] = g2val

                # Param columns (carried through by make_contrast if they start with "param_")
                # shared -> both sides
                for k, v in shared.items():
                    ref[f"param_{k}"] = v
                    cmp[f"param_{k}"] = v
                # side1/side2 -> store base name (without trailing 1/2) separately on each side
                for k, v in side1.items():
                    ref[f"param_{k}"] = v
                for k, v in side2.items():
                    cmp[f"param_{k}"] = v
                # meta params -> both sides
                for k, v in meta_params.items():
                    ref[f"param_{k}"] = v
                    cmp[f"param_{k}"] = v

                # "signal" columns
                ref["value"] = float(contrast[i])
                ref["value_norm"] = float(contrast[i])
                ref["S0"] = 1.0

                cmp["value"] = 0.0
                cmp["value_norm"] = 0.0
                cmp["S0"] = 1.0

                rows_ref.append(ref)
                rows_cmp.append(cmp)

    df_ref = pd.DataFrame(rows_ref)
    df_cmp = pd.DataFrame(rows_cmp)

    # Canonical contrast format
    result = make_contrast(
        df_ref,
        df_cmp,
        axes=spec.axes,
        y_col="value",
        y_norm_col="value_norm",
        key_cols=spec.key_cols,
    ).df

    return result
