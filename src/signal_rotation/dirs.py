from __future__ import annotations
from pathlib import Path
import numpy as np


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def repo_root_from_src_layout() -> Path:
    # .../repo/src/nogse_table_tools/dirs.py -> parents[2] = repo/
    return Path(__file__).resolve().parents[2]


def load_dirs_csv(path: str | Path) -> np.ndarray:
    arr = np.loadtxt(Path(path), delimiter=",", dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"CSV inválido: esperaba Nx3, obtuve {arr.shape}")
    return _normalize_rows(arr)


def load_default_dirs(ndirs: int) -> np.ndarray:
    p = repo_root_from_src_layout() / "assets" / "dirs" / f"dirs_{ndirs}.csv"
    if not p.is_file():
        raise FileNotFoundError(
            f"No encontré {p}. Para ndirs={ndirs} tenés que crear ese CSV "
            f"(una fila por direction, en el orden direction=1..{ndirs})."
        )
    dirs = load_dirs_csv(p)
    if dirs.shape[0] != ndirs:
        raise ValueError(f"{p} tiene {dirs.shape[0]} filas, pero ndirs={ndirs}.")
    return dirs
