from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class OGSEFitConfig:
    # dónde guardás TODO lo derivado
    plot_dir_out: Path = Path("plots")

    # naming para globalfit
    fit_model: str = "free"          # "free" | "tort" | "nonparam" | "restricted" (cuando lo implementes)
    method: str = "g_lin_max"        # o "gthorsten" etc.

    # defaults para tc_fittings
    regions: list[str] = (
        "GenuCC", "AntCC", "MidCC", "PostCC", "SpleniumCC"
    )
    N_1: int = 8
    N_2: int = 4
    palette: list[str] = (
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"
    )
