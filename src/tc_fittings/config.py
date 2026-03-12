from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

def _default_experiments() -> Dict[str, List[float]]:
    return {
        "20220622_BRAIN": [40],
        "20230619_BRAIN-3": [55, 66.7],
        "20230623_BRAIN-4": [33, 100],
        "20230623_LUDG-2": [55, 66.7],
        "20230629_MBBL-2": [55, 66.7],
        "20230630_MBBL-3": [40, 100],
        "20230710_LUDG-3": [40, 100],
    }

def _default_regions() -> List[str]:
    return ["PostCC_norm", "MidPostCC_norm", "CentralCC_norm", "MidAntCC_norm", "AntCC_norm"]

def _default_palette() -> List[str]:
    return [
        "#a65628",
        "#e41a1c",
        "#ff7f00",
        "#984ea3",
        "#377eb8",
        "#999999",
    ]

@dataclass
class OGSEFitConfig:
    # Data selection
    experiments: Dict[str, List[float]] = field(default_factory=_default_experiments)
    regions: List[str] = field(default_factory=_default_regions)
    dirs: List[str] = field(default_factory=lambda: ["long", "tra"])

    # Contrast file naming
    N1: int = 8
    N2: int = 4
    method: str = "rot_tensor-solve"
    g_type_fit: str = "gthorsten"

    # Model selection
    fit_model: str = "rest"  # matches nogse_model_fitting.OGSE_contrast_vs_g_rest

    # Timing / constants
    TM: float = 10.0  # ms
    M0_value: float = 1.0
    tc_value: float = 5.0
    D0_value: float = 3.2e-12
    D0_fix: float = 3.2e-12  # m^2/s (used in length conversions)
    gamma: float = 267.5221900  # 1/ms.mT

    # Fit degrees of freedom
    D0_vary: bool = True
    tc_vary: bool = True
    M0_vary: bool = False

    # I/O
    correction_factors_xlsx: Path = Path("factores_correccion_gradiente.xlsx")
    plot_dir_out: Optional[Path] = None

    # Plot styling
    palette: List[str] = field(default_factory=_default_palette)

    # Optional steps that depend on external summary files
    run_alpha_macro_plots_if_available: bool = True

    def __post_init__(self) -> None:
        if self.plot_dir_out is None:
            self.plot_dir_out = Path(
                f"analysis/fit_OGSE_contrast_corr/{self.method}/"
                f"globalfit_OGSE-contrast_vs_g_{self.fit_model}_nonpar_v2"
            )
