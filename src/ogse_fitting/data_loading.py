from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from .config import OGSEFitConfig

def format_d_folder(d: float) -> str:
    # Notebook convention: 66.7 -> '66p7', integers as '40', '55', '100', etc.
    if abs(d - 66.7) < 1e-6:
        return "66p7"
    if float(d).is_integer():
        return str(int(d))
    # fallback: keep one decimal (matches notebook keys like d:.1f)
    return f"{d:g}"

def td_ms(d: float, TM: float) -> float:
    return 2.0 * float(d) + float(TM)

def direction_label_for_f(direction: str) -> str:
    return "longitudinal" if direction == "long" else "transversal"

def get_f_value(tabla_f: pd.DataFrame, name: str, td: float, direction: str) -> float:
    direccion_f = direction_label_for_f(direction)
    row_f = tabla_f[
        (tabla_f["Archivo_origen"] == name) &
        (tabla_f["td_ms"] == td) &
        (tabla_f["direction"] == direccion_f)
    ]
    if not row_f.empty:
        return float(row_f["factor correccion (f)"].values[0])
    print(f"[WARN] No se encontró factor para {name}, {direccion_f}, Td={td} ms. Se usará f=1")
    return 1.0

def load_correction_table(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)

def load_contrast_xlsx(path: Path, g_type_fit: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_excel(path)
    g1 = df[f"{g_type_fit}_1"].to_numpy()
    g2 = df[f"{g_type_fit}_2"].to_numpy()
    # Notebook: sqrt(2*g^2)
    g1 = np.sqrt(2.0 * g1**2)
    g2 = np.sqrt(2.0 * g2**2)
    return g1, g2, df

def load_all_curves(cfg: OGSEFitConfig, tabla_f: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    all_curves: Dict[str, pd.DataFrame] = {}
    for name, d_list in cfg.experiments.items():
        for d in d_list:
            d_float = float(d)
            d_folder = format_d_folder(d_float)
            td = td_ms(d_float, cfg.TM)

            base = Path(f"../analysis/OGSE_contrast/{cfg.method}/{name}/contrast_N{cfg.N_1}-N{cfg.N_2}_d={d_folder}/")
            for direction in cfg.dirs:
                fname = f"OGSE_contrast_N{cfg.N_1}-N{cfg.N_2}_dir={direction}_d={d_folder}_{cfg.method}.xlsx"
                fpath = base / fname
                if not fpath.exists():
                    print(f"⚠️ No encontrado: {fpath}")
                    continue

                g1, g2, df = load_contrast_xlsx(fpath, cfg.g_type_fit)

                for region in cfg.regions:
                    if region not in df.columns:
                        print(f"⚠️ roi {region} no encontrada en {fname}")
                        continue
                    signal = df[region].to_numpy()
                    key = f"{name}|{region}|{d_float:.1f}|{direction}"
                    all_curves[key] = pd.DataFrame({"g1": g1, "g2": g2, "signal": signal})
    return all_curves
