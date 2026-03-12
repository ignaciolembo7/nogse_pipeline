from __future__ import annotations
import os
import numpy as np
import pandas as pd
import lmfit
from nogse_models import nogse_model_fitting

from .config import OGSEFitConfig
from .data_loading import get_f_value, td_ms

def brain_from_experiment(name: str) -> str:
    if name in ["20220622_BRAIN", "20230619_BRAIN-3", "20230623_BRAIN-4"]:
        return "BRAIN"
    if name in ["20230623_LUDG-2", "20230710_LUDG-3"]:
        return "LUDG"
    if name in ["20230629_MBBL-2", "20230630_MBBL-3"]:
        return "MBBL"
    return "Unknown"

def get_model_func(cfg: OGSEFitConfig):
    # Central place to swap models.
    if cfg.fit_model == "rest":
        return nogse_model_fitting.OGSE_contrast_vs_g_rest
    raise ValueError(f"Unknown fit_model='{cfg.fit_model}'. Add it in get_model_func().")

def fit_global(cfg: OGSEFitConfig, all_curves: dict[str, pd.DataFrame], tabla_f: pd.DataFrame) -> pd.DataFrame:
    """
    FIT POR CURVA (rápido): equivalente a global fit cuando NO compartís parámetros entre curvas.
    """
    param_rows = []
    model_func = get_model_func(cfg)

    keys = sorted(all_curves.keys())
    if len(keys) == 0:
        raise FileNotFoundError("No se cargó ninguna curva (all_curves vacío). Revisá paths y method.")

    print(f"✅ Curvas a ajustar: {len(keys)}")

    for idx, key in enumerate(keys, 1):
        name, region, d_str, direction = key.split("|")
        d = float(d_str)
        td = td_ms(d, cfg.TM)
        brain = brain_from_experiment(name)

        f_valor = get_f_value(tabla_f, name, td, direction)

        df = all_curves[key]
        G1 = df["g1"].values * f_valor
        G2 = df["g2"].values * f_valor
        y  = df["signal"].values

        # Modelo por curva (TE, N1, N2 fijos en closure)
        def curve_model(G1, G2, tc, M0, D0):
            return model_func(td, G1, G2, cfg.N1, cfg.N2, tc, M0, D0)

        model = lmfit.Model(curve_model, independent_vars=["G1", "G2"])
        params = lmfit.Parameters()
        params.add("D0", value=cfg.D0_value, min=cfg.D0_value*0.01, max=cfg.D0_value*10, vary=cfg.D0_vary)
        params.add("tc", value=cfg.tc_value, min=0.1, max=1000.0, vary=cfg.tc_vary)
        params.add("M0", value=cfg.M0_value, min=0.0, max=2.0, vary=cfg.M0_vary)

        # Fit (normal). Si después querés Huber acá, te digo abajo cómo.
        result = model.fit(y, params, G1=G1, G2=G2)

        if idx % 10 == 0 or idx == 1 or idx == len(keys):
            print(f"[{idx}/{len(keys)}] {name} | {region} | d={d} | {direction} -> tc={result.params['tc'].value:.3f}")

        row = {
            "brain": brain,
            "Archivo_origen": name,
            "roi": region,
            "d": float(d),
            "td_ms": td,
            "direction": direction,
            "f": float(f_valor),
            "M0_fit": float(result.params["M0"].value),
            "M0_error": result.params["M0"].stderr,
            "tc_fit": float(result.params["tc"].value),
            "tc_error": result.params["tc"].stderr,
            "D0_fit": float(result.params["D0"].value),
            "D0_error": result.params["D0"].stderr,
        }
        param_rows.append(row)

    df_params = pd.DataFrame(param_rows).sort_values(by=["direction", "Archivo_origen", "roi", "d"])

    out_dir = cfg.plot_dir_out / "globalfits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = out_dir / f"globalfit_{cfg.fit_model}_{cfg.method}.xlsx"
    df_params.to_excel(out_xlsx, index=False)
    print(f"✅ Resultados guardados en {out_xlsx}")

    return df_params
