from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from ogse_fitting.config import OGSEFitConfig
from tc_fittings.tc_td_registry import METHODS
from tc_fittings.tc_td_pseudohuber import load_alpha_macro_summary

def _ensure_brain_col(df: pd.DataFrame, globalfit_path: Path) -> pd.DataFrame:
    df = df.copy()
    tag = globalfit_path.stem  # fallback

    if "source_file" in df.columns:
        # mejor: identificar por archivo de origen si está
        df["brain"] = df["source_file"].astype(str).apply(lambda s: Path(s).stem)
        return df

    if "brain" not in df.columns:
        df["brain"] = tag
        return df

    # existe, pero puede venir vacío/NaN
    b = df["brain"].astype(str)
    b = b.fillna(tag)
    b = b.replace({"nan": tag, "None": tag, "": tag})
    df["brain"] = b
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(METHODS.keys()))
    ap.add_argument("--k-last", type=int, default=None, help="Usar últimos K puntos (default: método).")
    ap.add_argument("--globalfit", default=None, help="Ruta al globalfit xlsx. Si no, usa la default de cfg.")
    args = ap.parse_args()

    cfg = OGSEFitConfig()

    globalfit_path = Path(args.globalfit) if args.globalfit else (
        cfg.plot_dir_out / "globalfits" / f"globalfit_{cfg.fit_model}_{cfg.method}.xlsx"
    )
    if not globalfit_path.exists():
        raise FileNotFoundError(f"No encuentro globalfit: {globalfit_path}")

    df_params = pd.read_excel(globalfit_path)
    df_params["roi"] = df_params["roi"].astype(str).str.replace("_norm","", regex=False)

    # ✅ make it generic: brain siempre existe y no es NaN
    df_params = _ensure_brain_col(df_params, globalfit_path)

    spec = METHODS[args.method]
    k_last = args.k_last if args.k_last is not None else spec.default_k_last

    alpha_macro_df = None
    summary_path = cfg.plot_dir_out / "summary_alpha_values.xlsx"
    if summary_path.exists():
        alpha_macro_df = load_alpha_macro_summary(summary_path)

    if spec.needs_alpha_macro and alpha_macro_df is None:
        raise FileNotFoundError(f"{args.method} requiere summary_alpha_values.xlsx en: {summary_path}")

    out_dir = cfg.plot_dir_out / "tc_vs_td" / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    spec.func(cfg, df_params, out_dir, k_last, alpha_macro_df)

if __name__ == "__main__":
    main()