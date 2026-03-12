from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _read_all_fit_params(root: Path) -> pd.DataFrame:
    files = sorted(root.rglob("fit_params.csv"))
    if not files:
        raise SystemExit(f"No encontré fit_params.csv debajo de: {root}")

    frames = []
    for fp in files:
        df = pd.read_csv(fp)

        # Normalizar D0 col
        if "D0_mm2_s" not in df.columns:
            d0c = _pick_col(df, ["D0_mm2_s", "D0", "D0_mm2_per_s"])
            if d0c is None:
                continue
            df = df.rename(columns={d0c: "D0_mm2_s"})

        # Normalizar max_dur col
        if "max_dur_ms" not in df.columns:
            mdc = _pick_col(df, ["max_dur_ms", "max_dur", "max_duration", "meta_max_dur", "param_max_dur"])
            if mdc is not None:
                df = df.rename(columns={mdc: "max_dur_ms"})

        # trazabilidad
        df["fit_params_file"] = str(fp)
        df["fit_folder"] = fp.parent.name

        frames.append(df)

    if not frames:
        raise SystemExit("Encontré fit_params.csv pero ninguno tenía columna D0 usable.")

    out = pd.concat(frames, ignore_index=True)
    return out


def _relevant_cols(df: pd.DataFrame) -> list[str]:
    """
    “Relevantes” = lo típico de tu fit_signal_vs_bval.py:
      direction, roi, g_type, fit_points, M0, M0_err, D0_mm2_s, D0_err, delta_ms, delta_app_ms, N, max_dur_ms
    y además cualquier param_* / meta_* si existen.
    """
    preferred = [
        "max_dur_ms",
        "roi",
        "direction",
        "axis",
        "g_type",
        "fit_points",
        "N",
        "delta_ms",
        "delta_app_ms",
        "M0",
        "M0_err",
        "D0_mm2_s",
        "D0_err_mm2_s",
        "D0_err",
        "stat",
        "fit_folder",
        "fit_params_file",
    ]
    cols = [c for c in preferred if c in df.columns]

    # sumá param_*/meta_* si existen
    extra = [c for c in df.columns if (c.startswith("param_") or c.startswith("meta_")) and c not in cols]
    cols.extend(sorted(extra))

    # finalmente, agrega el resto (por si querés todo)
    rest = [c for c in df.columns if c not in cols]
    cols.extend(sorted(rest))

    return cols


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Junta D0 de todos los fit_params.csv, y hace promedio por max_dur + plot D0 vs td."
    )
    ap.add_argument("--root", type=Path, default=None, help="root directory")
    ap.add_argument("--roi", type=str, default="Agua", help="ROI a filtrar (ALL para no filtrar).")
    ap.add_argument("--out_root", type=Path, default=None, help="out_root directory")
    ap.add_argument("--tm-ms", type=float, default=None, help="tm (ms). Si no lo pasás, lo pedimos por teclado.")
    ap.add_argument("--use-std", action="store_true", help="Barras de error = STD (default: SEM).")
    args = ap.parse_args()

    root = args.root
    out = args.out_root or (root / "D0_summary")
    out.mkdir(parents=True, exist_ok=True)

    df = _read_all_fit_params(root)

    # limpio tipos
    df["D0_mm2_s"] = pd.to_numeric(df["D0_mm2_s"], errors="coerce")
    if "max_dur_ms" in df.columns:
        df["max_dur_ms"] = pd.to_numeric(df["max_dur_ms"], errors="coerce")

    df = df[np.isfinite(df["D0_mm2_s"])].copy()
    if "max_dur_ms" not in df.columns:
        raise SystemExit("No encontré columna max_dur_ms en los fit_params.csv (no puedo ordenar/agrupaar).")
    df = df[np.isfinite(df["max_dur_ms"])].copy()

    # filtro ROI (agua)
    if args.roi.upper() != "ALL" and "roi" in df.columns:
        df = df[df["roi"].astype(str) == str(args.roi)].copy()

    if df.empty:
        raise SystemExit("No quedaron filas luego del filtro (¿ROI correcto?).")

    # TABLA A: todas las mediciones ordenadas por max_dur
    df_all = df[_relevant_cols(df)].sort_values(["max_dur_ms"], kind="stable").reset_index(drop=True)
    df_all.to_csv(out / "D0_all_measurements.csv", index=False)
    df_all.to_excel(out / "D0_all_measurements.xlsx", index=False)

    # TABLA B: promedio por max_dur
    g = df.groupby("max_dur_ms", as_index=False)
    summary = g["D0_mm2_s"].agg(["mean", "std", "count"]).reset_index()
    summary = summary.rename(columns={"mean": "D0_mean_mm2_s", "std": "D0_std_mm2_s", "count": "n"})
    summary["D0_sem_mm2_s"] = summary["D0_std_mm2_s"] / np.sqrt(summary["n"].clip(lower=1))
    summary = summary.sort_values("max_dur_ms", kind="stable").reset_index(drop=True)

    # tm: por teclado si no vino por argumento
    tm_ms = args.tm_ms
    if tm_ms is None:
        tm_ms = float(input("Ingresá tm (ms): ").strip())

    # td y plot
    summary["tm_ms"] = float(tm_ms)
    summary["td_ms"] = 2.0 * summary["max_dur_ms"] + float(tm_ms)

    summary.to_csv(out / "D0_mean_by_maxdur.csv", index=False)
    summary.to_excel(out / "D0_mean_by_maxdur.xlsx", index=False)

    x = summary["td_ms"].to_numpy(float)
    y = summary["D0_mean_mm2_s"].to_numpy(float)
    yerr = summary["D0_std_mm2_s"].to_numpy(float) if args.use_std else summary["D0_sem_mm2_s"].to_numpy(float)

    plt.figure(figsize=(7.5, 5.5))
    plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3)
    plt.xlabel("td_ms = 2*max_dur + tm")
    plt.ylabel("D0 (mm²/s)")
    plt.title(f"D0 vs td (ROI={args.roi}, tm={tm_ms} ms)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "D0_vs_td.png", dpi=200)
    plt.close()

    print(f"[OK] Tabla A (all):     {out / 'D0_all_measurements.xlsx'}")
    print(f"[OK] Tabla B (mean):    {out / 'D0_mean_by_maxdur.xlsx'}")
    print(f"[OK] Plot D0 vs td:     {out / 'D0_vs_td.png'}")


if __name__ == "__main__":
    main()
