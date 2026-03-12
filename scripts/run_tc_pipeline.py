from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# --- ensure src/ on path ---
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nogse_models.nogse_model_fitting import OGSE_contrast_vs_g_rest  # noqa: E402


def _read_table(p: Path) -> pd.DataFrame:
    suf = p.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(p)
    if suf == ".csv":
        return pd.read_csv(p)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(p, sheet_name=0)
    raise ValueError(f"Formato no soportado: {p}")


def _canonical_sheet(sheet: str) -> str:
    s = str(sheet).strip()
    for token in ["_duration", "_dur", "_version", "_ver", "_v"]:
        idx = s.find(token)
        if idx > 0:
            return s[:idx]
    return s.split("_")[0]


def _infer_sheet(df: pd.DataFrame, fallback: str) -> str:
    for c in ["sheet_1", "sheet"]:
        if c in df.columns:
            u = pd.Series(df[c]).dropna().astype(str).unique()
            if len(u) == 1:
                return _canonical_sheet(u[0])
    name = str(fallback)
    for token in ["_N", "_td", "_Td", "_TE"]:
        i = name.find(token)
        if i > 0:
            return _canonical_sheet(name[:i])
    return _canonical_sheet(name.split("_")[0])


def _infer_exp_id_from_path(p: Path) -> str:
    """
    ID estable que NO cambie con td.
    Ej: OGSEvsMax_N8-N4_td75p1_Hz65-35_dir1-2-3_long -> OGSEvsMax_N8-N4_Hz65-35_dir1-2-3
    """
    stem = p.stem
    stem = stem.replace(".long", "")
    stem = re.sub(r"_long$", "", stem)
    stem = re.sub(r"_td\d+(?:p\d+)?", "", stem)  # quitar td75p1
    stem = re.sub(r"^([A-Za-z0-9]+)\1(?=_)", r"\1", stem)  # OGSEvsMaxOGSEvsMax_ -> OGSEvsMax_
    return stem


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_float_unique(series: pd.Series) -> Optional[float]:
    v = pd.to_numeric(series, errors="coerce").dropna().unique()
    if len(v) == 1:
        return float(v[0])
    return None


def _infer_td_ms(group: pd.DataFrame) -> Optional[float]:
    # prefer td_ms or td_ms_1
    for c in ["td_ms", "td_ms_1"]:
        if c in group.columns:
            td = _to_float_unique(group[c])
            if td is not None:
                return td
    # fallback: 2*max_dur + tm
    d = None
    tm = None
    for c in ["max_dur_ms", "max_dur_ms_1"]:
        if c in group.columns:
            d = _to_float_unique(group[c])
            if d is not None:
                break
    for c in ["tm_ms", "tm_ms_1"]:
        if c in group.columns:
            tm = _to_float_unique(group[c])
            if tm is not None:
                break
    if d is not None and tm is not None:
        return 2.0 * d + tm
    return None


@dataclass(frozen=True)
class CorrLookup:
    tol_ms: float
    table: dict[tuple[str, int, str], float]


def _load_corr_xlsx(path: Path, sheet_name: str, tol_ms: float) -> CorrLookup:
    df = pd.read_excel(path, sheet_name=sheet_name)
    needed = {"sheet", "td_ms", "direction", "correction_factor"}
    miss = needed - set(df.columns)
    if miss:
        raise KeyError(f"En {path} faltan columnas {sorted(miss)}. Tengo {list(df.columns)}")

    df = df.copy()
    df["sheet"] = df["sheet"].astype(str).map(_canonical_sheet)
    df["direction"] = df["direction"].astype(str)
    df["td_ms"] = pd.to_numeric(df["td_ms"], errors="coerce")
    df["correction_factor"] = pd.to_numeric(df["correction_factor"], errors="coerce")
    df = df.dropna(subset=["sheet", "td_ms", "direction", "correction_factor"])

    df["_td_key"] = np.round(df["td_ms"].astype(float) / float(tol_ms)).astype(int)

    table: dict[tuple[str, int, str], float] = {}
    for sheet, td_key, direction, cf in df[["sheet", "_td_key", "direction", "correction_factor"]].itertuples(
        index=False, name=None
    ):
        table[(str(sheet), int(td_key), str(direction))] = float(cf)

    return CorrLookup(tol_ms=float(tol_ms), table=table)


def _corr_factor(corr: CorrLookup, sheet: str, td_ms: float, direction: str) -> float:
    td_key = int(round(float(td_ms) / float(corr.tol_ms)))
    key = (_canonical_sheet(sheet), td_key, str(direction))
    # notebook: si no hay factor, usa 1.0
    return float(corr.table.get(key, 1.0))


def _fit_restricted_curvefit(
    td_ms: float,
    G1: np.ndarray,
    G2: np.ndarray,
    y: np.ndarray,
    *,
    N1: int,
    N2: int,
    tc_value: float,
    D0_value: float,
    M0_value: float,
    tc_vary: bool,
    D0_vary: bool,
    M0_vary: bool,
    tc_bounds: tuple[float, float],
    D0_bounds: tuple[float, float],
    M0_bounds: tuple[float, float],
):
    try:
        from scipy.optimize import curve_fit  # type: ignore
    except Exception as e:
        raise RuntimeError("Necesitás SciPy: pip install scipy") from e

    p0 = []
    lo = []
    hi = []

    def model_wrapper(_dummy, *params):
        it = iter(params)
        tc = tc_value if not tc_vary else next(it)
        M0 = M0_value if not M0_vary else next(it)
        D0 = D0_value if not D0_vary else next(it)
        return OGSE_contrast_vs_g_rest(td_ms, G1, G2, N1, N2, tc, M0, D0)

    if tc_vary:
        p0.append(float(tc_value))
        lo.append(float(tc_bounds[0]))
        hi.append(float(tc_bounds[1]))
    if M0_vary:
        p0.append(float(M0_value))
        lo.append(float(M0_bounds[0]))
        hi.append(float(M0_bounds[1]))
    if D0_vary:
        p0.append(float(D0_value))
        lo.append(float(D0_bounds[0]))
        hi.append(float(D0_bounds[1]))

    if len(p0) == 0:
        yhat = OGSE_contrast_vs_g_rest(td_ms, G1, G2, N1, N2, float(tc_value), float(M0_value), float(D0_value))
        return {
            "tc_fit": float(tc_value),
            "tc_error": np.nan,
            "M0_fit": float(M0_value),
            "M0_error": np.nan,
            "D0_fit": float(D0_value),
            "D0_error": np.nan,
            "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
        }

    popt, pcov = curve_fit(
        model_wrapper,
        np.zeros_like(y),
        y,
        p0=p0,
        bounds=(lo, hi),
        maxfev=400000,
    )
    perr = np.sqrt(np.diag(pcov)) if pcov is not None and pcov.size else np.full(len(popt), np.nan)

    idx = 0
    tc_fit, tc_err = float(tc_value), np.nan
    M0_fit, M0_err = float(M0_value), np.nan
    D0_fit, D0_err = float(D0_value), np.nan

    if tc_vary:
        tc_fit = float(popt[idx])
        tc_err = float(perr[idx])
        idx += 1
    if M0_vary:
        M0_fit = float(popt[idx])
        M0_err = float(perr[idx])
        idx += 1
    if D0_vary:
        D0_fit = float(popt[idx])
        D0_err = float(perr[idx])
        idx += 1

    yhat = OGSE_contrast_vs_g_rest(td_ms, G1, G2, N1, N2, tc_fit, M0_fit, D0_fit)
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))

    return {
        "tc_fit": tc_fit,
        "tc_error": tc_err,
        "M0_fit": M0_fit,
        "M0_error": M0_err,
        "D0_fit": D0_fit,
        "D0_error": D0_err,
        "rmse": rmse,
    }


def _tc_at_max_from_notebook(
    *,
    td_ms: float,
    g_at_max_raw_mTpm: float,
    D0_fix_m2_ms: float,
    gamma_rad_ms_mT: float,
) -> tuple[float, float, float, float]:
    """
    Replica EXACTA del notebook:
      l_G = ((2^(3/2)) * D0_fix / (gamma * g_raw))^(1/3)
      l_d = sqrt(2*D0_fix*td)
      L_d = l_d/l_G
      L_cf = (3/2)^(1/4) * L_d^(-1/2)
      lcf_at_max = L_cf * l_G
      tc_at_max = lcf_at_max^2 / (2*D0_fix)
    """
    g_raw = float(g_at_max_raw_mTpm)
    if not np.isfinite(g_raw) or g_raw <= 0:
        return (np.nan, np.nan, np.nan, np.nan)

    D0 = float(D0_fix_m2_ms)
    gamma = float(gamma_rad_ms_mT)
    td = float(td_ms)

    l_G = ((2.0 ** (3.0 / 2.0)) * D0 / (gamma * g_raw)) ** (1.0 / 3.0)
    l_d = np.sqrt(2.0 * D0 * td)
    L_d = l_d / l_G
    L_cf = ((3.0 / 2.0) ** (1.0 / 4.0)) * (L_d ** (-1.0 / 2.0))
    lcf = L_cf * l_G
    tc_at_max = (lcf ** 2) / (2.0 * D0)
    return (float(l_G), float(lcf), float(tc_at_max), float(L_cf))


def main():
    ap = argparse.ArgumentParser(
        description="Fit OGSE contrast con modelo restricted y calcular tc_at_max como en el notebook."
    )
    ap.add_argument("contrast_long", nargs="+", type=Path, help="Uno o varios contrast_long (.parquet/.csv)")
    ap.add_argument("--corr-xlsx", type=Path, default=None)
    ap.add_argument("--corr-sheet", default="grad_correction")
    ap.add_argument("--corr-tol-ms", type=float, default=1e-3)

    ap.add_argument("--gbase", default="g_lin_max", help="g_lin_max | g_thorsten | ... (usa <gbase>_1/_2)")
    ap.add_argument("--ycol", default="value_norm")

    ap.add_argument("--out-xlsx", type=Path, default=Path("plots/globalfits/globalfit_restricted_g_lin_max.xlsx"))
    ap.add_argument("--overwrite", action="store_true", help="Sobrescribe el out-xlsx (default: append+dedup).")

    # si no están en la tabla, forzarlos
    ap.add_argument("--N1", type=int, default=None)
    ap.add_argument("--N2", type=int, default=None)

    # guesses
    ap.add_argument("--tc-value", type=float, default=5.0)
    ap.add_argument("--D0-value", type=float, default=3.2e-12)
    ap.add_argument("--M0-value", type=float, default=1.0)

    # vary flags (notebook: D0_vary=True, tc_vary=True, M0_vary=False)
    ap.add_argument("--tc-vary", action="store_true", default=True)
    ap.add_argument("--no-tc-vary", dest="tc_vary", action="store_false")
    ap.add_argument("--D0-vary", action="store_true", default=True)
    ap.add_argument("--no-D0-vary", dest="D0_vary", action="store_false")
    ap.add_argument("--M0-vary", action="store_true", default=False)
    ap.add_argument("--no-M0-vary", dest="M0_vary", action="store_false")

    # bounds (notebook-like)
    ap.add_argument("--tc-min", type=float, default=0.1)
    ap.add_argument("--tc-max", type=float, default=1000.0)
    ap.add_argument("--D0-min", type=float, default=3.2e-14)
    ap.add_argument("--D0-max", type=float, default=3.2e-11)
    ap.add_argument("--M0-min", type=float, default=0.0)
    ap.add_argument("--M0-max", type=float, default=2.0)

    # notebook constants
    ap.add_argument("--gamma", type=float, default=267.5221900, help="rad/(ms*mT) (igual al notebook)")
    ap.add_argument("--D0-fix", type=float, default=3.2e-12, help="m^2/ms (igual al notebook)")
    ap.add_argument("--grid-n", type=int, default=1000, help="puntos para buscar máximo del modelo")

    ap.add_argument("--stat", default="avg", help="filtrar stat si existe (default avg). Usa ALL para no filtrar.")
    ap.add_argument("--plots", action="store_true", help="guardar plots individuales")
    ap.add_argument("--plots-dir", type=Path, default=Path("plots/tc_pipeline/individual_plots"))

    args = ap.parse_args()

    out_xlsx: Path = args.out_xlsx
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    if args.plots:
        args.plots_dir.mkdir(parents=True, exist_ok=True)

    corr = None
    if args.corr_xlsx:
        corr = _load_corr_xlsx(args.corr_xlsx, sheet_name=args.corr_sheet, tol_ms=float(args.corr_tol_ms))

    g1c = f"{args.gbase}_1"
    g2c = f"{args.gbase}_2"

    rows = []

    for p in args.contrast_long:
        df = _read_table(p)

        if g1c not in df.columns or g2c not in df.columns:
            raise KeyError(f"[{p.name}] faltan columnas {g1c}/{g2c}. Columns={list(df.columns)[:40]}")
        if args.ycol not in df.columns:
            raise KeyError(f"[{p.name}] ycol='{args.ycol}' no existe. Columns={list(df.columns)[:40]}")
        if "roi" not in df.columns or "direction" not in df.columns:
            raise KeyError(f"[{p.name}] necesito columnas 'roi' y 'direction'.")

        df = df.copy()
        df["roi"] = df["roi"].astype(str)
        df["direction"] = df["direction"].astype(str)

        if "stat" in df.columns and str(args.stat).upper() != "ALL":
            df = df[df["stat"].astype(str) == str(args.stat)].copy()

        sheet = _infer_sheet(df, fallback=p.stem)
        exp_id = _infer_exp_id_from_path(p)
        brain = exp_id  # estable: sin td

        td_col = "td_ms" if "td_ms" in df.columns else ("td_ms_1" if "td_ms_1" in df.columns else None)
        group_cols = ["roi", "direction"]
        if td_col is not None:
            group_cols.append(td_col)

        for keys, gg in df.groupby(group_cols, sort=False):
            if td_col is None:
                roi, direction = keys
                td_ms = _infer_td_ms(gg)
            else:
                roi, direction, td_val = keys
                td_ms = float(td_val) if np.isfinite(float(td_val)) else _infer_td_ms(gg)

            if td_ms is None or not np.isfinite(td_ms):
                continue

            n1_col = _pick_col(gg, ["N_1", "N1"])
            n2_col = _pick_col(gg, ["N_2", "N2"])
            N1 = int(round(_to_float_unique(gg[n1_col]) if n1_col else (args.N1 if args.N1 is not None else np.nan)))
            N2 = int(round(_to_float_unique(gg[n2_col]) if n2_col else (args.N2 if args.N2 is not None else np.nan)))
            if not np.isfinite(N1) or not np.isfinite(N2):
                raise ValueError(f"[{p.name}] No pude inferir N1/N2 (ni en tabla ni via CLI).")

            y = pd.to_numeric(gg[args.ycol], errors="coerce").to_numpy(float)
            g1_raw = pd.to_numeric(gg[g1c], errors="coerce").to_numpy(float)
            g2_raw = pd.to_numeric(gg[g2c], errors="coerce").to_numpy(float)

            m = np.isfinite(y) & np.isfinite(g1_raw) & np.isfinite(g2_raw)
            y, g1_raw, g2_raw = y[m], g1_raw[m], g2_raw[m]
            if len(y) < 3:
                continue

            f_val = 1.0
            if corr is not None:
                f_val = _corr_factor(corr, sheet=sheet, td_ms=float(td_ms), direction=str(direction))

            # fit con g corregido (igual notebook)
            g1 = g1_raw * float(f_val)
            g2 = g2_raw * float(f_val)

            order = np.argsort(g1)
            y, g1, g2, g1_raw, g2_raw = y[order], g1[order], g2[order], g1_raw[order], g2_raw[order]

            fit = _fit_restricted_curvefit(
                float(td_ms),
                g1,
                g2,
                y,
                N1=N1,
                N2=N2,
                tc_value=float(args.tc_value),
                D0_value=float(args.D0_value),
                M0_value=float(args.M0_value),
                tc_vary=bool(args.tc_vary),
                D0_vary=bool(args.D0_vary),
                M0_vary=bool(args.M0_vary),
                tc_bounds=(float(args.tc_min), float(args.tc_max)),
                D0_bounds=(float(args.D0_min), float(args.D0_max)),
                M0_bounds=(float(args.M0_min), float(args.M0_max)),
            )

            tc_fit = float(fit["tc_fit"])
            D0_fit = float(fit["D0_fit"])
            M0_fit = float(fit["M0_fit"])

            # peak como notebook: máximo del modelo en grilla
            g1_fit = np.linspace(0.0, float(np.max(g1)), int(args.grid_n))
            g2_fit = np.linspace(0.0, float(np.max(g2)), int(args.grid_n))
            y_fit = OGSE_contrast_vs_g_rest(float(td_ms), g1_fit, g2_fit, N1, N2, tc_fit, M0_fit, D0_fit)

            i_max = int(np.nanargmax(y_fit))
            signal_max = float(y_fit[i_max])
            g_at_max_corr = float(g1_fit[i_max])
            g_at_max_raw = float(g_at_max_corr / float(f_val)) if float(f_val) != 0 else np.nan

            l_G, lcf_at_max, tc_at_max, L_cf = _tc_at_max_from_notebook(
                td_ms=float(td_ms),
                g_at_max_raw_mTpm=float(g_at_max_raw),
                D0_fix_m2_ms=float(args.D0_fix),
                gamma_rad_ms_mT=float(args.gamma),
            )

            rows.append(
                {
                    "brain": brain,
                    "Archivo_origen": exp_id,
                    "source_file": str(p),
                    "sheet": sheet,
                    "roi": str(roi),
                    "direction": str(direction),
                    "td_ms": float(td_ms),
                    "f": float(f_val),
                    "N1": int(N1),
                    "N2": int(N2),
                    "M0_fit": float(M0_fit),
                    "M0_error": float(fit.get("M0_error", np.nan)),
                    "tc_fit": float(tc_fit),
                    "tc_error": float(fit.get("tc_error", np.nan)),
                    "D0_fit": float(D0_fit),
                    "D0_error": float(fit.get("D0_error", np.nan)),
                    "rmse": float(fit.get("rmse", np.nan)),
                    "signal_max": float(signal_max),
                    "g_at_max": float(g_at_max_raw),
                    "l_G": float(l_G),
                    "L_cf": float(L_cf),
                    "lcf_at_max": float(lcf_at_max),
                    "tc_at_max": float(tc_at_max),
                }
            )

            if args.plots:
                try:
                    import matplotlib.pyplot as plt

                    plt.figure(figsize=(8, 6))
                    plt.plot(g1_raw, y, "o", label="datos (g raw)")
                    plt.plot(g1_fit / float(f_val), y_fit, "-", label="fit (grid, raw axis)")
                    plt.scatter([g_at_max_raw], [signal_max], s=60, c="k", label="max")
                    plt.xlabel("G [mT/m] (raw)")
                    plt.ylabel("contrast")
                    plt.title(f"{exp_id} | {roi} | dir {direction} | Td={td_ms:.3g} ms")
                    plt.grid(True)
                    plt.legend()
                    fn = args.plots_dir / f"{exp_id}__{roi}__dir{direction}__Td{td_ms:.3g}.png"
                    plt.tight_layout()
                    plt.savefig(fn, dpi=200)
                    plt.close()
                except Exception:
                    pass

    if not rows:
        raise RuntimeError("No se generaron filas. Revisá stat/ycol y que existan columnas td_ms y <gbase>_1/_2.")

    df_new = pd.DataFrame(rows)
    df_new = df_new.sort_values(["direction", "Archivo_origen", "roi", "td_ms"], kind="stable")

    # DEFAULT: append + dedup (no sobreescribe)
    if (not args.overwrite) and out_xlsx.exists():
        df_old = pd.read_excel(out_xlsx)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        key_cols = ["Archivo_origen", "roi", "direction", "td_ms"]
        df_all = df_all.drop_duplicates(subset=key_cols, keep="last")
        df_all = df_all.sort_values(["direction", "Archivo_origen", "roi", "td_ms"], kind="stable")
        df_all.to_excel(out_xlsx, index=False)
    else:
        df_new.to_excel(out_xlsx, index=False)

    print(f"OK: {out_xlsx}")
    print("Siguiente paso:")
    print(f'  python scripts/run_tc_vs_td.py --method pseudohuber_free --globalfit "{out_xlsx}"')


if __name__ == "__main__":
    main()