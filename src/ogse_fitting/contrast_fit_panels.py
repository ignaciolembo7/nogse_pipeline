from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ogse_fitting.fit_ogse_contrast import _gcols, _maybe_scale_g_thorsten, _model_yhat
from tc_fittings.contrast_fit_table import load_contrast_fit_params


@dataclass(frozen=True)
class PanelSpec:
    subj: str
    model: str
    gbase: str
    ycol: str
    xplot: str


def _sanitize_token(value: str) -> str:
    text = str(value).strip()
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "NA"


def _table_root(contrast_root: str | Path) -> Path:
    base = Path(contrast_root)
    if (base / "tables").is_dir():
        return base / "tables"
    return base


def _resolve_contrast_parquet(
    *,
    analysis_id: str,
    sheet: str | None,
    contrast_root: str | Path,
) -> Path:
    table_root = _table_root(contrast_root)
    target_name = f"{analysis_id}.long.parquet"

    if sheet:
        candidate = table_root / str(sheet) / target_name
        if candidate.exists():
            return candidate

    matches = sorted(table_root.glob(f"**/{target_name}"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No encontré tabla de contraste para analysis_id={analysis_id!r} en {table_root}"
        )
    raise FileNotFoundError(
        f"Encontré múltiples tablas para analysis_id={analysis_id!r} en {table_root}: "
        f"{[str(p) for p in matches[:5]]}"
    )


def _load_contrast_table_cached(
    path: Path,
    cache: dict[Path, pd.DataFrame],
) -> pd.DataFrame:
    cached = cache.get(path)
    if cached is not None:
        return cached
    df = pd.read_parquet(path)
    cache[path] = df
    return df


def _subset_group(df: pd.DataFrame, fit_row: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out = out[out["roi"].astype(str) == str(fit_row["roi"])].copy()
    out = out[out["direction"].astype(str) == str(fit_row["direction"])].copy()
    stat = fit_row.get("stat", None)
    if stat is not None and "stat" in out.columns and str(stat) != "nan":
        out = out[out["stat"].astype(str) == str(stat)].copy()
    return out


def _extract_plot_arrays(df_group: pd.DataFrame, fit_row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gbase = str(fit_row.get("gbase", "g_lin_max"))
    ycol = str(fit_row.get("ycol", "value_norm"))
    xplot = str(fit_row.get("xplot", "1"))

    y_alias = {"value_norm": "contrast_norm", "value": "contrast"}
    y_eff = ycol if ycol in df_group.columns else y_alias.get(ycol)
    if y_eff is None or y_eff not in df_group.columns:
        raise KeyError(f"No encuentro ycol={ycol!r} en tabla de contraste.")

    g1c, g2c = _gcols(gbase)
    if g1c not in df_group.columns or g2c not in df_group.columns:
        raise KeyError(f"No encuentro columnas {g1c!r}/{g2c!r} en tabla de contraste.")

    y = pd.to_numeric(df_group[y_eff], errors="coerce").to_numpy(dtype=float)
    G1 = pd.to_numeric(df_group[g1c], errors="coerce").to_numpy(dtype=float)
    G2 = pd.to_numeric(df_group[g2c], errors="coerce").to_numpy(dtype=float)

    G1 = _maybe_scale_g_thorsten(gbase, G1)
    G2 = _maybe_scale_g_thorsten(gbase, G2)

    f_corr = float(fit_row.get("f_corr", 1.0) or 1.0)
    G1 = G1 * f_corr
    G2 = G2 * f_corr

    m = np.isfinite(y) & np.isfinite(G1) & np.isfinite(G2)
    y, G1, G2 = y[m], G1[m], G2[m]

    x = G1 if xplot == "1" else G2
    order = np.argsort(x)
    return x[order], y[order], G1[order], G2[order]


def _build_fit_curve(fit_row: pd.Series, G1: np.ndarray, G2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if G1.size == 0 or G2.size == 0:
        return np.array([]), np.array([])

    td = float(fit_row["td_ms"])
    n_1 = int(fit_row["N_1"])
    n_2 = int(fit_row["N_2"])
    xplot = str(fit_row.get("xplot", "1"))

    frac = np.linspace(0.0, 1.0, 300)
    G1s = frac * float(np.nanmax(G1))
    G2s = frac * float(np.nanmax(G2))
    ys = _model_yhat(
        model=str(fit_row["model"]),
        td_ms=td,
        G1=G1s,
        G2=G2s,
        n_1=n_1,
        n_2=n_2,
        fit_row=fit_row.to_dict(),
    )
    xs = G1s if xplot == "1" else G2s
    return xs, ys


def _panel_specs(df: pd.DataFrame) -> list[PanelSpec]:
    specs: list[PanelSpec] = []
    for key, sub in df.groupby(["subj", "model", "gbase", "ycol", "xplot"], sort=True):
        subj, model, gbase, ycol, xplot = key
        if sub.empty:
            continue
        specs.append(
            PanelSpec(
                subj=str(subj),
                model=str(model),
                gbase=str(gbase),
                ycol=str(ycol),
                xplot=str(xplot),
            )
        )
    return specs


def _pick_order(requested: Iterable[str] | None, available: Iterable[str], *, preferred: tuple[str, ...] = ()) -> list[str]:
    avail = [str(x) for x in available if str(x) != ""]
    if requested is not None:
        out = [str(x) for x in requested if str(x) in set(avail)]
        return out
    pref = [x for x in preferred if x in avail]
    rest = [x for x in sorted(set(avail)) if x not in pref]
    return pref + rest


def plot_contrast_fit_panels(
    *,
    fits_root: str | Path,
    contrast_root: str | Path,
    out_dir: str | Path,
    pattern: str = "**/fit_params.*",
    models: list[str] | None = None,
    subjs: list[str] | None = None,
    rois: list[str] | None = None,
    directions: list[str] | None = None,
    exclude_td_ms: list[float] | None = None,
    ok_only: bool = True,
) -> list[Path]:
    df = load_contrast_fit_params(
        [fits_root],
        pattern=pattern,
        models=models,
        subjs=subjs,
        directions=directions,
        rois=rois,
        ok_only=ok_only,
    )

    if df.empty:
        raise ValueError("No quedó ningún fit válido después de filtrar.")

    if exclude_td_ms:
        td_vals = pd.to_numeric(df["td_ms"], errors="coerce")
        keep = np.ones(len(df), dtype=bool)
        for td_excl in exclude_td_ms:
            keep &= ~np.isclose(td_vals.to_numpy(dtype=float), float(td_excl), atol=1e-3, equal_nan=False)
        df = df.loc[keep].copy()
        if df.empty:
            raise ValueError("No quedó ningún fit válido después de excluir td_ms.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[Path, pd.DataFrame] = {}
    outputs: list[Path] = []

    for spec in _panel_specs(df):
        sub = df[
            (df["subj"].astype(str) == spec.subj)
            & (df["model"].astype(str) == spec.model)
            & (df["gbase"].astype(str) == spec.gbase)
            & (df["ycol"].astype(str) == spec.ycol)
            & (df["xplot"].astype(str) == spec.xplot)
        ].copy()
        if sub.empty:
            continue

        roi_order = _pick_order(rois, sub["roi"].astype(str).unique().tolist())
        dir_order = _pick_order(directions, sub["direction"].astype(str).unique().tolist(), preferred=("long", "tra"))
        if not roi_order or not dir_order:
            continue

        td_values = sorted(pd.to_numeric(sub["td_ms"], errors="coerce").dropna().unique().tolist())
        if not td_values:
            continue

        cmap = cm.get_cmap("viridis", max(2, len(td_values)))
        td_to_color = {float(td): cmap(i) for i, td in enumerate(td_values)}

        fig, axes = plt.subplots(
            len(roi_order),
            len(dir_order),
            figsize=(6.2 * len(dir_order), 3.6 * len(roi_order) + 1.0),
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        missing_items: list[str] = []

        for i, roi in enumerate(roi_order):
            for j, direction in enumerate(dir_order):
                ax = axes[i, j]
                sub_panel = sub[(sub["roi"].astype(str) == roi) & (sub["direction"].astype(str) == direction)].copy()
                sub_panel = sub_panel.sort_values("td_ms", kind="stable")

                if sub_panel.empty:
                    ax.set_title(f"{roi} | {direction}")
                    ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
                    ax.grid(True, alpha=0.25)
                    continue

                for _, row in sub_panel.iterrows():
                    td_val = float(row["td_ms"])
                    color = td_to_color.get(td_val, "#1f77b4")
                    try:
                        contrast_path = _resolve_contrast_parquet(
                            analysis_id=str(row["analysis_id"]),
                            sheet=row.get("sheet", None),
                            contrast_root=contrast_root,
                        )
                        contrast_df = _load_contrast_table_cached(contrast_path, cache)
                        df_group = _subset_group(contrast_df, row)
                        if df_group.empty:
                            missing_items.append(f"{row['analysis_id']} | {roi} | {direction} -> grupo vacío")
                            continue

                        x, y, G1, G2 = _extract_plot_arrays(df_group, row)
                        if x.size == 0:
                            missing_items.append(f"{row['analysis_id']} | {roi} | {direction} -> sin puntos finitos")
                            continue

                        xs, ys = _build_fit_curve(row, G1, G2)
                        ax.plot(x, y, "o", color=color, markersize=4, alpha=0.95)
                        ax.plot(xs, ys, "-", color=color, linewidth=1.8, alpha=0.95)
                    except Exception as exc:
                        missing_items.append(f"{row['analysis_id']} | {roi} | {direction} -> {exc}")

                ax.set_title(f"{roi} | {direction}", fontsize=11)
                ax.grid(True, alpha=0.25)
                if i == len(roi_order) - 1:
                    ax.set_xlabel(f"{spec.gbase}_{spec.xplot} [mT/m]", fontsize=11)
                if j == 0:
                    ax.set_ylabel(spec.ycol, fontsize=11)

        legend_handles = [
            Line2D([0], [0], color=td_to_color[float(td)], marker="o", linewidth=2, markersize=5, label=f"td={float(td):g} ms")
            for td in td_values
        ]
        fig.legend(
            handles=legend_handles,
            loc="center right",
            bbox_to_anchor=(0.995, 0.5),
            title="Curvas",
            fontsize=9,
            title_fontsize=10,
        )

        fig.suptitle(
            f"{spec.subj} | model={spec.model} | y={spec.ycol} | g={spec.gbase}_{spec.xplot}",
            fontsize=14,
        )
        plt.tight_layout(rect=[0.02, 0.02, 0.9, 0.95])

        out_path = out_dir / (
            f"contrast_fit_panels_subj={_sanitize_token(spec.subj)}"
            f"_model={_sanitize_token(spec.model)}"
            f"_g={_sanitize_token(spec.gbase)}"
            f"_y={_sanitize_token(spec.ycol)}"
            f"_x={_sanitize_token(spec.xplot)}.png"
        )
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        outputs.append(out_path)

        if missing_items:
            warnings.warn(
                "Algunos paneles/curvas no pudieron graficarse:\n" + "\n".join(missing_items[:15]),
                stacklevel=2,
            )

    if not outputs:
        raise ValueError("No pude generar ninguna figura con los filtros actuales.")

    return outputs
