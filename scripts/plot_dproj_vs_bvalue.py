from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

def iter_parquets(globs: list[str]):
    """Yield (exp_id, path, df) for each parquet matched by the globs."""
    paths = []
    for g in globs:
        paths += sorted(Path().glob(g))
    if not paths:
        raise FileNotFoundError("No encontré parquets con esos glob patterns.")

    for p in paths:
        name = p.name
        exp = re.sub(r"\.rot_tensor\.Dproj\.long\.parquet$", "", name)
        df = pd.read_parquet(p)
        yield exp, p, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True,
                    help="Ej: OGSE_signal/rotated/*BRAIN-3*.rot_tensor.Dproj.long.parquet")
    ap.add_argument("--roi", default="ALL",
                    help="Nombre de ROI (ej PostCC). Usá ALL para plotear todas.")
    ap.add_argument("--out_root", default="plots", help="Carpeta raíz de plots/")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for exp, path, df in iter_parquets(args.glob):
        # --- sanity checks
        needed = {"roi", "axis", "b_step", "bvalue", "D_proj"}
        miss = needed - set(df.columns)
        if miss:
            raise ValueError(f"En {path}: faltan columnas {sorted(miss)}")

        if "param_N" not in df.columns:
            raise ValueError(f"En {path}: falta param_N (necesaria para separar por N).")

        # --- output dir por experimento
        outdir = out_root / exp / "dproj"
        outdir.mkdir(parents=True, exist_ok=True)

        # --- ROIs a plotear
        rois = sorted(df["roi"].dropna().unique())
        if args.roi != "ALL":
            rois = [args.roi]

        # --- Ns disponibles
        Ns = sorted(df["param_N"].dropna().unique())

        # ===========================
        # FIGURA 1: x y z (solo x,z) para todos los N en una sola gráfica (por ROI)
        # ===========================
        axes_xz = ["x", "y","z"]
        for roi in rois:
            d = df[(df["roi"] == roi) & (df["axis"].isin(axes_xz))].copy()
            if d.empty:
                continue

            plt.figure(figsize=(10, 7))
            plt.title(f"{exp} | Dproj vs bvalue | ROI={roi} | (x,z) | todos los N", fontsize=14)
            plt.xlabel("bvalue [s/mm²]", fontsize=12)
            plt.ylabel("D_proj [mm²/s]", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.3)

            for axis in axes_xz:
                for N in Ns:
                    dAN = d[(d["axis"] == axis) & (d["param_N"] == N)].sort_values("bvalue")
                    if dAN.empty:
                        continue
                    plt.plot(
                        dAN["bvalue"], dAN["D_proj"],
                        marker="o",
                        label=f"{axis} | N={int(N)}"
                    )

            plt.legend(fontsize=9)
            plt.tight_layout()
            outpath = outdir / f"Dproj_vs_bvalue_xyz_allN_ROI-{roi}.png"
            plt.savefig(outpath, dpi=300)
            plt.close()
            print("Saved:", outpath)

        # ===========================
        # FIGURA 2: subplots por N, dentro comparar x,z,eig1,eig2,eig3 (por ROI)
        # ===========================
        axes_panel = ["x", "y", "z", "eig1", "eig2", "eig3"]
        label_map = {"eig1": "eig1 (λ1)", "eig2": "eig2 (λ2)", "eig3": "eig3 (λ3)"}

        # grid tipo notebook: ideal 1x4, si hay más Ns se arma en filas
        ncols = min(4, len(Ns))
        nrows = int(np.ceil(len(Ns) / ncols))

        for roi in rois:
            d = df[(df["roi"] == roi) & (df["axis"].isin(axes_panel))].copy()
            if d.empty:
                continue

            fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharey=True)
            axs = np.array(axs).reshape(-1)  # flatten

            fig.suptitle(f"{exp} | Dproj vs bvalue | ROI={roi} | panel por N", fontsize=16)

            for i, N in enumerate(Ns):
                ax = axs[i]
                dN = d[d["param_N"] == N]
                ax.set_title(f"N={int(N)}")
                ax.set_xlabel("bvalue [s/mm²]")
                ax.grid(True, linestyle="--", alpha=0.3)

                for axis in axes_panel:
                    dA = dN[dN["axis"] == axis].sort_values("bvalue")
                    if dA.empty:
                        continue
                    lab = label_map.get(axis, axis)
                    ax.plot(dA["bvalue"], dA["D_proj"], marker="o", label=lab)

                ax.legend(fontsize=9)

            # apagar subplots vacíos si Ns < nrows*ncols
            for j in range(len(Ns), len(axs)):
                axs[j].axis("off")

            axs[0].set_ylabel("D_proj [mm²/s]")
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            outpath = outdir / f"Dproj_vs_bvalue_panels_byN_ROI-{roi}.png"
            fig.savefig(outpath, dpi=300)
            plt.close(fig)
            print("Saved:", outpath)


if __name__ == "__main__":
    main()