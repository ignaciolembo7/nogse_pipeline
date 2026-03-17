from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_many(globs: list[str]) -> pd.DataFrame:
    parts = []
    for g in globs:
        for p in sorted(Path().glob(g)):
            parts.append(pd.read_parquet(p))
    if not parts:
        raise FileNotFoundError("No encontré parquets con esos glob patterns.")
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True,
                    help="Glob(s) tipo: OGSE_signal/rotated/*BRAIN-3*.rot_tensor.long.parquet")
    ap.add_argument("--roi", required=True, help="Ej: PostCC")
    ap.add_argument("--xcol", default="g_lin_max", help="Ej: g_lin_max | g | g_thorsten")
    ap.add_argument("--ycol", default="value_norm", help="Ej: value_norm | value")
    ap.add_argument("--directions", nargs="+", default=["x", "y", "z", "eig1", "eig2", "eig3", "long", "tra"])
    ap.add_argument("--out_root", default="plots", help="Carpeta raíz de plots/")
    args = ap.parse_args()

    df = load_many(args.glob)

    if "N" in df.columns:
        n_col = "N"
    elif "param_N" in df.columns:
        n_col = "param_N"
    else:
        raise ValueError("No encuentro columna N en el parquet.")

    # filtro ROI + direcciones rotadas
    df = df[(df["roi"] == args.roi) & (df["direction"].isin(args.directions))].copy()

    outdir = Path(args.out_root) / "rot_signal"
    outdir.mkdir(parents=True, exist_ok=True)

    Ns = sorted(df[n_col].dropna().unique())
    fig, axs = plt.subplots(1, len(Ns), figsize=(6 * len(Ns), 5), sharey=True)
    if len(Ns) == 1:
        axs = [axs]

    for i, N in enumerate(Ns):
        ax = axs[i]
        dN = df[df[n_col] == N]

        for direction in args.directions:
            dA = dN[dN["direction"] == direction].sort_values(args.xcol)
            ax.plot(dA[args.xcol], dA[args.ycol], marker="o", label=direction)

        ax.set_title(f"N={int(N)}")
        ax.set_xlabel(args.xcol)
        ax.grid(True, linestyle="--", alpha=0.3)

    axs[0].set_ylabel(args.ycol)
    axs[-1].legend(fontsize=9)
    fig.suptitle(f"Rotated signal vs {args.xcol} | ROI={args.roi}", fontsize=14)
    fig.tight_layout()

    outpath = outdir / f"rot_signal_vs_{args.xcol}_{args.roi}.png"
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print("Saved:", outpath)


if __name__ == "__main__":
    main()
