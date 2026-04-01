from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd

from signal_rotation.rotation_tensor import rotate_signals_tensor


def _infer_exp_dir(df: pd.DataFrame, long_parquet: Path) -> str:
    if "sheet" in df.columns:
        vals = pd.Series(df["sheet"]).dropna().astype(str).str.strip().unique().tolist()
        if len(vals) == 1 and vals[0]:
            return vals[0]

    parent = long_parquet.parent.name
    if parent and parent != ".":
        return parent

    stem = long_parquet.stem.replace(".long", "")
    if "_ep2d" in stem:
        return stem.split("_ep2d")[0]
    return stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("long_parquet", type=Path)
    ap.add_argument("--out_dir", type=Path, default=Path("analysis/ogse_experiments/data-rotated"))
    ap.add_argument("--solver", type=str, default="lstsq", choices=["lstsq", "solve"])
    ap.add_argument("--s0_mode", type=str, default="dir1", choices=["dir1", "mean"])
    ap.add_argument("--b_col", type=str, default="bvalue")
    ap.add_argument("--dirs_csv", type=Path, default=None, help="CSV Nx3 con direcciones (sin header). Si no se pasa, usa assets/dirs/dirs_{ndirs}.csv")
    args = ap.parse_args()

    df = pd.read_parquet(args.long_parquet)

    res = rotate_signals_tensor(
        df,
        solver=args.solver,
        s0_mode=args.s0_mode,
        b_col=args.b_col,
        dirs_csv=args.dirs_csv, 
    )

    exp_dir = args.out_dir / _infer_exp_dir(df, args.long_parquet)
    exp_dir.mkdir(parents=True, exist_ok=True)
    stem = args.long_parquet.stem.replace(".long", "")

    out_rot = exp_dir / f"{stem}.rot_tensor.long.parquet"
    out_dpr = exp_dir / f"{stem}.rot_tensor.Dproj.long.parquet"

    res.rotated_signal_long.to_parquet(out_rot, index=False)
    res.dproj_long.to_parquet(out_dpr, index=False)
    res.rotated_signal_long.to_excel(out_rot.with_suffix(".xlsx"), index=False)
    res.dproj_long.to_excel(out_dpr.with_suffix(".xlsx"), index=False)
    
    print("Saved rotated signals:", out_rot)
    print("Saved Dproj:", out_dpr)


if __name__ == "__main__":
    main()
