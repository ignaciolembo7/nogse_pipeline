from __future__ import annotations

import argparse
import sys
from pathlib import Path
import hashlib
import re

from simulations.nogse_contrast_sim import (
    SimSpec,
    list_models,
    load_module,
    simulate_contrast_long,
    write_parquet,
    parse_kv,
)

def _fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        s = f"{v:.12g}"  # compacto
        s = s.replace(".", "p")  # 0.3 -> 0p3 (más filename-friendly)
        return s
    return str(v)

def _sanitize(s: str) -> str:
    # caracteres inválidos en Windows + espacios
    s = s.strip().replace(" ", "")
    return re.sub(r'[<>:"/\\|?*\n\r\t]+', "_", s)

def build_out_path(
    out_arg: str,
    model: str,
    grid_name: str,
    grid_spec: str,
    grid2_name: str | None,
    grid2_spec: str | None,
    axes: list[str],
    rois: list[str],
    model_kwargs: dict,
    meta_params: dict,
    exp_prefix: str = "",
) -> Path:
    
    out_p = Path(out_arg)

    # out puede ser un directorio o un archivo .parquet
    if out_p.suffix.lower() == ".parquet":
        out_dir = out_p.parent
        prefix = out_p.stem
    else:
        out_dir = out_p
        prefix = ""

    # prepend exp_prefix (short, user-defined)
    if exp_prefix:
        prefix = exp_prefix if not prefix else f"{exp_prefix}__{prefix}"

    parts = []
    parts.append(f"grid-{grid_name}_{grid_spec}")
    if grid2_name:
        parts.append(f"grid2-{grid2_name}_{grid2_spec or grid_spec}")
    parts.append(f"axes-{'-'.join(axes)}")
    parts.append(f"rois-{'-'.join(rois)}")

    # IMPORTANT: incluir TODOS los kw y meta params en orden determinista
    for k in sorted(model_kwargs.keys()):
        parts.append(f"{k}-{_fmt(model_kwargs[k])}")
    for k in sorted(meta_params.keys()):
        parts.append(f"{k}-{_fmt(meta_params[k])}")

    tag = "__".join(_sanitize(p) for p in parts)
    base = f"{model}__{tag}"
    base = _sanitize(base)

    # evitar filenames gigantes: truncar y agregar hash
    maxlen = 180
    if len(base) > maxlen:
        h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
        base = base[:maxlen] + f"__h{h}"

    if prefix:
        base = _sanitize(prefix) + "__" + base

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{base}.contrast.long.parquet"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Simulate NOGSE/OGSE contrast curves using a model function from nogse_model_fitting.py "
            "and output a table in the exact format produced by contrast.make_contrast()."
        )
    )

    ap.add_argument("--model", type=str, help="Function name in the model module, e.g. NOGSE_contrast_vs_g_mixed")
    ap.add_argument("--model-module", type=str, default="nogse_fitting.nogse_model_fitting", help="Python module name to import")

    ap.add_argument("--model-path", type=str, default=None, help="Optional path to .py file (overrides --model-module)")

    ap.add_argument("--list-models", action="store_true", help="List available contrast model functions and exit")

    ap.add_argument("--out", type=str, default="OGSE_signal/contrast/simulated", help="Output directory OR a .parquet path prefix")
    ap.add_argument("--exp-prefix", type=str, default="", help="Short prefix to prepend to output filename (e.g. rest_D0-... )")


    ap.add_argument("--axes", nargs="+", default=None, help="Directions to write into 'direction' column (e.g. long tra x y z)")

    ap.add_argument("--rois", nargs="+", default=["SIM_ROI"])
    ap.add_argument("--stat", type=str, default="mean")

    ap.add_argument("--grid-name", type=str, default="G", help="Model argument name that receives the x-grid (e.g. G, Lc, bvalue, G1)")
    ap.add_argument("--grid", type=str, default="0:80:41", help="Grid spec: 'a:b:n' (linspace) or 'v1,v2,v3'")
    ap.add_argument("--grid2-name", type=str, default=None, help="Optional second grid arg name (e.g. G2)")
    ap.add_argument("--grid2", type=str, default=None, help="Optional second grid spec. If omitted but grid2-name is set, grid2=grid")

    ap.add_argument("--key-cols", nargs="+", default=["stat", "roi", "direction", "b_step"])

    ap.add_argument(
        "--kw",
        action="append",
        default=[],
        help="Model kwarg as key=value (repeatable). Example: --kw TE=54 --kw N=8 --kw tc=20",
    )
    ap.add_argument(
        "--meta",
        action="append",
        default=[],
        help="Extra param columns to carry through as param_<key> (repeatable). Example: --meta Delta=54 --meta delta=10 --meta td_ms=30",
    )

    args = ap.parse_args(argv)

    # list models
    mod = load_module(args.model_module, args.model_path)
    if args.list_models:
        for name in list_models(mod):
            print(name)
        return 0

    if not args.model:
        ap.error("--model is required unless --list-models is used")

    if args.axes is None:
        ap.error("--axes is required unless --list-models is used")


    model_kwargs = dict(parse_kv(x) for x in args.kw)
    meta_params = dict(parse_kv(x) for x in args.meta)

    out_path = build_out_path(
    out_arg=args.out,
    model=args.model,
    grid_name=args.grid_name,
    grid_spec=args.grid,
    grid2_name=args.grid2_name,
    grid2_spec=args.grid2,
    axes=args.axes,
    rois=args.rois,
    model_kwargs=model_kwargs,
    meta_params=meta_params,
    exp_prefix=args.exp_prefix,
    )

    spec = SimSpec(
        model_name=args.model,
        model_module=args.model_module,
        model_module_path=args.model_path,
        grid_name=args.grid_name,
        grid_spec=args.grid,
        grid2_name=args.grid2_name,
        grid2_spec=args.grid2,
        out_path=str(out_path),
        axes=tuple(args.axes),
        rois=tuple(args.rois),
        stat=args.stat,
        key_cols=tuple(args.key_cols),
        model_kwargs=model_kwargs,
        meta_params=meta_params,
    )

    df = simulate_contrast_long(spec)
    p = write_parquet(df, spec.out_path)
    csv_path = p.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] wrote {p} (rows={len(df)})")
    print(f"[OK] wrote {csv_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
