from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from data_processing.io import write_table_outputs
from data_processing.master_table import (
    build_analysis_id_from_columns,
    normalize_master_rows,
    validate_master_table,
    write_master_table,
)


SIGNAL_KEY = ["row_kind", "subj", "sheet", "roi", "direction", "stat", "td_ms", "N", "Hz", "b_step"]
DPROJ_KEY = ["row_kind", "subj", "sheet", "roi", "direction", "td_ms", "N", "Hz", "b_step"]
CONTRAST_KEY = [
    "row_kind",
    "subj",
    "sheet",
    "roi",
    "direction",
    "stat",
    "td_ms",
    "N_1",
    "N_2",
    "Hz_1",
    "Hz_2",
    "b_step",
]
SUMMARY_COLS = ["row_kind", "subj", "roi", "td_ms", "N", "direction"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0)
    raise ValueError(f"Unsupported table format: {path}")


def _is_table_file(path: Path) -> bool:
    if path.suffix.lower() not in {".parquet", ".csv", ".xlsx", ".xls"}:
        return False
    name = path.name
    if "fit_params" in name or "signal_fit_params" in name:
        return False
    return True


def _infer_row_kind(path: Path, experiment_root: Path) -> str | None:
    rel = path.relative_to(experiment_root)
    parts = rel.parts
    first = parts[0] if parts else ""
    name = path.name
    if "Dproj.long" in name:
        return "dproj"
    if first == "data-rotated":
        return "signal_rotated"
    if first == "data":
        return "signal"
    if first.startswith("contrast-data"):
        if name.endswith(".long.parquet") or name.endswith(".long.csv") or name.endswith(".long.xlsx"):
            return "contrast"
    return None


def _analysis_id(rows: pd.DataFrame, row_kind: str, path: Path) -> str:
    if "analysis_id" in rows.columns:
        vals = pd.Series(rows["analysis_id"]).dropna().astype(str)
        vals = vals[vals.str.strip() != ""].unique().tolist()
        if len(vals) == 1:
            return vals[0]
    if row_kind == "contrast":
        cols = ("subj", "sheet", "td_ms", "N_1", "N_2", "Hz_1", "Hz_2", "direction")
    else:
        cols = ("subj", "sheet", "type", "td_ms", "N", "Hz")
    try:
        return build_analysis_id_from_columns(rows, columns=[c for c in cols if c in rows.columns], prefix=row_kind)
    except Exception:
        stem = path.stem.replace(".long", "").replace(".rot_tensor", "")
        return f"{row_kind}_{stem}"


def _with_source_metadata(
    rows: pd.DataFrame,
    row_kind: str,
    path: Path,
    experiment_root: Path,
    *,
    hash_source_files: bool = False,
) -> pd.DataFrame:
    out = rows.copy()
    out["row_kind"] = row_kind
    out["source_file"] = path.name
    out["source_path"] = str(path.relative_to(experiment_root))
    out["source_hash"] = _sha256(path) if hash_source_files else pd.NA
    out["analysis_id"] = _analysis_id(out, row_kind, path)
    return out


def discover_tables(experiment_root: Path, *, include_xlsx_csv: bool = False) -> list[tuple[Path, str]]:
    roots = [experiment_root / "data", experiment_root / "data-rotated"]
    roots.extend(sorted(p for p in experiment_root.glob("contrast-data*") if p.is_dir()))
    suffixes = {".parquet"} if not include_xlsx_csv else {".parquet", ".csv", ".xlsx", ".xls"}

    found: list[tuple[Path, str]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in suffixes or not _is_table_file(path):
                continue
            row_kind = _infer_row_kind(path, experiment_root)
            if row_kind is None:
                continue
            found.append((path, row_kind))
    return found


def build_master_from_experiment(
    experiment_root: Path,
    *,
    include_xlsx_csv: bool = False,
    hash_source_files: bool = False,
    max_files: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []
    tables = discover_tables(experiment_root, include_xlsx_csv=include_xlsx_csv)
    if max_files is not None:
        tables = tables[: int(max_files)]

    for path, row_kind in tables:
        try:
            raw = _read_table(path)
            rows = _with_source_metadata(raw, row_kind, path, experiment_root, hash_source_files=hash_source_files)
            frames.append(rows)
            inventory.append({"path": str(path.relative_to(experiment_root)), "row_kind": row_kind, "rows": len(rows), "status": "ok"})
        except Exception as exc:
            inventory.append({"path": str(path.relative_to(experiment_root)), "row_kind": row_kind, "rows": 0, "status": f"error: {exc}"})

    if not frames:
        raise FileNotFoundError(f"No legacy tables found under {experiment_root}")

    normalized_frames = []
    for frame in frames:
        kind = str(frame["row_kind"].iloc[0])
        normalized_frames.append(normalize_master_rows(frame, row_kind=kind))
    master = validate_master_table(pd.concat(normalized_frames, ignore_index=True, sort=False))
    return master.reset_index(drop=True), pd.DataFrame(inventory)


def _key_for_kind(df: pd.DataFrame, row_kind: str) -> list[str]:
    if row_kind in {"signal", "signal_rotated"}:
        candidates = SIGNAL_KEY
    elif row_kind == "dproj":
        candidates = DPROJ_KEY
    elif row_kind == "contrast":
        candidates = CONTRAST_KEY
    else:
        candidates = ["row_kind", "analysis_id"]
    return [c for c in candidates if c in df.columns]


def validation_report(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    missing_rows = []
    required_by_kind = {
        "signal": ["stat", "roi", "direction", "b_step", "bvalue", "value"],
        "signal_rotated": ["stat", "roi", "direction", "b_step", "bvalue", "value"],
        "dproj": ["roi", "direction", "b_step", "bvalue", "D_proj"],
        "contrast": ["stat", "roi", "direction", "b_step", "value", "value_norm", "N_1", "N_2"],
    }
    for row_kind, required in required_by_kind.items():
        sub = master[master["row_kind"].astype(str).eq(row_kind)]
        if sub.empty:
            continue
        for col in required:
            if col not in sub.columns:
                missing_rows.append({"row_kind": row_kind, "column": col, "missing_cells": len(sub)})
            else:
                missing_rows.append({"row_kind": row_kind, "column": col, "missing_cells": int(sub[col].isna().sum())})

    exact_dups = master[master.duplicated(keep=False)].copy()

    key_dups = []
    for row_kind in sorted(master["row_kind"].dropna().astype(str).unique()):
        sub = master[master["row_kind"].astype(str).eq(row_kind)].copy()
        key = _key_for_kind(sub, row_kind)
        if not key:
            continue
        dup = sub[sub.duplicated(subset=key, keep=False)]
        if dup.empty:
            continue
        grouped = dup.groupby(key, dropna=False).size().reset_index(name="n_rows")
        grouped.insert(0, "duplicate_key", "|".join(key))
        key_dups.append(grouped)

    summary_cols = [c for c in SUMMARY_COLS if c in master.columns]
    summary = master.groupby(summary_cols, dropna=False).size().reset_index(name="rows") if summary_cols else pd.DataFrame()

    by_kind = master.groupby(["row_kind"], dropna=False).size().reset_index(name="rows")
    return {
        "summary": summary.sort_values(summary_cols, kind="stable") if summary_cols else summary,
        "by_row_kind": by_kind,
        "missing_required": pd.DataFrame(missing_rows),
        "exact_duplicates": exact_dups,
        "duplicate_keys": pd.concat(key_dups, ignore_index=True, sort=False) if key_dups else pd.DataFrame(),
    }


def write_report(report: dict[str, pd.DataFrame], out_dir: Path, *, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in report.items():
        path = out_dir / f"{prefix}_{name}.csv"
        df.to_csv(path, index=False)
    xlsx = out_dir / f"{prefix}_migration_report.xlsx"
    with pd.ExcelWriter(xlsx) as writer:
        for name, df in report.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


def migrate_one(experiment_root: Path, args: argparse.Namespace) -> None:
    out_master = Path(args.out) if args.out is not None else experiment_root / "master.long.parquet"
    report_dir = Path(args.report_dir) if args.report_dir is not None else experiment_root / "master_migration_report"

    master, inventory = build_master_from_experiment(
        experiment_root,
        include_xlsx_csv=bool(args.include_xlsx_csv),
        hash_source_files=bool(args.hash_source_files),
        max_files=args.max_files,
    )
    before = len(master)
    if args.drop_exact_duplicates:
        master = master.drop_duplicates().reset_index(drop=True)
    report = validation_report(master)
    report["inventory"] = inventory
    report["migration_stats"] = pd.DataFrame(
        [
            {"metric": "rows_before_exact_dedup", "value": before},
            {"metric": "rows_after_exact_dedup", "value": len(master)},
            {"metric": "exact_duplicate_rows", "value": len(report["exact_duplicates"])},
            {"metric": "duplicate_key_groups", "value": len(report["duplicate_keys"])},
        ]
    )

    if args.strict_duplicate_keys and not report["duplicate_keys"].empty:
        write_report(report, report_dir, prefix=experiment_root.parent.name)
        raise ValueError(f"Duplicate logical keys found. See report: {report_dir}")

    if not args.dry_run:
        write_master_table(master, out_master)
        if args.csv:
            write_table_outputs(master, out_master, csv_path=out_master.with_suffix(".csv"))
    write_report(report, report_dir, prefix=experiment_root.parent.name)

    print("Experiment:", experiment_root)
    print("Master:", out_master if not args.dry_run else f"{out_master} (dry-run)")
    print("Rows:", len(master))
    print("Report:", report_dir)


def _default_roots(project_root: Path) -> list[Path]:
    roots = []
    for family in ("brains", "phantoms"):
        root = project_root / "analysis" / family / "ogse_experiments"
        if root.exists():
            roots.append(root)
    return roots


def main() -> None:
    ap = argparse.ArgumentParser(description="Migrate legacy OGSE analysis tables into master.long.parquet.")
    ap.add_argument("experiment_roots", nargs="*", type=Path, help="analysis/<family>/ogse_experiments roots. Defaults to brains and phantoms.")
    ap.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root used to discover default analysis roots.")
    ap.add_argument("--out", type=Path, default=None, help="Output master parquet for a single experiment root.")
    ap.add_argument("--report-dir", type=Path, default=None, help="Report directory for a single experiment root.")
    ap.add_argument("--include-xlsx-csv", action="store_true", help="Also read xlsx/csv legacy tables. Default reads parquet only.")
    ap.add_argument("--hash-source-files", action="store_true", help="Compute SHA256 for each source table. Slower, useful for forensic audits.")
    ap.add_argument("--drop-exact-duplicates", action="store_true", help="Drop exact duplicate rows before writing.")
    ap.add_argument("--strict-duplicate-keys", action="store_true", help="Fail when duplicate logical keys are found.")
    ap.add_argument("--csv", action="store_true", help="Also write master CSV next to the parquet.")
    ap.add_argument("--dry-run", action="store_true", help="Build and report without writing master parquet.")
    ap.add_argument("--max-files", type=int, default=None, help="Debug limit on input files.")
    args = ap.parse_args()

    roots = args.experiment_roots or _default_roots(args.project_root)
    if not roots:
        raise FileNotFoundError("No experiment roots found.")
    if args.out is not None and len(roots) != 1:
        raise ValueError("--out can only be used with one experiment root.")
    if args.report_dir is not None and len(roots) != 1:
        raise ValueError("--report-dir can only be used with one experiment root.")

    for root in roots:
        migrate_one(root.resolve(), args)


if __name__ == "__main__":
    main()
