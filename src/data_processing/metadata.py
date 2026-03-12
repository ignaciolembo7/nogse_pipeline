from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re

@dataclass(frozen=True)
class FileMeta:
    filename: str
    tokens: list[str]
    parsed: dict[str, object]

# patrones típicos (si no matchea, no pasa nada)
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("date_yyyymmdd", re.compile(r"^(?P<v>\d{8})")),
    ("nbvals",        re.compile(r"(?P<v>\d+)bval", re.IGNORECASE)),
    ("ndirs",         re.compile(r"(?P<v>\d+)dir",  re.IGNORECASE)),
    ("d_ms",          re.compile(r"_d(?P<v>\d+)",   re.IGNORECASE)),
    ("ogse_hz",       re.compile(r"_Hz(?P<v>\d+)",  re.IGNORECASE)),
    ("bmax",          re.compile(r"_b(?P<v>\d+)",   re.IGNORECASE)),
    ("seq_id",        re.compile(r"_(?P<v>\d+)_results", re.IGNORECASE)),
]

def parse_filename_metadata(path: str | Path) -> FileMeta:
    p = Path(path)
    name = p.name

    # tokens crudos por si mañana aparece info nueva
    tokens = re.split(r"[_\-\s]+", p.stem)

    parsed: dict[str, object] = {"encoding": None, "has_qc": False}
    if "OGSE" in tokens:
        parsed["encoding"] = "OGSE"
    elif "PGSE" in tokens:
        parsed["encoding"] = "PGSE"

    parsed["has_qc"] = any("QC" in t.upper() for t in tokens)

    for key, rx in _PATTERNS:
        m = rx.search(name)
        if not m:
            continue
        v = m.group("v")
        # casteos típicos
        if key in {"nbvals", "ndirs", "d_ms", "ogse_hz", "bmax", "seq_id"}:
            parsed[key] = int(v)
        else:
            parsed[key] = v

    return FileMeta(filename=name, tokens=tokens, parsed=parsed)
