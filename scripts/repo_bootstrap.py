"""Import-path guard for repository command-line scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def prepend_sys_path(path: Path) -> None:
    path_text = str(path)
    if path_text in sys.path:
        sys.path.remove(path_text)
    sys.path.insert(0, path_text)


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if SRC_ROOT.is_dir():
    prepend_sys_path(SRC_ROOT)
