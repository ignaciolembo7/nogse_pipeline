"""Import-path guard for direct signal-extraction scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def prepend_sys_path(path: Path) -> None:
    path_text = str(path)
    if path_text in sys.path:
        sys.path.remove(path_text)
    sys.path.insert(0, path_text)


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent

prepend_sys_path(SRC_ROOT)
prepend_sys_path(SCRIPT_DIR)
