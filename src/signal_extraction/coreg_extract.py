from __future__ import annotations

import sys

try:
    import repo_bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import repo_bootstrap  # noqa: F401

from coreg_extract_brain import main as brain_main
from coreg_extract_phantom import main as phantom_main


def main() -> None:
    argv = sys.argv[1:]

    if "--phantom-direct" in argv:
        sys.argv = [sys.argv[0]] + [a for a in argv if a != "--phantom-direct"]
        phantom_main()
    else:
        brain_main()


if __name__ == "__main__":
    main()
