"""Small command-planning utilities for DWI preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Command:
    """A shell command with optional expected outputs."""

    label: str
    argv: tuple[str, ...]
    outputs: tuple[Path, ...] = ()

    def as_shell(self) -> str:
        return " ".join(shlex.quote(str(item)) for item in self.argv)


def ensure_parent_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def require_tools(tool_names: Iterable[str]) -> None:
    missing = [tool for tool in tool_names if shutil.which(tool) is None]
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise RuntimeError(f"Missing required external tool(s): {missing_text}")


def run_commands(commands: Sequence[Command], *, dry_run: bool, overwrite: bool) -> None:
    for command in commands:
        if command.outputs and not overwrite:
            existing = [path for path in command.outputs if path.exists()]
            if existing:
                existing_text = ", ".join(str(path) for path in existing)
                raise FileExistsError(
                    f"Refusing to overwrite existing output(s) for {command.label}: {existing_text}"
                )

        ensure_parent_dirs(command.outputs)
        print(f"[{command.label}] {command.as_shell()}")

        if dry_run:
            continue

        subprocess.run(command.argv, check=True)
