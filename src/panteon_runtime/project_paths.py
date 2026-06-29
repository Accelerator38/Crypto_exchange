from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_project_root(start: str | os.PathLike[str] | None = None) -> Path:
    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent

    markers = ("settings.txt", "Start_panteon.py")
    for candidate in (current, *current.parents):
        if all((candidate / marker).exists() for marker in markers):
            return candidate

    return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root()
RUNTIME_DIR = Path(__file__).resolve().parent
STATE_DIR = PROJECT_ROOT / "state"
MEMORY_DIR = STATE_DIR / "memory"
RESULTS_DIR = PROJECT_ROOT / "Results"


def add_runtime_paths() -> None:
    paths = (
        RUNTIME_DIR,
        PROJECT_ROOT / "Genetics_DL_Agents",
        PROJECT_ROOT / "Retrodate_cryptotrade",
        PROJECT_ROOT.parent / "Retrodate_cryptotrade",
        PROJECT_ROOT.parent / "Genetics_DL_Agents",
        PROJECT_ROOT / ".venv" / "Lib" / "site-packages",
    )
    for path in reversed(paths):
        if path.exists():
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)


def project_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))
