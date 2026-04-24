"""Small .env loader for local launch scripts.

The loader intentionally protects real environment variables from the shell or
IDE run configuration. Values from .env.local may override values loaded from
.env, but not values that already existed before this loader ran.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_local_env(
    base_dir: str | os.PathLike[str] | None = None,
    filename=None,
) -> None:
    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    filenames = (".env", ".env.local") if filename is None else filename
    if isinstance(filenames, (str, os.PathLike)):
        filenames = (filenames,)

    protected_keys = {key for key, value in os.environ.items() if str(value).strip()}
    for env_name in filenames:
        env_path = root / env_name
        if not env_path.is_file():
            continue

        for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in protected_keys:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if not value:
                continue
            os.environ[key] = value
