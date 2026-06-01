"""Cross-process single-instance guard for live entrypoints."""

from __future__ import annotations

import atexit
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Union


_ACTIVE_LOCKS: Dict[str, "SingleInstanceLock"] = {}
_WINDOWS_LOCK_OFFSET = 4096


def _lock_key(path: Path) -> str:
    text = str(path.resolve())
    return text.lower() if os.name == "nt" else text


def _clean_metadata_value(value: object) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def _lock_metadata_payload(name: str) -> bytes:
    argv = " ".join(_clean_metadata_value(item) for item in sys.argv)
    lines = (
        f"pid={os.getpid()}",
        f"parent_pid={os.getppid()}",
        f"name={name}",
        f"sys_executable={_clean_metadata_value(sys.executable)}",
        f"sys_prefix={_clean_metadata_value(sys.prefix)}",
        f"sys_base_prefix={_clean_metadata_value(getattr(sys, 'base_prefix', ''))}",
        f"argv={argv}",
    )
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    if len(payload) >= _WINDOWS_LOCK_OFFSET:
        payload = payload[: _WINDOWS_LOCK_OFFSET - 2] + b"\n"
    return payload


@dataclass
class SingleInstanceLock:
    name: str
    path: Path
    info_path: Path
    _fh: BinaryIO
    _key: str
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._fh.seek(_WINDOWS_LOCK_OFFSET)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._released = True
            _ACTIVE_LOCKS.pop(self._key, None)
            try:
                self._fh.close()
            except Exception:
                pass
            try:
                self.info_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


def acquire_single_instance(
    name: str,
    *,
    lock_dir: Union[str, Path],
) -> Optional[SingleInstanceLock]:
    """Acquire a non-blocking process lock; return None when already running."""
    lock_root = Path(lock_dir)
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"{name}.lock"
    info_path = lock_root / f"{name}.lock.info"
    key = _lock_key(lock_path)
    if key in _ACTIVE_LOCKS:
        return None

    fh = open(lock_path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt

            fh.seek(_WINDOWS_LOCK_OFFSET)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None

    fh.seek(0)
    fh.truncate()
    payload = _lock_metadata_payload(name)
    fh.write(payload)
    fh.flush()
    info_path.write_bytes(payload)

    lock = SingleInstanceLock(
        name=name,
        path=lock_path,
        info_path=info_path,
        _fh=fh,
        _key=key,
    )
    _ACTIVE_LOCKS[key] = lock
    atexit.register(lock.release)
    return lock
