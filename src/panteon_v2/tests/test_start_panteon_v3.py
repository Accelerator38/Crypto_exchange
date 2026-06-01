from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch


def _load_launcher():
    root = Path(__file__).resolve().parents[3]
    path = root / "Start_panteon_v3.py"
    spec = importlib.util.spec_from_file_location("start_panteon_v3", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v3_launcher_enables_mexc_and_bitget_by_default():
    launcher = _load_launcher()

    assert launcher.EXCHANGES["MEXC"] == "ON"
    assert launcher.EXCHANGES["BITGET"] == "ON"
    assert launcher._enabled_exchanges() == ("MEXC", "BITGET")


def test_v3_launcher_skips_exchange_when_instance_lock_is_held(tmp_path):
    launcher = _load_launcher()
    lock_path = tmp_path / "panteon_v2_mexc.lock"

    with patch.object(launcher, "_lock_path", return_value=lock_path), patch.object(
        launcher,
        "_lock_is_held",
        return_value=True,
    ), patch.object(launcher, "_spawn_child") as spawn:
        result = launcher._launch_exchange("MEXC", dry_run=False)

    spawn.assert_not_called()
    assert result.status == "already_running"
    assert result.exchange == "MEXC"
