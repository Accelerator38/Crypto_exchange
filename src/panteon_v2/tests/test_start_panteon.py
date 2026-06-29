from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch


def _load_launcher():
    root = Path(__file__).resolve().parents[3]
    path = root / "Start_panteon.py"
    spec = importlib.util.spec_from_file_location("start_panteon", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _FakePreflight:
    passed: bool
    reasons: tuple[str, ...] = ("matrix_failed",)


def test_unified_launcher_defaults_to_bitget_only():
    launcher = _load_launcher()

    assert launcher.BITGET == "ON"
    assert launcher.MEXC == "OFF"
    assert launcher._enabled_exchanges() == ("BITGET",)


def test_parse_trade_regime_multi():
    launcher = _load_launcher()

    parsed = launcher._parse_trade_regime("multi")

    assert parsed.mode == "multi"
    assert parsed.singleton_actor == ""


def test_parse_trade_regime_singlton_actor():
    launcher = _load_launcher()

    parsed = launcher._parse_trade_regime("singlton(GeneticsCore)")

    assert parsed.mode == "singlton"
    assert parsed.singleton_actor == "GeneticsCore"


def test_parse_trade_regime_singlton_normalizes_common_actor_alias():
    launcher = _load_launcher()

    parsed = launcher._parse_trade_regime("singlton(genetic_core)")

    assert parsed.mode == "singlton"
    assert parsed.singleton_actor == "GeneticsCore"


def test_unified_launcher_worker_args_use_new_entrypoint():
    launcher = _load_launcher()

    assert launcher._worker_args("BITGET") == (
        "Start_panteon.py",
        "--worker",
        "BITGET",
    )


def test_unified_launcher_blocks_live_when_preflight_fails(tmp_path):
    launcher = _load_launcher()
    lock_path = tmp_path / "panteon_v2_bitget.lock"

    with patch.object(launcher, "_lock_path", return_value=lock_path), patch.object(
        launcher,
        "_lock_is_held",
        return_value=False,
    ), patch.dict(os.environ, {"BITGET_TRADING_MODE": "live_futures"}), patch.object(
        launcher,
        "_live_preflight_result",
        return_value=_FakePreflight(False),
    ), patch.object(
        launcher,
        "_spawn_child",
    ) as spawn:
        result = launcher._launch_exchange(
            "BITGET",
            dry_run=False,
            python_executable="python-test",
        )

    spawn.assert_not_called()
    assert result.status == "preflight_failed"
    assert "live preflight failed" in result.message


def test_unified_launcher_main_returns_nonzero_when_preflight_fails(tmp_path):
    launcher = _load_launcher()
    lock_path = tmp_path / "panteon_v2_bitget.lock"

    with patch.object(launcher, "_lock_path", return_value=lock_path), patch.object(
        launcher,
        "_lock_is_held",
        return_value=False,
    ), patch.dict(os.environ, {"BITGET_TRADING_MODE": "live_futures"}), patch.object(
        launcher,
        "_live_preflight_result",
        return_value=_FakePreflight(False),
    ), patch.object(
        launcher,
        "_spawn_child",
    ) as spawn:
        exit_code = launcher.main(["--only", "BITGET"])

    spawn.assert_not_called()
    assert exit_code == 2
