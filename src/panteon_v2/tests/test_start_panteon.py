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


def test_parse_trade_regime_policy():
    launcher = _load_launcher()

    parsed = launcher._parse_trade_regime("policy")

    assert parsed.mode == "policy"
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


def test_parse_trade_regime_accepts_correct_singleton_spelling():
    launcher = _load_launcher()

    parsed = launcher._parse_trade_regime("singleton(genetic_core)")

    assert parsed.mode == "singlton"
    assert parsed.singleton_actor == "GeneticsCore"


def test_unified_launcher_worker_args_use_new_entrypoint():
    launcher = _load_launcher()

    assert launcher._worker_args("BITGET") == (
        "Start_panteon.py",
        "--worker",
        "BITGET",
    )


def test_unified_launcher_worker_args_preserve_singleton_trade_regime():
    launcher = _load_launcher()
    parsed = launcher._parse_trade_regime("singleton(genetic_core)")

    assert launcher._worker_args("BITGET", parsed) == (
        "Start_panteon.py",
        "--worker",
        "BITGET",
        "--trade-regime",
        "singlton(GeneticsCore)",
    )


def test_unified_launcher_worker_args_preserve_policy_trade_regime():
    launcher = _load_launcher()
    parsed = launcher._parse_trade_regime("policy")

    assert launcher._worker_args("BITGET", parsed) == (
        "Start_panteon.py",
        "--worker",
        "BITGET",
        "--trade-regime",
        "policy",
    )


def test_unified_launcher_main_accepts_trade_regime_override_for_dry_run(capsys):
    launcher = _load_launcher()

    with patch.object(launcher, "_lock_is_held", return_value=False), patch.dict(
        os.environ,
        {"BITGET_TRADING_MODE": "paper"},
    ):
        exit_code = launcher.main(
            [
                "--only",
                "BITGET",
                "--dry-run",
                "--trade-regime",
                "singleton(genetic_core)",
            ]
        )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "trade_regime=singlton actor=GeneticsCore" in out
    assert (
        "would launch Start_panteon.py --worker BITGET "
        "--trade-regime singlton(GeneticsCore)"
    ) in out


def test_unified_launcher_dry_run_reports_blocked_legacy_bitget_live(capsys):
    launcher = _load_launcher()

    with patch.object(launcher, "_lock_is_held", return_value=False), patch.dict(
        os.environ,
        {"BITGET_TRADING_MODE": "live_futures"},
    ):
        exit_code = launcher.main(
            ["--only", "BITGET", "--dry-run", "--trade-regime", "multi"]
        )

    out = capsys.readouterr().out
    assert exit_code == 2
    assert "BITGET: blocked" in out
    assert "PANTEON_TRADE_REGIME=policy" in out


def test_unified_launcher_blocks_live_when_preflight_fails(tmp_path):
    launcher = _load_launcher()
    lock_path = tmp_path / "panteon_v2_bitget.lock"
    parsed = launcher._parse_trade_regime("policy")

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
            parsed_regime=parsed,
        )

    spawn.assert_not_called()
    assert result.status == "preflight_failed"
    assert "live preflight failed" in result.message


def test_unified_launcher_policy_dry_run_still_requires_preflight(tmp_path):
    launcher = _load_launcher()
    lock_path = tmp_path / "panteon_v2_bitget.lock"
    parsed = launcher._parse_trade_regime("policy")

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
            dry_run=True,
            parsed_regime=parsed,
        )

    spawn.assert_not_called()
    assert result.status == "preflight_failed"


def test_unified_launcher_blocks_legacy_live_singleton_even_with_override(tmp_path):
    launcher = _load_launcher()
    lock_path = tmp_path / "panteon_v2_bitget.lock"
    parsed = launcher._parse_trade_regime("singlton(GeneticsCore)")

    with patch.object(launcher, "_lock_path", return_value=lock_path), patch.object(
        launcher,
        "_lock_is_held",
        return_value=False,
    ), patch.dict(
        os.environ,
        {
            "BITGET_TRADING_MODE": "live_futures",
            "PANTEON_ALLOW_LIVE_SINGLETON": "1",
        },
        clear=True,
    ), patch.object(
        launcher,
        "_live_preflight_result",
    ) as preflight, patch.object(
        launcher,
        "_spawn_child",
    ) as spawn:
        result = launcher._launch_exchange(
            "BITGET",
            dry_run=False,
            parsed_regime=parsed,
        )

    preflight.assert_not_called()
    spawn.assert_not_called()
    assert result.status == "blocked"
    assert "BITGET live requires PANTEON_TRADE_REGIME=policy" in result.message
    assert "virtual-only" in result.message


def test_unified_launcher_spawns_bitget_live_policy_only_after_preflight(tmp_path):
    launcher = _load_launcher()
    lock_path = tmp_path / "panteon_v2_bitget.lock"
    parsed = launcher._parse_trade_regime("policy")

    with patch.object(launcher, "_lock_path", return_value=lock_path), patch.object(
        launcher,
        "_lock_is_held",
        return_value=False,
    ), patch.dict(os.environ, {"BITGET_TRADING_MODE": "live_futures"}), patch.object(
        launcher,
        "_live_preflight_result",
        return_value=_FakePreflight(True, ()),
    ) as preflight, patch.object(
        launcher,
        "_spawn_child",
        return_value=12345,
    ) as spawn:
        result = launcher._launch_exchange(
            "BITGET",
            dry_run=False,
            python_executable="python-test",
            parsed_regime=parsed,
        )

    preflight.assert_called_once_with("BITGET", "live_futures")
    spawn.assert_called_once_with("BITGET", "python-test", parsed)
    assert result.status == "launched"
    assert result.pid == 12345


def test_unified_launcher_rejects_policy_in_virtual_mode(tmp_path):
    launcher = _load_launcher()
    lock_path = tmp_path / "panteon_v2_bitget.lock"
    parsed = launcher._parse_trade_regime("policy")

    with patch.object(launcher, "_lock_path", return_value=lock_path), patch.object(
        launcher,
        "_lock_is_held",
        return_value=False,
    ), patch.dict(os.environ, {"BITGET_TRADING_MODE": "paper"}), patch.object(
        launcher,
        "_live_preflight_result",
    ) as preflight, patch.object(
        launcher,
        "_spawn_child",
    ) as spawn:
        result = launcher._launch_exchange(
            "BITGET",
            dry_run=False,
            parsed_regime=parsed,
        )

    preflight.assert_not_called()
    spawn.assert_not_called()
    assert result.status == "blocked"
    assert "reserved for non-virtual BITGET" in result.message


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
        exit_code = launcher.main(["--only", "BITGET", "--trade-regime", "policy"])

    spawn.assert_not_called()
    assert exit_code == 2
