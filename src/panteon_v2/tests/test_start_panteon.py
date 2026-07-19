from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest


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


def test_parse_trade_regime_rejects_removed_policy_mode():
    launcher = _load_launcher()

    with pytest.raises(ValueError, match="multi.*singleton"):
        launcher._parse_trade_regime("policy")


def test_parse_trade_regime_singlton_actor():
    launcher = _load_launcher()

    parsed = launcher._parse_trade_regime("singlton(GeneticsCore)")

    assert parsed.mode == "singleton"
    assert parsed.singleton_actor == "GeneticsCore"


def test_parse_trade_regime_singlton_normalizes_common_actor_alias():
    launcher = _load_launcher()

    parsed = launcher._parse_trade_regime("singlton(genetic_core)")

    assert parsed.mode == "singleton"
    assert parsed.singleton_actor == "GeneticsCore"


def test_parse_trade_regime_accepts_correct_singleton_spelling():
    launcher = _load_launcher()

    parsed = launcher._parse_trade_regime("singleton(genetic_core)")

    assert parsed.mode == "singleton"
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
        "singleton(GeneticsCore)",
    )


def test_unified_launcher_worker_args_normalize_multitrade_alias():
    launcher = _load_launcher()
    parsed = launcher._parse_trade_regime("multitrade")

    assert launcher._worker_args("BITGET", parsed) == (
        "Start_panteon.py",
        "--worker",
        "BITGET",
    )


def test_player_snapshots_are_isolated_by_execution_scope():
    launcher = _load_launcher()

    assert launcher._player_snapshot_name("BITGET", "paper_live_feed") == (
        "bitget_paper_players_v1_snapshot.json"
    )
    assert launcher._player_snapshot_name("BITGET", "demo_futures") == (
        "bitget_demo_players_v1_snapshot.json"
    )
    assert launcher._player_snapshot_name("BITGET", "live_futures") == (
        "bitget_live_players_v1_snapshot.json"
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
    assert "trade_regime=singleton actor=GeneticsCore" in out
    assert (
        "would launch Start_panteon.py --worker BITGET "
        "--trade-regime singleton(GeneticsCore)"
    ) in out


def test_bitget_demo_singleton_dry_run_uses_demo_credentials_without_live_override(
    tmp_path,
):
    launcher = _load_launcher()
    parsed = launcher._parse_trade_regime("singleton(ResearchValidatorAgent)")
    demo_env = {
        "BITGET_TRADING_MODE": "demo_futures",
        "BITGET_DEMO_API_KEY": "demo-key",
        "BITGET_DEMO_SECRET_KEY": "demo-secret",
        "BITGET_DEMO_PASSPHRASE": "demo-passphrase",
    }

    with patch.object(
        launcher,
        "_lock_path",
        return_value=tmp_path / "panteon_v2_bitget.lock",
    ), patch.object(
        launcher,
        "_load_env",
        return_value=None,
    ), patch.object(
        launcher,
        "_lock_is_held",
        return_value=False,
    ), patch.dict(
        os.environ,
        demo_env,
        clear=True,
    ), patch.object(
        launcher,
        "_live_preflight_result",
        wraps=launcher._live_preflight_result,
    ) as preflight:
        result = launcher._launch_exchange(
            "BITGET",
            dry_run=True,
            parsed_regime=parsed,
        )

    assert result.status == "dry_run"
    preflight.assert_called_once_with("BITGET", "demo_futures")
    assert "PANTEON_ALLOW_LIVE_SINGLETON" not in result.message


def test_bitget_demo_dry_run_fails_closed_without_separate_demo_credentials(
    tmp_path,
):
    launcher = _load_launcher()
    parsed = launcher._parse_trade_regime("singleton(ResearchValidatorAgent)")

    with patch.object(
        launcher,
        "_lock_path",
        return_value=tmp_path / "panteon_v2_bitget.lock",
    ), patch.object(
        launcher,
        "_load_env",
        return_value=None,
    ), patch.object(
        launcher,
        "_lock_is_held",
        return_value=False,
    ), patch.dict(
        os.environ,
        {
            "BITGET_TRADING_MODE": "demo_futures",
            "BITGET_API_KEY": "live-key-must-not-be-used",
            "BITGET_SECRET_KEY": "live-secret-must-not-be-used",
            "BITGET_PASSPHRASE": "live-pass-must-not-be-used",
        },
        clear=True,
    ):
        result = launcher._launch_exchange(
            "BITGET",
            dry_run=True,
            parsed_regime=parsed,
        )

    assert result.status == "credentials_missing"
    assert "BITGET_DEMO_API_KEY" in result.message


def test_bitget_demo_main_returns_nonzero_without_demo_credentials(capsys):
    launcher = _load_launcher()

    with patch.object(
        launcher,
        "_load_env",
        return_value=None,
    ), patch.object(
        launcher,
        "_lock_is_held",
        return_value=False,
    ), patch.dict(
        os.environ,
        {"BITGET_TRADING_MODE": "demo_futures"},
        clear=True,
    ):
        exit_code = launcher.main(
            [
                "--only",
                "BITGET",
                "--dry-run",
                "--trade-regime",
                "singleton(ResearchValidatorAgent)",
            ]
        )

    assert exit_code == 2
    assert "credentials_missing" in capsys.readouterr().out


def test_unified_launcher_dry_run_requires_live_preflight(capsys):
    launcher = _load_launcher()

    with patch.object(launcher, "_lock_is_held", return_value=False), patch.dict(
        os.environ,
        {"BITGET_TRADING_MODE": "live_futures"},
    ), patch.object(
        launcher,
        "_live_preflight_result",
        return_value=_FakePreflight(False),
    ):
        exit_code = launcher.main(
            ["--only", "BITGET", "--dry-run", "--trade-regime", "multi"]
        )

    out = capsys.readouterr().out
    assert exit_code == 2
    assert "BITGET: preflight_failed" in out
    assert "live preflight failed" in out


def test_unified_launcher_blocks_live_when_preflight_fails(tmp_path):
    launcher = _load_launcher()
    lock_path = tmp_path / "panteon_v2_bitget.lock"
    parsed = launcher._parse_trade_regime("multi")

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


def test_unified_launcher_multi_dry_run_still_requires_preflight(tmp_path):
    launcher = _load_launcher()
    lock_path = tmp_path / "panteon_v2_bitget.lock"
    parsed = launcher._parse_trade_regime("multi")

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


def test_unified_launcher_spawns_live_singleton_with_override_and_preflight(tmp_path):
    launcher = _load_launcher()
    lock_path = tmp_path / "panteon_v2_bitget.lock"
    parsed = launcher._parse_trade_regime("singleton(GeneticsCore)")

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
        return_value=_FakePreflight(True, ()),
    ) as preflight, patch.object(
        launcher,
        "_spawn_child",
        return_value=12345,
    ) as spawn:
        result = launcher._launch_exchange(
            "BITGET",
            dry_run=False,
            parsed_regime=parsed,
        )

    preflight.assert_called_once_with("BITGET", "live_futures")
    spawn.assert_called_once()
    assert result.status == "launched"
    assert result.pid == 12345


def test_unified_launcher_spawns_bitget_live_multi_only_after_preflight(tmp_path):
    launcher = _load_launcher()
    lock_path = tmp_path / "panteon_v2_bitget.lock"
    parsed = launcher._parse_trade_regime("multi")

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


def test_unified_launcher_blocks_live_singleton_without_override(tmp_path):
    launcher = _load_launcher()
    lock_path = tmp_path / "panteon_v2_bitget.lock"
    parsed = launcher._parse_trade_regime("singleton(GeneticsCore)")

    with patch.object(launcher, "_lock_path", return_value=lock_path), patch.object(
        launcher,
        "_lock_is_held",
        return_value=False,
    ), patch.dict(
        os.environ,
        {"BITGET_TRADING_MODE": "live_futures"},
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
    assert "PANTEON_ALLOW_LIVE_SINGLETON=1" in result.message


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
        exit_code = launcher.main(["--only", "BITGET", "--trade-regime", "multi"])

    spawn.assert_not_called()
    assert exit_code == 2
