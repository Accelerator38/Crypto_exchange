"""Tests for Retrodate market benchmark helpers."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import panteon_v2.analysis.retrodate_market_runner as runner
from panteon_v2.analysis.soft_allocator import ShadowPnLEvent, write_shadow_pnl_events
from panteon_v2.analysis.retrodate_market_runner import (
    RetrodateFileSelection,
    RetrodateMarketConfig,
    RetrodateRunSummary,
    RetrodateSnapshotState,
    load_retrodate_year_snapshots,
    select_retrodate_files,
)
from panteon_v2.app.agent_bootstrap import (
    ActionFilterAgent,
    register_experimental_flash_agents,
)
from panteon_v2.analysis.retrodate_validator import RetrodateDirReport
from panteon_v2.analysis.retrodate_validator import RetrodateValidationError
from panteon_v2.attribution import (
    CandidateRejected,
    CandidateScored,
    EventLog,
    ExecutionAttributed,
    PositionClosed,
    RegimeDetected,
    ShadowActorUpdated,
)
from panteon_v2.domain.types import Action, Regime


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "open", "high", "low", "close", "volume", "symbol", "datetime"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(timestamp: int, close: float, symbol: str) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": close * 10,
        "symbol": symbol,
        "datetime": "",
    }


def test_write_walk_forward_report_from_event_log(tmp_path):
    event_log = EventLog()
    event_log.emit(RegimeDetected(bar=1, regime=Regime.BULLISH))
    event_log.emit(
        PositionClosed(
            bar=2,
            sym="BTC/USDT",
            realized_pnl=5.0,
            by_player="Panteon_Flash",
            by_agent="AgentA",
        )
    )

    report_path = runner._write_walk_forward_report_from_event_log(
        tmp_path,
        event_log,
    )

    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    assert data["by_regime"]["bullish"]["closed_trades"] == 1
    assert data["by_regime"]["bullish"]["net_pnl"] == 5.0


def test_walk_forward_report_from_event_log_does_not_write_full_events_jsonl(tmp_path):
    event_log = EventLog()
    event_log.emit(RegimeDetected(bar=1, regime=Regime.BULLISH))
    event_log.emit(
        PositionClosed(
            bar=2,
            sym="BTC/USDT",
            realized_pnl=10.0,
            by_player="Panteon_Flash",
            by_agent="AgentA",
        )
    )

    report_path = runner._write_walk_forward_report_from_event_log(
        tmp_path,
        event_log,
    )

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    assert report["totals"]["closed_trades"] == 1
    assert report["by_regime"]["bullish"]["net_pnl"] == 10.0
    assert report["event_logs"] == []
    assert not (tmp_path / "events.jsonl").exists()


def test_load_retrodate_year_snapshots_builds_stride_multi_symbol_bars(tmp_path):
    csv_path = tmp_path / "crypto_1m_2025_all_symbols.csv"
    _write_rows(
        csv_path,
        [
            _row(1735689600000, 100.0, "BTC/USDT"),
            _row(1735689600000, 10.0, "ETH/USDT"),
            _row(1735691400000, 999.0, "BTC/USDT"),
            _row(1735693200000, 110.0, "BTC/USDT"),
            _row(1735693200000, 11.0, "ETH/USDT"),
        ],
    )

    snapshots = load_retrodate_year_snapshots(csv_path, stride_minutes=60, state=RetrodateSnapshotState())

    assert [snap.bar for snap in snapshots] == [1, 2]
    assert snapshots[0].prices == {"BTC/USDT": 100.0, "ETH/USDT": 10.0}
    assert snapshots[0].volumes == {"BTC/USDT": 1000.0, "ETH/USDT": 100.0}
    assert snapshots[0].timestamp.isoformat() == "2025-01-01T00:00:00+00:00"
    assert snapshots[0].month == 1
    assert snapshots[1].prices["BTC/USDT"] == 110.0


def test_load_retrodate_year_snapshots_populates_lookback_returns(tmp_path):
    csv_path = tmp_path / "crypto_1m_2025_all_symbols.csv"
    start_ts = 1735689600000
    _write_rows(
        csv_path,
        [
            _row(start_ts + index * 60 * 60 * 1000, 100.0 + index, "BTC/USDT")
            for index in range(13)
        ],
    )

    snapshots = load_retrodate_year_snapshots(csv_path, stride_minutes=60, state=RetrodateSnapshotState())

    assert snapshots[-1].lookback_returns_pct["BTC/USDT"][12] == pytest.approx(12.0)
    assert snapshots[-1].lookback_volatility_pct["BTC/USDT"][12] == pytest.approx(
        0.9493931889785356
    )


def test_load_retrodate_year_snapshots_keeps_state_across_files(tmp_path):
    first = tmp_path / "crypto_1m_2025_all_symbols.csv"
    second = tmp_path / "crypto_1m_2026_all_symbols.csv"
    _write_rows(first, [_row(1735689600000, 100.0, "BTC/USDT")])
    _write_rows(second, [_row(1767225600000, 120.0, "BTC/USDT")])
    state = RetrodateSnapshotState()

    first_snapshots = load_retrodate_year_snapshots(first, stride_minutes=60, state=state)
    second_snapshots = load_retrodate_year_snapshots(second, stride_minutes=60, state=state)

    assert first_snapshots[0].bar == 1
    assert second_snapshots[0].bar == 2


def test_select_retrodate_files_excludes_invalid_requested_years(tmp_path):
    _write_rows(
        tmp_path / "crypto_1m_2025_all_symbols.csv",
        [_row(1735689600000, 100.0, "BTC/USDT")],
    )
    _write_rows(
        tmp_path / "crypto_1m_2026_all_symbols.csv",
        [_row(1640995200000, 100.0, "BTC/USDT")],
    )

    selection = select_retrodate_files(
        RetrodateMarketConfig(data_dir=tmp_path, years=(2025, 2026), stride_minutes=60)
    )

    assert [path.name for path in selection.valid_files] == ["crypto_1m_2025_all_symbols.csv"]
    assert [item.path.name for item in selection.excluded_files] == ["crypto_1m_2026_all_symbols.csv"]
    assert selection.executed_years == (2025,)
    assert selection.missing_years == (2026,)


def test_select_retrodate_files_can_fail_on_invalid_requested_years(tmp_path):
    _write_rows(
        tmp_path / "crypto_1m_2026_all_symbols.csv",
        [_row(1640995200000, 100.0, "BTC/USDT")],
    )

    with pytest.raises(RetrodateValidationError):
        select_retrodate_files(
            RetrodateMarketConfig(
                data_dir=tmp_path,
                years=(2026,),
                stride_minutes=60,
                invalid_policy="fail",
            )
        )


def test_retrodate_regime_classifier_detects_large_btc_drop(tmp_path):
    csv_path = tmp_path / "crypto_1m_2025_all_symbols.csv"
    rows = []
    timestamp = 1735689600000
    for index in range(25):
        close = 100.0 if index < 24 else 90.0
        rows.append(_row(timestamp + index * 60 * 60 * 1000, close, "BTC/USDT"))
    _write_rows(csv_path, rows)

    snapshots = load_retrodate_year_snapshots(csv_path, stride_minutes=60, state=RetrodateSnapshotState())

    assert snapshots[-1].regime == Regime.CRASH


def test_cli_config_accepts_use_v3_rolling_score_flag():
    config = runner._parse_cli_config([
        "--years",
        "2025",
        "--use-v3-rolling-score",
        "--use-v3-shadow-rolling-score",
        "--use-v3-soft-shadow-score",
        "--use-v3-entry-causal-score",
        "--enable-v3-shadow-flat-handoff",
        "--enable-v3-shadow-fresh-handoff",
        "--v3-shadow-fresh-handoff-max-age-bars",
        "1",
        "--v3-shadow-fresh-handoff-allow-nonpositive-unrealized",
        "--v3-shadow-rolling-window-bars",
        "24",
        "--v3-shadow-rolling-min-closed-trades",
        "50",
        "--v3-entry-causal-min-filled",
        "2",
        "--v3-entry-causal-actionability-weight",
        "0.5",
        "--enable-actionable-fallback",
        "--actionable-fallback-min-score",
        "0.25",
        "--actionable-fallback-require-has-data",
        "--enable-current-actionable-candidate-layer",
        "--use-v3-executable-soft-confirmed-score",
        "--solo-agent-candidate-limit",
        "6",
        "--enable-fixed-agent-players",
        "--fixed-agent-player-set",
        "Fixed_AB=AgentA,AgentB",
    ])

    assert config.use_v3_rolling_score is True
    assert config.use_v3_shadow_rolling_score is True
    assert config.use_v3_soft_shadow_score is True
    assert config.use_v3_entry_causal_score is True
    assert config.v3_shadow_position_gate_enabled is True
    assert config.v3_shadow_flat_handoff_enabled is True
    assert config.v3_shadow_fresh_handoff_enabled is True
    assert config.v3_shadow_fresh_handoff_max_age_bars == 1
    assert config.v3_shadow_fresh_handoff_require_positive_unrealized is False
    assert config.v3_shadow_rolling_window_bars == 24
    assert config.v3_shadow_rolling_min_closed_trades == 50
    assert config.v3_entry_causal_min_filled == 2
    assert config.v3_entry_causal_actionability_weight == 0.5
    assert config.actionable_fallback_enabled is True
    assert config.actionable_fallback_min_score == 0.25
    assert config.actionable_fallback_require_has_data is True
    assert config.current_actionable_candidate_layer_enabled is True
    assert config.use_v3_executable_soft_confirmed_score is True
    assert config.solo_agent_candidate_limit == 6
    assert config.fixed_agent_players_enabled is True
    assert config.fixed_agent_player_sets == (("Fixed_AB", ("AgentA", "AgentB")),)


def test_cli_and_flash_allocator_config_accept_flash_flags():
    config = runner._parse_cli_config([
        "--years",
        "2025",
        "--enable-flash",
        "--flash-min-score-to-trade",
        "0.15",
        "--flash-actionable-bonus",
        "0.35",
        "--flash-no-data-score",
        "-0.25",
        "--flash-min-closed-trades-to-trade",
        "8",
        "--flash-min-pnl-pct-to-trade",
        "0.2",
        "--enable-flash-shadow-confirmation",
        "--flash-shadow-min-score",
        "0.1",
        "--flash-shadow-min-closed-trades",
        "55",
        "--flash-shadow-min-full-open-closed-trades",
        "2",
        "--enable-flash-symbol-shadow-confirmation",
        "--enable-flash-shadow-actor-fallback-confirmation",
        "--enable-flash-shadow-base-fallback-confirmation",
        "--enable-flash-shadow-signal-handoff",
        "--flash-shadow-actor-fallback-min-base-score",
        "3.5",
        "--flash-shadow-base-fallback-actor-key",
        "agent:StrongActor",
        "--flash-shadow-base-fallback-actor-key",
        "ensemble:SafeSolo",
        "--flash-actor-switch-margin",
        "0.45",
        "--flash-anchor-actor-key",
        "ensemble:Solo_MomentumScalper",
        "--flash-portfolio-actor-key",
        "ensemble:Solo_MomentumScalper",
        "--flash-portfolio-actor-key",
        "ensemble:Solo_LiveCrashHunter",
        "--enable-flash-portfolio-shadow-bootstrap-min-closed",
        "--flash-anchor-min-score-to-trade",
        "1.5",
        "--flash-anchor-shadow-min-score",
        "-0.1",
        "--flash-anchor-min-score-advantage",
        "0.8",
        "--enable-flash-prefer-solo-player-wrappers",
        "--enable-flash-prefer-proven-solo-player-wrappers",
        "--flash-proven-solo-min-score-advantage",
        "0.75",
        "--flash-max-signals-per-actor",
        "2",
        "--enable-flash-overextension-guard",
        "--flash-overextension-lookback-bars",
        "12",
        "--flash-short-overextension-return-floor-pct",
        "-8.0",
        "--flash-long-overextension-return-ceiling-pct",
        "8.0",
        "--enable-flash-overextension-volatility-normalized",
        "--flash-short-overextension-z-floor",
        "-2.5",
        "--flash-long-overextension-z-ceiling",
        "2.5",
        "--flash-overextension-min-volatility-pct",
        "0.2",
        "--enable-flash-shadow-quality-confirmation",
        "--flash-shadow-min-win-rate-pct",
        "60.0",
        "--flash-shadow-max-recent-downside-usd",
        "5.0",
        "--flash-shadow-min-pnl-per-trade-lcb-usd",
        "0.05",
        "--flash-shadow-pnl-per-trade-lcb-z",
        "1.0",
        "--flash-shadow-pnl-per-trade-lcb-penalty-floor-usd",
        "0.0",
        "--flash-shadow-pnl-per-trade-lcb-penalty-weight",
        "20.0",
        "--flash-deny-signal-key",
        "agent:MomentumScalper|ATOM/USDT|SPOT_BUY_FULL",
        "--flash-deny-signal-key",
        "agent:MomentumScalper|MATIC/USDT|SPOT_BUY_FULL",
        "--flash-terminal-deny-signal-key",
        "ensemble:Solo_LiveCrashHunter|ATOM/USDT|FUT_SHORT_FULL",
        "--flash-denied-open-symbol",
        "atom/usdt",
        "--flash-denied-open-regime",
        "neutral",
        "--enable-flash-degradation-guard",
        "--enable-flash-degradation-actor-guard",
        "--flash-degradation-actor-scope",
        "actor_regime",
        "--flash-degradation-signal-cooldown-bars",
        "144",
        "--flash-degradation-actor-cooldown-bars",
        "72",
        "--enable-flash-degradation-symbol-guard",
        "--flash-degradation-symbol-cooldown-bars",
        "48",
        "--flash-degradation-symbol-window-closed-trades",
        "2",
        "--flash-degradation-symbol-min-closed-trades",
        "2",
        "--flash-degradation-symbol-max-recent-pnl-usd",
        "-10.0",
        "--flash-degradation-window-closed-trades",
        "3",
        "--flash-degradation-min-closed-trades",
        "3",
        "--flash-degradation-max-recent-pnl-usd",
        "-25.0",
        "--flash-degradation-reserve-actor-cap",
        "--enable-flash-degradation-recovery",
        "--flash-degradation-recovery-min-closed-trades",
        "3",
        "--flash-degradation-recovery-min-recent-pnl-usd",
        "0.0",
        "--enable-flash-stale-position-exit",
        "--flash-stale-position-exit-max-age-bars",
        "168",
    ])

    flash_config = runner._build_flash_allocator_config(config)

    assert config.flash_enabled is True
    assert flash_config.min_score_to_trade == 0.15
    assert flash_config.actionable_bonus == 0.35
    assert flash_config.no_data_score == -0.25
    assert flash_config.min_closed_trades_to_trade == 8
    assert flash_config.min_pnl_pct_to_trade == 0.2
    assert flash_config.shadow_confirmation_enabled is True
    assert flash_config.shadow_symbol_confirmation_enabled is True
    assert flash_config.shadow_actor_fallback_confirmation_enabled is True
    assert flash_config.shadow_base_fallback_confirmation_enabled is True
    assert flash_config.shadow_signal_handoff_enabled is True
    assert flash_config.shadow_actor_fallback_min_base_score == 3.5
    assert flash_config.shadow_base_fallback_actor_keys == (
        "agent:StrongActor",
        "ensemble:SafeSolo",
    )
    assert flash_config.actor_switch_margin == 0.45
    assert flash_config.anchor_actor_keys == ("ensemble:Solo_MomentumScalper",)
    assert flash_config.portfolio_actor_keys == (
        "ensemble:Solo_MomentumScalper",
        "ensemble:Solo_LiveCrashHunter",
    )
    assert flash_config.portfolio_shadow_bootstrap_min_closed_enabled is True
    assert flash_config.anchor_min_score_to_trade == 1.5
    assert flash_config.anchor_shadow_min_score == -0.1
    assert flash_config.anchor_min_score_advantage == 0.8
    assert flash_config.prefer_solo_player_wrappers_enabled is True
    assert flash_config.prefer_proven_solo_player_wrappers_enabled is True
    assert flash_config.proven_solo_min_score_advantage == 0.75
    assert flash_config.shadow_confirmation_min_score == 0.1
    assert flash_config.shadow_confirmation_min_closed_trades == 55
    assert flash_config.shadow_confirmation_min_full_open_closed_trades == 2
    assert flash_config.max_signals_per_actor == 2
    assert flash_config.open_overextension_guard_enabled is True
    assert flash_config.overextension_lookback_bars == 12
    assert flash_config.short_overextension_return_floor_pct == -8.0
    assert flash_config.long_overextension_return_ceiling_pct == 8.0
    assert flash_config.overextension_volatility_normalized_enabled is True
    assert flash_config.short_overextension_z_floor == -2.5
    assert flash_config.long_overextension_z_ceiling == 2.5
    assert flash_config.overextension_min_volatility_pct == 0.2
    assert flash_config.shadow_quality_confirmation_enabled is True
    assert flash_config.shadow_confirmation_min_win_rate_pct == 60.0
    assert flash_config.shadow_confirmation_max_recent_downside_usd == 5.0
    assert flash_config.shadow_confirmation_min_pnl_per_trade_lcb_usd == 0.05
    assert flash_config.shadow_confirmation_pnl_per_trade_lcb_z == 1.0
    assert flash_config.shadow_confirmation_pnl_per_trade_lcb_penalty_floor_usd == 0.0
    assert flash_config.shadow_confirmation_pnl_per_trade_lcb_penalty_weight == 20.0
    assert flash_config.denied_signal_keys == (
        "agent:MomentumScalper|ATOM/USDT|SPOT_BUY_FULL",
        "agent:MomentumScalper|MATIC/USDT|SPOT_BUY_FULL",
    )
    assert flash_config.terminal_denied_signal_keys == (
        "ensemble:Solo_LiveCrashHunter|ATOM/USDT|FUT_SHORT_FULL",
    )
    assert flash_config.denied_open_symbols == ("ATOM/USDT",)
    assert flash_config.denied_open_regimes == (Regime.NEUTRAL,)
    assert flash_config.degradation_guard_enabled is True
    assert flash_config.degradation_actor_guard_enabled is True
    assert flash_config.degradation_actor_scope == "actor_regime"
    assert flash_config.degradation_signal_cooldown_bars == 144
    assert flash_config.degradation_actor_cooldown_bars == 72
    assert flash_config.degradation_symbol_guard_enabled is True
    assert flash_config.degradation_symbol_cooldown_bars == 48
    assert flash_config.degradation_symbol_window_closed_trades == 2
    assert flash_config.degradation_symbol_min_closed_trades == 2
    assert flash_config.degradation_symbol_max_recent_pnl_usd == -10.0
    assert flash_config.degradation_window_closed_trades == 3
    assert flash_config.degradation_min_closed_trades == 3
    assert flash_config.degradation_max_recent_pnl_usd == -25.0
    assert flash_config.degradation_reserve_actor_cap is True
    assert flash_config.degradation_recovery_enabled is True
    assert flash_config.degradation_recovery_min_closed_trades == 3
    assert flash_config.degradation_recovery_min_recent_pnl_usd == 0.0
    assert config.flash_stale_position_exit_enabled is True
    assert config.flash_stale_position_exit_max_age_bars == 168
    assert config.flash_stale_position_exit_require_nonpositive_unrealized is True


def test_cli_and_flash_allocator_config_accept_promotion_manifest_flags(tmp_path):
    manifest_path = tmp_path / "flash_promotion_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "allowed_signal_keys": [
                    "agent:MomentumScalper|BTC/USDT|FUT_LONG_FULL"
                ]
            }
        ),
        encoding="utf-8",
    )

    config = runner._parse_cli_config([
        "--years",
        "2025",
        "--enable-flash-promotion-manifest",
        "--flash-promotion-manifest-path",
        str(manifest_path),
        "--flash-promotion-min-full-closed-trades",
        "3",
        "--flash-promotion-min-latest-closed-trades",
        "2",
        "--flash-promotion-min-full-pnl-pct",
        "0.1",
        "--flash-promotion-min-latest-pnl-pct",
        "0.2",
        "--flash-promotion-min-full-pnl-per-trade-lcb-pct",
        "0.03",
        "--flash-promotion-min-latest-pnl-per-trade-lcb-pct",
        "0.01",
        "--flash-promotion-pnl-per-trade-lcb-z",
        "1.0",
        "--flash-promotion-max-drawdown-pct",
        "12.5",
        "--flash-promotion-min-win-rate-pct",
        "45.0",
        "--flash-promotion-max-recent-downside-usd",
        "3.5",
        "--flash-promoted-signal-key",
        "agent:FundingArb|ETH/USDT|FUT_SHORT_FULL",
        "--enable-flash-earned-cap-overrides",
        "--flash-promoted-actor-cap-override",
        "agent:FundingArb=3",
    ])
    flash_config = runner._build_flash_allocator_config(config)

    assert config.flash_promotion_manifest_enabled is True
    assert config.flash_promotion_manifest_path == manifest_path
    assert config.flash_promotion_min_full_closed_trades == 3
    assert config.flash_promotion_min_latest_closed_trades == 2
    assert config.flash_promotion_min_full_pnl_pct == 0.1
    assert config.flash_promotion_min_latest_pnl_pct == 0.2
    assert config.flash_promotion_min_full_pnl_per_trade_lcb_pct == 0.03
    assert config.flash_promotion_min_latest_pnl_per_trade_lcb_pct == 0.01
    assert config.flash_promotion_pnl_per_trade_lcb_z == 1.0
    assert config.flash_promotion_max_drawdown_pct == 12.5
    assert config.flash_promotion_min_win_rate_pct == 45.0
    assert config.flash_promotion_max_recent_downside_usd == 3.5
    assert config.flash_promoted_signal_keys == (
        "agent:FundingArb|ETH/USDT|FUT_SHORT_FULL",
        "agent:MomentumScalper|BTC/USDT|FUT_LONG_FULL",
    )
    assert config.flash_earned_cap_overrides_enabled is True
    assert config.flash_promoted_actor_cap_overrides == ("agent:FundingArb=3",)
    assert flash_config.promotion_manifest_enabled is True
    assert flash_config.promoted_actor_cap_overrides == ("agent:FundingArb=3",)
    manifest_config = runner._flash_promotion_manifest_config(config)
    assert manifest_config.min_full_pnl_per_trade_lcb_pct == 0.03
    assert manifest_config.min_latest_pnl_per_trade_lcb_pct == 0.01
    assert manifest_config.pnl_per_trade_lcb_z == 1.0


def test_flash_manifest_keys_combine_manifest_and_explicit_values(tmp_path):
    manifest_path = tmp_path / "flash_promotion_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "allowed_signal_keys": [
                    "agent:Alpha|BTC/USDT|FUT_LONG_FULL",
                    "agent:Alpha|BTC/USDT|FUT_LONG_FULL",
                    "",
                ],
                "ignored": ["agent:Ignored|BTC/USDT|FUT_LONG_FULL"],
            }
        ),
        encoding="utf-8",
    )

    config = RetrodateMarketConfig(
        flash_promotion_manifest_enabled=True,
        flash_promotion_manifest_path=manifest_path,
        flash_promoted_signal_keys=("agent:Beta|ETH/USDT|FUT_SHORT_FULL", ""),
    )

    assert config.flash_promoted_signal_keys == (
        "agent:Alpha|BTC/USDT|FUT_LONG_FULL",
        "agent:Beta|ETH/USDT|FUT_SHORT_FULL",
    )


def test_flash_earned_cap_auto_derives_actor_cap_from_allowed_keys():
    config = RetrodateMarketConfig(
        flash_promotion_manifest_enabled=True,
        flash_earned_cap_overrides_enabled=True,
        flash_promoted_signal_keys=(
            "agent:Alpha|BTC/USDT|FUT_LONG_FULL",
            "agent:Alpha|ETH/USDT|FUT_SHORT_FULL",
            "agent:Beta|BTC/USDT|FUT_LONG_FULL",
            "agent:Gamma|BTC/USDT|FUT_LONG_FULL",
            "agent:Gamma|ETH/USDT|FUT_SHORT_FULL",
        ),
        flash_promoted_actor_cap_overrides=("agent:Gamma=4",),
    )

    flash_config = runner._build_flash_allocator_config(config)

    assert flash_config.promoted_actor_cap_overrides == (
        "agent:Gamma=4",
        "agent:Alpha=2",
    )


def test_retro_config_allows_short_degradation_window_when_recovery_disabled():
    config = RetrodateMarketConfig(
        flash_degradation_window_closed_trades=1,
        flash_degradation_min_closed_trades=1,
        flash_degradation_recovery_enabled=False,
    )

    assert config.flash_degradation_window_closed_trades == 1
    assert config.flash_degradation_recovery_min_closed_trades == 3


def test_cli_can_enable_experimental_flash_actor_layer():
    config = runner._parse_cli_config([
        "--years",
        "2025",
        "--enable-experimental-flash-actors",
    ])

    assert config.experimental_flash_actors_enabled is True
    assert config.experimental_flash_real_actors_enabled is False

    default_config = runner._parse_cli_config(["--years", "2025"])
    assert default_config.experimental_flash_actors_enabled is False
    assert default_config.experimental_flash_real_actors_enabled is False


def test_cli_can_allow_experimental_flash_actors_to_trade():
    config = runner._parse_cli_config([
        "--years",
        "2025",
        "--enable-experimental-flash-actors",
        "--allow-experimental-flash-real-actors",
    ])

    assert config.experimental_flash_actors_enabled is True
    assert config.experimental_flash_real_actors_enabled is True


def test_experimental_flash_actors_default_to_shadow_only_policy():
    shadow_only = runner._parse_cli_config([
        "--years",
        "2025",
        "--enable-experimental-flash-actors",
    ])
    strategist = runner._build_strategist_config(shadow_only)

    assert "MomentumScalperShortOnly" in strategist.hard_policy_deny_labels
    assert "MomentumScalperSpotQuality" in strategist.hard_policy_deny_labels
    assert "VolBreakoutSpotOnly" in strategist.hard_policy_deny_labels

    real_enabled = runner._parse_cli_config([
        "--years",
        "2025",
        "--enable-experimental-flash-actors",
        "--allow-experimental-flash-real-actors",
    ])
    strategist = runner._build_strategist_config(real_enabled)

    assert "MomentumScalperShortOnly" not in strategist.hard_policy_deny_labels


def test_shadow_only_experimental_flash_registry_does_not_mutate_real_registry():
    class StaticAgent:
        def __init__(self, label):
            self.label = label

        def act(self, market):
            return {symbol: Action.HOLD for symbol in market.prices}

    real_registry = runner.AgentRegistry()
    real_registry.register(StaticAgent("MomentumScalper"))
    real_registry.register(StaticAgent("VolBreakoutHunter"))
    real_registry.register(StaticAgent("ResearchValidatorAgent"))
    real_registry.register(StaticAgent("FundingArb"))
    real_registry.register(StaticAgent("CrashPanicShortAgent"))

    shadow_registry = runner._clone_registry_with_experimental_flash_agents(real_registry)
    new_labels = {
        "MomentumScalperShortCrashOnly",
        "MomentumScalperShortBearOnly",
        "MomentumScalperSpotPullbackOnly",
        "ResearchValidatorNeutralOnly",
        "FundingArbBearOnly",
        "CrashPanicCrashOnly",
    }

    assert real_registry.has("MomentumScalper")
    assert not real_registry.has("MomentumScalperShortOnly")
    assert all(not real_registry.has(label) for label in new_labels)
    assert shadow_registry.has("MomentumScalper")
    assert shadow_registry.has("MomentumScalperShortOnly")
    assert shadow_registry.has("MomentumScalperSpotQuality")
    assert shadow_registry.has("VolBreakoutSpotOnly")
    assert all(shadow_registry.has(label) for label in new_labels)
    assert new_labels.issubset(set(runner.experimental_flash_agent_labels()))
    assert shadow_registry.get("MomentumScalper") is not real_registry.get("MomentumScalper")


def test_shadow_registry_omits_uncopyable_agents_instead_of_sharing_originals():
    class StaticAgent:
        def __init__(self, label):
            self.label = label

        def act(self, market):
            return {symbol: Action.HOLD for symbol in market.prices}

    class UncopyableMutableAgent:
        label = "MomentumScalper"

        def __init__(self):
            self.state = []

        def __deepcopy__(self, memo):
            raise TypeError("cannot copy mutable handle")

        def act(self, market):
            self.state.append(market.bar)
            return {symbol: Action.FUT_LONG_FULL for symbol in market.prices}

    real_registry = runner.AgentRegistry()
    base_agent = UncopyableMutableAgent()
    cloneable_agent = StaticAgent("FundingArb")
    real_registry.register(base_agent)
    real_registry.register(cloneable_agent)

    shadow_registry = runner._clone_registry_with_experimental_flash_agents(real_registry)

    assert real_registry.get("MomentumScalper") is base_agent
    assert shadow_registry.get("MomentumScalper") is None
    assert not shadow_registry.has("MomentumScalperShortOnly")
    assert not shadow_registry.has("MomentumScalperSpotQuality")
    assert shadow_registry.get("FundingArb") is not cloneable_agent
    assert shadow_registry.has("FundingArbBearOnly")


def test_action_filter_agent_holds_actions_outside_allowed_regimes():
    class StaticAgent:
        label = "Base"

        def act(self, market):
            return {"BTC/USDT": Action.FUT_SHORT_FULL}

    agent = ActionFilterAgent(
        label="CrashOnly",
        base_agent=StaticAgent(),
        allowed_actions=(Action.FUT_SHORT_FULL,),
        allowed_regimes=("crash",),
    )
    neutral_market = runner.MarketSnapshot(
        bar=1,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        regime=Regime.NEUTRAL,
        prices={"BTC/USDT": 100.0},
        volumes={"BTC/USDT": 10.0},
    )
    crash_market = runner.MarketSnapshot(
        bar=2,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        regime=Regime.CRASH,
        prices={"BTC/USDT": 90.0},
        volumes={"BTC/USDT": 20.0},
    )

    assert agent.act(neutral_market) == {"BTC/USDT": Action.HOLD}
    assert agent.act(crash_market) == {"BTC/USDT": Action.FUT_SHORT_FULL}
    assert agent.clone_for_shadow().allowed_regimes == ("crash",)


def test_experimental_flash_wrapper_skips_uncopyable_base_agent():
    class UncopyableMutableAgent:
        label = "MomentumScalper"

        def __init__(self):
            self.state = []

        def __deepcopy__(self, memo):
            raise TypeError("cannot copy mutable handle")

        def act(self, market):
            self.state.append(market.bar)
            return {"BTC/USDT": Action.FUT_SHORT_FULL}

    registry = runner.AgentRegistry()
    base_agent = UncopyableMutableAgent()
    registry.register(base_agent)

    registered = register_experimental_flash_agents(registry, skip_missing=True)

    assert registered == []
    assert not registry.has("MomentumScalperShortOnly")
    assert not registry.has("MomentumScalperSpotQuality")
    assert registry.get("MomentumScalper") is base_agent

    with pytest.raises(ValueError, match="could not be cloned"):
        register_experimental_flash_agents(registry, skip_missing=False)


def test_action_filter_agent_allowed_regime_alias_matches_bearish_market():
    class StaticAgent:
        label = "Base"

        def act(self, market):
            return {"BTC/USDT": Action.FUT_SHORT_FULL}

    agent = ActionFilterAgent(
        label="BearOnly",
        base_agent=StaticAgent(),
        allowed_actions=(Action.FUT_SHORT_FULL,),
        allowed_regimes=("bear",),
    )
    market = runner.MarketSnapshot(
        bar=1,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        regime=Regime.BEARISH,
        prices={"BTC/USDT": 100.0},
        volumes={"BTC/USDT": 10.0},
    )

    assert agent.allowed_regimes == ("bearish",)
    assert agent.act(market) == {"BTC/USDT": Action.FUT_SHORT_FULL}


def test_action_filter_agent_allowed_regime_alias_matches_crash_market():
    class StaticAgent:
        label = "Base"

        def act(self, market):
            return {"BTC/USDT": Action.FUT_SHORT_FULL}

    agent = ActionFilterAgent(
        label="PanicOnly",
        base_agent=StaticAgent(),
        allowed_actions=(Action.FUT_SHORT_FULL,),
        allowed_regimes=("panic",),
    )
    market = runner.MarketSnapshot(
        bar=1,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        regime=Regime.CRASH,
        prices={"BTC/USDT": 100.0},
        volumes={"BTC/USDT": 10.0},
    )

    assert agent.allowed_regimes == ("crash",)
    assert agent.act(market) == {"BTC/USDT": Action.FUT_SHORT_FULL}


def test_experimental_flash_player_sets_reference_wrapper_agents():
    fixed_sets = runner._experimental_flash_fixed_agent_player_sets()
    rotating_sets = runner._experimental_flash_rotating_agent_player_sets()

    assert ("Fixed_ProfitTriad", (
        "MomentumScalperShortOnly",
        "FundingArb",
        "CrashPanicShortAgent",
    )) in fixed_sets
    assert any(
        label == "Experimental_FlashEdgeRotator"
        and "MomentumScalperSpotQuality" in fallback
        for label, _mapping, fallback in rotating_sets
    )


def test_fixed_agent_players_default_to_multiple_experiment_sets():
    config = RetrodateMarketConfig(fixed_agent_players_enabled=True)

    assert config.fixed_agent_players_enabled is True
    assert len(config.fixed_agent_player_sets) >= 5
    assert any(label == "Fixed_BullBreakout" for label, _agents in config.fixed_agent_player_sets)
    assert any(label == "Fixed_NeutralValidator" for label, _agents in config.fixed_agent_player_sets)


def test_probation_loss_kill_defaults_include_experimental_composites():
    config = runner._parse_cli_config(["--years", "2025"])
    strategist_config = runner._build_strategist_config(config)

    assert strategist_config.v3_probation_loss_kill_label_prefixes == (
        "Solo_",
        "Fixed_",
        "Antonius_",
        "Optimal_",
    )


def test_retrodate_strategist_config_denies_weak_production_profiles():
    config = runner._parse_cli_config(["--years", "2025"])
    strategist_config = runner._build_strategist_config(config)

    deny = set(strategist_config.hard_policy_deny_labels)
    assert {"DefaultEnsemble", "DefensiveResearch", "TrendResearch"}.issubset(deny)
    assert {"Solo_PlayerFunding", "Solo_LiveAfterShock"}.issubset(deny)
    assert strategist_config.hard_policy_experimental_min_bar == 0


def test_cli_hard_policy_deny_label_extends_default_denylist():
    config = runner._parse_cli_config([
        "--years",
        "2025",
        "--hard-policy-deny-label",
        "Solo_LiveVolCompress",
        "--hard-policy-deny-label",
        "Solo_VolBreakoutHunter",
    ])
    strategist_config = runner._build_strategist_config(config)

    deny = set(strategist_config.hard_policy_deny_labels)
    assert {"DefaultEnsemble", "Solo_PlayerFunding"}.issubset(deny)
    assert {"Solo_LiveVolCompress", "Solo_VolBreakoutHunter"}.issubset(deny)


def test_build_strategist_config_passes_use_v3_rolling_score():
    config = RetrodateMarketConfig(
        use_v3_rolling_score=True,
        use_v3_shadow_rolling_score=True,
        use_v3_soft_shadow_score=True,
        use_v3_executable_soft_confirmed_score=True,
        use_v3_entry_causal_score=True,
        current_actionable_candidate_layer_enabled=True,
        v3_shadow_position_gate_enabled=True,
        v3_shadow_flat_handoff_enabled=True,
        v3_shadow_fresh_handoff_enabled=True,
        v3_shadow_fresh_handoff_max_age_bars=1,
        v3_shadow_rolling_window_bars=24,
        v3_shadow_rolling_min_closed_trades=50,
        v3_entry_causal_min_filled=2,
        v3_entry_causal_actionability_weight=0.5,
        solo_agent_candidate_limit=6,
    )

    strategist_config = runner._build_strategist_config(config)

    assert strategist_config.use_v3_rolling_score is True
    assert strategist_config.use_v3_shadow_rolling_score is True
    assert strategist_config.use_v3_soft_shadow_score is True
    assert strategist_config.v3_current_actionable_gate_enabled is True
    assert strategist_config.use_v3_entry_causal_score is True
    assert strategist_config.v3_shadow_position_gate_enabled is True
    assert strategist_config.v3_shadow_flat_handoff_enabled is True
    assert strategist_config.v3_shadow_fresh_handoff_enabled is True
    assert strategist_config.v3_shadow_fresh_handoff_max_age_bars == 1
    assert strategist_config.v3_shadow_rolling_window_bars == 24
    assert strategist_config.v3_shadow_rolling_min_closed_trades == 50
    assert strategist_config.v3_entry_causal_min_filled == 2
    assert strategist_config.v3_entry_causal_actionability_weight == 0.5
    assert config.solo_agent_candidate_limit == 6


def test_cli_and_strategist_config_accept_executable_soft_top1_score_flag():
    config = runner._parse_cli_config([
        "--years",
        "2025",
        "--use-v3-executable-soft-top1-score",
    ])

    strategist_config = runner._build_strategist_config(config)

    assert config.use_v3_executable_soft_top1_score is True
    assert strategist_config.use_v3_rolling_score is True
    assert strategist_config.use_v3_soft_shadow_score is True
    assert strategist_config.v3_current_actionable_gate_enabled is True
    assert strategist_config.v3_shadow_rolling_window_bars == 24
    assert strategist_config.v3_shadow_rolling_min_closed_trades == 20


def test_run_summary_records_effective_executable_soft_top1_strategy_config(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    summary = RetrodateRunSummary(
        output_dir=output_dir,
        report_path=output_dir / "analysis.md",
        summary_path=output_dir / "run_summary.json",
        requested_years=(2025,),
        executed_years=(2025,),
        excluded_files=(),
        missing_years=(),
        bars_processed=0,
        registered_agents=(),
        player_profile_count=0,
        first_timestamp="",
        last_timestamp="",
        stride_minutes=60,
        max_bars=None,
        step_errors=(),
    )
    config = RetrodateMarketConfig(
        use_v3_executable_soft_top1_score=True,
        flash_denied_signal_keys=(
            "agent:MomentumScalper|ATOM/USDT|SPOT_BUY_FULL",
        ),
        flash_terminal_denied_signal_keys=(
            "ensemble:Solo_LiveCrashHunter|ATOM/USDT|FUT_SHORT_FULL",
        ),
        flash_denied_open_symbols=("ATOM/USDT",),
        flash_degradation_symbol_guard_enabled=True,
        flash_degradation_symbol_cooldown_bars=48,
        flash_degradation_symbol_window_closed_trades=2,
        flash_degradation_symbol_min_closed_trades=2,
        flash_degradation_symbol_max_recent_pnl_usd=-10.0,
        flash_promotion_manifest_enabled=True,
        flash_promotion_manifest_path=output_dir / "flash_promotion_manifest.json",
        flash_promoted_signal_keys=(
            "agent:FundingArb|BTC/USDT|FUT_SHORT_FULL",
        ),
        flash_earned_cap_overrides_enabled=True,
        flash_promoted_actor_cap_overrides=("agent:FundingArb=3",),
    )
    selection = RetrodateFileSelection(
        report=RetrodateDirReport(path=tmp_path, files=[]),
        valid_reports=(),
        excluded_files=(),
        missing_years=(),
    )

    runner._write_run_summary(summary.summary_path, summary, selection, config)

    data = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    assert data["use_v3_executable_soft_top1_score"] is True
    assert data["effective_v3_shadow_rolling_window_bars"] == 24
    assert data["effective_v3_shadow_rolling_min_closed_trades"] == 20
    assert data["effective_v3_current_actionable_gate_enabled"] is True
    assert data["flash_denied_signal_keys"] == [
        "agent:MomentumScalper|ATOM/USDT|SPOT_BUY_FULL",
    ]
    assert data["flash_terminal_denied_signal_keys"] == [
        "ensemble:Solo_LiveCrashHunter|ATOM/USDT|FUT_SHORT_FULL",
    ]
    assert data["flash_denied_open_symbols"] == ["ATOM/USDT"]
    assert data["flash_degradation_symbol_guard_enabled"] is True
    assert data["flash_degradation_symbol_cooldown_bars"] == 48
    assert data["flash_degradation_symbol_window_closed_trades"] == 2
    assert data["flash_degradation_symbol_min_closed_trades"] == 2
    assert data["flash_degradation_symbol_max_recent_pnl_usd"] == -10.0
    assert data["flash_promotion_manifest_enabled"] is True
    assert data["flash_promotion_manifest_path"] == str(
        output_dir / "flash_promotion_manifest.json"
    )
    assert data["flash_promoted_signal_keys"] == [
        "agent:FundingArb|BTC/USDT|FUT_SHORT_FULL",
    ]
    assert data["flash_earned_cap_overrides_enabled"] is True
    assert data["flash_promoted_actor_cap_overrides"] == ["agent:FundingArb=3"]


def test_cli_and_strategist_config_accept_real_loss_rescue_flags():
    config = runner._parse_cli_config([
        "--years",
        "2025",
        "--enable-v3-real-loss-rescue",
        "--v3-real-loss-rescue-min-virtual-pnl-pct",
        "12.5",
        "--v3-real-loss-rescue-max-virtual-dd-pct",
        "40",
        "--v3-real-loss-rescue-min-actionable-share",
        "0.15",
        "--v3-real-loss-rescue-min-recent-filled",
        "3",
        "--v3-real-loss-rescue-max-real-loss-pct",
        "-2.5",
        "--allow-genetics-real-loss-rescue",
    ])

    strategist_config = runner._build_strategist_config(config)

    assert config.v3_real_loss_rescue_enabled is True
    assert strategist_config.v3_real_loss_rescue_enabled is True
    assert strategist_config.v3_real_loss_rescue_min_virtual_pnl_pct == 12.5
    assert strategist_config.v3_real_loss_rescue_max_virtual_dd_pct == 40.0
    assert strategist_config.v3_real_loss_rescue_min_actionable_share == 0.15
    assert strategist_config.v3_real_loss_rescue_min_recent_filled == 3
    assert strategist_config.v3_real_loss_rescue_max_real_loss_pct == -2.5
    assert config.v3_real_loss_rescue_allow_genetics is True
    assert strategist_config.v3_real_loss_rescue_allow_genetics is True


def test_cli_and_strategist_config_accept_probation_shadow_rescue_flags():
    config = runner._parse_cli_config([
        "--years",
        "2025",
        "--enable-v3-probation-shadow-rescue",
        "--v3-probation-shadow-rescue-min-virtual-pnl-pct",
        "12.5",
        "--v3-probation-shadow-rescue-max-virtual-dd-pct",
        "35",
        "--v3-probation-shadow-rescue-min-actionable-share",
        "0.2",
        "--v3-probation-shadow-rescue-min-recent-filled",
        "4",
        "--v3-probation-shadow-rescue-min-recent-pnl-usd",
        "3.5",
        "--allow-genetics-probation-shadow-rescue",
    ])

    strategist_config = runner._build_strategist_config(config)

    assert config.v3_probation_shadow_rescue_enabled is True
    assert strategist_config.v3_probation_shadow_rescue_enabled is True
    assert strategist_config.v3_probation_shadow_rescue_min_virtual_pnl_pct == 12.5
    assert strategist_config.v3_probation_shadow_rescue_max_virtual_dd_pct == 35.0
    assert strategist_config.v3_probation_shadow_rescue_min_actionable_share == 0.2
    assert strategist_config.v3_probation_shadow_rescue_min_recent_filled == 4
    assert strategist_config.v3_probation_shadow_rescue_min_recent_pnl_usd == 3.5
    assert config.v3_probation_shadow_rescue_allow_genetics is True
    assert strategist_config.v3_probation_shadow_rescue_allow_genetics is True


def test_cli_and_live_execution_config_accept_genetics_probation_execution_flags():
    config = runner._parse_cli_config([
        "--years",
        "2025",
        "--max-new-opens-per-bar",
        "3",
        "--risk-max-open-positions",
        "16",
        "--enable-genetics-probation-execution",
        "--genetics-probation-risk-mult",
        "0.20",
        "--genetics-probation-max-real-trades",
        "7",
        "--disable-genetics-probation-shadow-confirmation",
    ])

    live_config = runner._build_live_execution_config(config)
    risk_config = runner._build_risk_config(config)

    assert config.max_new_opens_per_bar == 3
    assert config.risk_max_open_positions == 16
    assert live_config.max_new_opens_per_bar == 3
    assert risk_config.max_open_positions == 16
    assert config.genetics_probation_execution_enabled is True
    assert config.genetics_probation_risk_mult == 0.20
    assert config.genetics_probation_max_real_trades == 7
    assert config.genetics_probation_require_shadow_confirmation is False
    assert live_config.genetics_probation_execution_enabled is True
    assert live_config.genetics_probation_labels == ("GeneticsResearch",)
    assert live_config.genetics_probation_allowed_regimes == ("bearish", "crash")
    assert live_config.genetics_probation_risk_mult == 0.20
    assert live_config.genetics_probation_max_real_trades == 7
    assert live_config.genetics_probation_require_shadow_confirmation is False


def test_cli_and_strategist_config_accept_profit_lock_and_probation_kill_flags():
    config = runner._parse_cli_config([
        "--years",
        "2025",
        "--v3-probation-loss-kill-min-closed-trades",
        "2",
        "--v3-probation-loss-kill-pnl-pct",
        "-0.15",
        "--v3-probation-loss-kill-win-rate-pct",
        "50",
        "--v3-probation-loss-kill-label-prefix",
        "Solo_",
        "--v3-probation-loss-kill-label-prefix",
        "MeanRevResearch",
        "--v3-realized-profit-lock-min-closed-trades",
        "20",
        "--v3-realized-profit-lock-min-peak-pnl-pct",
        "0.75",
        "--v3-realized-profit-lock-max-giveback-pct",
        "0.55",
        "--v3-realized-profit-lock-floor-pnl-pct",
        "0.25",
        "--enable-v3-panteon-equity-guard",
        "--v3-panteon-equity-guard-min-peak-pnl-pct",
        "3.0",
        "--v3-panteon-equity-guard-max-giveback-pct",
        "1.1",
        "--v3-panteon-equity-guard-floor-pnl-pct",
        "2.4",
        "--v3-panteon-equity-guard-cooldown-bars",
        "168",
    ])

    strategist_config = runner._build_strategist_config(config)

    assert config.v3_probation_loss_kill_min_closed_trades == 2
    assert strategist_config.v3_probation_loss_kill_min_closed_trades == 2
    assert strategist_config.v3_probation_loss_kill_pnl_pct == -0.15
    assert strategist_config.v3_probation_loss_kill_win_rate_pct == 50.0
    assert strategist_config.v3_probation_loss_kill_label_prefixes == (
        "Solo_",
        "MeanRevResearch",
    )
    assert strategist_config.v3_realized_profit_lock_min_closed_trades == 20
    assert strategist_config.v3_realized_profit_lock_min_peak_pnl_pct == 0.75
    assert strategist_config.v3_realized_profit_lock_max_giveback_pct == 0.55
    assert strategist_config.v3_realized_profit_lock_floor_pnl_pct == 0.25
    assert strategist_config.v3_panteon_equity_guard_enabled is True
    assert strategist_config.v3_panteon_equity_guard_min_peak_pnl_pct == 3.0
    assert strategist_config.v3_panteon_equity_guard_max_giveback_pct == 1.1
    assert strategist_config.v3_panteon_equity_guard_floor_pnl_pct == 2.4
    assert strategist_config.v3_panteon_equity_guard_cooldown_bars == 168


def test_cli_can_apply_probation_loss_kill_to_all_labels():
    config = runner._parse_cli_config([
        "--years",
        "2025",
        "--v3-probation-loss-kill-all-labels",
    ])

    strategist_config = runner._build_strategist_config(config)

    assert config.v3_probation_loss_kill_label_prefixes == ()
    assert strategist_config.v3_probation_loss_kill_label_prefixes == ()


def test_analysis_report_includes_soft_allocator_summary(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "status.json").write_text(
        json.dumps({
            "current_leader": "Leader",
            "live_session": {
                "panteon_owned_pnl_pct": 1.5,
                "panteon_owned_realized_pnl_usd": 15.0,
                "real_closed_trades": 3,
                "panteon_owned_positions_count": 0,
            },
            "shadow": {"actors": 2},
        }),
        encoding="utf-8",
    )
    (output_dir / "leaderboard_agents.json").write_text(
        json.dumps({"agents": {}}),
        encoding="utf-8",
    )
    (output_dir / "leaderboard_players.json").write_text(
        json.dumps({"players": {}}),
        encoding="utf-8",
    )
    (output_dir / "soft_allocator_report.json").write_text(
        json.dumps({
            "best_single_label": "Solo_A",
            "best_single_pnl_usd": 100.0,
            "best_single_pnl_pct": 10.0,
            "best_single_max_drawdown_pct": 4.0,
            "best_policy_name": "soft_top3_decayed",
            "best_policy_pnl_usd": 130.0,
            "best_policy_pnl_pct": 13.0,
            "best_policy_max_drawdown_pct": 3.0,
            "regret_vs_best_single_usd": 0.0,
            "beats_best_single": True,
        }),
        encoding="utf-8",
    )
    (output_dir / "perfect_panteon_report.json").write_text(
        json.dumps({
            "pnl_usd": 200.0,
            "pnl_pct": 20.0,
            "max_drawdown_pct": 2.5,
            "month_count": 2,
            "profitable_months": 2,
            "cash_months": 0,
            "closed_trades": 11.0,
        }),
        encoding="utf-8",
    )
    summary = RetrodateRunSummary(
        output_dir=output_dir,
        report_path=output_dir / "analysis_report.md",
        summary_path=output_dir / "run_summary.json",
        requested_years=(2025,),
        executed_years=(2025,),
        excluded_files=(),
        missing_years=(),
        bars_processed=1,
        registered_agents=("A",),
        player_profile_count=1,
        first_timestamp="2025-01-01T00:00:00+00:00",
        last_timestamp="2025-01-01T01:00:00+00:00",
        stride_minutes=60,
        max_bars=None,
        step_errors=(),
    )
    selection = RetrodateFileSelection(
        report=RetrodateDirReport(path=tmp_path, files=[]),
        valid_reports=(),
        excluded_files=(),
        missing_years=(),
    )

    runner._write_analysis_report(summary.report_path, summary, selection)

    report = summary.report_path.read_text(encoding="utf-8")
    assert "## Soft Allocator Simulation" in report
    assert "`soft_top3_decayed`" in report
    assert "Beats best single: yes" in report
    assert "## Perfect Panteon Monthly Oracle" in report
    assert "$200.00" in report
    assert "available 2026 file contains 2022-2023 timestamps" not in report


def test_analysis_report_and_summary_include_allocation_diagnostics(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "status.json").write_text(
        json.dumps({
            "current_leader": "NoTrade",
            "live_session": {
                "panteon_owned_pnl_pct": -1.0,
                "panteon_owned_realized_pnl_usd": -10.0,
                "real_closed_trades": 2,
                "panteon_owned_positions_count": 0,
            },
            "shadow": {"actors": 2},
        }),
        encoding="utf-8",
    )
    (output_dir / "leaderboard_agents.json").write_text(
        json.dumps({"agents": {}}),
        encoding="utf-8",
    )
    (output_dir / "leaderboard_players.json").write_text(
        json.dumps({"players": {}}),
        encoding="utf-8",
    )
    (output_dir / "allocation_diagnostics.json").write_text(
        json.dumps({
            "bars": 4,
            "no_trade_bars": 1,
            "no_trade_share_pct": 25.0,
            "raw_zero_bars": 3,
            "raw_zero_share_pct": 75.0,
            "filled_zero_bars": 3,
            "filled_zero_share_pct": 75.0,
            "leaders": {
                "Alpha": {
                    "bars": 3,
                    "bar_share_pct": 75.0,
                    "raw_zero_share_pct": 66.6666666667,
                    "filled_zero_share_pct": 66.6666666667,
                    "raw_signals": 2,
                    "signals": 1,
                    "filled": 1,
                }
            },
        }),
        encoding="utf-8",
    )
    summary = RetrodateRunSummary(
        output_dir=output_dir,
        report_path=output_dir / "analysis_report.md",
        summary_path=output_dir / "run_summary.json",
        requested_years=(2025,),
        executed_years=(2025,),
        excluded_files=(),
        missing_years=(),
        bars_processed=4,
        registered_agents=("A",),
        player_profile_count=1,
        first_timestamp="2025-01-01T00:00:00+00:00",
        last_timestamp="2025-01-01T04:00:00+00:00",
        stride_minutes=60,
        max_bars=None,
        step_errors=(),
    )
    selection = RetrodateFileSelection(
        report=RetrodateDirReport(path=tmp_path, files=[]),
        valid_reports=(),
        excluded_files=(),
        missing_years=(),
    )

    runner._write_run_summary(
        summary.summary_path,
        summary,
        selection,
        RetrodateMarketConfig(data_dir=tmp_path, years=(2025,), solo_agent_candidate_limit=6),
    )
    runner._write_analysis_report(summary.report_path, summary, selection)

    run_summary = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    report = summary.report_path.read_text(encoding="utf-8")
    assert run_summary["solo_agent_candidate_limit"] == 6
    assert run_summary["fixed_agent_players_enabled"] is False
    assert run_summary["fixed_agent_player_count"] == 0
    assert run_summary["allocation_diagnostics"]["no_trade_share_pct"] == 25.0
    assert "## Allocation Diagnostics" in report
    assert "NoTrade share: 25.00%" in report
    assert "`allocation_diagnostics.json`" in report


def test_shadow_pnl_export_can_write_agent_stream(tmp_path):
    path = write_shadow_pnl_events(
        tmp_path,
        runner.shadow_pnl_events_from_shadow_updates(
            [
                ShadowActorUpdated(
                    bar=1,
                    actor_type="agent",
                    actor_label="MomentumScalper",
                    regime="bearish",
                    realized_pnl_usd=4.0,
                    closed_trades=2,
                    winning_trades=1,
                ),
                ShadowActorUpdated(
                    bar=1,
                    actor_type="player",
                    actor_label="MeanRevResearch",
                    regime="bearish",
                    realized_pnl_usd=8.0,
                    closed_trades=3,
                    winning_trades=2,
                ),
            ],
            actor_type="agent",
        ),
        filename="shadow_agent_pnl_events.jsonl",
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert path.name == "shadow_agent_pnl_events.jsonl"
    assert len(rows) == 1
    assert rows[0]["timestamp"]
    rows[0]["timestamp"] = "<timestamp>"
    assert rows == [{
        "bar": 1,
        "label": "MomentumScalper",
        "timestamp": "<timestamp>",
        "regime": "bearish",
        "pnl_usd": 4.0,
        "closed_trades": 2,
        "wins": 1,
    }]


def test_candidate_diagnostics_export_summarizes_actionability(tmp_path):
    path = runner.write_candidate_diagnostics(
        tmp_path,
        candidate_events=[
            CandidateScored(
                bar=1,
                player_label="Fixed_OIBreakoutCore",
                selected_by_pantheon=True,
                score=2.0,
                has_data=True,
                closed_trades=12,
                signals=8,
                recent_bars=4,
                recent_actionable_bars=2,
                actionable_share=0.5,
                recent_filled=3,
                recent_pnl_usd=7.0,
            ),
            CandidateScored(
                bar=1,
                player_label="Solo_LiveOIBreakout",
                selected_by_pantheon=False,
                score=1.0,
                has_data=True,
                closed_trades=10,
                signals=4,
                recent_bars=4,
                recent_actionable_bars=4,
                actionable_share=1.0,
                recent_filled=4,
                recent_pnl_usd=5.0,
            ),
        ],
        rejection_events=[
            CandidateRejected(
                bar=1,
                player_label="Fixed_Bad",
                reason="hard policy test",
            ),
            CandidateRejected(
                bar=2,
                player_label="LiveVolCompress",
                reason="flash:inactive",
            ),
            CandidateRejected(
                bar=3,
                player_label="LiveVolCompress",
                reason="flash:pnl_below_threshold",
            ),
            CandidateRejected(
                bar=4,
                player_label="MomentumScalper",
                reason="flash:shadow_unconfirmed",
            ),
            CandidateRejected(
                bar=5,
                player_label="Solo_MomentumScalper",
                reason="flash:quarantined",
            ),
        ],
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    fixed = data["candidates"]["Fixed_OIBreakoutCore"]
    solo = data["candidates"]["Solo_LiveOIBreakout"]

    assert data["candidate_rows"] == 2
    assert data["selected_rows"] == 1
    assert data["rejection_rows"] == 5
    assert fixed["group_type"] == "fixed"
    assert fixed["selected_bars"] == 1
    assert fixed["avg_actionable_share"] == 0.5
    assert fixed["avg_recent_filled"] == 3.0
    assert fixed["avg_recent_pnl_usd"] == 7.0
    assert solo["group_type"] == "solo"
    assert data["rejections"]["Fixed_Bad"]["count"] == 1
    assert data["rejection_summary"]["reason_counts"] == {
        "flash:inactive": 1,
        "flash:pnl_below_threshold": 1,
        "flash:quarantined": 1,
        "flash:shadow_unconfirmed": 1,
        "hard policy test": 1,
    }
    assert data["rejection_summary"]["flash_reason_counts"] == {
        "inactive": 1,
        "pnl_below_threshold": 1,
        "quarantined": 1,
        "shadow_unconfirmed": 1,
    }
    assert data["rejection_summary"]["flash_active_reason_counts"] == {
        "pnl_below_threshold": 1,
        "shadow_unconfirmed": 1,
    }


def test_flash_attribution_summary_groups_selection_execution_and_realized_pnl(tmp_path):
    causal_path = tmp_path / "causal_entry_decisions.jsonl"
    causal_rows = [
        {
            "bar": 1,
            "regime": "bullish",
            "flash_decisions": [
                {
                    "symbol": "BTC/USDT",
                    "selected_actor": "Alpha",
                    "actor_type": "agent",
                    "score": 2.5,
                    "action": "FUT_LONG_FULL",
                    "signal": {
                        "id": 10,
                        "sym": "BTC/USDT",
                        "action": "FUT_LONG_FULL",
                        "by_player": "Alpha",
                        "by_agent": "Alpha",
                    },
                    "candidates": [
                        {
                            "label": "Alpha",
                            "actor_type": "agent",
                            "actor_key": "agent:Alpha",
                            "rank": 1,
                            "shadow_score": 1.5,
                            "shadow_closed_trades": 70,
                        }
                    ],
                }
            ],
        },
        {
            "bar": 2,
            "regime": "bearish",
            "flash_decisions": [
                {
                    "symbol": "ETH/USDT",
                    "selected_actor": "Alpha",
                    "actor_type": "agent",
                    "score": 1.0,
                    "action": "FUT_SHORT_FULL",
                    "signal": {
                        "id": 11,
                        "sym": "ETH/USDT",
                        "action": "FUT_SHORT_FULL",
                        "by_player": "Alpha",
                        "by_agent": "Alpha",
                    },
                    "candidates": [
                        {
                            "label": "Alpha",
                            "actor_type": "agent",
                            "actor_key": "agent:Alpha",
                            "rank": 1,
                            "shadow_score": 0.75,
                            "shadow_closed_trades": 55,
                        }
                    ],
                }
            ],
        },
    ]
    causal_path.write_text(
        "\n".join(json.dumps(row) for row in causal_rows) + "\n",
        encoding="utf-8",
    )

    path = runner.write_flash_attribution_summary(
        tmp_path,
        execution_events=[
            ExecutionAttributed(
                bar=1,
                signal_id=10,
                sym="BTC/USDT",
                action="FUT_LONG_FULL",
                status="filled",
            ),
            ExecutionAttributed(
                bar=2,
                signal_id=11,
                sym="ETH/USDT",
                action="FUT_SHORT_FULL",
                status="blocked",
                reason="risk limit",
            ),
        ],
        position_closed_events=[
            PositionClosed(
                bar=3,
                open_signal_id=10,
                close_signal_id=12,
                sym="BTC/USDT",
                side="long",
                realized_pnl=7.25,
                by_player="Alpha",
                by_agent="Alpha",
            )
        ],
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["summary"]["selected_signals"] == 2
    assert data["summary"]["filled_signals"] == 1
    assert data["summary"]["blocked_signals"] == 1
    assert data["summary"]["closed_trades"] == 1
    assert data["summary"]["realized_pnl_usd"] == 7.25
    btc = data["rows"][0]
    eth = data["rows"][1]
    assert btc["actor_key"] == "agent:Alpha"
    assert btc["symbol"] == "BTC/USDT"
    assert btc["action"] == "FUT_LONG_FULL"
    assert btc["selected_signals"] == 1
    assert btc["filled_signals"] == 1
    assert btc["closed_trades"] == 1
    assert btc["realized_pnl_usd"] == 7.25
    assert btc["avg_shadow_score"] == 1.5
    assert eth["symbol"] == "ETH/USDT"
    assert eth["blocked_signals"] == 1
    assert eth["failure_reasons"]["risk limit"] == 1


def test_flash_attribution_summary_splits_guard_filtered_from_missing_execution(tmp_path):
    causal_path = tmp_path / "causal_entry_decisions.jsonl"
    causal_row = {
        "bar": 1,
        "regime": "bullish",
        "signal_filter_details": ["duplicate_open:ETH/USDT:Alpha"],
        "executable_signals": [
            {
                "id": 10,
                "sym": "BTC/USDT",
                "action": "FUT_LONG_FULL",
                "by_player": "Alpha",
                "by_agent": "Alpha",
            },
            {
                "id": 12,
                "sym": "SOL/USDT",
                "action": "FUT_LONG_FULL",
                "by_player": "Alpha",
                "by_agent": "Alpha",
            },
        ],
        "flash_decisions": [
            {
                "symbol": "BTC/USDT",
                "selected_actor": "Alpha",
                "actor_type": "agent",
                "score": 2.5,
                "action": "FUT_LONG_FULL",
                "signal": {
                    "id": 10,
                    "sym": "BTC/USDT",
                    "action": "FUT_LONG_FULL",
                    "by_player": "Alpha",
                    "by_agent": "Alpha",
                },
            },
            {
                "symbol": "ETH/USDT",
                "selected_actor": "Alpha",
                "actor_type": "agent",
                "score": 2.0,
                "action": "FUT_LONG_FULL",
                "signal": {
                    "id": 11,
                    "sym": "ETH/USDT",
                    "action": "FUT_LONG_FULL",
                    "by_player": "Alpha",
                    "by_agent": "Alpha",
                },
            },
            {
                "symbol": "SOL/USDT",
                "selected_actor": "Alpha",
                "actor_type": "agent",
                "score": 1.5,
                "action": "FUT_LONG_FULL",
                "signal": {
                    "id": 12,
                    "sym": "SOL/USDT",
                    "action": "FUT_LONG_FULL",
                    "by_player": "Alpha",
                    "by_agent": "Alpha",
                },
            },
        ],
    }
    causal_path.write_text(json.dumps(causal_row) + "\n", encoding="utf-8")

    path = runner.write_flash_attribution_summary(
        tmp_path,
        execution_events=[
            ExecutionAttributed(
                bar=1,
                signal_id=10,
                sym="BTC/USDT",
                action="FUT_LONG_FULL",
                status="filled",
            ),
        ],
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["summary"]
    assert summary["selected_signals"] == 3
    assert summary["executable_selected_signals"] == 2
    assert summary["selected_filtered_before_execution"] == 1
    assert summary["missing_execution_signals"] == 1
    assert summary["filter_detail_counts"] == {"duplicate_open": 1}

    by_symbol = {row["symbol"]: row for row in data["rows"]}
    assert by_symbol["ETH/USDT"]["selected_filtered_before_execution"] == 1
    assert by_symbol["SOL/USDT"]["executable_selected_signals"] == 1


def test_signal_key_shadow_report_groups_symbol_action_outcomes(tmp_path):
    path = runner.write_flash_signal_key_shadow_report(
        tmp_path,
        shadow_updates=[
            ShadowActorUpdated(
                bar=1,
                actor_type="agent",
                actor_label="Alpha",
                regime="bullish",
                symbol_action_outcomes=(("BTC/USDT", "long", 6.0, 3, 2),),
            ),
            ShadowActorUpdated(
                bar=2,
                actor_type="agent",
                actor_label="Alpha",
                regime="bullish",
                symbol_action_outcomes=(("BTC/USDT", "long", -1.0, 1, 0),),
            ),
        ],
        initial_capital=100.0,
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    row = data["rows"][0]

    assert row["signal_key"] == "agent:Alpha|BTC/USDT|FUT_LONG_FULL"
    assert row["full_pnl_pct"] == 5.0
    assert row["full_closed_trades"] == 4
    assert row["win_rate_pct"] == 50.0


def test_signal_key_shadow_report_computes_pnl_per_trade_lcb(tmp_path):
    path = runner.write_flash_signal_key_shadow_report(
        tmp_path,
        shadow_updates=[
            ShadowActorUpdated(
                bar=1,
                actor_type="agent",
                actor_label="Alpha",
                regime="bullish",
                symbol_action_outcomes=(("BTC/USDT", "long", 6.0, 3, 2),),
            ),
            ShadowActorUpdated(
                bar=2,
                actor_type="agent",
                actor_label="Alpha",
                regime="bullish",
                symbol_action_outcomes=(("BTC/USDT", "long", -1.0, 1, 0),),
            ),
        ],
        initial_capital=100.0,
        promotion_manifest_config=runner.PromotionManifestConfig(
            pnl_per_trade_lcb_z=1.0,
        ),
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    row = data["rows"][0]

    assert row["full_pnl_per_trade_mean_pct"] == pytest.approx(1.25)
    assert row["full_pnl_per_trade_lcb_pct"] == pytest.approx(0.5)
    assert row["latest_pnl_per_trade_mean_pct"] == pytest.approx(1.25)
    assert row["latest_pnl_per_trade_lcb_pct"] == pytest.approx(0.5)


def test_signal_key_shadow_report_maps_player_actor_key_to_allocator_ensemble(tmp_path):
    path = runner.write_flash_signal_key_shadow_report(
        tmp_path,
        shadow_updates=[
            ShadowActorUpdated(
                bar=1,
                actor_type="player",
                actor_label="Fixed_Core",
                regime="neutral",
                symbol_action_outcomes=(("ETH/USDT", "SPOT_BUY_FULL", 3.0, 2, 1),),
            )
        ],
        initial_capital=100.0,
    )

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["rows"][0]["actor_label"] == "Fixed_Core"
    assert data["rows"][0]["actor_type"] == "ensemble"
    assert data["rows"][0]["source_actor_type"] == "player"
    assert data["rows"][0]["actor_key"] == "ensemble:Fixed_Core"
    assert data["rows"][0]["signal_key"] == "ensemble:Fixed_Core|ETH/USDT|SPOT_BUY_FULL"


def test_signal_key_shadow_report_skips_non_finite_pnl_and_writes_strict_json(tmp_path):
    path = runner.write_flash_signal_key_shadow_report(
        tmp_path,
        shadow_updates=[
            ShadowActorUpdated(
                bar=1,
                actor_type="agent",
                actor_label="BadNan",
                regime="neutral",
                symbol_action_outcomes=(("BTC/USDT", "long", float("nan"), 2, 1),),
            ),
            ShadowActorUpdated(
                bar=2,
                actor_type="agent",
                actor_label="BadInf",
                regime="neutral",
                symbol_action_outcomes=(("ETH/USDT", "long", float("inf"), 2, 1),),
            ),
            ShadowActorUpdated(
                bar=3,
                actor_type="agent",
                actor_label="Good",
                regime="neutral",
                symbol_action_outcomes=(("SOL/USDT", "long", 4.0, float("inf"), 1),),
            ),
        ],
        initial_capital=100.0,
    )

    report_text = path.read_text(encoding="utf-8")
    manifest_text = (tmp_path / "flash_promotion_manifest.json").read_text(
        encoding="utf-8"
    )
    assert "NaN" not in report_text
    assert "Infinity" not in report_text
    assert "NaN" not in manifest_text
    assert "Infinity" not in manifest_text

    data = json.loads(report_text)
    manifest = json.loads(manifest_text)
    assert [row["actor_label"] for row in data["rows"]] == ["Good"]
    assert data["rows"][0]["full_pnl_usd"] == 4.0
    assert data["rows"][0]["full_closed_trades"] == 0
    assert isinstance(manifest, dict)


def test_signal_key_shadow_report_uses_latest_half_year_when_timestamps_exist(tmp_path):
    path = runner.write_flash_signal_key_shadow_report(
        tmp_path,
        shadow_updates=[
            ShadowActorUpdated(
                bar=1,
                timestamp=datetime(2025, 12, 31, 23, tzinfo=timezone.utc),
                actor_type="agent",
                actor_label="Alpha",
                regime="bearish",
                symbol_action_outcomes=(("BTC/USDT", "short", 9.0, 20, 12),),
            ),
            ShadowActorUpdated(
                bar=2,
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                actor_type="agent",
                actor_label="Alpha",
                regime="bearish",
                symbol_action_outcomes=(("BTC/USDT", "short", 4.0, 5, 3),),
            ),
            ShadowActorUpdated(
                bar=3,
                timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc),
                actor_type="agent",
                actor_label="Alpha",
                regime="bearish",
                symbol_action_outcomes=(("BTC/USDT", "short", -1.0, 5, 2),),
            ),
        ],
        initial_capital=100.0,
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    row = data["rows"][0]

    assert data["summary"]["latest_period"] == "2026-H1"
    assert row["action"] == "FUT_SHORT_FULL"
    assert row["latest_pnl_usd"] == 3.0
    assert row["latest_pnl_pct"] == 3.0
    assert row["latest_closed_trades"] == 10
    assert row["recent_downside_usd"] == 1.0


def test_signal_key_shadow_report_rejects_invalid_initial_capital(tmp_path):
    with pytest.raises(ValueError, match="initial_capital must be > 0"):
        runner.write_flash_signal_key_shadow_report(
            tmp_path,
            shadow_updates=[],
            initial_capital=0.0,
        )


def test_signal_key_shadow_report_writes_promotion_manifest(tmp_path):
    path = runner.write_flash_signal_key_shadow_report(
        tmp_path,
        shadow_updates=[
            ShadowActorUpdated(
                bar=1,
                actor_type="agent",
                actor_label="Allowed",
                regime="bullish",
                symbol_action_outcomes=(("BTC/USDT", "long", 8.0, 60, 40),),
            ),
            ShadowActorUpdated(
                bar=2,
                actor_type="agent",
                actor_label="Rejected",
                regime="bullish",
                symbol_action_outcomes=(("ETH/USDT", "long", 1.0, 1, 1),),
            ),
        ],
        initial_capital=100.0,
    )

    report = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = tmp_path / "flash_promotion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert report["summary"]["signal_key_count"] == 2
    assert manifest["allowed_signal_keys"] == [
        "agent:Allowed|BTC/USDT|FUT_LONG_FULL"
    ]
    assert manifest["rejected"] == [
        {
            "signal_key": "agent:Rejected|ETH/USDT|FUT_LONG_FULL",
            "reason": "full_closed_trades_below_gate",
        }
    ]
    assert (tmp_path / "flash_signal_key_shadow_report.md").exists()
    assert (tmp_path / "flash_promotion_manifest.md").exists()


def test_signal_key_shadow_report_uses_custom_promotion_manifest_gates(tmp_path):
    runner.write_flash_signal_key_shadow_report(
        tmp_path,
        shadow_updates=[
            ShadowActorUpdated(
                bar=1,
                actor_type="agent",
                actor_label="ShortRunWinner",
                regime="bullish",
                symbol_action_outcomes=(("BTC/USDT", "short", 1.5, 2, 1),),
            ),
        ],
        initial_capital=100.0,
        promotion_manifest_config=runner.PromotionManifestConfig(
            min_full_closed_trades=1,
            min_latest_closed_trades=1,
            min_win_rate_pct=0.0,
            min_win_rate_lcb_pct=0.0,
        ),
    )

    manifest = json.loads(
        (tmp_path / "flash_promotion_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["allowed_signal_keys"] == [
        "agent:ShortRunWinner|BTC/USDT|FUT_SHORT_FULL"
    ]


def test_experimental_flash_shadow_report_gates_shadow_labels(tmp_path):
    path = runner.write_experimental_flash_shadow_report(
        tmp_path,
        shadow_agent_pnl_events=[
            ShadowPnLEvent(bar=1, label="GoodWrapper", pnl_usd=8.0, closed_trades=2, wins=2),
            ShadowPnLEvent(bar=2, label="GoodWrapper", pnl_usd=-2.0, closed_trades=1, wins=0),
            ShadowPnLEvent(bar=3, label="GoodWrapper", pnl_usd=6.0, closed_trades=1, wins=1),
            ShadowPnLEvent(bar=1, label="BadWrapper", pnl_usd=-4.0, closed_trades=2, wins=0),
            ShadowPnLEvent(bar=2, label="BadWrapper", pnl_usd=1.0, closed_trades=1, wins=1),
            ShadowPnLEvent(bar=1, label="LeakyWrapper", pnl_usd=9.0, closed_trades=3, wins=3),
        ],
        experimental_labels=("GoodWrapper", "BadWrapper", "LeakyWrapper"),
        initial_capital=100.0,
        flash_attribution={
            "actor_summary": {
                "agent:LeakyWrapper": {
                    "selected_signals": 2,
                    "closed_trades": 1,
                    "realized_pnl_usd": 3.0,
                }
            }
        },
        min_closed_trades=3,
        min_pnl_pct=5.0,
        max_drawdown_pct=15.0,
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    by_label = {row["label"]: row for row in data["labels"]}

    assert data["summary"]["label_count"] == 3
    assert data["summary"]["shadow_gate_passed"] == 2
    assert data["summary"]["promote_to_real"] == 1
    assert by_label["GoodWrapper"]["shadow_pnl_pct"] == 12.0
    assert by_label["GoodWrapper"]["shadow_gate_passed"] is True
    assert by_label["GoodWrapper"]["real_selected_signals"] == 0
    assert by_label["GoodWrapper"]["promote_to_real"] is True
    assert by_label["BadWrapper"]["shadow_gate_passed"] is False
    assert by_label["LeakyWrapper"]["shadow_gate_passed"] is True
    assert by_label["LeakyWrapper"]["real_selected_signals"] == 2
    assert by_label["LeakyWrapper"]["promote_to_real"] is False
    assert (tmp_path / "experimental_flash_shadow_report.md").exists()


def test_component_benchmark_report_marks_panteon_under_best_component(tmp_path):
    path = runner.write_component_benchmark_report(
        tmp_path,
        panteon_pnl_usd=10.0,
        initial_capital=100.0,
        shadow_agent_pnl_events=[
            ShadowPnLEvent(bar=1, label="Alpha", pnl_usd=12.0, closed_trades=3, wins=2),
            ShadowPnLEvent(bar=2, label="Alpha", pnl_usd=3.0, closed_trades=1, wins=1),
            ShadowPnLEvent(bar=1, label="Beta", pnl_usd=4.0, closed_trades=2, wins=1),
        ],
        shadow_player_pnl_events=[
            ShadowPnLEvent(bar=1, label="PlayerOne", pnl_usd=7.0, closed_trades=2, wins=1),
        ],
        min_alpha_pct=2.0,
    )

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["summary"]["panteon_pnl_pct"] == 10.0
    assert data["summary"]["best_component_label"] == "Alpha"
    assert data["summary"]["best_component_pnl_pct"] == 15.0
    assert data["summary"]["panteon_alpha_pct"] == -5.0
    assert data["summary"]["panteon_beats_best_component"] is False
    assert (tmp_path / "component_benchmark_report.md").exists()


def test_component_benchmark_report_allows_player_to_be_best_component(tmp_path):
    path = runner.write_component_benchmark_report(
        tmp_path,
        panteon_pnl_usd=8.0,
        initial_capital=100.0,
        shadow_agent_pnl_events=[
            ShadowPnLEvent(bar=1, label="AgentOne", pnl_usd=5.0, closed_trades=2, wins=1),
        ],
        shadow_player_pnl_events=[
            ShadowPnLEvent(bar=1, label="PlayerOne", pnl_usd=11.0, closed_trades=3, wins=2),
        ],
    )

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["summary"]["best_component_type"] == "player"
    assert data["summary"]["best_component_label"] == "PlayerOne"
    assert data["summary"]["best_component_pnl_pct"] == 11.0


def test_component_benchmark_report_excludes_non_deployable_labels(tmp_path):
    path = runner.write_component_benchmark_report(
        tmp_path,
        panteon_pnl_usd=8.0,
        initial_capital=100.0,
        shadow_agent_pnl_events=[
            ShadowPnLEvent(bar=1, label="AgentOne", pnl_usd=5.0, closed_trades=2, wins=1),
        ],
        shadow_player_pnl_events=[
            ShadowPnLEvent(bar=1, label="DeniedPlayer", pnl_usd=20.0, closed_trades=4, wins=4),
            ShadowPnLEvent(bar=1, label="DeployablePlayer", pnl_usd=7.0, closed_trades=3, wins=2),
        ],
        excluded_component_labels=("DeniedPlayer",),
    )

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["summary"]["best_component_type"] == "player"
    assert data["summary"]["best_component_label"] == "DeployablePlayer"
    assert data["summary"]["best_component_pnl_pct"] == pytest.approx(7.0)
    assert data["summary"]["excluded_component_labels"] == ["DeniedPlayer"]
    assert [
        (row["actor_type"], row["label"])
        for row in data["components"]
    ] == [
        ("player", "DeployablePlayer"),
        ("agent", "AgentOne"),
    ]


def test_component_benchmark_report_ties_sort_deterministically(tmp_path):
    path = runner.write_component_benchmark_report(
        tmp_path,
        panteon_pnl_usd=10.0,
        initial_capital=100.0,
        shadow_agent_pnl_events=[
            ShadowPnLEvent(bar=1, label="Zulu", pnl_usd=5.0, closed_trades=1, wins=1),
            ShadowPnLEvent(bar=1, label="Alpha", pnl_usd=5.0, closed_trades=1, wins=0),
        ],
        shadow_player_pnl_events=[
            ShadowPnLEvent(bar=1, label="Alpha", pnl_usd=5.0, closed_trades=1, wins=1),
        ],
    )

    data = json.loads(path.read_text(encoding="utf-8"))

    assert [
        (row["actor_type"], row["label"])
        for row in data["components"]
    ] == [
        ("agent", "Alpha"),
        ("agent", "Zulu"),
        ("player", "Alpha"),
    ]
    assert data["summary"]["best_component_type"] == "agent"
    assert data["summary"]["best_component_label"] == "Alpha"


def test_component_benchmark_report_handles_empty_component_streams(tmp_path):
    path = runner.write_component_benchmark_report(
        tmp_path,
        panteon_pnl_usd=3.0,
        initial_capital=100.0,
        shadow_agent_pnl_events=[],
        shadow_player_pnl_events=[],
    )

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["components"] == []
    assert data["summary"]["best_component_label"] == ""
    assert data["summary"]["best_component_type"] == ""
    assert data["summary"]["best_component_pnl_pct"] == 0.0
    assert data["summary"]["panteon_alpha_pct"] == 3.0
    assert data["summary"]["panteon_beats_best_component"] is True


def test_component_benchmark_report_rejects_invalid_initial_capital(tmp_path):
    with pytest.raises(ValueError, match="initial_capital must be > 0"):
        runner.write_component_benchmark_report(
            tmp_path,
            panteon_pnl_usd=3.0,
            initial_capital=0.0,
            shadow_agent_pnl_events=[],
            shadow_player_pnl_events=[],
        )


def test_component_benchmark_panteon_pnl_requires_live_session_field():
    assert runner._live_session_panteon_pnl_usd({"live_session": {}}) is None
    assert runner._live_session_panteon_pnl_usd({"live_session": "bad"}) is None
    assert runner._live_session_panteon_pnl_usd({}) is None
    assert runner._live_session_panteon_pnl_usd({
        "live_session": {"panteon_owned_realized_pnl_usd": -2.5}
    }) == -2.5
    assert runner._live_session_panteon_pnl_usd({
        "live_session": {
            "panteon_owned_realized_pnl_usd": -2.5,
            "panteon_owned_total_pnl_usd": 4.0,
        }
    }) == 4.0


def test_component_benchmark_integration_skips_missing_panteon_pnl(tmp_path):
    step_errors = []

    path = runner._write_component_benchmark_report_from_status(
        tmp_path,
        {"live_session": {}},
        initial_capital=100.0,
        shadow_agent_pnl_events=[
            ShadowPnLEvent(bar=1, label="Alpha", pnl_usd=5.0, closed_trades=1, wins=1),
        ],
        shadow_player_pnl_events=[],
        step_errors=step_errors,
    )

    assert path is None
    assert step_errors == ["component_benchmark_failed: missing_panteon_pnl"]
    assert not (tmp_path / "component_benchmark_report.json").exists()
    assert not (tmp_path / "component_benchmark_report.md").exists()


def test_standalone_vs_flash_selected_report_compares_target_actors(tmp_path):
    path = runner.write_standalone_vs_flash_selected_report(
        tmp_path,
        shadow_agent_pnl_events=[
            ShadowPnLEvent(
                bar=1,
                label="LiveOIBreakout",
                pnl_usd=11.0,
                closed_trades=5,
                wins=3,
            ),
        ],
        shadow_player_pnl_events=[
            ShadowPnLEvent(
                bar=1,
                label="Solo_MomentumScalper",
                pnl_usd=18.0,
                closed_trades=6,
                wins=4,
            ),
        ],
        flash_attribution={
            "actor_summary": {
                "ensemble:Solo_MomentumScalper": {
                    "actor_label": "Solo_MomentumScalper",
                    "actor_type": "ensemble",
                    "selected_signals": 3,
                    "executable_selected_signals": 3,
                    "selected_filtered_before_execution": 0,
                    "filled_signals": 2,
                    "closed_trades": 2,
                    "realized_pnl_usd": 2.0,
                },
                "agent:LiveOIBreakout": {
                    "actor_label": "LiveOIBreakout",
                    "actor_type": "agent",
                    "selected_signals": 4,
                    "executable_selected_signals": 4,
                    "selected_filtered_before_execution": 0,
                    "filled_signals": 3,
                    "closed_trades": 3,
                    "realized_pnl_usd": -1.0,
                },
            },
            "rows": [
                {
                    "actor_key": "ensemble:Solo_MomentumScalper",
                    "actor_label": "Solo_MomentumScalper",
                    "actor_type": "ensemble",
                    "symbol": "BTC/USDT",
                    "action": "FUT_LONG_FULL",
                    "selected_signals": 2,
                    "closed_trades": 1,
                    "realized_pnl_usd": 4.0,
                },
                {
                    "actor_key": "ensemble:Solo_MomentumScalper",
                    "actor_label": "Solo_MomentumScalper",
                    "actor_type": "ensemble",
                    "symbol": "ETH/USDT",
                    "action": "FUT_SHORT_FULL",
                    "selected_signals": 1,
                    "closed_trades": 1,
                    "realized_pnl_usd": -2.0,
                },
                {
                    "actor_key": "agent:LiveOIBreakout",
                    "actor_label": "LiveOIBreakout",
                    "actor_type": "agent",
                    "symbol": "SOL/USDT",
                    "action": "FUT_LONG_FULL",
                    "selected_signals": 4,
                    "closed_trades": 3,
                    "realized_pnl_usd": -1.0,
                },
            ],
        },
        initial_capital=100.0,
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    by_label = {row["label"]: row for row in data["actors"]}

    assert data["summary"]["target_count"] == 2
    assert data["summary"]["targets_with_flash_selection"] == 2
    assert data["summary"]["total_standalone_pnl_pct"] == pytest.approx(29.0)
    assert data["summary"]["total_flash_selected_pnl_pct"] == 1.0
    assert data["summary"]["total_selection_alpha_pct"] == pytest.approx(-28.0)
    assert by_label["Solo_MomentumScalper"]["standalone_actor_type"] == "player"
    assert by_label["Solo_MomentumScalper"]["flash_actor_key"] == "ensemble:Solo_MomentumScalper"
    assert by_label["Solo_MomentumScalper"]["standalone_pnl_pct"] == 18.0
    assert by_label["Solo_MomentumScalper"]["flash_selected_pnl_pct"] == 2.0
    assert by_label["Solo_MomentumScalper"]["selection_alpha_pct"] == -16.0
    assert by_label["LiveOIBreakout"]["flash_actor_key"] == "agent:LiveOIBreakout"
    assert by_label["LiveOIBreakout"]["standalone_pnl_pct"] == 11.0
    assert by_label["LiveOIBreakout"]["flash_selected_pnl_pct"] == -1.0
    assert by_label["LiveOIBreakout"]["selection_alpha_pct"] == -12.0
    assert by_label["Solo_MomentumScalper"]["worst_signal_keys"][0]["signal_key"] == (
        "ensemble:Solo_MomentumScalper|ETH/USDT|FUT_SHORT_FULL"
    )
    assert (tmp_path / "standalone_vs_flash_selected_report.md").exists()


def test_final_artifacts_continue_to_oracle_after_experimental_failure(tmp_path, monkeypatch):
    calls = []
    step_errors = []

    def fake_flash_signal(*args, **kwargs):
        calls.append("flash_signal")

    def fake_component(*args, **kwargs):
        calls.append("component")

    def fake_standalone_vs_flash(*args, **kwargs):
        calls.append("standalone_vs_flash")

    def fake_experimental(*args, **kwargs):
        calls.append("experimental")
        raise RuntimeError("experimental boom")

    def fake_oracle(output_dir, trading_log, **kwargs):
        calls.append("oracle")
        path = Path(output_dir) / "oracle_mismatch_report.json"
        path.write_text(json.dumps({"summary": {"mismatch_rows": 0}}), encoding="utf-8")
        return path

    monkeypatch.setattr(runner, "write_flash_signal_key_shadow_report", fake_flash_signal)
    monkeypatch.setattr(runner, "_write_component_benchmark_report_from_status", fake_component)
    monkeypatch.setattr(runner, "write_standalone_vs_flash_selected_report", fake_standalone_vs_flash)
    monkeypatch.setattr(runner, "write_experimental_flash_shadow_report", fake_experimental)
    monkeypatch.setattr(runner, "write_oracle_mismatch_report", fake_oracle)

    runner._write_retrodate_final_artifact_reports(
        tmp_path,
        config=RetrodateMarketConfig(experimental_flash_actors_enabled=True),
        shadow_updates=[],
        shadow_agent_pnl_events=[],
        shadow_player_pnl_events=[],
        candidate_rejections=[],
        step_errors=step_errors,
    )

    assert calls == ["flash_signal", "component", "standalone_vs_flash", "experimental", "oracle"]
    assert step_errors == [
        "experimental_flash_shadow_report_failed: RuntimeError: experimental boom"
    ]
    assert (tmp_path / "oracle_mismatch_report.json").exists()


def test_final_artifacts_record_oracle_failure_independently(tmp_path, monkeypatch):
    calls = []
    step_errors = []

    def fake_flash_signal(*args, **kwargs):
        calls.append("flash_signal")

    def fake_component(*args, **kwargs):
        calls.append("component")

    def fake_standalone_vs_flash(*args, **kwargs):
        calls.append("standalone_vs_flash")

    def fake_experimental(*args, **kwargs):
        calls.append("experimental")

    def fake_oracle(*args, **kwargs):
        calls.append("oracle")
        raise ValueError("oracle boom")

    monkeypatch.setattr(runner, "write_flash_signal_key_shadow_report", fake_flash_signal)
    monkeypatch.setattr(runner, "_write_component_benchmark_report_from_status", fake_component)
    monkeypatch.setattr(runner, "write_standalone_vs_flash_selected_report", fake_standalone_vs_flash)
    monkeypatch.setattr(runner, "write_experimental_flash_shadow_report", fake_experimental)
    monkeypatch.setattr(runner, "write_oracle_mismatch_report", fake_oracle)

    runner._write_retrodate_final_artifact_reports(
        tmp_path,
        config=RetrodateMarketConfig(experimental_flash_actors_enabled=True),
        shadow_updates=[],
        shadow_agent_pnl_events=[],
        shadow_player_pnl_events=[],
        candidate_rejections=[],
        step_errors=step_errors,
    )

    assert calls == ["flash_signal", "component", "standalone_vs_flash", "experimental", "oracle"]
    assert step_errors == ["oracle_mismatch_report_failed: ValueError: oracle boom"]


def test_experimental_flash_shadow_report_includes_latest_half_year_gate(tmp_path):
    path = runner.write_experimental_flash_shadow_report(
        tmp_path,
        shadow_agent_pnl_events=[
            ShadowPnLEvent(
                bar=1,
                label="GoodWrapper",
                timestamp="2025-12-31T23:00:00+00:00",
                pnl_usd=-6.0,
                closed_trades=2,
                wins=0,
            ),
            ShadowPnLEvent(
                bar=2,
                label="GoodWrapper",
                timestamp="2026-01-01T00:00:00+00:00",
                pnl_usd=5.0,
                closed_trades=2,
                wins=2,
            ),
            ShadowPnLEvent(
                bar=3,
                label="GoodWrapper",
                timestamp="2026-05-01T00:00:00+00:00",
                pnl_usd=2.0,
                closed_trades=1,
                wins=1,
            ),
            ShadowPnLEvent(
                bar=2,
                label="BadWrapper",
                timestamp="2026-01-01T00:00:00+00:00",
                pnl_usd=-3.0,
                closed_trades=2,
                wins=0,
            ),
        ],
        experimental_labels=("GoodWrapper", "BadWrapper"),
        initial_capital=100.0,
        min_closed_trades=3,
        min_pnl_pct=1.0,
        max_drawdown_pct=10.0,
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    latest = data["summary"]["latest_period"]
    periods = {
        (row["label"], row["period"]): row
        for row in data["periods"]
    }

    assert latest == "2026-H1"
    assert data["summary"]["latest_period_gate_passed"] == 1
    assert periods[("GoodWrapper", "2026-H1")]["pnl_pct"] == pytest.approx(7.0)
    assert periods[("GoodWrapper", "2026-H1")]["gate_passed"] is True
    assert periods[("BadWrapper", "2026-H1")]["gate_passed"] is False


def test_analysis_report_and_summary_include_flash_attribution_summary(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "status.json").write_text(
        json.dumps({
            "current_leader": "Panteon_Flash",
            "live_session": {
                "panteon_owned_pnl_pct": 1.0,
                "panteon_owned_realized_pnl_usd": 10.0,
                "real_closed_trades": 1,
                "panteon_owned_positions_count": 0,
            },
            "shadow": {"actors": 2},
        }),
        encoding="utf-8",
    )
    (output_dir / "leaderboard_agents.json").write_text(
        json.dumps({"agents": {}}),
        encoding="utf-8",
    )
    (output_dir / "leaderboard_players.json").write_text(
        json.dumps({"players": {}}),
        encoding="utf-8",
    )
    (output_dir / "flash_attribution_summary.json").write_text(
        json.dumps({
            "summary": {
                "selected_signals": 3,
                "filled_signals": 2,
                "blocked_signals": 1,
                "rejected_signals": 0,
                "pending_signals": 0,
                "closed_trades": 1,
                "realized_pnl_usd": 4.0,
            },
            "rows": [
                {
                    "actor_key": "agent:Alpha",
                    "actor_label": "Alpha",
                    "actor_type": "agent",
                    "symbol": "BTC/USDT",
                    "action": "FUT_LONG_FULL",
                    "selected_signals": 2,
                    "filled_signals": 2,
                    "blocked_signals": 0,
                    "closed_trades": 1,
                    "realized_pnl_usd": 4.0,
                }
            ],
        }),
        encoding="utf-8",
    )
    summary = RetrodateRunSummary(
        output_dir=output_dir,
        report_path=output_dir / "analysis_report.md",
        summary_path=output_dir / "run_summary.json",
        requested_years=(2025,),
        executed_years=(2025,),
        excluded_files=(),
        missing_years=(),
        bars_processed=4,
        registered_agents=("A",),
        player_profile_count=1,
        first_timestamp="2025-01-01T00:00:00+00:00",
        last_timestamp="2025-01-01T04:00:00+00:00",
        stride_minutes=60,
        max_bars=None,
        step_errors=(),
    )
    selection = RetrodateFileSelection(
        report=RetrodateDirReport(path=tmp_path, files=[]),
        valid_reports=(),
        excluded_files=(),
        missing_years=(),
    )

    runner._write_run_summary(
        summary.summary_path,
        summary,
        selection,
        RetrodateMarketConfig(data_dir=tmp_path, years=(2025,)),
    )
    runner._write_analysis_report(summary.report_path, summary, selection)

    run_summary = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    report = summary.report_path.read_text(encoding="utf-8")
    assert run_summary["flash_attribution_summary"]["selected_signals"] == 3
    assert "## Flash Attribution" in report
    assert "Selected signals: 3" in report
    assert "`agent:Alpha` / `BTC/USDT` / `FUT_LONG_FULL`" in report
    assert "`flash_attribution_summary.json`" in report


def test_analysis_report_and_summary_include_experimental_flash_shadow_report(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "status.json").write_text(
        json.dumps({
            "current_leader": "Panteon_Flash",
            "live_session": {
                "panteon_owned_pnl_pct": 1.0,
                "panteon_owned_realized_pnl_usd": 10.0,
                "real_closed_trades": 1,
                "panteon_owned_positions_count": 0,
            },
            "shadow": {"actors": 2},
        }),
        encoding="utf-8",
    )
    (output_dir / "leaderboard_agents.json").write_text(
        json.dumps({"agents": {}}),
        encoding="utf-8",
    )
    (output_dir / "leaderboard_players.json").write_text(
        json.dumps({"players": {}}),
        encoding="utf-8",
    )
    (output_dir / "experimental_flash_shadow_report.json").write_text(
        json.dumps({
            "summary": {
                "label_count": 2,
                "shadow_gate_passed": 1,
                "promote_to_real": 0,
                "real_selected_signals": 3,
            },
            "labels": [
                {
                    "label": "GoodWrapper",
                    "shadow_pnl_pct": 8.0,
                    "closed_trades": 10,
                    "max_drawdown_pct": 3.0,
                    "real_selected_signals": 0,
                    "promote_to_real": True,
                }
            ],
        }),
        encoding="utf-8",
    )
    summary = RetrodateRunSummary(
        output_dir=output_dir,
        report_path=output_dir / "analysis_report.md",
        summary_path=output_dir / "run_summary.json",
        requested_years=(2025,),
        executed_years=(2025,),
        excluded_files=(),
        missing_years=(),
        bars_processed=4,
        registered_agents=("A",),
        player_profile_count=1,
        first_timestamp="2025-01-01T00:00:00+00:00",
        last_timestamp="2025-01-01T04:00:00+00:00",
        stride_minutes=60,
        max_bars=None,
        step_errors=(),
    )
    selection = RetrodateFileSelection(
        report=RetrodateDirReport(path=tmp_path, files=[]),
        valid_reports=(),
        excluded_files=(),
        missing_years=(),
    )

    runner._write_run_summary(
        summary.summary_path,
        summary,
        selection,
        RetrodateMarketConfig(data_dir=tmp_path, years=(2025,)),
    )
    runner._write_analysis_report(summary.report_path, summary, selection)

    run_summary = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    report = summary.report_path.read_text(encoding="utf-8")
    assert run_summary["experimental_flash_shadow_report"]["label_count"] == 2
    assert "## Experimental Flash Shadow Gate" in report
    assert "Promote-to-real labels: 0" in report
    assert "`GoodWrapper`: shadow PnL 8.00%" in report


def test_analysis_report_and_summary_include_component_benchmark_report(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "status.json").write_text(
        json.dumps({
            "current_leader": "Panteon_Flash",
            "live_session": {
                "panteon_owned_pnl_pct": 1.0,
                "panteon_owned_realized_pnl_usd": 10.0,
                "real_closed_trades": 1,
                "panteon_owned_positions_count": 0,
            },
            "shadow": {"actors": 2},
        }),
        encoding="utf-8",
    )
    (output_dir / "leaderboard_agents.json").write_text(
        json.dumps({"agents": {}}),
        encoding="utf-8",
    )
    (output_dir / "leaderboard_players.json").write_text(
        json.dumps({"players": {}}),
        encoding="utf-8",
    )
    (output_dir / "component_benchmark_report.json").write_text(
        json.dumps({
            "summary": {
                "panteon_pnl_usd": 10.0,
                "panteon_pnl_pct": 10.0,
                "best_component_label": "Alpha",
                "best_component_type": "agent",
                "best_component_pnl_usd": 15.0,
                "best_component_pnl_pct": 15.0,
                "panteon_alpha_pct": -5.0,
                "min_alpha_pct": 2.0,
                "panteon_beats_best_component": False,
            },
            "components": [
                {
                    "actor_type": "agent",
                    "label": "Alpha",
                    "pnl_pct": 15.0,
                    "closed_trades": 4,
                    "win_rate_pct": 75.0,
                }
            ],
        }),
        encoding="utf-8",
    )
    (output_dir / "component_benchmark_report.md").write_text(
        "# Component Benchmark\n",
        encoding="utf-8",
    )
    summary = RetrodateRunSummary(
        output_dir=output_dir,
        report_path=output_dir / "analysis_report.md",
        summary_path=output_dir / "run_summary.json",
        requested_years=(2025,),
        executed_years=(2025,),
        excluded_files=(),
        missing_years=(),
        bars_processed=4,
        registered_agents=("A",),
        player_profile_count=1,
        first_timestamp="2025-01-01T00:00:00+00:00",
        last_timestamp="2025-01-01T04:00:00+00:00",
        stride_minutes=60,
        max_bars=None,
        step_errors=(),
    )
    selection = RetrodateFileSelection(
        report=RetrodateDirReport(path=tmp_path, files=[]),
        valid_reports=(),
        excluded_files=(),
        missing_years=(),
    )

    runner._write_run_summary(
        summary.summary_path,
        summary,
        selection,
        RetrodateMarketConfig(data_dir=tmp_path, years=(2025,)),
    )
    runner._write_analysis_report(summary.report_path, summary, selection)

    run_summary = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    report = summary.report_path.read_text(encoding="utf-8")
    assert run_summary["component_benchmark_report"]["best_component_label"] == "Alpha"
    assert "## Component Benchmark" in report
    assert "Panteon alpha: -5.00%" in report
    assert "`agent:Alpha`: 15.00%" in report
    assert "`component_benchmark_report.json`" in report
    assert "`component_benchmark_report.md`" in report


def test_analysis_report_and_summary_include_standalone_vs_flash_report(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "status.json").write_text(
        json.dumps({
            "current_leader": "Panteon_Flash",
            "live_session": {
                "panteon_owned_pnl_pct": 1.0,
                "panteon_owned_realized_pnl_usd": 10.0,
                "real_closed_trades": 1,
                "panteon_owned_positions_count": 0,
            },
            "shadow": {"actors": 2},
        }),
        encoding="utf-8",
    )
    (output_dir / "leaderboard_agents.json").write_text(
        json.dumps({"agents": {}}),
        encoding="utf-8",
    )
    (output_dir / "leaderboard_players.json").write_text(
        json.dumps({"players": {}}),
        encoding="utf-8",
    )
    (output_dir / "standalone_vs_flash_selected_report.json").write_text(
        json.dumps({
            "summary": {
                "target_count": 2,
                "targets_with_flash_selection": 2,
                "total_standalone_pnl_pct": 18.15,
                "total_flash_selected_pnl_pct": 3.75,
                "total_selection_alpha_pct": -14.4,
            },
            "actors": [
                {
                    "label": "Solo_MomentumScalper",
                    "standalone_pnl_pct": 18.15,
                    "flash_selected_pnl_pct": 3.75,
                    "selection_alpha_pct": -14.4,
                    "flash_selected_signals": 10,
                }
            ],
        }),
        encoding="utf-8",
    )
    (output_dir / "standalone_vs_flash_selected_report.md").write_text(
        "# Standalone vs Flash Selected\n",
        encoding="utf-8",
    )
    summary = RetrodateRunSummary(
        output_dir=output_dir,
        report_path=output_dir / "analysis_report.md",
        summary_path=output_dir / "run_summary.json",
        requested_years=(2025,),
        executed_years=(2025,),
        excluded_files=(),
        missing_years=(),
        bars_processed=4,
        registered_agents=("A",),
        player_profile_count=1,
        first_timestamp="2025-01-01T00:00:00+00:00",
        last_timestamp="2025-01-01T04:00:00+00:00",
        stride_minutes=60,
        max_bars=None,
        step_errors=(),
    )
    selection = RetrodateFileSelection(
        report=RetrodateDirReport(path=tmp_path, files=[]),
        valid_reports=(),
        excluded_files=(),
        missing_years=(),
    )

    runner._write_run_summary(
        summary.summary_path,
        summary,
        selection,
        RetrodateMarketConfig(data_dir=tmp_path, years=(2025,)),
    )
    runner._write_analysis_report(summary.report_path, summary, selection)

    run_summary = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    report = summary.report_path.read_text(encoding="utf-8")
    assert run_summary["standalone_vs_flash_selected_report"]["target_count"] == 2
    assert "## Standalone vs Flash Selected" in report
    assert "Selection alpha: -14.40%" in report
    assert "`Solo_MomentumScalper`: standalone 18.15%" in report
    assert "`standalone_vs_flash_selected_report.json`" in report


def test_candidate_group_type_marks_regime_switch_and_rotator() -> None:
    assert runner._candidate_group_type("Antonius_strategy") == "regime_switch"
    assert runner._candidate_group_type("Perfect_OIBreakout") == "regime_switch"
    assert runner._candidate_group_type("Optimal_StaticRotator") == "rotating"


def test_oracle_mismatch_report_marks_perfect_leader_without_current_signal(tmp_path):
    trading_log = tmp_path / "trading.log"
    trading_log.write_text(
        "2025-01-01 00:00:00  bar=1  regime=bullish  leader=NoTrade  "
        "selected_leader=NoTrade  executed_leader=NoTrade  raw_signals=0  "
        "signals=0  filled=0\n",
        encoding="utf-8",
    )

    path = runner.write_oracle_mismatch_report(
        tmp_path,
        trading_log,
        shadow_updates=[
            ShadowActorUpdated(
                bar=1,
                timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
                actor_type="player",
                actor_label="PerfectPlayer",
                regime="bullish",
                signals=0,
                filled=0,
                realized_pnl_usd=10.0,
                closed_trades=2,
                winning_trades=2,
            )
        ],
        candidate_rejections=[],
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["summary"]["mismatch_rows"] == 1
    assert data["summary"]["reason_counts"]["no current signal"] == 1
    row = data["rows"][0]
    assert row["real_leader"] == "NoTrade"
    assert row["perfect_leader"] == "PerfectPlayer"
    assert row["reason"] == "no current signal"
