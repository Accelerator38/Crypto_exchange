from __future__ import annotations

import json
from pathlib import Path


def test_current_regime_series_normalizes_retrodate_symbols(tmp_path: Path) -> None:
    from tools.build_panteon_current_regime_breakdown_report import (
        build_current_regime_series,
    )

    csv_path = tmp_path / "crypto_1m_2022_all_symbols.csv"
    csv_path.write_text(
        "\n".join(
            [
                "timestamp,open,high,low,close,volume,symbol,datetime",
                "1640995200000,100,100,100,100,10,BTC/USDT,2022-01-01 00:00:00+00:00",
                "1640995200000,50,50,50,50,20,ETH/USDT,2022-01-01 00:00:00+00:00",
                "1640998800000,101,101,101,101,11,BTC/USDT,2022-01-01 01:00:00+00:00",
                "1640998800000,49,49,49,49,21,ETH/USDT,2022-01-01 01:00:00+00:00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_current_regime_series(
        [csv_path],
        stride_minutes=60,
        detector_kwargs={"hysteresis_bars": 1, "min_history_bars": 2},
    )

    assert [row["bar"] for row in rows] == [1, 2]
    assert rows[0]["symbol_count"] == 2
    assert set(rows[1]["symbol_regimes"]) == {"BTC", "ETH"}
    assert rows[1]["regime"] in {
        "bullish",
        "bearish",
        "neutral",
        "crash",
        "range_low_vol",
        "choppy_down",
        "choppy_up",
        "mixed_rotational",
    }


def test_current_regime_series_updates_detector_on_source_minutes_but_samples_stride(
    tmp_path: Path,
) -> None:
    from tools.build_panteon_current_regime_breakdown_report import (
        build_current_regime_series,
    )

    csv_path = tmp_path / "crypto_1m_2022_all_symbols.csv"
    rows = ["timestamp,open,high,low,close,volume,symbol,datetime"]
    for minute in range(61):
        timestamp = 1640995200000 + minute * 60_000
        rows.append(
            f"{timestamp},100,100,100,{100 + minute},10,BTC/USDT,2022-01-01 00:{minute:02d}:00+00:00"
        )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    series = build_current_regime_series(
        [csv_path],
        stride_minutes=60,
        source_update_minutes=1,
        detector_kwargs={"hysteresis_bars": 1, "min_history_bars": 2},
    )

    assert [row["timestamp_ms"] for row in series] == [
        1640995200000,
        1640998800000,
    ]
    assert series[1]["symbol_count"] == 1


def test_current_regime_series_groups_symbol_sorted_retrodate_rows(tmp_path: Path) -> None:
    from tools.build_panteon_current_regime_breakdown_report import (
        build_current_regime_series,
    )

    csv_path = tmp_path / "crypto_1m_2022_all_symbols.csv"
    csv_path.write_text(
        "\n".join(
            [
                "timestamp,open,high,low,close,volume,symbol,datetime",
                "1640995200000,100,100,100,100,10,BTC/USDT,2022-01-01 00:00:00+00:00",
                "1640998800000,101,101,101,101,11,BTC/USDT,2022-01-01 01:00:00+00:00",
                "1640995200000,50,50,50,50,20,ETH/USDT,2022-01-01 00:00:00+00:00",
                "1640998800000,49,49,49,49,21,ETH/USDT,2022-01-01 01:00:00+00:00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_current_regime_series(
        [csv_path],
        stride_minutes=60,
        source_update_minutes=1,
        detector_kwargs={"hysteresis_bars": 1, "min_history_bars": 2},
    )

    assert [row["bar"] for row in rows] == [1, 2]
    assert rows[0]["symbol_count"] == 2
    assert rows[1]["symbol_count"] == 2


def test_shadow_breakdown_uses_current_bar_regime_override(tmp_path: Path) -> None:
    from tools.build_panteon_current_regime_breakdown_report import (
        _shadow_breakdown_with_regime_override,
    )

    events_path = tmp_path / "shadow_agent_pnl_events.jsonl"
    events_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "bar": 1,
                        "label": "AgentA",
                        "regime": "neutral",
                        "pnl_usd": 2.5,
                        "closed_trades": 1,
                        "wins": 1,
                    }
                ),
                json.dumps(
                    {
                        "bar": 2,
                        "label": "AgentA",
                        "regime": "bearish",
                        "pnl_usd": -1.0,
                        "closed_trades": 1,
                        "wins": 0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = _shadow_breakdown_with_regime_override(
        events_path,
        initial_capital=100.0,
        actor_type="agent",
        regime_by_bar={1: "range_low_vol", 2: "mixed_rotational"},
    )

    by_regime = {row["regime"]: row for row in rows}
    assert by_regime["range_low_vol"]["pnl_pct"] == 2.5
    assert by_regime["range_low_vol"]["closed_trades"] == 1
    assert by_regime["mixed_rotational"]["pnl_pct"] == -1.0
    assert by_regime["mixed_rotational"]["losses"] == 1
