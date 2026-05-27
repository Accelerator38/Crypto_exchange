import json
import math
import importlib.util
from pathlib import Path

from panteon_v2.analysis.flash_offline_context_attribution import (
    write_offline_context_attribution_summary,
)


ROOT = Path(__file__).resolve().parents[3]


def _signal(signal_id: int, *, bar: int, action: str, price: float) -> dict:
    return {
        "id": signal_id,
        "bar": bar,
        "sym": "BTC/USDT",
        "action": action,
        "price": price,
        "regime": "bullish",
        "by_player": "Solo_MomentumScalper",
        "by_agent": "Solo_MomentumScalper",
        "timestamp": f"2025-01-01T0{bar}:00:00+00:00",
    }


def test_offline_context_attribution_replays_executable_close_to_open_context(tmp_path):
    open_signal = _signal(1, bar=1, action="FUT_LONG_FULL", price=100.0)
    close_signal = _signal(2, bar=2, action="FUT_CLOSE_ALL", price=120.0)
    rows = [
        {
            "bar": 1,
            "timestamp": "2025-01-01T01:00:00+00:00",
            "regime": "bullish",
            "executable_signals": [open_signal],
            "flash_decisions": [
                {
                    "selected_actor": "Solo_MomentumScalper",
                    "actor_type": "ensemble",
                    "symbol": "BTC/USDT",
                    "action": "FUT_LONG_FULL",
                    "score": 5.0,
                    "signal": open_signal,
                    "candidates": [
                        {
                            "label": "Solo_MomentumScalper",
                            "actor_type": "ensemble",
                            "actor_key": "ensemble:Solo_MomentumScalper",
                            "shadow_score": 5.0,
                            "shadow_closed_trades": 10,
                            "shadow_pnl_per_trade_lcb_usd": 1.0,
                        }
                    ],
                }
            ],
        },
        {
            "bar": 2,
            "timestamp": "2025-01-01T02:00:00+00:00",
            "regime": "bullish",
            "executable_signals": [close_signal],
            "flash_decisions": [],
        },
    ]
    with (tmp_path / "causal_entry_decisions.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    result = write_offline_context_attribution_summary(
        tmp_path,
        filename="flash_attribution_summary_offline.json",
    )

    data = json.loads(result.path.read_text(encoding="utf-8"))
    summary = data["summary"]
    assert result.rows_read == 2
    assert result.replayed_signals == 2
    assert result.status_counts == {"filled": 2, "pending": 0, "rejected": 0, "blocked": 0}
    assert summary["selected_signals"] == 1
    assert summary["filled_signals"] == 1
    assert summary["unattributed_execution_events"] == 1
    assert summary["closed_trades"] == 1
    assert math.isclose(summary["realized_pnl_usd"], 19.868, abs_tol=1e-12)

    context_rows = data["context_rows"]
    assert len(context_rows) == 1
    assert context_rows[0]["context_key"] == (
        "ensemble:Solo_MomentumScalper|BTC/USDT|FUT_LONG_FULL|bullish"
    )
    assert context_rows[0]["closed_trades"] == 1
    assert math.isclose(context_rows[0]["realized_pnl_usd"], 19.868, abs_tol=1e-12)


def test_offline_context_attribution_cli_writes_summary(tmp_path):
    open_signal = _signal(1, bar=1, action="FUT_LONG_FULL", price=100.0)
    with (tmp_path / "causal_entry_decisions.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "bar": 1,
            "regime": "bullish",
            "executable_signals": [open_signal],
            "flash_decisions": [
                {
                    "selected_actor": "Solo_MomentumScalper",
                    "actor_type": "ensemble",
                    "symbol": "BTC/USDT",
                    "action": "FUT_LONG_FULL",
                    "signal": open_signal,
                }
            ],
        }) + "\n")

    module = _load_offline_context_tool()
    rc = module.main([
        str(tmp_path),
        "--filename",
        "offline_context.json",
    ])

    assert rc == 0
    data = json.loads((tmp_path / "offline_context.json").read_text(encoding="utf-8"))
    assert data["summary"]["selected_signals"] == 1
    assert len(data["context_rows"]) == 1


def _load_offline_context_tool():
    path = ROOT / "tools" / "build_flash_offline_context_attribution.py"
    spec = importlib.util.spec_from_file_location(
        "build_flash_offline_context_attribution_tool",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
