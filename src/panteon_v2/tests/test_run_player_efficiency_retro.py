from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    path = ROOT / "tools" / "run_player_efficiency_retro.py"
    spec = importlib.util.spec_from_file_location(
        "run_player_efficiency_retro",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _summary_fixture(module, tmp_path: Path, window_pnls, *, stressed_pnl: float):
    _write_json(
        tmp_path / "status.json",
        {
            "real_trades": {"closed": 10},
            "pnl_usd": 10.0,
            "panteon_realized_max_drawdown_pct": 2.0,
            "shadow": {"actors": 3},
        },
    )
    _write_json(tmp_path / "leaderboard_players.json", {"players": {}})
    _write_json(
        tmp_path / "walk_forward_report.json",
        {
            "temporal_windows": {
                "windows": [
                    {
                        "window": f"window_{index:02d}",
                        "bar_start": index,
                        "bar_end": index,
                        "stats": {
                            "closed_trades": 2,
                            "net_pnl": pnl,
                            "expectancy": pnl / 2.0,
                        },
                    }
                    for index, pnl in enumerate(window_pnls, start=1)
                ]
            },
            "totals": {"net_pnl_minus_5bps": stressed_pnl},
        },
    )
    args = module._parser().parse_args([])
    args.min_bars_for_promotion = 1
    args.min_real_closed = 1
    args.min_shadow_closed = 0
    args.max_drawdown_pct = 10.0
    args.min_positive_window_ratio = 0.8
    args.require_latest_window_positive = True
    args.required_extra_cost_bps = 5
    summary = SimpleNamespace(
        output_dir=tmp_path,
        bars_processed=100,
        step_errors=(),
        summary_path=tmp_path / "run_summary.json",
    )
    return module._promotion_summary(summary, args)


def test_promotion_requires_temporal_stability_and_cost_stress(tmp_path):
    module = _load_tool()

    payload = _summary_fixture(
        module,
        tmp_path,
        (2.0, 2.0, -1.0, 2.0, 2.0),
        stressed_pnl=3.0,
    )

    assert payload["promotion_verdict"]["passed"]
    assert payload["temporal_stability"]["positive_window_ratio"] == 0.8
    assert payload["temporal_stability"]["latest_window_positive"]
    assert payload["cost_stress"]["net_pnl_usd"] == 3.0


def test_promotion_rejects_unstable_latest_window_and_cost_stress(tmp_path):
    module = _load_tool()

    payload = _summary_fixture(
        module,
        tmp_path,
        (2.0, 2.0, -1.0, 2.0, -1.0),
        stressed_pnl=-0.5,
    )

    reasons = payload["promotion_verdict"]["fail_reasons"]
    assert "positive_window_ratio<0.8" in reasons
    assert "latest_window_not_positive" in reasons
    assert "pnl_after_extra_5bps<=0" in reasons


def test_existing_summary_resumes_completed_run_without_replay(tmp_path):
    module = _load_tool()
    _write_json(tmp_path / "status.json", {})
    _write_json(tmp_path / "leaderboard_players.json", {"players": {}})
    _write_json(tmp_path / "walk_forward_report.json", {"totals": {}})
    summary_path = tmp_path / "run_summary.json"
    _write_json(
        summary_path,
        {
            "output_dir": str(tmp_path),
            "bars_processed": 123,
            "step_errors": [],
        },
    )

    summary = module._existing_summary(str(tmp_path))

    assert summary.output_dir == tmp_path.resolve()
    assert summary.summary_path == summary_path.resolve()
    assert summary.bars_processed == 123
    assert summary.step_errors == ()
