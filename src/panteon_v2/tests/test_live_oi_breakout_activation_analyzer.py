from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "analyze_live_oi_breakout_activation.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "analyze_live_oi_breakout_activation",
        TOOL_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _decision(symbol: str, diagnostics: dict) -> dict:
    return {
        "symbol": symbol,
        "selected_actor": "NoTrade",
        "top_rejected_candidates": [
            {
                "label": "LiveOIBreakout",
                "actor_key": "agent:LiveOIBreakout",
                "action": "HOLD",
                "reason": "inactive",
                "agent_diagnostics": diagnostics,
            }
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_analyzer_summarizes_real_checks_waits_and_momentum_near_misses(tmp_path):
    tool = _load_tool()
    path = tmp_path / "run" / "causal_entry_decisions.jsonl"
    _write_jsonl(
        path,
        [
            {
                "bar": 100,
                "flash_decisions": [
                    _decision(
                        "BTC",
                        {
                            "bar": 100,
                            "reason": "momentum_below_threshold",
                            "check_interval": 60,
                            "action": 0,
                            "momentum": 0.0014,
                            "momentum_threshold": 0.002,
                            "volume_spike": True,
                            "oi_expansion": False,
                        },
                    )
                ],
            },
            {
                "bar": 101,
                "flash_decisions": [
                    _decision(
                        "BTC",
                        {
                            "bar": 101,
                            "reason": "check_interval_wait",
                            "check_interval": 60,
                            "bars_since_check": 1,
                            "wait_bars": 59,
                            "action": 0,
                        },
                    )
                ],
            },
            {
                "bar": 160,
                "flash_decisions": [
                    _decision(
                        "ETH",
                        {
                            "bar": 160,
                            "reason": "no_volume_or_oi_breakout",
                            "check_interval": 60,
                            "action": 0,
                            "momentum": -0.003,
                            "momentum_threshold": 0.002,
                            "volume_spike": False,
                            "oi_expansion": False,
                        },
                    )
                ],
            },
        ],
    )

    report = tool.analyze_paths(
        [path],
        actor_label="LiveOIBreakout",
        mom_min_grid=(0.001, 0.002),
        min_real_checks_per_symbol=2,
    )

    assert report["rows"] == 3
    assert report["candidate_diagnostics"] == 3
    assert report["wait_check_count"] == 1
    assert report["real_check_count"] == 2
    assert report["effective_checks_by_symbol"] == {"BTC": 1, "ETH": 1}
    assert report["reason_counts"] == {
        "check_interval_wait": 1,
        "momentum_below_threshold": 1,
        "no_volume_or_oi_breakout": 1,
    }
    assert report["momentum"]["abs_max"] == 0.003
    assert report["near_miss_grid"]["0.001000"]["would_pass_breakout_and_momentum"] == 1
    assert report["near_miss_grid"]["0.001000"]["long_count"] == 1
    assert report["near_miss_grid"]["0.002000"]["would_pass_breakout_and_momentum"] == 0
    assert report["hard_blocked"] is True
    assert "insufficient_real_checks" in report["activation_blockers"]
    assert "zero_candidate_signals" in report["activation_blockers"]


def test_analyzer_accepts_directory_inputs_and_writes_json_and_markdown(tmp_path):
    tool = _load_tool()
    path = tmp_path / "MEXC" / "session_v2" / "causal_entry_decisions.jsonl"
    _write_jsonl(
        path,
        [
            {
                "bar": 200,
                "flash_decisions": [
                    _decision(
                        "SOL",
                        {
                            "bar": 200,
                            "reason": "candidate_long",
                            "check_interval": 60,
                            "action": 2,
                            "momentum": 0.0031,
                            "momentum_threshold": 0.002,
                            "volume_spike": True,
                            "oi_expansion": False,
                        },
                    )
                ],
            }
        ],
    )

    paths = tool.resolve_causal_paths([tmp_path])
    report = tool.analyze_paths(
        paths,
        actor_label="LiveOIBreakout",
        mom_min_grid=(0.002,),
        min_real_checks_per_symbol=1,
    )
    out_json, out_md = tool.write_reports(report, tmp_path / "report.json")

    assert paths == [path]
    assert report["hard_blocked"] is False
    assert report["candidate_signal_count"] == 1
    assert out_json.exists()
    assert out_md.exists()
    assert "LiveOIBreakout Activation Report" in out_md.read_text(encoding="utf-8")


def test_analyzer_blocks_when_each_file_has_too_few_real_checks_even_if_aggregate_passes(tmp_path):
    tool = _load_tool()
    for exchange in ("MEXC", "BITGET"):
        _write_jsonl(
            tmp_path / exchange / "session_v2" / "causal_entry_decisions.jsonl",
            [
                {
                    "bar": 300,
                    "flash_decisions": [
                        _decision(
                            "BTC",
                            {
                                "bar": 300,
                                "reason": "momentum_below_threshold",
                                "check_interval": 60,
                                "action": 0,
                                "momentum": 0.001,
                                "momentum_threshold": 0.002,
                                "volume_spike": True,
                                "oi_expansion": False,
                            },
                        )
                    ],
                }
            ],
        )

    report = tool.analyze_paths(
        tool.resolve_causal_paths([tmp_path]),
        actor_label="LiveOIBreakout",
        min_real_checks_per_symbol=2,
    )

    assert report["effective_checks_by_symbol"] == {"BTC": 2}
    assert report["sufficient_real_checks"] is True
    assert report["min_effective_checks_per_file_symbol"] == 1
    assert report["sufficient_real_checks_per_file"] is False
    assert "insufficient_real_checks_per_file" in report["activation_blockers"]


def test_analyzer_treats_dominant_filter_as_warning_when_candidate_signals_exist(tmp_path):
    tool = _load_tool()
    path = tmp_path / "run" / "causal_entry_decisions.jsonl"
    _write_jsonl(
        path,
        [
            {
                "bar": 400,
                "flash_decisions": [
                    _decision(
                        "BTC",
                        {
                            "bar": 400,
                            "reason": "momentum_below_threshold",
                            "check_interval": 1,
                            "action": 0,
                            "momentum": 0.0002,
                            "momentum_threshold": 0.0015,
                            "volume_spike": True,
                            "oi_expansion": False,
                        },
                    )
                ],
            },
            {
                "bar": 401,
                "flash_decisions": [
                    _decision(
                        "BTC",
                        {
                            "bar": 401,
                            "reason": "candidate_long",
                            "check_interval": 1,
                            "action": 2,
                            "momentum": 0.002,
                            "momentum_threshold": 0.0015,
                            "volume_spike": True,
                            "oi_expansion": False,
                        },
                    )
                ],
            },
        ],
    )

    report = tool.analyze_paths(
        [path],
        actor_label="LiveOIBreakout",
        min_real_checks_per_symbol=1,
    )

    assert report["candidate_signal_count"] == 1
    assert report["hard_blocked"] is False
    assert report["activation_blockers"] == []
    assert report["diagnostic_warnings"] == ["dominant_filter:momentum_below_threshold"]


def test_activation_report_marks_confirmed_range_transition_breakout(tmp_path):
    tool = _load_tool()
    path = tmp_path / "causal_entry_decisions.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({
                "timestamp": "2026-07-06T00:00:00+00:00",
                "actor_label": "LiveOIBreakout",
                "symbol": "BTC",
                "regime": "range_low_vol",
                "next_regime": "bullish",
                "action": "FUT_LONG_FULL",
                "agent_diagnostics": {
                    "reason": "candidate_long",
                    "momentum": 0.004,
                    "volume_spike": True,
                    "oi_expansion": True,
                    "close_outside_range": True,
                    "spread_ok": True,
                    "slippage_ok": True,
                },
            })
        ]) + "\n",
        encoding="utf-8",
    )

    report = tool.analyze_paths(
        [path],
        actor_label="LiveOIBreakout",
        min_real_checks_per_symbol=1,
    )

    assert report["transition_breakouts"]["confirmed"] == 1
    assert report["transition_breakouts"]["by_symbol"]["BTC"] == 1
    assert report["hard_blocked"] is False
