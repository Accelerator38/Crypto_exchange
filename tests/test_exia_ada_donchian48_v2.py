from __future__ import annotations

import json
from pathlib import Path

from exia.contracts import load_candidate_spec
from tools.run_exia_experiment_v1 import _taxonomy_api


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = ROOT / "freqtrade_pilot" / "candidates"
STRATEGY_PATH = (
    ROOT
    / "freqtrade_pilot"
    / "user_data"
    / "strategies"
    / "ExiaAdaDonchian48V2.py"
)


def test_ada_candidate_is_single_symbol_and_pins_taxonomy_v2() -> None:
    spec = load_candidate_spec(
        CANDIDATE_DIR / "exia_ada_donchian48_v2.json",
        root=ROOT,
    )

    assert spec.status == "DEVELOPMENT"
    assert spec.dataset["symbols"] == ["ADA"]
    assert spec.taxonomy["version"] == "exia.market_mode.v2"
    assert spec.allowed_market_states == ("TREND_UP", "TREND_DOWN")
    assert spec.parameters == {
        "pair": "ADA/USDT:USDT",
        "breakout_hours": 48,
        "max_holding_hours": 168,
        "stoploss": -0.05,
    }
    assert spec.risk["max_open_trades"] == 1
    assert spec.risk["leverage"] == 1
    assert spec.source["paper_allowed"] is False
    assert spec.source["live_allowed"] is False
    assert spec.source["orders_enabled"] is False
    assert spec.source["promotion_authority"] is False

    taxonomy_states, _ = _taxonomy_api(spec)
    assert set(spec.allowed_market_states) < set(taxonomy_states)


def test_ada_family_has_exactly_one_preregistered_variant() -> None:
    family = json.loads(
        (
            CANDIDATE_DIR / "exia_ada_donchian48_family_v2.json"
        ).read_text(encoding="utf-8")
    )

    assert family["maximum_variants"] == 1
    assert family["candidate_ids"] == ["exia_ada_donchian48_v2"]
    assert family["selection_window"] == "development"
    assert family["selection_cost"] == "stress"
    assert family["maximum_candidates_opened_on_validation"] == 1
    assert family["selection_gate"] == {
        "min_closed_trades": 20,
        "min_mean_net_bps": 0,
        "min_block_lcb_net_bps": 0,
        "max_drawdown": 0.2,
    }
    assert family["paper_allowed"] is False
    assert family["live_allowed"] is False
    assert family["orders_enabled"] is False
    assert family["promotion_authority"] is False


def test_ada_strategy_is_pair_locked_and_dry_run_only() -> None:
    source = STRATEGY_PATH.read_text(encoding="utf-8")

    assert "class ExiaAdaDonchian48V2(IStrategy)" in source
    assert 'allowed_pair = "ADA/USDT:USDT"' in source
    assert 'metadata.get("pair") != self.allowed_pair' in source
    assert 'self.config.get("dry_run") is not True' in source
    assert "runtime_startup_bars" in source
    assert "return 1.0" in source
    assert "stoploss = -0.05" in source
    assert "max_holding_hours = 168" in source
    assert '"stoploss_on_exchange": True' in source


def test_generated_ada_decision_is_fail_closed_when_present() -> None:
    path = (
        ROOT
        / "Reports"
        / "Exia"
        / "ada_donchian48_v2"
        / "development_report.json"
    )
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["status"] in {
        "SELECTED_FOR_VALIDATION",
        "NO_CANDIDATE_FOR_VALIDATION",
    }
    assert report["taxonomy_runtime"]["status"] == "PASS_RESTART_STABLE"
    candidate = report["candidates"][0]
    if not candidate["passes_alpha_gate"]:
        assert candidate["lookahead_status"] == "NOT_RUN_ALPHA_FAILED"
    assert report["paper_allowed"] is False
    assert report["live_allowed"] is False
    assert report["orders_enabled"] is False
    assert report["promotion_authority"] is False
