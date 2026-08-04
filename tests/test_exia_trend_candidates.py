from __future__ import annotations

import json
from pathlib import Path

from exia.contracts import load_candidate_spec
from tools.evaluate_exia_trend_family_v1 import _gate_checks


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = ROOT / "freqtrade_pilot" / "candidates"
STRATEGY_PATH = (
    ROOT
    / "freqtrade_pilot"
    / "user_data"
    / "strategies"
    / "ExiaTrendCandidatesV1.py"
)
CANDIDATE_IDS = (
    "exia_trend_state_onset_v1",
    "exia_trend_donchian48_v1",
    "exia_trend_pullback_reclaim_v1",
)


def test_trend_candidate_specs_pin_one_shared_implementation() -> None:
    class_names = set()
    for candidate_id in CANDIDATE_IDS:
        spec = load_candidate_spec(CANDIDATE_DIR / f"{candidate_id}.json", root=ROOT)
        assert spec.status == "DEVELOPMENT"
        assert spec.family == "trend_following"
        assert spec.trial_family == "exia_trend_v1"
        assert spec.allowed_market_states == ("TREND_UP", "TREND_DOWN")
        assert spec.baseline_id == "no_trade_cash_v1"
        assert spec.risk["max_open_trades"] == 1
        assert spec.risk["leverage"] == 1
        assert spec.source["paper_allowed"] is False
        assert spec.source["live_allowed"] is False
        assert spec.source["orders_enabled"] is False
        assert spec.source["promotion_authority"] is False
        class_names.add(spec.strategy["class_name"])

    assert len(class_names) == len(CANDIDATE_IDS)


def test_trend_family_is_preregistered_and_bounded() -> None:
    registry = json.loads(
        (CANDIDATE_DIR / "exia_trend_family_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert registry["schema_version"] == "exia.candidate_family.v1"
    assert registry["candidate_ids"] == list(CANDIDATE_IDS)
    assert registry["maximum_variants"] == len(CANDIDATE_IDS) == 3
    assert registry["selection_window"] == "development"
    assert registry["selection_cost"] == "stress"
    assert registry["maximum_candidates_opened_on_validation"] == 1
    assert registry["selection_gate"] == {
        "min_closed_trades": 20,
        "min_mean_net_bps": 0,
        "min_block_lcb_net_bps": 0,
        "max_drawdown": 0.2,
    }
    assert registry["paper_allowed"] is False
    assert registry["live_allowed"] is False
    assert registry["orders_enabled"] is False
    assert registry["promotion_authority"] is False


def test_trend_strategy_is_dry_run_only_with_fixed_risk() -> None:
    source = STRATEGY_PATH.read_text(encoding="utf-8")

    for class_name in (
        "ExiaTrendStateOnsetV1",
        "ExiaTrendDonchian48V1",
        "ExiaTrendPullbackReclaimV1",
    ):
        assert f"class {class_name}(ExiaTrendBaseV1)" in source
    assert 'self.config.get("dry_run") is not True' in source
    assert "return 1.0" in source
    assert "stoploss = -0.05" in source
    assert "max_holding_hours = 168" in source
    assert '"stoploss_on_exchange": True' in source


def test_experiment_runner_supports_development_only_scoring() -> None:
    source = (ROOT / "tools" / "run_exia_experiment_v1.py").read_text(
        encoding="utf-8"
    )

    assert '"--splits"' in source
    assert 'default="all"' in source
    assert '"DEVELOPMENT_SCORED"' in source
    assert 'if selected_splits == ("development",)' in source


def test_family_gate_requires_positive_lcb_and_baseline_differential() -> None:
    gate = {
        "min_closed_trades": 20,
        "min_mean_net_bps": 0,
        "min_block_lcb_net_bps": 0,
        "max_drawdown": 0.2,
    }
    row = {
        "closed_trades": 100,
        "mean_net_bps": 20.0,
        "block_lcb_net_bps": -1.0,
        "baseline_differential_lcb_bps": 5.0,
        "max_drawdown": 0.01,
        "status": "FAIL",
    }
    checks = _gate_checks(row, gate)

    assert checks["minimum_closed_trades"] is True
    assert checks["positive_mean_after_stress_costs"] is True
    assert checks["positive_block_lcb"] is False
    assert all(checks.values()) is False


def test_generated_trend_family_decision_is_fail_closed_when_present() -> None:
    report_path = (
        ROOT
        / "Reports"
        / "Exia"
        / "trend_family_v1"
        / "development_report.json"
    )
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["status"] in {
        "SELECTED_FOR_VALIDATION",
        "NO_CANDIDATE_FOR_VALIDATION",
    }
    assert report["paper_allowed"] is False
    assert report["live_allowed"] is False
    assert report["orders_enabled"] is False
    assert report["promotion_authority"] is False
    assert len(report["candidates"]) == 3
