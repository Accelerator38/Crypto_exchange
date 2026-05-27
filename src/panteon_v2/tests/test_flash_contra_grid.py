import json

from panteon_v2.analysis.flash_contra_grid import (
    build_flash_contra_grid,
    write_flash_contra_grid,
)


def _intersection_row(
    genetics_signal_key: str,
    *,
    bar: int = 42,
    regime: str = "neutral",
    relationship: str = "same_action",
    selected_pnl_usd: float = -5.0,
    selected_closed: int = 1,
    genetics_lcb_pct: float = -0.25,
    genetics_closed: int = 12,
    symbol: str = "ADA/USDT",
    action: str = "SPOT_BUY_FULL",
) -> dict:
    return {
        "bar": bar,
        "regime": regime,
        "symbol": symbol,
        "selected_signal_key": f"agent:FlashTeacher|{symbol}|{action}",
        "selected_actor_key": "agent:FlashTeacher",
        "selected_actor_label": "FlashTeacher",
        "selected_actor_type": "agent",
        "selected_action": action,
        "selected_side": "long",
        "selected_score": 1.5,
        "selected_realized_pnl_usd": selected_pnl_usd,
        "selected_closed_trades": selected_closed,
        "relationship": relationship,
        "genetics_signal_key": genetics_signal_key,
        "genetics_actor_key": genetics_signal_key.split("|", 1)[0],
        "genetics_actor_label": "GeneticsBest",
        "genetics_action": action,
        "genetics_side": "long",
        "genetics_full_closed_trades": genetics_closed,
        "genetics_full_pnl_pct": -2.0,
        "genetics_full_pnl_per_trade_lcb_pct": genetics_lcb_pct,
        "genetics_latest_closed_trades": genetics_closed,
        "genetics_latest_pnl_pct": -1.0,
        "genetics_latest_pnl_per_trade_lcb_pct": genetics_lcb_pct,
        "genetics_max_drawdown_pct": 1.2,
        "genetics_recent_downside_usd": -4.0,
    }


def test_contra_grid_proposes_negative_same_action_selected_loser():
    key = "genetics:Overall|ADA/USDT|SPOT_BUY_FULL"

    report = build_flash_contra_grid(
        intersection_rows=[
            _intersection_row(key, selected_pnl_usd=-7.5, genetics_lcb_pct=-0.3)
        ],
        min_genetics_closed=10,
    )

    assert report["summary"]["contra_candidates"] == 1
    candidate = report["contra_candidates"][0]
    assert candidate["contra_signal_key"] == key
    assert candidate["mode"] == "static_no_backfill"
    assert candidate["requires_validation"] is True
    assert candidate["selected_loss_usd"] == -7.5
    assert candidate["genetics_min_lcb_pct"] == -0.3
    assert "--enable-flash-genetics-confirmation-contra-no-backfill" in candidate[
        "recommended_flags"
    ]


def test_contra_grid_ignores_selected_winner_even_with_negative_genetics_lcb():
    key = "genetics:Overall|ADA/USDT|SPOT_BUY_FULL"

    report = build_flash_contra_grid(
        intersection_rows=[
            _intersection_row(key, selected_pnl_usd=3.0, genetics_lcb_pct=-0.3)
        ],
        min_genetics_closed=10,
    )

    assert report["summary"]["contra_candidates"] == 0
    assert report["contra_candidates"] == []


def test_contra_grid_ignores_sparse_or_nonnegative_genetics_lcb():
    sparse_key = "genetics:Sparse|ADA/USDT|SPOT_BUY_FULL"
    positive_key = "genetics:Positive|ADA/USDT|SPOT_BUY_FULL"

    report = build_flash_contra_grid(
        intersection_rows=[
            _intersection_row(sparse_key, genetics_lcb_pct=-0.3, genetics_closed=3),
            _intersection_row(positive_key, genetics_lcb_pct=0.05, genetics_closed=20),
        ],
        min_genetics_closed=10,
    )

    assert report["summary"]["contra_candidates"] == 0
    assert report["summary"]["rejected_sparse_genetics"] == 1
    assert report["summary"]["rejected_nonnegative_genetics_lcb"] == 1


def test_contra_grid_rejects_candidate_without_stable_period_support():
    key = "genetics:Unstable|ADA/USDT|SPOT_BUY_FULL"

    report = build_flash_contra_grid(
        intersection_rows=[
            _intersection_row(key, bar=10, selected_pnl_usd=-2.0),
            _intersection_row(key, bar=20, selected_pnl_usd=-3.0),
        ],
        min_genetics_closed=10,
        support_period_bars=100,
        min_support_periods=2,
    )

    assert report["summary"]["contra_candidates"] == 0
    assert report["summary"]["rejected_unstable_support_periods"] == 1
    assert report["contra_candidates"] == []


def test_contra_grid_accepts_candidate_with_stable_period_and_regime_support():
    key = "genetics:Stable|ADA/USDT|SPOT_BUY_FULL"

    report = build_flash_contra_grid(
        intersection_rows=[
            _intersection_row(key, bar=10, regime="bearish", selected_pnl_usd=-2.0),
            _intersection_row(key, bar=120, regime="neutral", selected_pnl_usd=-3.0),
        ],
        min_genetics_closed=10,
        support_period_bars=100,
        min_support_periods=2,
        min_regimes=2,
    )

    assert report["summary"]["contra_candidates"] == 1
    candidate = report["contra_candidates"][0]
    assert candidate["support_period_count"] == 2
    assert candidate["support_periods"] == [0, 1]
    assert candidate["regimes"] == ["bearish", "neutral"]


def test_write_contra_grid_outputs_json_and_markdown(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "Results" / "neiro_genetics" / "FlashContraGrid"
    run_dir.mkdir()
    key = "genetics:Overall|ADA/USDT|SPOT_BUY_FULL"
    (run_dir / "flash_genetics_intersection_report.json").write_text(
        json.dumps({"rows": [_intersection_row(key)]}),
        encoding="utf-8",
    )

    json_path, md_path = write_flash_contra_grid(run_dir, output_dir=output_dir)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_path.parent == output_dir
    assert md_path.parent == output_dir
    assert data["summary"]["contra_candidates"] == 1
    assert "requires validation" in md_path.read_text(encoding="utf-8")
