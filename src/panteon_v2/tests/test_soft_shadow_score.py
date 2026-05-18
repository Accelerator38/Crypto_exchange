from panteon_v2.selection.soft_shadow_score import SoftShadowScoreState


def test_soft_shadow_score_uses_previous_events_only_in_same_regime():
    state = SoftShadowScoreState(window_bars=24, min_closed_trades=2)
    state.update(bar=10, label="Alpha", regime="bullish", pnl_usd=5.0, closed_trades=1)
    state.update(bar=11, label="Alpha", regime="bullish", pnl_usd=7.0, closed_trades=1)
    state.update(bar=12, label="Alpha", regime="bullish", pnl_usd=99.0, closed_trades=10)
    state.update(bar=11, label="Beta", regime="neutral", pnl_usd=99.0, closed_trades=10)

    assert state.score(label="Alpha", regime="bullish", current_bar=12) == 12.0
    assert state.score(label="Beta", regime="bullish", current_bar=12) == 0.0


def test_soft_shadow_score_uses_lifetime_trade_count_but_scores_recent_pnl():
    state = SoftShadowScoreState(window_bars=3, min_closed_trades=2)
    state.update(bar=1, label="Alpha", regime="bullish", pnl_usd=10.0, closed_trades=10)
    state.update(bar=4, label="Alpha", regime="bullish", pnl_usd=2.0, closed_trades=1)

    assert state.score(label="Alpha", regime="bullish", current_bar=5) == 2.0
    assert state.score_with_trade_count(
        label="Alpha",
        regime="bullish",
        current_bar=5,
    ) == (2.0, 11)
