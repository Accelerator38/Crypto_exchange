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


def test_soft_shadow_score_can_scope_by_symbol():
    state = SoftShadowScoreState(window_bars=24, min_closed_trades=1)
    state.update(
        bar=10,
        label="Alpha",
        regime="bullish",
        symbol="BTC",
        pnl_usd=2.0,
        closed_trades=1,
    )
    state.update(
        bar=10,
        label="Alpha",
        regime="bullish",
        symbol="ETH",
        pnl_usd=-1.0,
        closed_trades=1,
    )

    assert state.score_with_trade_count(
        label="Alpha",
        regime="bullish",
        symbol="BTC",
        current_bar=11,
    ) == (2.0, 1)
    assert state.score_with_trade_count(
        label="Alpha",
        regime="bullish",
        symbol="ETH",
        current_bar=11,
    ) == (-1.0, 1)
    assert state.score_with_trade_count(
        label="Alpha",
        regime="bullish",
        symbol="SOL",
        current_bar=11,
    ) == (0.0, 0)


def test_soft_shadow_score_can_scope_by_symbol_and_action():
    state = SoftShadowScoreState(window_bars=24, min_closed_trades=1)
    state.update(
        bar=10,
        label="Alpha",
        regime="bullish",
        symbol="BTC",
        action="long",
        pnl_usd=2.0,
        closed_trades=1,
    )
    state.update(
        bar=10,
        label="Alpha",
        regime="bullish",
        symbol="BTC",
        action="short",
        pnl_usd=-1.0,
        closed_trades=1,
    )

    assert state.score_with_trade_count(
        label="Alpha",
        regime="bullish",
        symbol="BTC",
        action="long",
        current_bar=11,
    ) == (2.0, 1)
    assert state.score_with_trade_count(
        label="Alpha",
        regime="bullish",
        symbol="BTC",
        action="short",
        current_bar=11,
    ) == (-1.0, 1)


def test_soft_shadow_stats_include_win_rate_and_recent_downside():
    state = SoftShadowScoreState(window_bars=24, min_closed_trades=1)
    state.update(
        bar=10,
        label="Alpha",
        regime="bullish",
        symbol="BTC",
        action="long",
        pnl_usd=5.0,
        closed_trades=2,
        winning_trades=2,
    )
    state.update(
        bar=11,
        label="Alpha",
        regime="bullish",
        symbol="BTC",
        action="long",
        pnl_usd=-3.0,
        closed_trades=2,
        winning_trades=0,
    )

    stats = state.stats(
        label="Alpha",
        regime="bullish",
        symbol="BTC",
        action="long",
        current_bar=12,
    )

    assert stats.score == 2.0
    assert stats.closed_trades == 4
    assert stats.winning_trades == 2
    assert stats.losing_trades == 2
    assert stats.win_rate_pct == 50.0
    assert stats.recent_downside_usd == 3.0


def test_soft_shadow_stats_include_pnl_per_trade_lcb_inputs():
    state = SoftShadowScoreState(window_bars=24, min_closed_trades=1)
    state.update(bar=10, label="Alpha", regime="bullish", pnl_usd=2.0, closed_trades=1)
    state.update(bar=11, label="Alpha", regime="bullish", pnl_usd=4.0, closed_trades=1)
    state.update(bar=12, label="Alpha", regime="bullish", pnl_usd=6.0, closed_trades=1)

    stats = state.stats(label="Alpha", regime="bullish", current_bar=13)
    payload = stats.as_confirmation_payload()

    assert stats.pnl_per_trade_mean_usd == 4.0
    assert stats.pnl_per_trade_std_usd == 2.0
    assert payload["pnl_per_trade_mean_usd"] == 4.0
    assert payload["pnl_per_trade_std_usd"] == 2.0
    assert payload["pnl_per_trade_lcb_usd"] == 4.0 - 2.0 / (3 ** 0.5)
    # 3 события по 1 сделке → samples == closed → когерентный LCB совпадает с legacy.
    assert stats.pnl_per_trade_samples == 3
    assert payload["pnl_per_trade_lcb_event_usd"] == 4.0 - 2.0 / (3 ** 0.5)


def test_coherent_event_lcb_differs_when_events_have_many_trades():
    # Phase 1 / B3: высокочастотный актор — мало событий, но много сделок в каждом.
    # Legacy-LCB делит на √closed (переоценка уверенности), event-LCB — на √samples.
    state = SoftShadowScoreState(window_bars=100, min_closed_trades=1)
    state.update(bar=10, label="Busy", regime="bullish", pnl_usd=20.0, closed_trades=10)
    state.update(bar=11, label="Busy", regime="bullish", pnl_usd=-10.0, closed_trades=10)

    stats = state.stats(label="Busy", regime="bullish", current_bar=20)
    # 2 события → samples == 2, closed == 20.
    assert stats.closed_trades == 20
    assert stats.pnl_per_trade_samples == 2
    # event-LCB строго НИЖЕ (консервативнее) legacy-LCB, т.к. √2 < √20.
    assert stats.pnl_per_trade_lcb_event_usd < stats.pnl_per_trade_lcb_usd
    # Сходимость: при равном знаменателе значения совпали бы.
    expected_event = stats.pnl_per_trade_mean_usd - stats.pnl_per_trade_std_usd / (2 ** 0.5)
    assert abs(stats.pnl_per_trade_lcb_event_usd - expected_event) < 1e-9
