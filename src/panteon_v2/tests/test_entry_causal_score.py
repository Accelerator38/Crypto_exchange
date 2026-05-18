from __future__ import annotations

from panteon_v2.selection.entry_causal_score import EntryCausalScoreState


def test_entry_causal_score_discounts_sparse_actionability() -> None:
    state = EntryCausalScoreState(window_bars=10, min_closed_trades=1, min_filled=1)
    for bar in range(1, 6):
        state.update(
            bar=bar,
            label="Sparse",
            regime="bullish",
            pnl_usd=100.0 if bar == 5 else 0.0,
            closed_trades=2 if bar == 5 else 0,
            signals=1 if bar == 5 else 0,
            filled=1 if bar == 5 else 0,
        )
        state.update(
            bar=bar,
            label="Dense",
            regime="bullish",
            pnl_usd=6.0,
            closed_trades=1,
            signals=1,
            filled=1,
        )

    sparse = state.score_with_stats(label="Sparse", regime="bullish", current_bar=6)
    dense = state.score_with_stats(label="Dense", regime="bullish", current_bar=6)

    assert sparse.recent_pnl_usd == 100.0
    assert dense.recent_pnl_usd == 30.0
    assert sparse.actionable_share < dense.actionable_share
    assert dense.score > sparse.score


def test_entry_causal_score_requires_lifetime_closed_trades() -> None:
    state = EntryCausalScoreState(window_bars=5, min_closed_trades=3, min_filled=1)
    state.update(
        bar=1,
        label="Candidate",
        regime="bearish",
        pnl_usd=50.0,
        closed_trades=1,
        signals=1,
        filled=1,
    )

    stats = state.score_with_stats(label="Candidate", regime="bearish", current_bar=2)

    assert stats.has_data is False
    assert stats.score == 0.0
