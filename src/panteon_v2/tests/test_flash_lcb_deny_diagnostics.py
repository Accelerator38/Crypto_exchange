from panteon_v2.analysis.flash_lcb_deny_diagnostics import (
    build_flash_lcb_diagnostics,
)


def _shadow(
    signal_key,
    *,
    lcb,
    closed=12,
    pnl_usd=-10.0,
    win_rate=40.0,
):
    actor_key, symbol, action = signal_key.split("|")
    actor_type, actor_label = actor_key.split(":", 1)
    return {
        "signal_key": signal_key,
        "actor_key": actor_key,
        "actor_label": actor_label,
        "actor_type": actor_type,
        "symbol": symbol,
        "action": action,
        "full_pnl_usd": pnl_usd,
        "full_closed_trades": closed,
        "full_pnl_per_trade_lcb_pct": lcb,
        "win_rate_pct": win_rate,
    }


def _selected(signal_key, *, pnl_usd, selected=1, closed=1):
    actor_key, symbol, action = signal_key.split("|")
    actor_type, actor_label = actor_key.split(":", 1)
    return {
        "actor_key": actor_key,
        "actor_label": actor_label,
        "actor_type": actor_type,
        "symbol": symbol,
        "action": action,
        "selected_signals": selected,
        "closed_trades": closed,
        "realized_pnl_usd": pnl_usd,
    }


def test_lcb_diagnostics_keeps_sparse_alpha_out_of_deny_candidates():
    signal_key = "agent:Trend|APT/USDT|FUT_SHORT_FULL"

    report = build_flash_lcb_diagnostics(
        shadow_rows=[
            _shadow(signal_key, lcb=-0.2, closed=19, pnl_usd=-16.0),
        ],
        attribution_rows=[
            _selected(signal_key, pnl_usd=12.0),
        ],
        min_shadow_closed=10,
    )

    assert report["summary"]["sparse_alpha_selected_winners"] == 1
    assert report["summary"]["deny_candidates"] == 0
    assert report["sparse_alpha_selected_winners"][0]["signal_key"] == signal_key


def test_lcb_diagnostics_proposes_reliable_negative_selected_loser():
    signal_key = "agent:Scalper|APT/USDT|FUT_SHORT_FULL"

    report = build_flash_lcb_diagnostics(
        shadow_rows=[
            _shadow(signal_key, lcb=-0.15, closed=22, pnl_usd=-21.0),
        ],
        attribution_rows=[
            _selected(signal_key, pnl_usd=-2.5),
        ],
        min_shadow_closed=10,
    )

    assert report["summary"]["reliable_negative_selected_losers"] == 1
    assert report["summary"]["deny_candidates"] == 1
    assert report["deny_candidates"][0]["signal_key"] == signal_key
    assert report["deny_candidates"][0]["requires_retro_validation"] is True


def test_lcb_diagnostics_treats_missing_shadow_for_winner_as_sparse():
    signal_key = "ensemble:Solo_A|ICP/USDT|FUT_SHORT_FULL"

    report = build_flash_lcb_diagnostics(
        shadow_rows=[],
        attribution_rows=[
            _selected(signal_key, pnl_usd=5.0),
        ],
        min_shadow_closed=10,
    )

    assert report["summary"]["selected_rows_without_shadow"] == 1
    assert report["summary"]["sparse_alpha_selected_winners"] == 1
    assert report["sparse_alpha_selected_winners"][0]["shadow_status"] == "missing"
