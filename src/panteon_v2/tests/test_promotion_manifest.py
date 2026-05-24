import pytest

from panteon_v2.selection.promotion_manifest import (
    PromotionManifestConfig,
    build_promotion_manifest,
)


def _valid_row(signal_key="agent:MomentumScalper|BTC/USDT|FUT_SHORT_FULL"):
    return {
        "signal_key": signal_key,
        "full_pnl_pct": 8.0,
        "latest_pnl_pct": 1.5,
        "full_closed_trades": 80,
        "latest_closed_trades": 12,
        "max_drawdown_pct": 6.0,
        "win_rate_pct": 70.0,
        "full_pnl_per_trade_lcb_pct": 0.05,
        "latest_pnl_per_trade_lcb_pct": 0.01,
        "recent_downside_usd": 1.0,
    }


def test_promotion_manifest_allows_only_full_and_latest_positive_keys():
    rows = [
        {
            "signal_key": "agent:MomentumScalper|BTC/USDT|FUT_SHORT_FULL",
            "full_pnl_pct": 8.0,
            "latest_pnl_pct": 1.5,
            "full_closed_trades": 80,
            "latest_closed_trades": 12,
            "max_drawdown_pct": 6.0,
            "win_rate_pct": 70.0,
            "full_pnl_per_trade_lcb_pct": 0.05,
            "latest_pnl_per_trade_lcb_pct": 0.01,
            "recent_downside_usd": 1.0,
        },
        {
            "signal_key": "agent:MomentumScalper|ATOM/USDT|SPOT_BUY_FULL",
            "full_pnl_pct": 5.0,
            "latest_pnl_pct": -0.5,
            "full_closed_trades": 90,
            "latest_closed_trades": 20,
            "max_drawdown_pct": 7.0,
            "win_rate_pct": 70.0,
            "full_pnl_per_trade_lcb_pct": 0.04,
            "latest_pnl_per_trade_lcb_pct": 0.01,
            "recent_downside_usd": 3.0,
        },
    ]

    manifest = build_promotion_manifest(rows, PromotionManifestConfig())

    assert manifest.allowed_signal_keys == (
        "agent:MomentumScalper|BTC/USDT|FUT_SHORT_FULL",
    )
    rejected = {row.signal_key: row.reason for row in manifest.rejected}
    assert rejected["agent:MomentumScalper|ATOM/USDT|SPOT_BUY_FULL"] == "latest_pnl_below_gate"


def test_promotion_manifest_returns_sorted_unique_allowed_keys_and_skips_empty_keys():
    rows = [
        _valid_row("agent:MomentumScalper|SOL/USDT|SPOT_BUY_FULL"),
        _valid_row("agent:MomentumScalper|BTC/USDT|FUT_SHORT_FULL"),
        _valid_row("agent:MomentumScalper|BTC/USDT|FUT_SHORT_FULL"),
        {**_valid_row(""), "signal_key": ""},
        {**_valid_row("   "), "signal_key": "   "},
        {**_valid_row(), "signal_key": None},
    ]

    manifest = build_promotion_manifest(rows)

    assert manifest.allowed_signal_keys == (
        "agent:MomentumScalper|BTC/USDT|FUT_SHORT_FULL",
        "agent:MomentumScalper|SOL/USDT|SPOT_BUY_FULL",
    )
    assert manifest.rejected == ()


@pytest.mark.parametrize(
    ("field", "bad_value", "expected_reason"),
    [
        ("full_closed_trades", 49, "full_closed_trades_below_gate"),
        ("latest_closed_trades", 9, "latest_closed_trades_below_gate"),
        ("full_pnl_pct", -0.1, "full_pnl_below_gate"),
        ("latest_pnl_pct", -0.1, "latest_pnl_below_gate"),
        ("max_drawdown_pct", 25.1, "drawdown_above_gate"),
        ("win_rate_pct", 51.9, "win_rate_below_gate"),
        ("recent_downside_usd", 5.1, "recent_downside_above_gate"),
    ],
)
def test_promotion_manifest_reports_rejection_reasons_in_gate_order(
    field, bad_value, expected_reason
):
    row = _valid_row()
    row[field] = bad_value

    manifest = build_promotion_manifest([row])

    assert manifest.allowed_signal_keys == ()
    assert manifest.rejected[0].reason == expected_reason


def test_promotion_manifest_uses_bayesian_win_rate_lcb_for_borderline_samples():
    row = {
        **_valid_row("agent:Borderline|BTC/USDT|FUT_LONG_FULL"),
        "full_closed_trades": 50,
        "latest_closed_trades": 10,
        "win_rate_pct": 52.0,
    }

    manifest = build_promotion_manifest(
        [row],
        PromotionManifestConfig(
            min_win_rate_pct=52.0,
            min_win_rate_lcb_pct=50.0,
        ),
    )

    assert manifest.allowed_signal_keys == ()
    assert manifest.rejected[0].reason == "win_rate_lcb_below_gate"


def test_promotion_manifest_uses_pnl_per_trade_lcb_gate():
    good = _valid_row("agent:Good|BTC/USDT|FUT_LONG_FULL")
    weak_full = {
        **_valid_row("agent:WeakFull|BTC/USDT|FUT_LONG_FULL"),
        "full_pnl_per_trade_lcb_pct": -0.001,
    }
    weak_latest = {
        **_valid_row("agent:WeakLatest|BTC/USDT|FUT_LONG_FULL"),
        "latest_pnl_per_trade_lcb_pct": -0.001,
    }

    manifest = build_promotion_manifest(
        [good, weak_full, weak_latest],
        PromotionManifestConfig(
            min_full_pnl_per_trade_lcb_pct=0.001,
            min_latest_pnl_per_trade_lcb_pct=0.001,
        ),
    )

    rejected = {row.signal_key: row.reason for row in manifest.rejected}
    assert manifest.allowed_signal_keys == ("agent:Good|BTC/USDT|FUT_LONG_FULL",)
    assert rejected["agent:WeakFull|BTC/USDT|FUT_LONG_FULL"] == (
        "full_pnl_per_trade_lcb_below_gate"
    )
    assert rejected["agent:WeakLatest|BTC/USDT|FUT_LONG_FULL"] == (
        "latest_pnl_per_trade_lcb_below_gate"
    )


def test_promotion_manifest_does_not_require_pnl_lcb_fields_when_gates_disabled():
    row = _valid_row("agent:OldCaller|BTC/USDT|FUT_LONG_FULL")
    row.pop("full_pnl_per_trade_lcb_pct")
    row.pop("latest_pnl_per_trade_lcb_pct")

    manifest = build_promotion_manifest([row], PromotionManifestConfig())

    assert manifest.allowed_signal_keys == ("agent:OldCaller|BTC/USDT|FUT_LONG_FULL",)
    assert manifest.rejected == ()


@pytest.mark.parametrize(
    ("field", "expected_reason"),
    [
        ("full_pnl_per_trade_lcb_pct", "full_pnl_per_trade_lcb_below_gate"),
        ("latest_pnl_per_trade_lcb_pct", "latest_pnl_per_trade_lcb_below_gate"),
    ],
)
def test_promotion_manifest_requires_pnl_lcb_fields_when_gates_enabled(
    field, expected_reason
):
    row = _valid_row()
    row.pop(field)

    manifest = build_promotion_manifest(
        [row],
        PromotionManifestConfig(
            min_full_pnl_per_trade_lcb_pct=0.001,
            min_latest_pnl_per_trade_lcb_pct=0.001,
        ),
    )

    assert manifest.allowed_signal_keys == ()
    assert manifest.rejected[0].reason == expected_reason


def test_promotion_manifest_as_dict_returns_json_serializable_lists():
    manifest = build_promotion_manifest(
        [
            _valid_row("agent:MomentumScalper|BTC/USDT|FUT_SHORT_FULL"),
            {**_valid_row("agent:MomentumScalper|ATOM/USDT|SPOT_BUY_FULL"), "latest_pnl_pct": -1.0},
        ]
    )

    assert manifest.as_dict() == {
        "allowed_signal_keys": [
            "agent:MomentumScalper|BTC/USDT|FUT_SHORT_FULL",
        ],
        "rejected": [
            {
                "signal_key": "agent:MomentumScalper|ATOM/USDT|SPOT_BUY_FULL",
                "reason": "latest_pnl_below_gate",
            }
        ],
    }


def test_promotion_manifest_malformed_numeric_values_reject_conservatively():
    manifest = build_promotion_manifest(
        [
            {
                **_valid_row("agent:MomentumScalper|ETH/USDT|FUT_LONG_FULL"),
                "full_closed_trades": "not-a-number",
            }
        ]
    )

    assert manifest.allowed_signal_keys == ()
    assert manifest.rejected[0].reason == "full_closed_trades_below_gate"


@pytest.mark.parametrize(
    ("field", "bad_value", "expected_reason"),
    [
        ("full_pnl_pct", "not-a-number", "full_pnl_below_gate"),
        ("full_pnl_pct", "nan", "full_pnl_below_gate"),
        ("latest_pnl_pct", "not-a-number", "latest_pnl_below_gate"),
        ("latest_pnl_pct", "nan", "latest_pnl_below_gate"),
        ("max_drawdown_pct", "not-a-number", "drawdown_above_gate"),
        ("max_drawdown_pct", "nan", "drawdown_above_gate"),
        ("win_rate_pct", "not-a-number", "win_rate_below_gate"),
        ("win_rate_pct", "nan", "win_rate_below_gate"),
        ("recent_downside_usd", "not-a-number", "recent_downside_above_gate"),
        ("recent_downside_usd", "nan", "recent_downside_above_gate"),
    ],
)
def test_promotion_manifest_malformed_float_values_reject_by_field_gate(
    field, bad_value, expected_reason
):
    row = _valid_row()
    row[field] = bad_value

    manifest = build_promotion_manifest([row])

    assert manifest.allowed_signal_keys == ()
    assert manifest.rejected[0].reason == expected_reason


@pytest.mark.parametrize(
    ("field", "bad_value", "expected_reason"),
    [
        ("full_pnl_per_trade_lcb_pct", "not-a-number", "full_pnl_per_trade_lcb_below_gate"),
        ("full_pnl_per_trade_lcb_pct", "nan", "full_pnl_per_trade_lcb_below_gate"),
        ("latest_pnl_per_trade_lcb_pct", "not-a-number", "latest_pnl_per_trade_lcb_below_gate"),
        ("latest_pnl_per_trade_lcb_pct", "nan", "latest_pnl_per_trade_lcb_below_gate"),
    ],
)
def test_promotion_manifest_malformed_pnl_lcb_values_reject_when_gates_enabled(
    field, bad_value, expected_reason
):
    row = _valid_row()
    row[field] = bad_value

    manifest = build_promotion_manifest(
        [row],
        PromotionManifestConfig(
            min_full_pnl_per_trade_lcb_pct=0.001,
            min_latest_pnl_per_trade_lcb_pct=0.001,
        ),
    )

    assert manifest.allowed_signal_keys == ()
    assert manifest.rejected[0].reason == expected_reason


def test_promotion_manifest_sorts_duplicate_rejected_keys_by_reason():
    signal_key = "agent:MomentumScalper|ETH/USDT|FUT_LONG_FULL"
    rows = [
        {**_valid_row(signal_key), "win_rate_pct": 51.0},
        {**_valid_row(signal_key), "latest_pnl_pct": -1.0},
        {**_valid_row(signal_key), "full_pnl_pct": -1.0},
    ]

    manifest = build_promotion_manifest(rows)

    assert [(row.signal_key, row.reason) for row in manifest.rejected] == [
        (signal_key, "full_pnl_below_gate"),
        (signal_key, "latest_pnl_below_gate"),
        (signal_key, "win_rate_below_gate"),
    ]
