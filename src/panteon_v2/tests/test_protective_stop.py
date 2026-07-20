from __future__ import annotations

import pytest

from panteon_v2.policy.protective_stop import (
    ProtectiveStopError,
    intrabar_stop_fill_reference,
    protective_stop_breached,
    protective_stop_price,
)


def test_short_stop_uses_single_manifest_percentage():
    stop = protective_stop_price(100.0, "short", 1.2)

    assert stop == pytest.approx(101.2)
    assert protective_stop_breached("short", 101.19, stop) is False
    assert protective_stop_breached("short", 101.2, stop) is True


def test_intrabar_short_gap_uses_worse_open_reference():
    fill = intrabar_stop_fill_reference(
        "short",
        101.2,
        open_price=102.5,
        high=103.0,
        low=101.8,
        close=102.0,
    )

    assert fill == pytest.approx(102.5)


def test_intrabar_long_breach_uses_stop_when_bar_does_not_gap():
    fill = intrabar_stop_fill_reference(
        "long",
        98.8,
        open_price=100.0,
        high=100.5,
        low=98.0,
        close=99.0,
    )

    assert fill == pytest.approx(98.8)


def test_inconsistent_ohlc_is_rejected():
    with pytest.raises(ProtectiveStopError, match="OHLC is inconsistent"):
        intrabar_stop_fill_reference(
            "short",
            101.2,
            open_price=100.0,
            high=99.0,
            low=98.0,
            close=100.0,
        )
