"""Tiny deterministic regressions; run directly, never through collector/pytest."""
import runpy
from pathlib import Path

M = runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/exia/genetic_primus/market_fragment_v1.py"))
T = 1733025600000000
SHA = "1" * 64


def rows(received=T, exchange=None, base=100):
    shared = {"exchange": "bitget-futures", "symbol": "BTCUSDT",
              "timestamp": str(received - 10 if exchange is None else exchange),
              "local_timestamp": str(received), "is_snapshot": "true", "amount": "1"}
    return [dict(shared, side=side, price=str(price)) for side, price in
            (("ask", base + 1), ("ask", base + 2), ("bid", base), ("bid", base - 1))]


def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected fail-closed ValueError")


def test_snapshot_replacement():
    events = M["book_snapshots"](rows() + rows(T + 100, base=200), SHA, 2)
    selected = M["asof"](events, T + 100, 1000)
    assert selected["bids"] == [["200", "1"], ["199", "1"]]
    selected["bids"][0][0] = "999"
    assert events[-1]["bids"][0][0] == "200"


def test_incomplete_group():
    reject(lambda: M["book_snapshots"](rows()[:-1], SHA, 2))


def test_duplicate_price():
    data = rows(); data[1]["price"] = data[0]["price"]
    reject(lambda: M["book_snapshots"](data, SHA, 2))


def test_crossed_book():
    data = rows(); data[2]["price"] = "103"
    reject(lambda: M["book_snapshots"](data, SHA, 2))


def test_delta_not_snapshot():
    data = rows(); data[1]["is_snapshot"] = "false"
    reject(lambda: M["book_snapshots"](data, SHA, 2))


def test_ambiguous_receipt():
    reject(lambda: M["book_snapshots"](rows() + rows(exchange=T - 5), SHA, 2))


def test_future_exchange_before_receive_rejected():
    reject(lambda: M["book_snapshots"](rows(exchange=T + 1), SHA, 2))


def test_stale_and_missing():
    events = M["book_snapshots"](rows(T - 100), SHA, 2)
    assert M["asof"](events, T - 101, 1000) is None
    assert M["asof"](events, T, 99) is None
    assert M["asof"](events, T, 100) is None  # exchange is older than receipt
    assert M["asof"](events, T, 110) is not None


def test_no_future_lookahead():
    past = M["book_snapshots"](rows(T - 100), SHA, 2)
    # A late arrival carrying an old exchange timestamp must not leak backwards.
    future = M["book_snapshots"](rows(T + 1, exchange=T - 50, base=999), SHA, 2)
    assert M["asof"](past + future, T, 1000) == M["asof"](past, T, 1000)
    assert M["asof"](past + future, T + 1, 1000) == future[0]


def test_nonfinite_and_zero():
    for value in ("NaN", "Infinity", "1e3", "0", "-1"):
        reject(lambda: M["number"](value))


def test_event_tampering():
    events = M["book_snapshots"](rows(), SHA, 2)
    events[0]["bids"][0][1] = "1000"
    reject(lambda: M["asof"](events, T, 1000))


def test_mark_is_not_settlement():
    row = dict(rows()[0], mark_price="100", funding_rate="-0.001", funding_timestamp=str(T + 1000))
    events = M["mark_events"]([row], SHA)
    assert not events[0]["is_funding_settlement"]
    assert M["asof"](events, T - 1, 1000) is None
    assert M["asof"](events, T, 1000)["mark_price"] == "100"
    reject(lambda: M["mark_events"]([row, row], SHA))
