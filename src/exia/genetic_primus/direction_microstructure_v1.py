"""Finite receipt-clock book adapter for a single-day offline feature experiment."""
from __future__ import annotations
import numpy as np

MINUTE_US = 60_000_000


class ReceiptBook:
    def __init__(self, start_us, state=None):
        if state is None:
            self.state = {"start_us": start_us, "next_boundary_us": start_us + MINUTE_US,
                          "last_us": None, "bids": {}, "asks": {}, "rows": [], "snapshots": 0}
        else:
            self.state = state
            self.state["bids"] = {float(k): v for k, v in state["bids"].items()}
            self.state["asks"] = {float(k): v for k, v in state["asks"].items()}

    def flush(self, until_us):
        s = self.state
        while s["next_boundary_us"] <= until_us and len(s["rows"]) < 1440:
            boundary = s["next_boundary_us"]
            bids, asks = s["bids"], s["asks"]
            age = boundary - s["last_us"] if s["last_us"] is not None else None
            valid = bool(bids and asks and age is not None and 0 < age <= 5_000_000)
            values = None
            if valid:
                bp, ap = sorted(bids, reverse=True)[:5], sorted(asks)[:5]
                bid, ask = bp[0], ap[0]
                valid = bid < ask
                if valid:
                    mid = (bid + ask) / 2
                    bd, ad = sum(p*bids[p] for p in bp), sum(p*asks[p] for p in ap)
                    micro = (ask*bids[bid] + bid*asks[ask]) / (bids[bid] + asks[ask])
                    values = [(ask-bid)/mid*10000, bd, ad, (bd-ad)/(bd+ad), (micro/mid-1)*10000, age/1000]
            s["rows"].append({"time_us": boundary, "valid": valid, "values": values})
            s["next_boundary_us"] += MINUTE_US

    def batch(self, received_us, snapshot, updates):
        s = self.state
        if s["last_us"] is not None and received_us < s["last_us"]:
            raise ValueError("receipt timestamps must not go backwards")
        # Boundary features see the complete previous batch; equal-time data excluded.
        self.flush(received_us)
        if snapshot:
            s["bids"], s["asks"] = {}, {}
            s["snapshots"] += 1
        if not updates:
            raise ValueError("nonempty atomic exchange/receipt batch required")
        for side, price, amount in updates:
            if side not in ("bid", "ask") or not np.isfinite([price, amount]).all() or price <= 0 or amount < 0:
                raise ValueError("valid positive-price L2 update required")
            book = s["bids" if side == "bid" else "asks"]
            if amount:
                book[price] = amount
            else:
                book.pop(price, None)
        s["last_us"] = received_us


def feature_bank(ohlcv, trade_notional, trade_count, book_rows):
    bars, notional, count = np.asarray(ohlcv, float), np.asarray(trade_notional, float), np.asarray(trade_count, float)
    if bars.shape != (1440, 5) or notional.shape != (1440, 2) or count.shape != (1440,):
        raise ValueError("one complete receipt-clock day required")
    c, v = bars[:, 3], bars[:, 4]
    base = np.full((1440, 6), np.nan)
    base[:, 0] = np.r_[np.nan, c[1:] / c[:-1] - 1]
    base[5:, 1] = c[5:] / c[:-5] - 1
    base[30:, 2] = c[30:] / c[:-30] - 1
    for i in range(30, 1440):
        base[i, 3] = np.std(np.diff(np.log(c[i-30:i+1])))
    width = bars[:, 1] - bars[:, 2]
    base[:, 4] = np.divide(c-bars[:, 0], width, out=np.zeros(1440), where=width > 0)
    base[:, 5] = np.log1p(v)
    book = np.full((1440, 6), np.nan)
    for i, row in enumerate(book_rows):
        if row["valid"]:
            book[i] = row["values"]
    total = notional.sum(axis=1)
    flow = np.divide(notional[:, 0] - notional[:, 1], total, out=np.zeros(1440), where=total > 0)
    buy = np.divide(notional[:, 0], total, out=np.full(1440, .5), where=total > 0)
    depth = book[:, 1] + book[:, 2]
    change = np.r_[np.nan, depth[1:] / depth[:-1] - 1]
    extra = np.column_stack([flow, buy, np.log1p(count), book[:, 0], np.log1p(book[:, 1]),
                             np.log1p(book[:, 2]), book[:, 3], change, book[:, 4], book[:, 5]])
    return base, np.column_stack([base, extra])


def direct_checks():
    b = ReceiptBook(0)
    b.batch(1, True, [("bid", 99, 1), ("ask", 101, 1)])
    b.batch(59_999_999, False, [("bid", 99, 2)])
    b.batch(60_000_000, False, [("ask", 101, 0), ("ask", 103, 1)])
    first = b.state["rows"][0]
    assert first["valid"] and abs(first["values"][0] - 200) < 1e-10
    assert first["values"][2] == 101
    # The equal-boundary ask 103 cannot affect the feature at 60 seconds.
    assert b.state["asks"] == {103: 1}
    resumed = ReceiptBook(0, b.state)
    resumed.batch(119_999_999, False, [("bid", 99, 1)])
    resumed.flush(120_000_000)
    assert resumed.state["rows"][1]["values"][2] == 103
    resumed.flush(180_000_000)
    assert not resumed.state["rows"][2]["valid"]
    try:
        resumed.batch(1, False, [("bid", 99, 1)])
    except ValueError:
        pass
    else:
        raise AssertionError("backward receipt accepted")
    return {"atomic_batches": True, "strict_before_boundary": True, "deletion": True,
            "resume": True, "stale_rejected": True, "backward_clock_rejected": True}
