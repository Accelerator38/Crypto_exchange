"""Pure market-only snapshot/as-of checks; no model, orders, I/O, or runtime."""
import hashlib
import json
import re
from decimal import Decimal


SAFETY = {name: False for name in (
    "network_allowed", "credentials_allowed", "orders_enabled", "paper_allowed",
    "live_allowed", "promotion_authority", "runtime_authority",
)}


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def number(value, positive=True):
    if not isinstance(value, str) or len(value) > 32:
        raise ValueError("expected bounded decimal string")
    if not re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", value):
        raise ValueError("invalid decimal syntax")
    parsed = Decimal(value)
    if positive and parsed <= 0:
        raise ValueError("expected positive price/quantity")
    return parsed


def timestamp(value):
    if isinstance(value, str) and value.isascii() and value.isdigit() and len(value) == 16:
        return int(value)
    if type(value) is int and 10**15 <= value < 10**16:
        return value
    raise ValueError("expected integer UTC microseconds")


def identity(row):
    if row.get("exchange") != "bitget-futures" or row.get("symbol") != "BTCUSDT":
        raise ValueError("unexpected market/instrument")
    exchange = timestamp(row.get("timestamp"))
    received = timestamp(row.get("local_timestamp"))
    if exchange > received:
        raise ValueError("exchange timestamp after receipt")
    return received, exchange


def _seal(event):
    return dict(event, event_sha256=object_sha(event))


def _snapshot(group, key, levels, source_hash):
    if len(group) != levels * 2:
        raise ValueError("incomplete or ambiguous snapshot group")
    sides = {"ask": [], "bid": []}
    seen = {"ask": set(), "bid": set()}
    for row in group:
        side = row.get("side")
        if row.get("is_snapshot") != "true" or side not in sides:
            raise ValueError("snapshot-only fragment requires explicit snapshot rows")
        price, amount = number(row.get("price")), number(row.get("amount"))
        if price in seen[side]:
            raise ValueError("duplicate price in snapshot")
        seen[side].add(price)
        sides[side].append((price, row["price"], row["amount"]))
    if any(len(rows) != levels for rows in sides.values()):
        raise ValueError("wrong side depth")
    ask = sorted(sides["ask"], key=lambda x: x[0])
    bid = sorted(sides["bid"], key=lambda x: x[0], reverse=True)
    if bid[0][0] >= ask[0][0]:
        raise ValueError("locked or crossed snapshot")
    # Each complete group is a NEW entire observed-depth state. Never merge
    # prices from the previous snapshot or invent liquidity beyond this depth.
    return _seal({"kind": "book_snapshot", "symbol": "BTCUSDT",
                  "received_ts_us": key[0], "exchange_ts_us": key[1],
                  "observed_levels_per_side": levels, "full_depth_verified": False,
                  "asks": [[x[1], x[2]] for x in ask],
                  "bids": [[x[1], x[2]] for x in bid],
                  "source_sha256": source_hash})


def book_snapshots(rows, source_hash, levels=15):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 8192:
        raise ValueError("book row budget exceeded or empty")
    if type(levels) is not int or not 1 <= levels <= 25:
        raise ValueError("invalid observed depth")
    events, group, previous = [], [], None
    for row in rows:
        key = identity(row)
        if previous is not None and key[0] < previous[0]:
            raise ValueError("receipt order regression")
        if previous is not None and key != previous:
            # Distinct snapshots with indistinguishable receipt times would
            # require source sequence evidence, which this CSV does not supply.
            if key[0] == previous[0]:
                raise ValueError("ambiguous same-receipt snapshots")
            events.append(_snapshot(group, previous, levels, source_hash))
            group = []
        group.append(row)
        previous = key
    events.append(_snapshot(group, previous, levels, source_hash))
    return events


def mark_events(rows, source_hash):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 256:
        raise ValueError("mark row budget exceeded or empty")
    events, previous = [], None
    for row in rows:
        received, exchange = identity(row)
        if previous is not None and received <= previous:
            raise ValueError("duplicate or regressing mark receipt time")
        number(row.get("mark_price"))
        number(row.get("funding_rate"), positive=False)
        funding = timestamp(row.get("funding_timestamp"))
        events.append(_seal({"kind": "mark", "symbol": "BTCUSDT",
                            "received_ts_us": received, "exchange_ts_us": exchange,
                            "mark_price": row["mark_price"],
                            "reported_funding_rate": row["funding_rate"],
                            "reported_funding_timestamp_us": funding,
                            "is_funding_settlement": False, "source_sha256": source_hash}))
        previous = received
    return events


def asof(events, cutoff_us, max_age_us):
    cutoff_us = timestamp(cutoff_us)
    if type(max_age_us) is not int or not 0 <= max_age_us <= 5000000:
        raise ValueError("invalid age budget")
    if not isinstance(events, list) or len(events) > 8192:
        raise ValueError("as-of event budget exceeded")
    candidate, previous = None, None
    for event in events:
        received = timestamp(event["received_ts_us"])
        exchange = timestamp(event["exchange_ts_us"])
        if exchange > received or (previous is not None and received <= previous):
            raise ValueError("invalid event chronology")
        body = {k: v for k, v in event.items() if k != "event_sha256"}
        if object_sha(body) != event.get("event_sha256"):
            raise ValueError("event hash mismatch")
        if received <= cutoff_us and exchange <= cutoff_us:
            candidate = event
        previous = received
    if candidate is None or any(cutoff_us - candidate[k] > max_age_us
                                for k in ("received_ts_us", "exchange_ts_us")):
        return None
    # Return a detached state; callers cannot mutate the input event history.
    return json.loads(json.dumps(candidate))


def inspect_fragment(book_rows, ticker_rows, plan):
    start, end, cutoff = (timestamp(plan[k]) for k in ("start_us", "end_exclusive_us", "cutoff_us"))
    if not start < cutoff < end or end - start > 10000000:
        raise ValueError("invalid fragment interval")
    for rows in (book_rows, ticker_rows):
        for row in rows:
            received, _ = identity(row)
            if not start <= received < end:
                raise ValueError("row outside fixed receipt-time interval")
    books = book_snapshots(book_rows, plan["sources"]["l2"]["sha256"], plan["levels_per_side"])
    marks = mark_events(ticker_rows, plan["sources"]["ticker"]["sha256"])
    book = asof(books, cutoff, plan["book_max_age_us"])
    mark = asof(marks, cutoff, plan["mark_max_age_us"])
    if book is None or mark is None:
        raise ValueError("no sufficiently fresh book/mark at the fixed cutoff")
    if not any(e["received_ts_us"] > cutoff for e in books) or not any(e["received_ts_us"] > cutoff for e in marks):
        raise ValueError("fragment must straddle cutoff for a causal check")
    # Independently verify that deleting all later arrivals does not change the view.
    for events, selected, age in ((books, book, plan["book_max_age_us"]),
                                  (marks, mark, plan["mark_max_age_us"])):
        prefix = [e for e in events if e["received_ts_us"] <= cutoff]
        if asof(prefix, cutoff, age) != selected:
            raise ValueError("future-event dependence")
    body = {"schema": "genetic_primus.market_only_fragment/1", "plan_sha256": object_sha(plan),
            "safety": dict(SAFETY), "qualification_eligible": False,
            "trading_authorized": False, "full8_admission_eligible": False,
            "funding_settlements_verified": False, "latency_calibrated": False,
            "source_authenticity_verified": False, "book_snapshots": books, "marks": marks,
            "asof": {"cutoff_us": cutoff, "book": book, "mark": mark,
                     "future_prefix_invariance": True}}
    return dict(body, fragment_sha256=object_sha(body))
