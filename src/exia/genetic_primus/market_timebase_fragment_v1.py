"""Bounded, pure provider-clock diagnostics; not an execution/funding adapter."""
import re
from copy import deepcopy

from .market_fragment_v1 import identity, number, object_sha, timestamp

TRADE_HEADER = ("exchange", "symbol", "timestamp", "local_timestamp", "id",
                "side", "price", "amount")
TICKER_HEADER = ("exchange", "symbol", "timestamp", "local_timestamp",
                 "funding_timestamp", "funding_rate", "predicted_funding_rate",
                 "open_interest", "last_price", "index_price", "mark_price")
SAFETY = dict.fromkeys(("credentials_allowed", "orders_enabled", "execution_allowed",
                       "paper", "live", "promotion_authority", "training_allowed",
                       "financial_evaluation_allowed", "network_allowed"), False)


def seal(body):
    result = deepcopy(body)
    result["sha256"] = object_sha(result)
    return result


def verify(value):
    if not isinstance(value, dict):
        raise ValueError("expected sealed object")
    if object_sha({k: v for k, v in value.items() if k != "sha256"}) != value.get("sha256"):
        raise ValueError("object hash mismatch")


def _source(source_hash):
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ValueError("invalid prefix hash")


def _rows(rows, header, source_hash):
    _source(source_hash)
    if not isinstance(rows, list) or not 1 <= len(rows) <= 256:
        raise ValueError("row budget exceeded or empty")
    previous = None
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(header):
            raise ValueError("unexpected header/fields")
        received, exchange = identity(row)
        if previous is not None and received < previous:
            raise ValueError("regressing provider receipt")
        previous = received
        yield received, exchange, row


def _optional_number(value, *, nonnegative=False, positive=False):
    if value == "":
        return None
    parsed = number(value, positive=positive)
    if nonnegative and parsed < 0:
        raise ValueError("negative quantity")
    return value  # Preserve exact decimal representation; missing is never zero.


def _series(kind, events, rows, source_hash, **extra):
    # Without a subsequent receive-time delimiter the last batch may be partial.
    final_received = identity(rows[-1])[0]
    complete = [event for event in events if event["received_ts_us"] < final_received]
    return seal({"schema": "provider_clock_series/1", "kind": kind,
                 "source_prefix_sha256": source_hash, "source_row_count": len(rows),
                 "source_row_bank_sha256": object_sha(rows), "events": complete,
                 "dropped_final_events": len(events) - len(complete),
                 "final_receive_batch_complete": False,
                 "last_source_received_ts_us": final_received,
                 "strategy_availability_verified": False,
                 "financial_replay_ready": False, "safety": SAFETY, **extra})


def trade_events(rows, source_hash):
    events, seen, duplicates = [], {}, []
    previous_exchange, exchange_regressions = None, 0
    for ordinal, (received, exchange, row) in enumerate(_rows(rows, TRADE_HEADER, source_hash)):
        trade_id = row["id"]
        if not isinstance(trade_id, str) or not re.fullmatch(r"[1-9][0-9]{0,31}", trade_id):
            raise ValueError("invalid trade id")
        if row["side"] not in ("buy", "sell"):
            raise ValueError("invalid trade side")
        number(row["price"])
        number(row["amount"])
        payload_hash = object_sha({k: v for k, v in row.items() if k != "local_timestamp"})
        if trade_id in seen:
            if seen[trade_id] != payload_hash:
                raise ValueError("conflicting trade id")
            duplicates.append({"source_row_ordinal": ordinal, "trade_id": trade_id,
                               "source_row_sha256": object_sha(row)})
        else:
            seen[trade_id] = payload_hash
            events.append(seal({"kind": "trade", "symbol": "BTCUSDT",
                                "received_ts_us": received, "exchange_ts_us": exchange,
                                "trade_id": trade_id, "side": row["side"],
                                "price": row["price"], "amount": row["amount"],
                                "source_row_ordinal": ordinal,
                                "source_row_sha256": object_sha(row),
                                "source_prefix_sha256": source_hash,
                                "within_batch_order_verified": False,
                                "strategy_availability_verified": False}))
        if previous_exchange is not None and exchange < previous_exchange:
            exchange_regressions += 1
        previous_exchange = exchange
    return _series("trade", events, rows, source_hash, duplicate_rows=duplicates,
                   exchange_time_regressions=exchange_regressions,
                   side_semantics_empirically_verified=False)


def ticker_events(rows, source_hash):
    events = []
    for ordinal, (received, exchange, row) in enumerate(_rows(rows, TICKER_HEADER, source_hash)):
        funding_time = None if row["funding_timestamp"] == "" else timestamp(row["funding_timestamp"])
        relation = ("missing" if funding_time is None else "future" if funding_time > received
                    else "equal" if funding_time == received else "past")
        events.append(seal({"kind": "ticker", "symbol": "BTCUSDT",
                            "received_ts_us": received, "exchange_ts_us": exchange,
                            "mark_price": _optional_number(row["mark_price"], positive=True),
                            "index_price": _optional_number(row["index_price"], positive=True),
                            "last_price": _optional_number(row["last_price"], positive=True),
                            "open_interest": _optional_number(row["open_interest"], nonnegative=True),
                            "reported_funding_rate": _optional_number(row["funding_rate"]),
                            "predicted_funding_rate": _optional_number(row["predicted_funding_rate"]),
                            "reported_funding_timestamp_us": funding_time,
                            "funding_time_relation_to_receipt": relation,
                            "is_funding_settlement": False,
                            "funding_cashflow_verified": False,
                            "source_row_ordinal": ordinal,
                            "source_row_sha256": object_sha(row),
                            "source_prefix_sha256": source_hash,
                            "strategy_availability_verified": False}))
    return _series("ticker", events, rows, source_hash)


def receive_asof(events, cutoff_us, max_age_us, *, allow_batch=False):
    """Latest completed provider batch strictly before cutoff; ties are not ordered."""
    cutoff_us = timestamp(cutoff_us)
    if type(max_age_us) is not int or not 0 <= max_age_us <= 5_000_000:
        raise ValueError("invalid freshness budget")
    if type(allow_batch) is not bool:
        raise ValueError("invalid batch policy")
    if not isinstance(events, list) or len(events) > 256:
        raise ValueError("event budget exceeded")
    candidate, candidate_received, previous = [], None, None
    for event in events:
        verify(event)
        received = timestamp(event["received_ts_us"])
        exchange = timestamp(event["exchange_ts_us"])
        if exchange > received or (previous is not None and received < previous):
            raise ValueError("invalid event chronology")
        if received <= cutoff_us:
            if received != candidate_received:
                candidate = []
                candidate_received = received
            candidate.append(event)
        previous = received
    status = "missing"
    if candidate:
        if candidate_received == cutoff_us:
            status = "ambiguous_clock_boundary"
        elif any(cutoff_us - e[key] > max_age_us for e in candidate
                 for key in ("received_ts_us", "exchange_ts_us")):
            status = "stale"
        elif len(candidate) > 1 and not allow_batch:
            status = "ambiguous_batch"
        else:
            status = "fresh_batch" if len(candidate) > 1 else "fresh"
    fresh = status in ("fresh", "fresh_batch")
    return {"status": status, "events": deepcopy(candidate) if fresh else [],
            "candidate_received_ts_us": candidate_received,
            "candidate_event_count": len(candidate),
            "within_batch_order_verified": False,
            "strategy_availability_verified": False}


def inspect_timebase(books, trades, tickers, *, max_age_us=500_000):
    """Engineering joins only; no fills, NAV, labels, or funding cashflows."""
    for series in (trades, tickers):
        verify(series)
        if series.get("financial_replay_ready") is not False or series.get("safety") != SAFETY:
            raise ValueError("unexpected authority")
    if trades.get("kind") != "trade" or tickers.get("kind") != "ticker":
        raise ValueError("unexpected stream kind")
    if not isinstance(books, list) or not 1 <= len(books) <= 256:
        raise ValueError("book event budget exceeded")
    for book in books:
        if book.get("kind") != "engineering_book_fragment" or book.get("symbol") != "BTCUSDT":
            raise ValueError("unexpected book identity")
        _source(book.get("source_prefix_sha256"))
        for side in ("bids", "asks"):
            levels = book.get(side)
            if not isinstance(levels, list) or not 1 <= len(levels) <= 25:
                raise ValueError("invalid observed depth")
            previous_price = None
            for level in levels:
                if not isinstance(level, list) or len(level) != 2:
                    raise ValueError("invalid price level")
                price = number(level[0])
                number(level[1])
                if previous_price is not None and (price >= previous_price if side == "bids"
                                                   else price <= previous_price):
                    raise ValueError("unsorted/duplicate book levels")
                previous_price = price
        if number(book["bids"][0][0]) >= number(book["asks"][0][0]):
            raise ValueError("invalid book spread")
    # Validate even if no trade falls inside the book prefix.
    receive_asof(books, books[-1]["received_ts_us"], max_age_us)
    joins, comparisons = [], []
    for book in books:
        cutoff = book["received_ts_us"]
        mark = receive_asof(tickers["events"], cutoff, max_age_us)
        trade = receive_asof(trades["events"], cutoff, max_age_us, allow_batch=True)
        if mark["events"] and mark["events"][0]["mark_price"] is None:
            mark["status"], mark["events"] = "missing_mark_price", []
        joins.append({"book_sha256": book["sha256"], "provider_cutoff_us": cutoff,
                      "ticker": mark, "trades": trade})
    first, last = books[0]["received_ts_us"], books[-1]["received_ts_us"]
    for trade in trades["events"]:
        if first <= trade["received_ts_us"] <= last:
            quote = receive_asof(books, trade["received_ts_us"], max_age_us)
            comparison = {"trade_sha256": trade["sha256"],
                          "trade_received_ts_us": trade["received_ts_us"],
                          "quote_status": quote["status"], "outside_observed_spread": None}
            if quote["events"]:
                book = quote["events"][0]
                bid, ask = number(book["bids"][0][0]), number(book["asks"][0][0])
                if bid >= ask:
                    raise ValueError("invalid book spread")
                comparison.update({"book_sha256": book["sha256"],
                                   "outside_observed_spread": not bid <= number(trade["price"]) <= ask})
            comparisons.append(comparison)
    return seal({"schema": "market_timebase_diagnostic/1", "books": len(books),
                 "complete_trade_events": len(trades["events"]),
                 "complete_ticker_events": len(tickers["events"]),
                 "book_prefix_span_us": last - first, "joins": joins,
                 "trade_quote_comparisons": comparisons,
                 "cross_stream_order_verified": False,
                 "provider_delay_is_order_latency": False,
                 "funding_settlements_verified": False,
                 "financial_replay_ready": False, "safety": SAFETY})
