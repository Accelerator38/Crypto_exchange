"""Explicit bounded tests: no collector, network, pytest, sleep, or market replay."""
import unittest
from copy import deepcopy

from exia.genetic_primus.market_timebase_fragment_v1 import (
    SAFETY, inspect_timebase, receive_asof, seal, ticker_events, trade_events, verify,
)

BASE = 1733011200000000
HASH = "a" * 64


def trade(offset, trade_id="1", **changes):
    row = {"exchange": "bitget-futures", "symbol": "BTCUSDT",
           "timestamp": str(BASE + offset - 10), "local_timestamp": str(BASE + offset),
           "id": trade_id, "side": "buy", "price": "100.10", "amount": "0.010"}
    row.update(changes)
    return row


def ticker(offset, **changes):
    row = {"exchange": "bitget-futures", "symbol": "BTCUSDT",
           "timestamp": str(BASE + offset - 10), "local_timestamp": str(BASE + offset),
           "funding_timestamp": str(BASE + 28_800_000_000), "funding_rate": "0.000195",
           "predicted_funding_rate": "", "open_interest": "123.00", "last_price": "100.10",
           "index_price": "100.0", "mark_price": "100.20"}
    row.update(changes)
    return row


def book(offset, **changes):
    body = {"kind": "engineering_book_fragment", "symbol": "BTCUSDT",
            "received_ts_us": BASE + offset, "exchange_ts_us": BASE + offset - 10,
            "bids": [["100.10", "1"]], "asks": [["100.20", "1"]],
            "source_prefix_sha256": HASH}
    body.update(changes)
    return seal(body)


def fixtures(**trade_changes):
    return ([book(1000), book(2000)],
            trade_events([trade(1500, **trade_changes), trade(3000, "2")], HASH),
            ticker_events([ticker(1400), ticker(3100)], HASH))


class TimebaseTests(unittest.TestCase):
    def test_trade_exact_precision_final_batch_discarded(self):
        result = trade_events([trade(1001), trade(2002, "2")], HASH)
        self.assertEqual(result["events"][0]["received_ts_us"], BASE + 1001)
        self.assertEqual(result["events"][0]["amount"], "0.010")
        self.assertEqual(result["dropped_final_events"], 1)

    def test_trade_equal_receipts_preserved(self):
        result = trade_events([trade(1000), trade(1000, "2"), trade(2000, "3")], HASH)
        self.assertEqual([e["trade_id"] for e in result["events"]], ["1", "2"])

    def test_trade_zero_or_negative_size_rejected(self):
        for value in ("0", "-0.1"):
            with self.assertRaises(ValueError):
                trade_events([trade(1000, amount=value)], HASH)

    def test_invalid_side(self):
        with self.assertRaises(ValueError):
            trade_events([trade(1000, side="ask")], HASH)

    def test_invalid_id(self):
        for value in ("", "0", "1e3", "1" * 33):
            with self.assertRaises(ValueError):
                trade_events([trade(1000, id=value)], HASH)

    def test_milliseconds_rejected(self):
        with self.assertRaises(ValueError):
            trade_events([trade(1000, timestamp="1733011200000")], HASH)

    def test_regressing_receipt(self):
        with self.assertRaises(ValueError):
            trade_events([trade(2000), trade(1000, "2")], HASH)

    def test_regressing_exchange_is_recorded_not_reordered(self):
        result = trade_events([trade(1000), trade(2000, "2", timestamp=str(BASE + 900)),
                               trade(3000, "3")], HASH)
        self.assertEqual(result["exchange_time_regressions"], 1)
        self.assertEqual([e["trade_id"] for e in result["events"]], ["1", "2"])

    def test_exchange_after_receipt_rejected(self):
        with self.assertRaises(ValueError):
            trade_events([trade(1000, timestamp=str(BASE + 1001))], HASH)

    def test_duplicate_same_payload_not_double_counted(self):
        repeat = trade(2000, timestamp=str(BASE + 990))
        result = trade_events([trade(1000), repeat, trade(3000, "2")], HASH)
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(len(result["duplicate_rows"]), 1)

    def test_conflicting_duplicate_id_rejected(self):
        with self.assertRaises(ValueError):
            trade_events([trade(1000), trade(1000, price="101")], HASH)

    def test_raw_input_not_mutated(self):
        rows = [trade(1000), trade(2000, "2")]
        before = deepcopy(rows)
        trade_events(rows, HASH)
        self.assertEqual(rows, before)

    def test_extra_header_rejected(self):
        with self.assertRaises(ValueError):
            trade_events([trade(1000, extra="0")], HASH)

    def test_row_budget(self):
        for rows in ([], [trade(i + 1000, str(i + 1)) for i in range(257)]):
            with self.assertRaises(ValueError):
                trade_events(rows, HASH)

    def test_missing_prediction_is_not_zero(self):
        result = ticker_events([ticker(1000), ticker(2000)], HASH)
        self.assertIsNone(result["events"][0]["predicted_funding_rate"])

    def test_signed_and_zero_rates_preserved(self):
        result = ticker_events([ticker(1000, funding_rate="-0.001"),
                                ticker(2000, funding_rate="0"), ticker(3000)], HASH)
        self.assertEqual([e["reported_funding_rate"] for e in result["events"]], ["-0.001", "0"])

    def test_future_funding_timestamp_not_settlement(self):
        event = ticker_events([ticker(1000), ticker(2000)], HASH)["events"][0]
        self.assertEqual(event["reported_funding_timestamp_us"], BASE + 28_800_000_000)
        self.assertEqual(event["funding_time_relation_to_receipt"], "future")
        self.assertFalse(event["is_funding_settlement"])

    def test_equal_funding_timestamp_still_not_settlement(self):
        event = ticker_events([ticker(1000, funding_timestamp=str(BASE + 1000)),
                               ticker(2000)], HASH)["events"][0]
        self.assertEqual(event["funding_time_relation_to_receipt"], "equal")
        self.assertFalse(event["funding_cashflow_verified"])

    def test_past_funding_timestamp_still_not_settlement(self):
        event = ticker_events([ticker(1000, funding_timestamp=str(BASE)),
                               ticker(2000)], HASH)["events"][0]
        self.assertEqual(event["funding_time_relation_to_receipt"], "past")
        self.assertFalse(event["is_funding_settlement"])

    def test_missing_funding_remains_missing(self):
        event = ticker_events([ticker(1000, funding_timestamp="", funding_rate=""),
                               ticker(2000)], HASH)["events"][0]
        self.assertIsNone(event["reported_funding_timestamp_us"])
        self.assertIsNone(event["reported_funding_rate"])

    def test_nonfinite_and_exponent_prices_rejected(self):
        for value in ("NaN", "Infinity", "1e2", ".1"):
            with self.assertRaises(ValueError):
                ticker_events([ticker(1000, mark_price=value)], HASH)

    def test_open_interest_zero_allowed_negative_rejected(self):
        self.assertEqual(ticker_events([ticker(1000, open_interest="0"), ticker(2000)],
                                       HASH)["events"][0]["open_interest"], "0")
        with self.assertRaises(ValueError):
            ticker_events([ticker(1000, open_interest="-1")], HASH)

    def test_missing_prices_not_invented(self):
        event = ticker_events([ticker(1000, mark_price="", index_price="", last_price=""),
                               ticker(2000)], HASH)["events"][0]
        self.assertTrue(all(event[k] is None for k in ("mark_price", "index_price", "last_price")))

    def test_same_receive_tickers_are_ambiguous(self):
        events = ticker_events([ticker(1000), ticker(1000, mark_price="101"),
                                ticker(2000)], HASH)["events"]
        self.assertEqual(receive_asof(events, BASE + 1500, 5000)["status"], "ambiguous_batch")

    def test_future_receipt_not_selected_by_exchange_time(self):
        events = ticker_events([ticker(2000, timestamp=str(BASE + 500)),
                                ticker(3000)], HASH)["events"]
        self.assertEqual(receive_asof(events, BASE + 1500, 5000)["status"], "missing")

    def test_exact_cross_stream_boundary_ambiguous(self):
        self.assertEqual(receive_asof([book(1000)], BASE + 1000, 5000)["status"],
                         "ambiguous_clock_boundary")

    def test_trade_batch_all_events_returned_without_order_claim(self):
        events = trade_events([trade(1000), trade(1000, "2"), trade(2000, "3")], HASH)["events"]
        result = receive_asof(events, BASE + 1500, 5000, allow_batch=True)
        self.assertEqual(result["status"], "fresh_batch")
        self.assertEqual(len(result["events"]), 2)
        self.assertFalse(result["within_batch_order_verified"])

    def test_stale_receipt_and_exchange(self):
        self.assertEqual(receive_asof([book(1000)], BASE + 2000, 999)["status"], "stale")
        self.assertEqual(receive_asof([book(1900, exchange_ts_us=BASE)], BASE + 2000,
                                     1000)["status"], "stale")

    def test_event_hash_tamper(self):
        event = book(1000)
        event["received_ts_us"] += 1
        with self.assertRaises(ValueError):
            receive_asof([event], BASE + 2000, 5000)

    def test_return_is_detached(self):
        event = book(1000)
        result = receive_asof([event], BASE + 2000, 5000)
        result["events"][0]["bids"][0][0] = "1"
        self.assertEqual(event["bids"][0][0], "100.10")

    def test_event_receipt_regression(self):
        with self.assertRaises(ValueError):
            receive_asof([book(2000), book(1000)], BASE + 3000, 5000)

    def test_age_and_event_bounds(self):
        for age in (-1, True, 5_000_001):
            with self.assertRaises(ValueError):
                receive_asof([], BASE, age)
        with self.assertRaises(ValueError):
            receive_asof([book(1000)] * 257, BASE, 0)

    def test_trade_outside_quote_is_diagnostic_not_fill(self):
        result = inspect_timebase(*fixtures(price="101"))
        self.assertTrue(result["trade_quote_comparisons"][0]["outside_observed_spread"])
        self.assertFalse(result["financial_replay_ready"])
        self.assertFalse(result["funding_settlements_verified"])

    def test_missing_mark_join_explicit(self):
        books, trades, _ = fixtures()
        tickers = ticker_events([ticker(1400, mark_price=""), ticker(3100)], HASH)
        result = inspect_timebase(books, trades, tickers)
        self.assertEqual(result["joins"][-1]["ticker"]["status"], "missing_mark_price")

    def test_empty_complete_series_never_qualifies(self):
        result = inspect_timebase([book(1000)], trade_events([trade(1000)], HASH),
                                  ticker_events([ticker(1000)], HASH))
        self.assertEqual(result["complete_trade_events"], 0)
        self.assertEqual(result["safety"], SAFETY)
        self.assertTrue(all(v is False for v in result["safety"].values()))

    def test_locked_book_rejected_without_trade_overlap(self):
        with self.assertRaises(ValueError):
            inspect_timebase([book(1000, asks=[["100.10", "1"]])],
                             trade_events([trade(5000)], HASH), ticker_events([ticker(5000)], HASH))

    def test_series_hash_tamper(self):
        books, trades, tickers = fixtures()
        trades["events"] = []
        with self.assertRaises(ValueError):
            inspect_timebase(books, trades, tickers)

    def test_deterministic_and_sealed(self):
        first = inspect_timebase(*fixtures())
        self.assertEqual(first, inspect_timebase(*fixtures()))
        verify(first)

    def test_singleton_prefix_last_batch_not_published(self):
        self.assertEqual(trade_events([trade(1000)], HASH)["events"], [])
        self.assertEqual(ticker_events([ticker(1000)], HASH)["events"], [])

    def test_duplicate_delimiter_completes_prior_unique_batch(self):
        result = trade_events([trade(1000), trade(2000, timestamp=str(BASE + 990))], HASH)
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["dropped_final_events"], 0)

    def test_invalid_source_hash(self):
        with self.assertRaises(ValueError):
            trade_events([trade(1000)], "bad")

    def test_unsorted_book_rejected(self):
        books, trades, tickers = fixtures()
        books[0] = book(1000, bids=[["99", "1"], ["100", "1"]])
        with self.assertRaises(ValueError):
            inspect_timebase(books, trades, tickers)


if __name__ == "__main__":
    unittest.main()
