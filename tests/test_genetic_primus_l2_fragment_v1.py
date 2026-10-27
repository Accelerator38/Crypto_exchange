"""Explicit offline unittest: bounded bytes, synthetic faults, no tape collector."""
from copy import deepcopy
import gzip
import io
import unittest

from exia.genetic_primus.bounded_gzip_csv_v1 import read_prefix
from exia.genetic_primus.l2_fragment_reconstruction_v1 import L2Fragment
from exia.genetic_primus.unified_measurement_v1 import identity

BASE = 1733011200000000


def arguments(**overrides):
    return {"source_prefix_sha256": identity({"synthetic": "prefix"}), "snapshot_levels_per_side": 2,
            "max_state_levels_per_side": 16, "max_rows_per_message": 16, "max_rows": 64,
            "max_events": 16, "max_receive_gap_us": 2000000, **overrides}


def row(t=100, side="ask", price="100.1", amount="1", snapshot=True, exchange=None):
    return {"exchange": "bitget-futures", "symbol": "BTCUSDT", "timestamp": str(BASE + (t - 10 if exchange is None else exchange)),
            "local_timestamp": str(BASE + t), "is_snapshot": "true" if snapshot else "false",
            "side": side, "price": price, "amount": amount}


def snapshot(t=100, *, replacement=False, exchange=None):
    levels = (("ask", "101.1", "1"), ("ask", "101.2", "2"), ("bid", "101", "3"), ("bid", "100.9", "4")) if replacement else (
        ("ask", "100.1", "1"), ("ask", "100.2", "2"), ("bid", "99.9", "3"), ("bid", "99.8", "4"))
    return [row(t, side, price, amount, exchange=exchange) for side, price, amount in levels]


def delta(t, levels):
    return [row(t, side, price, amount, snapshot=False) for side, price, amount in levels]


def drive(groups, **overrides):
    engine = L2Fragment(**arguments(**overrides))
    for group in groups:
        for item in group:
            engine.ingest(item)
    # A later source row closes the previous group; this last group is unclosed.
    last = max(int(item["local_timestamp"]) - BASE for group in groups for item in group)
    engine.ingest(row(last + 100))
    engine.finish()
    return engine


class PrefixReaderTests(unittest.TestCase):
    def read(self, data, **kwargs):
        return read_prefix(io.BytesIO(gzip.compress(data)), **kwargs)

    def test_small_complete_archive_preserves_strings(self):
        r = self.read(b"a,b\n01,0.0100\n")
        self.assertEqual(r["rows"], [{"a": "01", "b": "0.0100"}])
        self.assertTrue(r["entire_archive_read_and_crc_verified"])
        self.assertFalse(r["full_archive_hash_verified"])

    def test_row_budget_never_claims_archive_crc(self):
        r = self.read(b"a,b\n1,2\n3,4\n", max_rows=1)
        self.assertEqual(r["row_count"], 1)
        self.assertEqual(r["stop_reason"], "row_budget")
        self.assertFalse(r["entire_archive_read_and_crc_verified"])

    def test_high_ratio_gzip_obeys_exact_inflate_cap(self):
        r = self.read(b"a,b\n" + (b"1," + b"x" * 100 + b"\n") * 2000, max_uncompressed_bytes=64)
        self.assertEqual(r["uncompressed_bytes_inflated"], 64)
        self.assertEqual(r["stop_reason"], "uncompressed_byte_budget")
        self.assertLessEqual(r["compressed_bytes_read"], 65536)

    def test_compressed_cap_cannot_be_crossed_to_get_header(self):
        with self.assertRaises(ValueError):
            self.read(b"a,b\n1,2\n", max_compressed_bytes=5)

    def test_concatenated_gzip_members(self):
        raw = io.BytesIO(gzip.compress(b"a,b\n1,2\n") + gzip.compress(b"3,4\n"))
        r = read_prefix(raw)
        self.assertEqual(r["row_count"], 2)
        self.assertEqual(r["verified_gzip_members"], 2)
        self.assertTrue(r["entire_archive_read_and_crc_verified"])

    def test_crc_corruption_detected_when_trailer_is_read(self):
        bad = bytearray(gzip.compress(b"a,b\n1,2\n"))
        bad[-8] ^= 1
        with self.assertRaises(ValueError):
            read_prefix(io.BytesIO(bad))

    def test_truncation_not_silently_accepted_as_complete(self):
        with self.assertRaises(ValueError):
            read_prefix(io.BytesIO(gzip.compress(b"a,b\n1,2\n")[:-4]))

    def test_exact_fields_and_unique_header(self):
        for bad in (b"a,b\n1\n", b"a,a\n1,2\n", b'a,b\n"unterminated,2\n'):
            with self.assertRaises(ValueError):
                self.read(bad)

    def test_line_budget_and_invalid_utf8(self):
        for data in (b"a,b\n1," + b"x" * 100 + b"\n", b"a,b\n1,\xff\n"):
            with self.assertRaises(ValueError):
                self.read(data, max_line_bytes=16)

    def test_invalid_limits_and_nonzero_stream_position(self):
        for value in (True, -1, 4097):
            with self.assertRaises(ValueError):
                self.read(b"a,b\n", max_rows=value)
        raw = io.BytesIO(gzip.compress(b"a,b\n"))
        raw.seek(1)
        with self.assertRaises(ValueError):
            read_prefix(raw)


class ReconstructionTests(unittest.TestCase):
    def test_snapshot_replaces_all_old_levels(self):
        r = drive([snapshot(), snapshot(200, replacement=True)]).result()
        self.assertEqual(r["snapshot_count"], 2)
        self.assertEqual(r["events"][-1]["bids"][0], ["101", "3"])
        self.assertNotIn(["99.9", "3"], r["events"][-1]["bids"])

    def test_delta_absolute_replacement_and_revealed_deep_level(self):
        groups = [snapshot(), delta(200, [("ask", "100.1", "0.25"), ("ask", "100.3", "3")]),
                  delta(300, [("ask", "100.1", "0")])]
        r = drive(groups).result()
        self.assertEqual(r["events"][1]["asks"][0], ["100.1", "0.25"])
        self.assertEqual(r["events"][-1]["asks"], [["100.2", "2"], ["100.3", "3"]])
        self.assertEqual((r["delta_count"], r["delete_count"]), (2, 1))

    def test_unknown_delete_is_idempotent_but_reported(self):
        r = drive([snapshot(), delta(200, [("ask", "109", "0")])]).result()
        self.assertEqual(r["unknown_delete_count"], 1)
        self.assertEqual(r["events"][-1]["asks"][0], ["100.1", "1"])

    def test_delta_without_base_never_constructs_liquidity(self):
        r = drive([delta(100, [("ask", "100.1", "1")])]).result()
        self.assertFalse(r["book_valid_at_last_completed_group"])
        self.assertEqual(r["events"], [])

    def test_incomplete_snapshot_not_published(self):
        r = drive([snapshot()[:-1]]).result()
        self.assertEqual(r["snapshot_count"], 0)
        self.assertFalse(r["book_valid_at_last_completed_group"])

    def test_duplicate_snapshot_price_rejected(self):
        group = snapshot()
        group[1] = deepcopy(group[0])
        self.assertEqual(drive([group]).result()["events"], [])

    def test_unclosed_last_group_discarded_even_if_it_looks_complete(self):
        e = L2Fragment(**arguments())
        for item in snapshot():
            e.ingest(item)
        e.finish()
        self.assertEqual((e.result()["dropped_final_rows"], len(e.result()["events"])), (4, 0))

    def test_gap_blocks_deltas_until_new_snapshot(self):
        e = L2Fragment(**arguments())
        for item in snapshot() + delta(200, [("ask", "100.1", "2")]):
            e.ingest(item)
        e.invalidate_gap()
        for item in delta(300, [("ask", "100.1", "9")]) + snapshot(400) + [row(500)]:
            e.ingest(item)
        e.finish()
        r = e.result()
        self.assertEqual(r["delta_count"], 0)
        self.assertTrue(r["book_valid_at_last_completed_group"])
        self.assertEqual(r["events"][-1]["asks"][0], ["100.1", "1"])
        with self.assertRaises(ValueError):
            e.checkpoint()

    def test_observed_silence_invalidates_without_claiming_packet_loss(self):
        r = drive([snapshot(), delta(3000000, [("ask", "100.1", "9")]), snapshot(3000100)]).result()
        self.assertTrue(any(x["reason"].startswith("observed_silence") for x in r["faults"]))
        self.assertEqual(r["delta_count"], 0)
        self.assertTrue(r["book_valid_at_last_completed_group"])

    def test_crossed_delta_invalidates_and_fresh_snapshot_recovers(self):
        r = drive([snapshot(), delta(200, [("bid", "100.5", "2")]), snapshot(300)]).result()
        self.assertEqual(len(r["events"]), 2)
        self.assertTrue(r["book_valid_at_last_completed_group"])

    def test_full_delta_message_is_atomic_not_transient_crossed_rows(self):
        updates = [("bid", "100.5", "2"), ("ask", "100.1", "0"),
                   ("ask", "100.2", "0"), ("ask", "101", "3")]
        r = drive([snapshot(), delta(200, updates)]).result()
        self.assertEqual(r["delta_count"], 1)
        self.assertEqual(r["faults"], [])

    def test_exchange_watermark_never_moves_backward_after_bad_snapshot(self):
        r = drive([snapshot(), snapshot(200, exchange=80), snapshot(300, exchange=85), snapshot(400)]).result()
        self.assertEqual(r["snapshot_count"], 2)
        self.assertEqual(len(r["faults"]), 2)

    def test_row_receipt_regression_clears_live_book_state(self):
        e = L2Fragment(**arguments())
        for item in snapshot() + [row(200)]:
            e.ingest(item)
        with self.assertRaises(ValueError):
            e.ingest(row(150))
        self.assertFalse(e.result()["book_valid_at_last_completed_group"])

    def test_invalid_market_timestamp_size_and_flag_fail_closed(self):
        for key, value in (("symbol", "ETHUSDT"), ("timestamp", "bad"), ("amount", "-1"),
                           ("amount", "0"), ("price", "NaN"), ("is_snapshot", "unknown")):
            item = row()
            item[key] = value
            e = L2Fragment(**arguments())
            with self.assertRaises(ValueError):
                e.ingest(item)
            self.assertFalse(e.result()["book_valid_at_last_completed_group"])

    def test_duplicate_delta_price_does_not_use_last_row_wins(self):
        r = drive([snapshot(), delta(200, [("bid", "99.9", "2"), ("bid", "99.9", "4")])]).result()
        self.assertEqual(r["delta_count"], 0)
        self.assertFalse(r["book_valid_at_last_completed_group"])

    def test_depth_and_event_bounds_do_not_silently_evict(self):
        r = drive([snapshot(), delta(200, [("ask", "100.3", "1")])], max_state_levels_per_side=2).result()
        self.assertFalse(r["book_valid_at_last_completed_group"])
        r = drive([snapshot(), snapshot(200)], max_events=1).result()
        self.assertEqual(len(r["events"]), 1)
        self.assertFalse(r["book_valid_at_last_completed_group"])

    def test_row_and_message_bounds(self):
        for kwargs in ({"max_rows": 4}, {"max_rows_per_message": 4}):
            e = L2Fragment(**arguments(**kwargs))
            with self.assertRaises(ValueError):
                for item in snapshot() + [row(100, side="ask", price="100.3")]:
                    e.ingest(item)
            self.assertFalse(e.result()["book_valid_at_last_completed_group"])

    def test_checkpoint_inside_snapshot_restores_exactly(self):
        rows = snapshot() + snapshot(200) + delta(300, [("ask", "100.1", "2")]) + [row(400)]
        direct = L2Fragment(**arguments())
        for item in rows[:6]:
            direct.ingest(item)
        checkpoint = direct.checkpoint()
        resumed = L2Fragment.restore(checkpoint, **arguments())
        for item in rows[6:]:
            direct.ingest(item)
            resumed.ingest(item)
        direct.finish()
        resumed.finish()
        self.assertEqual(direct.result(), resumed.result())
        bad = deepcopy(checkpoint)
        bad["rows"][0]["amount"] = "9"
        with self.assertRaises(ValueError):
            L2Fragment.restore(bad, **arguments())
        with self.assertRaises(ValueError):
            L2Fragment.restore(checkpoint, **arguments(source_prefix_sha256="0" * 64))

    def test_microseconds_preserved_and_no_financial_qualification(self):
        r = drive([snapshot(111999)]).result()
        self.assertEqual(r["events"][0]["received_ts_us"], BASE + 111999)
        self.assertFalse(r["venue_sequence_verified"])
        self.assertFalse(r["financial_replay_ready"])
        self.assertFalse(any(r["safety"].values()))

    def test_silent_lost_delta_is_not_detectable_without_sequence(self):
        full = drive([snapshot(), delta(200, [("ask", "100.1", "9")]),
                      delta(300, [("bid", "99.9", "1")])]).result()
        lost = drive([snapshot(), delta(300, [("bid", "99.9", "1")])]).result()
        self.assertEqual(full["faults"], [])
        self.assertEqual(lost["faults"], [])
        self.assertNotEqual(full["events"][-1]["asks"], lost["events"][-1]["asks"])
        self.assertFalse(lost["venue_sequence_verified"])

    def test_input_and_result_copies_and_finish_not_a_flush_loop(self):
        source = snapshot()
        e = L2Fragment(**arguments())
        for item in source:
            e.ingest(item)
        source[0]["amount"] = "999"
        e.ingest(row(200))
        e.finish()
        r = e.result()
        self.assertEqual(r["events"][0]["asks"][0][1], "1")
        r["events"][0]["asks"][0][1] = "0"
        self.assertEqual(e.result()["events"][0]["asks"][0][1], "1")
        with self.assertRaises(ValueError):
            e.finish()
        with self.assertRaises(ValueError):
            e.ingest(row(300))


if __name__ == "__main__":
    unittest.main()
