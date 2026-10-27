"""Bounded snapshot/delta engineering state; no file IO, orders or prices/PnL model.

Reuses the frozen snapshot-only validator. Missing venue sequence is explicit:
silence and injected faults can invalidate a book, but silent lost deltas cannot
be detected reliably from this CSV. Never claims complete market reconstruction.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal

from .market_fragment_v1 import book_snapshots, identity as row_clocks, number
from .unified_measurement_v1 import SAFETY, identity

HEADER = ("exchange", "symbol", "timestamp", "local_timestamp", "is_snapshot", "side", "price", "amount")


def bounded(value, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError("positive integer within hard bound required")
    return value


class L2Fragment:
    def __init__(self, *, source_prefix_sha256, snapshot_levels_per_side=15,
                 max_state_levels_per_side=128, max_rows_per_message=256,
                 max_rows=512, max_events=256, max_receive_gap_us=2000000):
        if not isinstance(source_prefix_sha256, str) or len(source_prefix_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_prefix_sha256):
            raise ValueError("explicit source PREFIX hash required")
        self.arguments = dict(source_prefix_sha256=source_prefix_sha256,
            snapshot_levels_per_side=bounded(snapshot_levels_per_side, 25),
            max_state_levels_per_side=bounded(max_state_levels_per_side, 256),
            max_rows_per_message=bounded(max_rows_per_message, 512), max_rows=bounded(max_rows, 4096),
            max_events=bounded(max_events, 512), max_receive_gap_us=bounded(max_receive_gap_us, 60000000))
        if max_rows_per_message < 2 * snapshot_levels_per_side or max_state_levels_per_side < snapshot_levels_per_side:
            raise ValueError("bounds cannot hold a complete snapshot")
        self.rows, self.pending, self.events, self.faults = [], [], [], []
        self.book = {"bid": {}, "ask": {}}
        self.valid, self.finished = False, False
        self.pending_key = self.last_row_received = self.last_committed_key = None
        self.snapshot_count = self.delta_count = self.delete_count = self.unknown_delete_count = self.dropped_final_rows = 0

    def _fault(self, reason):
        if len(self.faults) >= self.arguments["max_events"]:
            raise ValueError("fault bound exceeded")
        self.valid = False
        self.book = {"bid": {}, "ask": {}}
        self.faults.append({"reason": reason, "after_accepted_rows": len(self.rows)})

    def invalidate_gap(self, reason="explicit_gap_or_disconnect"):
        """Called only on external evidence; not a sequence detector or reconnect."""
        if self.finished or not isinstance(reason, str) or not 1 <= len(reason) <= 128:
            raise ValueError("bounded gap reason before finish required")
        self.pending, self.pending_key = [], None
        self._fault(reason)

    def ingest(self, row):
        if self.finished:
            raise ValueError("fragment already finished")
        try:
            if len(self.rows) >= self.arguments["max_rows"] or not isinstance(row, dict) or set(row) != set(HEADER):
                raise ValueError("row bound or exact source schema mismatch")
            received, exchange = row_clocks(row)
            if self.last_row_received is not None and received < self.last_row_received:
                raise ValueError("source row receipt regression")
            if row["is_snapshot"] not in ("true", "false") or row["side"] not in ("bid", "ask"):
                raise ValueError("explicit snapshot flag and bid/ask required")
            number(row["price"])
            amount = number(row["amount"], positive=False)
            if amount < 0 or (row["is_snapshot"] == "true" and amount == 0):
                raise ValueError("negative amount or zero snapshot liquidity")
            key = (received, exchange, row["is_snapshot"])
            if self.pending_key is not None and key != self.pending_key:
                self._commit()
                self.pending = []
            if len(self.pending) >= self.arguments["max_rows_per_message"]:
                raise ValueError("message row bound exceeded")
            self.pending_key = key
            self.pending.append(deepcopy(row))
            self.rows.append(deepcopy(row))
            self.last_row_received = received
        except (ValueError, KeyError, TypeError):
            self.pending, self.pending_key = [], None
            self._fault("rejected_source_row_or_bound")
            raise

    def _commit(self):
        key, group = self.pending_key, self.pending
        received, exchange, snapshot = key
        previous = self.last_committed_key
        if previous is not None and (received <= previous[0] or exchange < previous[1]):
            self.last_committed_key = (max(received, previous[0]), max(exchange, previous[1]), snapshot)
            self._fault("ambiguous_same_receipt_or_exchange_regression")
            return
        self.last_committed_key = key
        if previous is not None and received - previous[0] > self.arguments["max_receive_gap_us"]:
            self._fault("observed_silence_gap_not_proof_of_packet_loss")
        try:
            if snapshot == "true":
                observed = book_snapshots(group, self.arguments["source_prefix_sha256"],
                                          levels=self.arguments["snapshot_levels_per_side"])[0]
                candidate = {side: {Decimal(p): (p, q) for p, q in observed[side + "s"]} for side in ("bid", "ask")}
            else:
                if not self.valid:
                    self._fault("delta_without_valid_snapshot")
                    return
                candidate, seen, deleted, unknown = deepcopy(self.book), set(), 0, 0
                for row in group:
                    side, price = row["side"], number(row["price"])
                    if (side, price) in seen:
                        raise ValueError("duplicate delta price in inferred message")
                    seen.add((side, price))
                    if number(row["amount"], positive=False) == 0:
                        deleted += 1
                        unknown += int(price not in candidate[side])
                        candidate[side].pop(price, None)
                    else:
                        # Delta quantity is ABSOLUTE replacement, never an increment.
                        candidate[side][price] = (row["price"], row["amount"])
            if not all(candidate.values()) or max(candidate["bid"]) >= min(candidate["ask"]):
                raise ValueError("empty side or crossed/locked reconstructed book")
            if any(len(side) > self.arguments["max_state_levels_per_side"] for side in candidate.values()):
                raise ValueError("state depth bound exceeded; no silent eviction")
            if len(self.events) >= self.arguments["max_events"]:
                raise ValueError("event bound exceeded")
        except ValueError:
            self._fault("incomplete_ambiguous_crossed_or_oversized_message")
            return
        self.book, self.valid = candidate, True
        self.snapshot_count += int(snapshot == "true")
        self.delta_count += int(snapshot == "false")
        if snapshot == "false":
            self.delete_count += deleted
            self.unknown_delete_count += unknown
        body = {"kind": "engineering_book_fragment", "symbol": "BTCUSDT",
                "received_ts_us": received, "exchange_ts_us": exchange,
                "source_prefix_sha256": self.arguments["source_prefix_sha256"], "source_message_rows_sha256": identity(group),
                "message_type": "snapshot" if snapshot == "true" else "delta",
                "source_rows_end_exclusive": len(self.rows), "venue_sequence_verified": False,
                "full_depth_verified": False, "strategy_availability_verified": False,
                "observed_state_levels": {side: len(candidate[side]) for side in ("bid", "ask")},
                **{side + "s": [list(candidate[side][p]) for p in sorted(candidate[side], reverse=side == "bid")
                                [:self.arguments["snapshot_levels_per_side"]]] for side in ("bid", "ask")}}
        self.events.append(body | {"sha256": identity(body)})

    def finish(self):
        if self.finished:
            raise ValueError("finish is not a repeated flush")
        # No next-message delimiter: never guess that the last message is complete.
        self.dropped_final_rows = len(self.pending)
        self.pending, self.pending_key, self.finished = [], None, True

    def result(self):
        body = {"schema": "l2_fragment_result/1", "arguments_sha256": identity(self.arguments),
                "accepted_rows": len(self.rows), "accepted_row_prefix_sha256": identity(self.rows),
                "snapshot_count": self.snapshot_count, "delta_count": self.delta_count,
                "delete_count": self.delete_count, "unknown_delete_count": self.unknown_delete_count,
                "dropped_final_rows": self.dropped_final_rows, "pending_rows": len(self.pending),
                "finished": self.finished, "book_valid_at_last_completed_group": self.valid,
                "events": self.events, "faults": self.faults, "venue_sequence_verified": False,
                "delta_semantics_empirically_verified": False,
                "full_depth_verified": False, "strategy_availability_verified": False,
                "financial_replay_ready": False, "safety": SAFETY}
        body = deepcopy(body)
        return body | {"sha256": identity(body)}

    def checkpoint(self):
        if self.faults:
            raise ValueError("no resumable checkpoint after source faults; start a newly evidenced segment")
        body = {"schema": "l2_fragment_checkpoint/1", "arguments_sha256": identity(self.arguments),
                "rows": deepcopy(self.rows), "finished": self.finished, "result_sha256": self.result()["sha256"]}
        return body | {"sha256": identity(body)}

    @classmethod
    def restore(cls, checkpoint, **arguments):
        if checkpoint["schema"] != "l2_fragment_checkpoint/1" or checkpoint["sha256"] != identity({k: v for k, v in checkpoint.items() if k != "sha256"}):
            raise ValueError("checkpoint integrity failure")
        engine = cls(**arguments)
        if checkpoint["arguments_sha256"] != identity(engine.arguments) or len(checkpoint["rows"]) > engine.arguments["max_rows"]:
            raise ValueError("checkpoint source/bounds mismatch")
        for row in checkpoint["rows"]:
            engine.ingest(row)
        if checkpoint["finished"] is True:
            engine.finish()
        if engine.result()["sha256"] != checkpoint["result_sha256"]:
            raise ValueError("checkpoint deterministic replay mismatch")
        return engine
