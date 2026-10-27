"""Compact-model-independent, bounded OFFLINE IOC execution kernel.

Reuses legacy Decimal quantization, NOT its hard-coded genome/model registry.
No IO/network/clock/sleep/async/runtime. This is fully-collateralized long/cash
engineering accounting, not venue margin/liquidation or execution qualification.
"""
from __future__ import annotations

import copy
from decimal import Decimal, localcontext
from .execution_replay_v1 import dec, normalize_order, round_step, plain
from .unified_measurement_v1 import SAFETY, BAR_MS, HORIZON_MS, identity

PRIORITY = {"funding": 0, "mark": 1, "intent": 2, "book": 3, "cancel": 4, "clock": 5}
OPEN = {"WAIT_REFERENCE", "SUBMITTED"}


def clock(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("nonnegative integer event clock required")
    return value


def digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("full lowercase SHA256 required")
    return value


def binding_from_compact_origin(origin, policy, *, signal_available_ms):
    """Read-only provenance bridge; never requalifies old next-4h-open labels."""
    from .compact_net_replay_v1 import validate_origin, ARMS
    validate_origin(origin, origin["origin_ms"])
    if policy not in ARMS or clock(signal_available_ms) < origin["origin_ms"]:
        raise ValueError("registered policy and post-close signal availability required")
    body = {"origin_closed_ms": origin["origin_ms"], "signal_available_ms": signal_available_ms,
            "train_end_ms": origin["training_identity"]["train_end_ms"],
            "label_cutoff_ms": origin["training_identity"]["cutoff_ms"],
            "feature_available_ms": origin["origin_ms"],
            "model_sha256": identity([m["sha256"] for m in origin["arms"][policy]["models"]]),
            "scaler_sha256": origin["scaler_sha256"], "feature_prefix_sha256": origin["feature_prefix_sha256"],
            "training_identity_sha256": identity(origin["training_identity"]),
            "target_clock": "next_4h_open_gross_three_day_diagnostic",
            "new_clock_labels_qualified": False}
    return body | {"sha256": identity(body)}


class CompactIOC:
    def __init__(self, *, contract, rules, fee_schedule, bindings, data_sha256, implementation_sha256):
        self.contract, self.rules, self.fees = map(copy.deepcopy, (contract, rules, fee_schedule))
        if (contract.get("schema") != "compact_ioc_engineering/1" or contract.get("safety") != SAFETY
                or contract.get("evidence_kind") not in {"synthetic", "observed_unqualified"}
                or contract.get("ohlcv_fill_fallback") is not False or contract.get("live_adapter") is not None
                or contract.get("requires_new_target_clock_and_labels_before_market_strategy_replay") is not True):
            raise ValueError("offline fail-closed contract required")
        for key, bound in (("max_events", 512), ("max_orders", 64), ("max_levels", 50)):
            if not 0 < clock(contract[key]) <= bound:
                raise ValueError("hard replay bound exceeded")
        for key in ("signal_compute_latency_ms", "order_latency_ms", "order_ttl_ms", "reference_wait_ms", "max_quote_age_ms", "max_mark_age_ms"):
            clock(contract[key])
        if (contract["order_ttl_ms"] <= contract["order_latency_ms"] or contract["reference_wait_ms"] <= 0
                or dec(contract["initial_cash_quote"]) <= 0 or dec(contract["max_order_notional_quote"]) <= 0
                or not 0 < dec(contract["participation"]) <= 1 or not 0 < dec(contract["max_gross_ratio"]) <= 1
                or not 0 < dec(contract["research_drawdown_gate"]) <= 1 or not 0 <= dec(contract["protection_bps"]) < 10000):
            raise ValueError("invalid engineering latency/capital/liquidity budget")
        if not rules or len(rules) > 8:
            raise ValueError("one to eight explicit instruments required")
        for rule in rules.values():
            if set(rule) != {"price_tick", "quantity_step", "minimum_quantity", "minimum_notional", "source_sha256", "valid_from_ms", "valid_until_exclusive_ms"}:
                raise ValueError("complete versioned instrument rules required")
            digest(rule["source_sha256"])
            if any(dec(rule[k]) <= 0 for k in ("price_tick", "quantity_step", "minimum_quantity", "minimum_notional")) or clock(rule["valid_from_ms"]) >= clock(rule["valid_until_exclusive_ms"]):
                raise ValueError("invalid rule interval/grid")
        if set(fee_schedule) != {"taker_fee", "source_sha256", "valid_from_ms", "valid_until_exclusive_ms"}:
            raise ValueError("explicit fee evidence validity interval required")
        digest(fee_schedule["source_sha256"])
        if not 0 <= dec(fee_schedule["taker_fee"]) < 1 or clock(fee_schedule["valid_from_ms"]) >= clock(fee_schedule["valid_until_exclusive_ms"]):
            raise ValueError("invalid fee schedule")
        self.bindings = {}
        for binding in bindings:
            body = {k: v for k, v in binding.items() if k != "sha256"}
            if binding["sha256"] != identity(body) or binding["sha256"] in self.bindings:
                raise ValueError("model binding identity/uniqueness")
            origin = clock(binding["origin_closed_ms"])
            if (origin % BAR_MS or binding["train_end_ms"] != origin - BAR_MS
                    or binding["label_cutoff_ms"] != origin - 8 * BAR_MS
                    or binding["feature_available_ms"] != origin
                    or clock(binding["signal_available_ms"]) < origin + contract["signal_compute_latency_ms"]
                    or binding["new_clock_labels_qualified"] is not False):
                raise ValueError("closed-feature availability, purge or unqualified-label gate")
            for key in ("model_sha256", "scaler_sha256", "feature_prefix_sha256", "training_identity_sha256"):
                digest(binding[key])
            self.bindings[binding["sha256"]] = copy.deepcopy(binding)
        if not 0 < len(self.bindings) <= 32:
            raise ValueError("bounded model binding registry required")
        self.binding_sha = identity({"contract": contract, "rules": rules, "fees": fee_schedule, "bindings": bindings,
                                     "data_sha256": digest(data_sha256), "implementation_sha256": digest(implementation_sha256)})
        self.events = []
        self.state = {"cash": dec(contract["initial_cash_quote"]), "positions": {s: Decimal(0) for s in rules},
                      "marks": {}, "orders": {}, "snapshots": {}, "last_book_exchange": {}, "ledger": [],
                      "last_clock": None, "peak": dec(contract["initial_cash_quote"]), "max_drawdown": Decimal(0),
                      "equity": [], "fees": Decimal(0), "funding": Decimal(0)}

    def _log(self, kind, ts, **fields):
        self.state["ledger"].append({"kind": kind, "ts": ts, **fields})

    def _end(self, order, status, reason, ts):
        order.update(status=status, reason=reason, finished_ms=ts)
        self._log("order_finished", ts, order_id=order["order_id"], status=status, reason=reason,
                  filled_quantity=order["filled"], canceled_quantity=order["remaining"])

    def _equity(self):
        return self.state["cash"] + sum((q * self.state["marks"][s]["price"] for s, q in self.state["positions"].items() if q), Decimal(0))

    def _reserved(self, exclude=None):
        return sum((o["remaining"] * o["limit"] * (1 + dec(self.fees["taker_fee"]))
                    for oid, o in self.state["orders"].items() if oid != exclude and o["status"] == "SUBMITTED" and o["side"] == "buy"), Decimal(0))

    def _fresh_marks(self, ts, symbol):
        needed = {s for s, q in self.state["positions"].items() if q} | {symbol}
        return all(s in self.state["marks"] and ts - self.state["marks"][s]["exchange_ms"] <= self.contract["max_mark_age_ms"] for s in needed)

    def _valid_evidence(self, symbol, reference_ms, exchange_ms, processing_ms):
        rule = self.rules[symbol]
        return (rule["valid_from_ms"] <= min(reference_ms, exchange_ms, processing_ms)
                and max(reference_ms, exchange_ms, processing_ms) < rule["valid_until_exclusive_ms"]
                and self.fees["valid_from_ms"] <= processing_ms < self.fees["valid_until_exclusive_ms"])

    def _risk_reason(self, order, quantity, price, ts):
        symbol = order["symbol"]
        if order["side"] == "sell":
            return None if quantity <= self.state["positions"][symbol] else "reduce_only_position_limit"
        if not self._fresh_marks(ts, symbol):
            return "missing_or_stale_mark"
        spend = quantity * price * (1 + dec(self.fees["taker_fee"]))
        if spend + self._reserved(exclude=order["order_id"]) > self.state["cash"]:
            return "cash_reservation_limit"
        equity = self._equity()
        if self.state["max_drawdown"] > dec(self.contract["research_drawdown_gate"]):
            return "research_drawdown_breach_new_buy_blocked"
        gross = sum((q * self.state["marks"][s]["price"] for s, q in self.state["positions"].items() if q), Decimal(0))
        added = quantity * self.state["marks"][symbol]["price"]
        post_equity = equity - spend + added
        if post_equity <= 0 or gross + added > dec(self.contract["max_gross_ratio"]) * post_equity:
            return "gross_exposure_limit"
        return None

    def ingest(self, event):
        if len(self.events) >= self.contract["max_events"]:
            raise ValueError("event bound reached")
        event = copy.deepcopy(event)
        kind, ts = event["kind"], clock(event["ts"])
        if kind not in PRIORITY:
            raise ValueError("unknown event kind; no OHLCV fallback")
        key = (ts, PRIORITY[kind])
        if self.state["last_clock"] is not None and key < self.state["last_clock"]:
            raise ValueError("received-event/funding priority regression")
        before = copy.deepcopy(self.state)
        try:
            with localcontext() as context:
                context.prec = 50
                for order in self.state["orders"].values():
                    deadline = order["reference_deadline_ms"] if order["status"] == "WAIT_REFERENCE" else order.get("expires_ms")
                    if order["status"] in OPEN and ts >= deadline:
                        self._end(order, "EXPIRED", "exclusive_deadline", ts)
                getattr(self, "_" + kind)(event)
                equity = self._equity()
                if self.state["cash"] < 0 or equity <= 0:
                    raise ValueError("borrowing/insolvency forbidden")
                self.state["peak"] = max(self.state["peak"], equity)
                dd = 1 - equity / self.state["peak"]
                self.state["max_drawdown"] = max(self.state["max_drawdown"], dd)
                self.state["equity"].append({"ts": ts, "equity_quote": equity, "cash_quote": self.state["cash"], "drawdown": dd})
                self.state["last_clock"] = key
        except (ValueError, KeyError, TypeError, ArithmeticError):
            self.state = before
            raise
        self.events.append(event)

    def _mark(self, event):
        symbol, ts, exchange = event["symbol"], event["ts"], clock(event["exchange_ms"])
        if symbol not in self.rules or exchange > ts or dec(event["price"]) <= 0:
            raise ValueError("invalid/future observed mark")
        if symbol in self.state["marks"] and exchange < self.state["marks"][symbol]["exchange_ms"]:
            raise ValueError("mark exchange-time regression")
        self.state["marks"][symbol] = {"price": dec(event["price"]), "exchange_ms": exchange}

    def _intent(self, event):
        oid, symbol = event["order_id"], event["symbol"]
        if not isinstance(oid, str) or not 0 < len(oid) <= 80 or oid in self.state["orders"] or len(self.state["orders"]) >= self.contract["max_orders"]:
            raise ValueError("order identity/bound")
        binding = self.bindings[event["model_binding_sha256"]]
        if (symbol not in self.rules or event["side"] not in {"buy", "sell"} or dec(event["quantity"]) <= 0
                or not binding["signal_available_ms"] <= event["ts"] < binding["origin_closed_ms"] + HORIZON_MS):
            raise ValueError("invalid/pre-signal intent or expired deployment")
        self.state["orders"][oid] = {"order_id": oid, "symbol": symbol, "side": event["side"], "requested": dec(event["quantity"]),
            "remaining": dec(event["quantity"]), "filled": Decimal(0), "status": "WAIT_REFERENCE", "intent_ms": event["ts"],
            "reference_deadline_ms": event["ts"] + self.contract["reference_wait_ms"], "model_binding_sha256": event["model_binding_sha256"],
            **{k: binding[k] for k in ("model_sha256", "scaler_sha256", "feature_prefix_sha256", "training_identity_sha256")}}
        self._log("intent", event["ts"], order_id=oid, model_sha256=binding["model_sha256"])

    def _book(self, event):
        symbol, ts, exchange, sid = event["symbol"], event["ts"], clock(event["exchange_ms"]), event["snapshot_id"]
        if symbol not in self.rules or exchange > ts or not isinstance(sid, str) or not 0 < len(sid) <= 80:
            raise ValueError("invalid/future book identity")
        parsed = {}
        for side in ("bids", "asks"):
            levels = event[side]
            if not isinstance(levels, list) or not 0 < len(levels) <= self.contract["max_levels"]:
                raise ValueError("bounded nonempty depth required")
            values = [(dec(p), dec(q)) for p, q in levels]
            if any(p <= 0 or q <= 0 or p != round_step(p, self.rules[symbol]["price_tick"]) or q != round_step(q, self.rules[symbol]["quantity_step"]) for p, q in values):
                raise ValueError("nonpositive or off-grid depth")
            prices = [p for p, _ in values]
            if len(set(prices)) != len(prices) or prices != sorted(prices, reverse=side == "bids"):
                raise ValueError("unsorted/duplicate book prices")
            parsed[side] = values
        if parsed["bids"][0][0] >= parsed["asks"][0][0]:
            raise ValueError("locked/crossed snapshot")
        snapshot_key = symbol + ":" + sid
        fingerprint = identity({"exchange_ms": exchange, "symbol": symbol, "bids": event["bids"], "asks": event["asks"]})
        old = self.state["snapshots"].get(snapshot_key)
        if old is not None and old["sha256"] != fingerprint:
            raise ValueError("snapshot identity reused with changed depth")
        if old is None:
            self.state["snapshots"][snapshot_key] = {"sha256": fingerprint, "used": {}}
        if exchange < self.state["last_book_exchange"].get(symbol, -1) or ts - exchange > self.contract["max_quote_age_ms"]:
            self._log("book_ignored", ts, snapshot_id=sid, reason="stale_or_regressed")
            return
        self.state["last_book_exchange"][symbol] = exchange
        for order in self.state["orders"].values():
            if order["symbol"] != symbol or order["status"] not in OPEN:
                continue
            if order["status"] == "WAIT_REFERENCE":
                if exchange < order["intent_ms"]:
                    continue
                if not self._valid_evidence(symbol, exchange, exchange, ts):
                    self._end(order, "REJECTED", "rule_or_fee_interval", ts)
                    continue
                reference = parsed["asks" if order["side"] == "buy" else "bids"][0][0]
                protection = dec(self.contract["protection_bps"]) / 10000
                limit = reference * (1 + protection if order["side"] == "buy" else 1 - protection)
                normalized = normalize_order(order["side"], str(order["requested"]), str(limit), self.rules[symbol])
                order.update(quantity=dec(normalized["quantity"]), remaining=dec(normalized["quantity"]), limit=dec(normalized["limit_price"]),
                             reference_snapshot_id=sid, reference_exchange_ms=exchange, reference_received_ms=ts,
                             submitted_ms=ts, arrival_ms=ts + self.contract["order_latency_ms"], expires_ms=ts + self.contract["order_ttl_ms"])
                reason = normalized["reason"]
                if reason is None and order["quantity"] * order["limit"] > dec(self.contract["max_order_notional_quote"]):
                    reason = "max_order_notional"
                if reason is None and order["side"] == "sell":
                    reserved_sells = sum((o["remaining"] for o in self.state["orders"].values() if o is not order and o["symbol"] == symbol and o["side"] == "sell" and o["status"] == "SUBMITTED"), Decimal(0))
                    if order["quantity"] + reserved_sells > self.state["positions"][symbol]:
                        reason = "reduce_only_position_reservation"
                reason = reason or self._risk_reason(order, order["quantity"], order["limit"], ts)
                if reason:
                    self._end(order, "REJECTED", reason, ts)
                    continue
                order["status"] = "SUBMITTED"
                self._log("submitted", ts, order_id=order["order_id"], quantity=order["quantity"], limit_price=order["limit"], arrival_ms=order["arrival_ms"])
                # Even zero latency cannot use the same snapshot twice as submission/fill evidence.
                continue
            if exchange < order["arrival_ms"] or ts < order["arrival_ms"] or sid == order["reference_snapshot_id"]:
                continue
            if not self._valid_evidence(symbol, order["reference_exchange_ms"], exchange, ts):
                self._end(order, "REJECTED", "rule_or_fee_interval", ts)
                continue
            stop_reason = "no_executable_liquidity_or_limit"
            for price, offered in parsed["asks" if order["side"] == "buy" else "bids"]:
                if (order["side"] == "buy" and price > order["limit"]) or (order["side"] == "sell" and price < order["limit"]):
                    break
                key = order["side"] + ":" + str(price)
                used = self.state["snapshots"][snapshot_key]["used"].get(key, Decimal(0))
                pool = round_step(offered * dec(self.contract["participation"]), self.rules[symbol]["quantity_step"])
                quantity = round_step(min(order["remaining"], max(Decimal(0), pool - used)), self.rules[symbol]["quantity_step"])
                if quantity == 0:
                    continue
                reason = self._risk_reason(order, quantity, price, ts)
                if reason:
                    stop_reason = reason
                    break
                fee = quantity * price * dec(self.fees["taker_fee"])
                delta = quantity * price + fee if order["side"] == "buy" else -(quantity * price - fee)
                self.state["cash"] -= delta
                self.state["positions"][symbol] += quantity if order["side"] == "buy" else -quantity
                self.state["fees"] += fee
                order["remaining"] -= quantity
                order["filled"] += quantity
                self.state["snapshots"][snapshot_key]["used"][key] = used + quantity
                self._log("fill", ts, order_id=order["order_id"], symbol=symbol, side=order["side"], quantity=quantity,
                          price=price, fee_quote=fee, snapshot_id=sid, exchange_ms=exchange,
                          model_sha256=order["model_sha256"], scaler_sha256=order["scaler_sha256"],
                          feature_prefix_sha256=order["feature_prefix_sha256"], training_identity_sha256=order["training_identity_sha256"],
                          actual_fill_verified=False)
                if order["remaining"] == 0:
                    break
            self._end(order, "FILLED" if order["remaining"] == 0 else "PARTIAL_CANCELED" if order["filled"] else "UNFILLED_CANCELED",
                      "requested_quantity_filled" if order["remaining"] == 0 else "one_snapshot_IOC_" + stop_reason + "_remainder_canceled", ts)

    def _funding(self, event):
        symbol, ts = event["symbol"], event["ts"]
        if symbol not in self.rules or event["settlement_ms"] != ts or clock(event["known_at_ms"]) > ts:
            raise ValueError("late/unknown funding settlement cannot be applied to current inventory")
        rate, mark = dec(event["signed_rate"]), dec(event["mark_price"])
        if not -1 < rate < 1 or mark <= 0:
            raise ValueError("invalid signed funding rate or observed mark")
        digest(event["source_sha256"])
        # Settlement evidence also supplies the contemporaneous observed mark.
        # Do not measure a fictitious DD using yesterday's cached mark after a
        # current-price funding debit, before the same-clock mark event arrives.
        self.state["marks"][symbol] = {"price": mark, "exchange_ms": ts}
        debit = self.state["positions"][symbol] * mark * rate
        if debit > self.state["cash"]:
            raise ValueError("insufficient funding cash; no implicit borrowing")
        for order in reversed(list(self.state["orders"].values())):
            if self._reserved() <= self.state["cash"] - debit:
                break
            if order["status"] == "SUBMITTED" and order["side"] == "buy":
                self._end(order, "CANCELED", "funding_cash_reservation", ts)
        self.state["cash"] -= debit
        self.state["funding"] += debit
        self._log("funding", ts, symbol=symbol, signed_debit_quote=debit, source_sha256=event["source_sha256"])

    def _cancel(self, event):
        order = self.state["orders"][event["order_id"]]
        if order["status"] in OPEN:
            self._end(order, "CANCELED", "explicit_cancel", event["ts"])

    def _clock(self, event):
        pass  # Expiration advances only through explicit event clocks, never wall-clock waits.

    def result(self):
        with localcontext() as context:
            context.prec = 50
            body = plain({"schema": "compact_ioc_result/1", "binding_sha256": self.binding_sha,
                "event_prefix_sha256": identity(self.events), "event_count": len(self.events),
                "cash_quote": self.state["cash"], "positions": self.state["positions"], "equity_quote": self._equity(),
                "fees_quote": self.state["fees"], "signed_funding_quote": self.state["funding"],
                "max_drawdown": self.state["max_drawdown"], "orders": self.state["orders"], "ledger": self.state["ledger"],
                "equity": self.state["equity"], "actual_fill_verified": False, "financial_strategy_qualified": False, "safety": SAFETY})
        return body | {"sha256": identity(body)}

    def checkpoint(self):
        body = {"schema": "compact_ioc_checkpoint/1", "binding_sha256": self.binding_sha,
                "events": copy.deepcopy(self.events), "result_sha256": self.result()["sha256"]}
        return body | {"sha256": identity(body)}

    @classmethod
    def restore(cls, checkpoint, **kwargs):
        if checkpoint["sha256"] != identity({k: v for k, v in checkpoint.items() if k != "sha256"}):
            raise ValueError("checkpoint integrity failure")
        engine = cls(**kwargs)
        if engine.binding_sha != checkpoint["binding_sha256"] or len(checkpoint["events"]) > engine.contract["max_events"]:
            raise ValueError("checkpoint contract/source/bounds changed")
        for event in checkpoint["events"]:
            engine.ingest(event)
        if engine.result()["sha256"] != checkpoint["result_sha256"]:
            raise ValueError("checkpoint replay did not reconcile")
        return engine
