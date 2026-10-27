"""Bounded, deterministic, OFFLINE linear-perpetual IOC execution replay.

No sockets, clock reads, sleeps, pandas, OHLCV fills, credentials or order APIs.
Books must be real observations for empirical work; fixtures stay synthetic.
Not a venue matching engine, margin/liquidation simulator or live adapter.
"""
from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR, localcontext

if __package__:
    from .release_contract_v1 import is_sha, object_sha
else:  # Standalone offline checker avoids importing historical package code.
    from release_contract_v1 import is_sha, object_sha

ZERO = Decimal("0")
ONE = Decimal("1")
PRIORITY = {"funding": 0, "mark": 1, "decision": 2, "intent": 3,
            "book": 4, "cancel": 5, "clock": 6}
OPEN = "PENDING"


def dec(value: object) -> Decimal:
    if not isinstance(value, (str, Decimal)):
        raise ValueError("decimal strings required; binary floats forbidden")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("invalid decimal") from exc
    # Internal divisions retain 50-digit precision before rounding to lot size.
    # Only external strings have the stricter 24-decimal-place input bound.
    exponent_floor = -100 if isinstance(value, Decimal) else -24
    if not result.is_finite() or result.adjusted() > 24 or result.as_tuple().exponent < exponent_floor:
        raise ValueError("nonfinite or excessive decimal precision")
    return result


def plain(value: object) -> object:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(name + " must be a nonnegative integer")
    return value


def ident(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 160:
        raise ValueError("bounded nonempty identifier required")
    return value


def round_step(value: str | Decimal, step: str | Decimal, *, up: bool = False) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 50
        v, s = dec(value), dec(step)
        if v < 0 or s <= 0:
            raise ValueError("nonnegative value and positive step required")
        return (v / s).to_integral_value(rounding=ROUND_CEILING if up else ROUND_FLOOR) * s


def normalize_order(side: str, quantity: str, limit_price: str, rule: dict) -> dict:
    """Never round through the caller's protection price or increase size."""
    if side not in ("buy", "sell"):
        raise ValueError("invalid side")
    q = round_step(quantity, rule["quantity_step"])
    p = round_step(limit_price, rule["price_tick"], up=side == "sell")
    with localcontext() as ctx:
        ctx.prec = 50
        reason = None
        if p <= 0 or q <= 0:
            reason = "rounded_to_zero"
        elif q < dec(rule["minimum_quantity"]):
            reason = "below_minimum_quantity"
        elif q * p < dec(rule["minimum_notional"]):
            reason = "below_minimum_notional"
    return {"quantity": format(q, "f"), "limit_price": format(p, "f"), "reason": reason}


def overlay_target(features: list, proposal: int, model: dict) -> int:
    """Causal RAW features -> train-only scaler -> frozen continuous overlay.

    Computing features/proposal still belongs to the pinned research source.
    Missing/nonfinite features abstain; malformed model identity is an error.
    """
    if type(proposal) is not int or proposal not in (-1, 0, 1):
        raise ValueError("invalid proposal")
    if len(features) != 4:
        raise ValueError("four features required")
    if model["policy"]["genome"] != [0, 0, 1, 1, -1]:
        raise ValueError("unsupported genome")
    try:
        values = [dec(v) for v in features]
    except ValueError:
        return 0
    with localcontext() as ctx:
        ctx.prec = 50
        med = [dec(v) for v in model["scaler"]["median"]]
        scale = [dec(v) for v in model["scaler"]["scale"]]
        if len(med) != 4 or len(scale) != 4 or any(v <= 0 for v in scale):
            raise ValueError("invalid scaler")
        x = [max(Decimal("-8"), min(Decimal("8"), (v - m) / s))
             for v, m, s in zip(values, med, scale)]
        return proposal if x[2] + x[3] - ONE > 0 else 0


def transition_order(*, target: int, current_quantity: str, sleeve_equity: str,
                     buy_limit: str, sell_limit: str, taker_fee: str, rule: dict) -> dict | None:
    """Sizing bridge: retain same-side actual quantity; flat closes reduce-only.

    Caller records every decision, including hold/abstain. New notional reserves
    entry fees. Opposite targets use one net-delta IOC; partial reversals are real
    residual positions, NOT assumed completed flips. No automatic retry/top-up.
    """
    if type(target) is not int or target not in (-1, 0, 1):
        raise ValueError("invalid target")
    with localcontext() as ctx:
        ctx.prec = 50
        q, eq, fee = dec(current_quantity), dec(sleeve_equity), dec(taker_fee)
        if eq <= 0 or fee < 0 or fee >= 1:
            raise ValueError("invalid sizing budget")
        if (q > 0 and target == 1) or (q < 0 and target == -1) or (q == 0 and target == 0):
            return None
        side = "buy" if target > 0 or (target == 0 and q < 0) else "sell"
        limit = buy_limit if side == "buy" else sell_limit
        p = dec(limit)
        if p <= 0:
            raise ValueError("positive protection price required")
        # On a reversal reserve the closing fee as well as the new entry fee.
        budget = max(ZERO, eq - abs(q) * p * fee)
        desired = ZERO if target == 0 else round_step(budget / (p * (ONE + fee)),
                                                     rule["quantity_step"]) * target
        normalized = normalize_order(side, format(abs(desired - q), "f"), limit, rule)
        return {"side": side, **normalized, "reduce_only": target == 0}


class Replay:
    """Transactional events, explicit ordering, bounded memory and no I/O.

    At equal timestamps: funding -> mark -> decision -> intent -> book -> cancel
    -> clock. Inputs in another order raise rather than silently use future data.
    `ts` is local receive time, book/mark `exchange_ms` cannot exceed it.
    Funding is supplied at settlement time; late funding requires a corrected
    replay, never charging today's position for yesterday's settlement.
    """

    def __init__(self, *, settings: dict, rules: dict, models: list[dict]):
        required = {"contract_sha256", "dataset_sha256", "fee_schedule_sha256",
                    "evidence_kind", "initial_cash", "taker_fee", "latency_ms",
                    "max_quote_age_ms", "max_mark_age_ms", "order_ttl_ms",
                    "participation", "max_order_notional", "max_gross_ratio",
                    "max_loss", "max_drawdown", "max_events", "max_levels",
                    "fee_valid_from_ms", "fee_valid_until_exclusive_ms"}
        if set(settings) != required:
            raise ValueError("explicit replay settings required")
        for name in ("contract_sha256", "dataset_sha256", "fee_schedule_sha256"):
            if not is_sha(settings[name]):
                raise ValueError("missing identity: " + name)
        if settings["evidence_kind"] not in ("synthetic", "observed_unqualified"):
            raise ValueError("replay cannot emit qualified evidence")
        for name in ("latency_ms", "max_quote_age_ms", "max_mark_age_ms", "order_ttl_ms",
                     "max_events", "max_levels"):
            integer(settings[name], name)
        if not 0 < settings["max_events"] <= 10000 or not 0 < settings["max_levels"] <= 200:
            raise ValueError("replay bounds exceeded")
        if settings["order_ttl_ms"] <= settings["latency_ms"]:
            raise ValueError("TTL must exceed latency")
        if integer(settings["fee_valid_from_ms"], "fee_valid_from_ms") >= integer(
                settings["fee_valid_until_exclusive_ms"], "fee_valid_until_exclusive_ms"):
            raise ValueError("explicit fee schedule validity interval required")
        for name in ("initial_cash", "max_order_notional", "max_loss"):
            if dec(settings[name]) <= 0:
                raise ValueError("positive budget required")
        for name in ("participation", "max_gross_ratio", "max_drawdown"):
            if not ZERO < dec(settings[name]) <= ONE:
                raise ValueError("fraction outside (0,1]")
        if dec(settings["max_drawdown"]) == ONE or not ZERO <= dec(settings["taker_fee"]) < ONE:
            raise ValueError("invalid fee/drawdown")
        if not rules or len(rules) > 8:
            raise ValueError("one to eight explicit instrument rules required")
        for symbol, rule in rules.items():
            ident(symbol)
            if set(rule) != {"price_tick", "quantity_step", "minimum_quantity",
                             "minimum_notional", "source_sha256", "valid_from_ms", "valid_until_exclusive_ms"}:
                raise ValueError("complete rule snapshot required")
            if not is_sha(rule["source_sha256"]) or any(dec(rule[k]) <= 0 for k in (
                    "price_tick", "quantity_step", "minimum_quantity", "minimum_notional")):
                raise ValueError("invalid rule snapshot")
            if integer(rule["valid_from_ms"], "valid_from_ms") >= integer(rule["valid_until_exclusive_ms"], "valid_until_exclusive_ms"):
                raise ValueError("explicit instrument validity interval required")
        self.settings, self.rules = copy.deepcopy(settings), copy.deepcopy(rules)
        self.models = {}
        intervals = []
        for signed in models:
            body, sha = signed["model"], signed["model_sha256"]
            if not is_sha(sha) or object_sha(body) != sha or sha in self.models:
                raise ValueError("invalid model identity")
            if body["contract_sha256"] != settings["contract_sha256"]:
                raise ValueError("model/contract mismatch")
            start = integer(body["origin_start_ms"], "origin_start_ms")
            end = integer(body["origin_end_exclusive_ms"], "origin_end_exclusive_ms")
            if (start % 14400000 or end - start != 259200000 or
                    body["train_end_exclusive_ms"] != start - 7 * 14400000 or
                    body["scaler"]["train_end_exclusive_ms"] != body["train_end_exclusive_ms"]):
                raise ValueError("model temporal leakage")
            if body["policy"]["genome"] != [0, 0, 1, 1, -1]:
                raise ValueError("unsupported model policy")
            if not is_sha(body["provenance"]["data_sha256"]):
                raise ValueError("model data identity missing")
            self.models[sha] = copy.deepcopy(body)
            intervals.append((start, end))
        if not intervals or any(b[0] < a[1] for a, b in zip(sorted(intervals), sorted(intervals)[1:])):
            raise ValueError("missing or overlapping model origins")
        self.binding = object_sha({"settings": settings, "rules": rules, "models": models})
        cash = dec(settings["initial_cash"])
        self.state = {"cash": cash, "positions": {s: {"quantity": ZERO, "entry": ZERO} for s in rules},
                      "marks": {}, "orders": {}, "decisions": {}, "ledger": [],
                      "peak": cash, "max_drawdown": ZERO, "risk_latched": False, "risk_reasons": [],
                      "last_key": [-1, -1], "book_ids": {}, "last_book_exchange": {}, "funding_ids": {}}
        self.events: list[dict] = []
        self.seen: dict[str, str] = {}

    def ingest(self, event: dict) -> None:
        """Reject malformed events without partial ledger/state mutation."""
        event = copy.deepcopy(event)
        eid = ident(event.get("id"))
        digest = object_sha(event)
        if eid in self.seen:
            if self.seen[eid] != digest:
                raise ValueError("event ID payload conflict")
            return
        if len(self.events) >= self.settings["max_events"]:
            raise ValueError("event budget exhausted; explicit bounded segment required")
        kind, ts = event.get("kind"), integer(event.get("ts"), "ts")
        if kind not in PRIORITY:
            raise ValueError("unsupported event kind")
        key = [ts, PRIORITY[kind]]
        if key < self.state["last_key"]:
            raise ValueError("out-of-order event; never infer past positions")
        trial = copy.copy(self)
        trial.state = copy.deepcopy(self.state)
        with localcontext() as ctx:
            ctx.prec = 50
            trial._advance(ts)
            getattr(trial, "_" + kind)(event)
            trial.state["last_key"] = key
        self.state = trial.state
        self.events.append(event)
        self.seen[eid] = digest

    def _row(self, kind: str, ts: int, **values) -> None:
        self.state["ledger"].append({"kind": kind, "ts": ts,
                                     "execution_binding_sha256": self.binding, **plain(values)})

    def _symbol(self, e: dict) -> str:
        if e.get("symbol") not in self.rules:
            raise ValueError("unknown instrument")
        return e["symbol"]

    def _advance(self, ts: int) -> None:
        for order in self.state["orders"].values():
            if order["status"] == OPEN and ts >= order["expiry_ms"]:
                self._finish(order, "EXPIRED", "ttl", ts)

    def _finish(self, order: dict, status: str, reason: str, ts: int) -> None:
        order["status"], order["reason"] = status, reason
        self._row("order_end", ts, client_oid=order["client_oid"], status=status,
                  reason=reason, unfilled_quantity=order["remaining"],
                  model_sha256=order["model_sha256"])

    def _equity(self) -> Decimal:
        total = self.state["cash"]
        for s, pos in self.state["positions"].items():
            if pos["quantity"]:
                total += pos["quantity"] * (self.state["marks"][s]["price"] - pos["entry"])
        return total

    def _gross(self) -> Decimal:
        return sum((abs(p["quantity"]) * self.state["marks"][s]["price"]
                    for s, p in self.state["positions"].items() if p["quantity"]), ZERO)

    def _fresh_marks(self, ts: int, symbol: str) -> bool:
        needed = {symbol} | {s for s, p in self.state["positions"].items() if p["quantity"]}
        return all(s in self.state["marks"] and
                   ts - self.state["marks"][s]["exchange_ms"] <= self.settings["max_mark_age_ms"]
                   for s in needed)

    def _risk(self, ts: int) -> None:
        eq = self._equity()
        self.state["peak"] = max(self.state["peak"], eq)
        dd = max(ZERO, (self.state["peak"] - eq) / self.state["peak"])
        self.state["max_drawdown"] = max(self.state["max_drawdown"], dd)
        breached = []
        if eq <= 0:
            breached.append("nonpositive_equity")
        if dec(self.settings["initial_cash"]) - eq > dec(self.settings["max_loss"]):
            breached.append("maximum_loss")
        if dd > dec(self.settings["max_drawdown"]):
            breached.append("maximum_drawdown")
        if self._gross() > max(ZERO, eq) * dec(self.settings["max_gross_ratio"]):
            breached.append("gross_exposure_budget")
        self.state["risk_reasons"] = sorted(set(self.state["risk_reasons"] + breached))
        self.state["risk_latched"] = bool(self.state["risk_reasons"])
        if self.state["risk_latched"]:
            for order in self.state["orders"].values():
                if order["status"] == OPEN and not order["reduce_only"]:
                    self._finish(order, "CANCELLED", "risk_latched", ts)

    def _mark(self, e: dict) -> None:
        s, ts = self._symbol(e), e["ts"]
        exchange = integer(e.get("exchange_ms"), "exchange_ms")
        price = dec(e.get("price"))
        if price <= 0 or exchange > ts:
            raise ValueError("invalid/future mark")
        if s in self.state["marks"] and exchange < self.state["marks"][s]["exchange_ms"]:
            raise ValueError("mark time regressed")
        self.state["marks"][s] = {"price": price, "exchange_ms": exchange}
        self._risk(ts)
        self._row("mark", ts, symbol=s, price=price, equity=self._equity(),
                  gross_notional=self._gross(), risk_latched=self.state["risk_latched"])

    def _decision(self, e: dict) -> None:
        s, ts, sha = self._symbol(e), e["ts"], e.get("model_sha256")
        if sha not in self.models:
            raise ValueError("unbound model")
        model = self.models[sha]
        if not model["origin_start_ms"] <= ts < model["origin_end_exclusive_ms"]:
            raise ValueError("model outside deployment origin")
        cutoff = integer(e.get("feature_cutoff_ms"), "feature_cutoff_ms")
        if cutoff > ts or cutoff % 14400000 or ts - cutoff >= 14400000:
            raise ValueError("features must end at an already closed 4h boundary")
        target = overlay_target(e.get("features", []), e.get("proposal"), model)
        decision = {"id": e["id"], "ts": ts, "symbol": s, "target": target,
                    "proposal": e["proposal"], "feature_values_sha256": object_sha(e["features"]),
                    "model_sha256": sha, "feature_cutoff_ms": cutoff,
                    "contract_sha256": self.settings["contract_sha256"],
                    "data_sha256": model["provenance"]["data_sha256"],
                    "execution_dataset_sha256": self.settings["dataset_sha256"]}
        decision["execution_binding_sha256"] = self.binding
        self.state["decisions"][e["id"]] = decision
        self._row("decision", ts, **{k: v for k, v in decision.items() if k != "ts"})

    def _intent(self, e: dict) -> None:
        s, ts, oid = self._symbol(e), e["ts"], ident(e.get("client_oid"))
        did = e.get("decision_id")
        if did not in self.state["decisions"]:
            raise ValueError("intent has no recorded decision")
        d = self.state["decisions"][did]
        if d["symbol"] != s or ts < d["ts"] or e.get("model_sha256") != d["model_sha256"]:
            raise ValueError("intent/decision identity mismatch")
        body = {k: v for k, v in e.items() if k not in ("id", "ts")}
        identity = object_sha(body)
        if oid in self.state["orders"]:
            if self.state["orders"][oid]["intent_sha256"] != identity:
                raise ValueError("client order ID payload conflict")
            return
        if type(e.get("reduce_only")) is not bool:
            raise ValueError("explicit reduce_only required")
        norm = normalize_order(e.get("side"), e.get("quantity"), e.get("limit_price"), self.rules[s])
        q, price = dec(norm["quantity"]), dec(norm["limit_price"])
        order = {"client_oid": oid, "intent_sha256": identity, "symbol": s,
                 "side": e["side"], "quantity": q, "remaining": q, "filled": ZERO,
                 "fill_notional": ZERO, "limit_price": price, "reduce_only": e["reduce_only"],
                 "model_sha256": d["model_sha256"], "decision_id": did,
                 "arrival_ms": ts + self.settings["latency_ms"],
                 "expiry_ms": min(ts + self.settings["order_ttl_ms"],
                                  self.models[d["model_sha256"]]["origin_end_exclusive_ms"],
                                  self.settings["fee_valid_until_exclusive_ms"],
                                  self.rules[s]["valid_until_exclusive_ms"]),
                 "status": OPEN, "reason": None}
        reason = norm["reason"]
        signed = ONE if order["side"] == "buy" else -ONE
        pos = self.state["positions"][s]["quantity"]
        if not order["reduce_only"] and d["target"] != int(signed):
            reason = "intent_contradicts_overlay_target"
        elif not (self.rules[s]["valid_from_ms"] <= ts < self.rules[s]["valid_until_exclusive_ms"] and
                  self.settings["fee_valid_from_ms"] <= ts < self.settings["fee_valid_until_exclusive_ms"]):
            reason = "rules_or_fees_outside_validity"
        elif order["reduce_only"] and (pos == 0 or pos * signed >= 0):
            reason = "reduce_only_would_not_reduce"
        elif ts >= order["expiry_ms"] or ts - d["ts"] >= self.settings["order_ttl_ms"]:
            reason = "stale_decision"
        elif any(o["status"] == OPEN and o["symbol"] == s for o in self.state["orders"].values()):
            reason = "pending_order_requires_reconciliation"
        elif not self._fresh_marks(ts, s):
            reason = "missing_or_stale_mark"
        elif self.state["risk_latched"] and not order["reduce_only"]:
            reason = "risk_latched"
        elif q * price > dec(self.settings["max_order_notional"]):
            reason = "maximum_order_notional"
        self.state["orders"][oid] = order
        self._row("order_accepted" if reason is None else "order_rejected", ts,
                  client_oid=oid, decision_id=did, symbol=s, side=order["side"],
                  quantity=q, limit_price=price, model_sha256=d["model_sha256"])
        if reason:
            self._finish(order, "REJECTED", reason, ts)

    def _book(self, e: dict) -> None:
        s, ts = self._symbol(e), e["ts"]
        exchange, sid = integer(e.get("exchange_ms"), "exchange_ms"), ident(e.get("snapshot_id"))
        if exchange > ts:
            raise ValueError("future book")
        parsed = {}
        for side in ("bids", "asks"):
            levels = e.get(side)
            if not isinstance(levels, list) or len(levels) > self.settings["max_levels"]:
                raise ValueError("invalid/bound-exceeding depth")
            values = []
            for level in levels:
                if not isinstance(level, list) or len(level) != 2:
                    raise ValueError("price/quantity level required")
                p, q = map(dec, level)
                if p <= 0 or q <= 0 or p != round_step(p, self.rules[s]["price_tick"]) or q != round_step(q, self.rules[s]["quantity_step"]):
                    raise ValueError("invalid or off-grid book level")
                values.append([p, q])
            if any((a[0] <= b[0] if side == "bids" else a[0] >= b[0]) for a, b in zip(values, values[1:])):
                raise ValueError("depth must be strictly ordered")
            parsed[side] = values
        if parsed["bids"] and parsed["asks"] and parsed["bids"][0][0] >= parsed["asks"][0][0]:
            raise ValueError("crossed book")
        book_key = s + ":" + sid
        digest = object_sha({k: v for k, v in e.items() if k not in ("id", "ts")})
        if book_key in self.state["book_ids"]:
            if self.state["book_ids"][book_key] != digest:
                raise ValueError("snapshot ID payload conflict")
            return  # Never replenish the same observed snapshot on redelivery.
        self.state["book_ids"][book_key] = digest
        if exchange < self.state["last_book_exchange"].get(s, -1):
            self._row("book_ignored", ts, symbol=s, reason="regressed_book_time")
            return
        self.state["last_book_exchange"][s] = exchange
        if ts - exchange > self.settings["max_quote_age_ms"]:
            self._row("book_ignored", ts, symbol=s, reason="stale_book")
            return
        for levels in parsed.values():
            for level in levels:
                level[1] = round_step(level[1] * dec(self.settings["participation"]), self.rules[s]["quantity_step"])
        for order in self.state["orders"].values():
            if order["status"] != OPEN or order["symbol"] != s or ts < order["arrival_ms"]:
                continue
            if exchange < order["arrival_ms"]:
                continue  # A delayed pre-arrival observation is not executable liquidity.
            if not self._fresh_marks(ts, s):
                self._finish(order, "CANCELLED", "missing_or_stale_mark", ts)
                continue
            signed = ONE if order["side"] == "buy" else -ONE
            for level in parsed["asks" if signed > 0 else "bids"]:
                p, available = level
                if (signed > 0 and p > order["limit_price"]) or (signed < 0 and p < order["limit_price"]):
                    break
                q = min(order["remaining"], available)
                if order["reduce_only"]:
                    pos = self.state["positions"][s]["quantity"]
                    q = min(q, abs(pos)) if pos * signed < 0 else ZERO
                if q <= 0:
                    continue
                reason = self._fill_risk(order, q, p)
                if reason:
                    self._finish(order, "CANCELLED", reason, ts)
                    break
                self._fill(order, q, p, ts, sid, exchange)
                level[1] -= q
                if order["status"] != OPEN or order["remaining"] == 0:
                    break
            if order["status"] == OPEN:
                self._finish(order, "FILLED" if order["remaining"] == 0 else "CANCELLED",
                             "complete" if order["remaining"] == 0 else "ioc_residual", ts)

    def _fill_risk(self, order: dict, q: Decimal, price: Decimal) -> str | None:
        if order["fill_notional"] + q * price > dec(self.settings["max_order_notional"]):
            return "maximum_order_notional"
        if order["reduce_only"]:
            return None  # Exits are allowed under a risk latch; never reverse.
        if self.state["risk_latched"]:
            return "risk_latched"
        s = order["symbol"]
        mark = self.state["marks"][s]["price"]
        signed_q = q if order["side"] == "buy" else -q
        before = self.state["positions"][s]["quantity"]
        after_equity = self._equity() + signed_q * (mark - price) - q * price * dec(self.settings["taker_fee"])
        after_gross = self._gross() + (abs(before + signed_q) - abs(before)) * mark
        if after_equity <= 0 or after_gross > after_equity * dec(self.settings["max_gross_ratio"]):
            return "gross_exposure_budget"
        if dec(self.settings["initial_cash"]) - after_equity > dec(self.settings["max_loss"]):
            return "projected_maximum_loss"
        if (self.state["peak"] - after_equity) / self.state["peak"] > dec(self.settings["max_drawdown"]):
            return "projected_maximum_drawdown"
        return None

    def _fill(self, order: dict, q: Decimal, price: Decimal, ts: int, sid: str, exchange: int) -> None:
        s = order["symbol"]
        pos = self.state["positions"][s]
        old, entry = pos["quantity"], pos["entry"]
        delta = q if order["side"] == "buy" else -q
        new, realized = old + delta, ZERO
        if old == 0 or old * delta > 0:
            pos["entry"] = (abs(old) * entry + q * price) / abs(new)
        else:
            realized = min(abs(old), q) * (price - entry) * (ONE if old > 0 else -ONE)
            if new == 0:
                pos["entry"] = ZERO
            elif new * old < 0:
                pos["entry"] = price
        pos["quantity"] = new
        fee = q * price * dec(self.settings["taker_fee"])
        self.state["cash"] += realized - fee
        order["remaining"] -= q
        order["filled"] += q
        order["fill_notional"] += q * price
        self._row("fill", ts, client_oid=order["client_oid"], decision_id=order["decision_id"],
                  symbol=s, signed_quantity=delta, price=price, notional=q * price, fee=fee,
                  realized_pnl=realized, wallet_cash=self.state["cash"], equity=self._equity(),
                  position_quantity=new, snapshot_id=sid, exchange_ms=exchange,
                  model_sha256=order["model_sha256"], contract_sha256=self.settings["contract_sha256"])
        # A fully filled order stays FILLED even if that fill trips a loss latch.
        if order["remaining"] == 0:
            self._finish(order, "FILLED", "complete", ts)
        self._risk(ts)

    def _funding(self, e: dict) -> None:
        s, ts = self._symbol(e), e["ts"]
        if integer(e.get("settlement_ms"), "settlement_ms") != ts:
            raise ValueError("late funding requires chronological corrected replay")
        rate, mark = dec(e.get("rate")), dec(e.get("settlement_mark"))
        if abs(rate) >= 1 or mark <= 0:
            raise ValueError("invalid funding settlement")
        key = s + ":" + str(ts)
        digest = object_sha({"rate": str(rate), "mark": str(mark)})
        if key in self.state["funding_ids"]:
            if self.state["funding_ids"][key] != digest:
                raise ValueError("funding settlement conflict")
            return
        self.state["funding_ids"][key] = digest
        q = self.state["positions"][s]["quantity"]
        payment = -q * mark * rate
        self.state["cash"] += payment
        self.state["marks"][s] = {"price": mark, "exchange_ms": ts}
        self._row("funding", ts, symbol=s, quantity_at_settlement=q, payment=payment,
                  settlement_mark=mark, rate=rate, wallet_cash=self.state["cash"])
        self._risk(ts)

    def _cancel(self, e: dict) -> None:
        oid = ident(e.get("client_oid"))
        if oid not in self.state["orders"]:
            raise ValueError("unknown order cancellation")
        order = self.state["orders"][oid]
        if order["status"] == OPEN:
            self._finish(order, "CANCELLED", "explicit_cancel", e["ts"])

    def _clock(self, e: dict) -> None:
        pass  # Only deterministic expiry in _advance; no wall clock.

    def result(self) -> dict:
        with localcontext() as ctx:
            ctx.prec = 50
            equity = self._equity()
        unresolved = [s for s, p in self.state["positions"].items() if p["quantity"]]
        pending = [oid for oid, o in self.state["orders"].items() if o["status"] == OPEN]
        now = self.state["last_key"][0]
        stale = [s for s in unresolved if now - self.state["marks"][s]["exchange_ms"] > self.settings["max_mark_age_ms"]]
        state = plain(self.state)
        return {"schema_version": "exia.genetic_primus.execution_replay/1",
                "binding_sha256": self.binding, "state_sha256": object_sha(state),
                "ledger_sha256": object_sha(state["ledger"]), "event_count": len(self.events),
                "wallet_cash": str(self.state["cash"]), "marked_equity": str(equity),
                "max_drawdown": str(self.state["max_drawdown"]),
                "risk_latched": self.state["risk_latched"], "unresolved_positions": unresolved,
                "risk_reasons": list(self.state["risk_reasons"]),
                "pending_orders": pending, "stale_marks": stale,
                "flat_and_reconciled": not (unresolved or pending or stale),
                "evidence_kind": self.settings["evidence_kind"],
                "execution_scope": "intent_level_diagnostic",
                "strategy_backtest_comparable": False,
                "limitations": ["snapshot_matching_not_venue_queue_simulation",
                                "latency_fees_rules_and_participation_are_supplied_not_calibrated",
                                "funding_feed_completeness_and_authenticity_not_verified",
                                "no_margin_liquidation_or_live_acknowledgement_adapter",
                                "no_automatic_retry_topup_or_terminal_forced_fill"],
                "qualification_eligible": False, "trading_authorized": False, "state": state}

    def checkpoint(self) -> dict:
        body = {"schema_version": "exia.genetic_primus.execution_checkpoint/1",
                "binding_sha256": self.binding, "events": copy.deepcopy(self.events),
                "state_sha256": self.result()["state_sha256"]}
        return {**body, "checkpoint_sha256": object_sha(body)}

    @classmethod
    def restore(cls, checkpoint: dict, *, settings: dict, rules: dict, models: list[dict]) -> "Replay":
        body = {k: v for k, v in checkpoint.items() if k != "checkpoint_sha256"}
        if body.get("schema_version") != "exia.genetic_primus.execution_checkpoint/1" or object_sha(body) != checkpoint.get("checkpoint_sha256"):
            raise ValueError("checkpoint integrity failure")
        result = cls(settings=settings, rules=rules, models=models)
        if result.binding != body.get("binding_sha256"):
            raise ValueError("checkpoint settings/rules/model mismatch")
        if not isinstance(body.get("events"), list) or len(body["events"]) > settings["max_events"]:
            raise ValueError("checkpoint event budget exceeded")
        for event in body["events"]:
            result.ingest(event)
        if result.result()["state_sha256"] != body.get("state_sha256"):
            raise ValueError("checkpoint replay state mismatch")
        return result
