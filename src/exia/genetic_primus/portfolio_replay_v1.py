"""Common OFFLINE full8 portfolio controller over frozen execution replay v1.

Equal initial sleeves, no capital transfer, no hindsight orders, no end fill.
GA/EMA/NoTrade are isolated counterfactuals of the SAME observed event stream.
This module does no I/O, training, scheduling, network or runtime integration.
"""
from __future__ import annotations

import copy
from decimal import Decimal, localcontext
from itertools import groupby

if __package__:
    from .execution_replay_v1 import Replay, dec, ident, integer, normalize_order, overlay_target, plain, transition_order
    from .release_contract_v1 import bind_model, is_sha, object_sha, validate_contract
else:
    from execution_replay_v1 import Replay, dec, ident, integer, normalize_order, overlay_target, plain, transition_order
    from release_contract_v1 import bind_model, is_sha, object_sha, validate_contract

ROLES = ("GA", "EMA", "NoTrade")
TF = 14400000
ZERO = Decimal("0")
ONE = Decimal("1")
INPUT_PRIORITY = {"funding": 0, "mark": 1, "signal": 2, "book": 4, "clock": 6}
PORTFOLIO_RULES = {
    "capital": "eight_equal_initial_virtual_sleeves_no_transfers",
    "sizing": "own_sleeve_marked_equity_reserve_fees",
    "same_direction": "hold_actual_filled_quantity_no_topup",
    "zero_target": "reduce_only_exit_even_when_sleeve_equity_nonpositive",
    "reversal": "one_net_delta_IOC_keep_partial_residual",
    "protection_price": "fresh_observed_mark_plus_minus_registered_fraction",
    "origins": "carry_cash_and_positions_no_free_reset",
    "end_of_input": "mark_and_report_residual_never_force_fill",
    "decision_grid": "full8_once_per_closed_4h_boundary",
    "pending": "TTL_strictly_less_than_one_decision_bar",
    "risk_latch": "block_new_risk_reduce_at_next_scheduled_decision",
    "event_order": "funding_marks_all_decisions_all_intents_books_clock",
    "counterfactuals": "independent_liquidity_and_wallet_same_market_inputs",
    "statistics": "descriptive_paired_equity_only_no_promotion_test",
}


def validate_portfolio_contract(contract: dict, release: dict) -> None:
    if not validate_contract(release)["structurally_valid"]:
        raise ValueError("invalid parent release contract")
    if contract.get("schema_version") != "exia.genetic_primus.portfolio_contract/1":
        raise ValueError("unsupported portfolio contract")
    if contract.get("release_contract_sha256") != object_sha(release):
        raise ValueError("parent release contract hash mismatch")
    if contract.get("rules") != PORTFOLIO_RULES or contract.get("roles") != list(ROLES):
        raise ValueError("portfolio rules are not frozen")
    safety = contract.get("safety")
    if not isinstance(safety, dict) or set(safety) != set(release["safety"]) or any(v is not False for v in safety.values()):
        raise ValueError("all portfolio safety permissions must remain false")
    if contract.get("qualifying_evidence") is not False:
        raise ValueError("offline portfolio replay cannot authorize promotion")


def ema_proposal(features: list) -> int:
    """Same raw-feature sign/efficiency rule as pinned measurement_lab_v2.

    No user-supplied per-agent direction. Missing features abstain for everyone.
    Feature extraction itself remains external and must have causal provenance.
    """
    if not isinstance(features, list) or len(features) != 4:
        raise ValueError("four raw features required")
    if any(v is not None and not isinstance(v, str) for v in features):
        raise ValueError("raw features must be decimal strings or explicit null")
    if any(v is None or v in ("NaN", "nan", "Infinity", "+Infinity", "-Infinity") for v in features):
        return 0
    try:
        gap, slope, ret, efficiency = (dec(v) for v in features)
    except ValueError as exc:
        raise ValueError("invalid raw feature decimal; use null for missing observations") from exc
    sign = lambda value: int(value > 0) - int(value < 0)
    direction = sign(gap)
    return direction if efficiency >= Decimal("0.30") and sign(slope) == direction and sign(ret) == direction else 0


def sleeve_order(*, target: int, quantity: str, equity: str, mark: str,
                 protection_fraction: str, taker_fee: str, rule: dict) -> dict | None:
    """Pure sizing; zero/negative collateral must not prevent reducing risk."""
    if type(target) is not int or target not in (-1, 0, 1):
        raise ValueError("invalid target")
    with localcontext() as ctx:
        ctx.prec = 50
        q, eq, price, fraction = dec(quantity), dec(equity), dec(mark), dec(protection_fraction)
        if price <= 0 or not ZERO <= fraction < ONE:
            raise ValueError("invalid protection price")
        buy = price * (ONE + fraction)
        sell = price * (ONE - fraction)
        if target == 0 or eq <= 0:
            if not q:
                return None
            side = "sell" if q > 0 else "buy"
            result = normalize_order(side, abs(q), sell if q > 0 else buy, rule)
            return {"side": side, **result, "reduce_only": True}
        return transition_order(target=target, current_quantity=q, sleeve_equity=eq,
                                buy_limit=buy, sell_limit=sell, taker_fee=taker_fee, rule=rule)


class _StrategyReplay(Replay):
    """Extend the frozen matching kernel without editing its signed source.

    Base constructor validates GA source manifests first. We then install new,
    explicitly role-tagged identities (EMA does not masquerade as a GA genome).
    All controller state is in `state`, preserving transactional ingest.
    """

    def __init__(self, *, settings, rules, models, role, profile, portfolio_sha, implementation_sha):
        super().__init__(settings=settings, rules=rules, models=models)
        if role not in ROLES:
            raise ValueError("unknown candidate")
        self.role = role
        self.profile = copy.deepcopy(profile)
        source_models = self.models
        self.models = {}
        for source_sha, body in source_models.items():
            model = {"schema_version": "exia.genetic_primus.portfolio_policy/1", "role": role,
                     "source_release_model_sha256": source_sha, "portfolio_contract_sha256": portfolio_sha,
                     "implementation_sha256": implementation_sha, "profile_sha256": object_sha(profile),
                     "origin_start_ms": body["origin_start_ms"],
                     "origin_end_exclusive_ms": body["origin_end_exclusive_ms"],
                     "train_end_exclusive_ms": body["train_end_exclusive_ms"],
                     "provenance": copy.deepcopy(body["provenance"]),
                     "policy": copy.deepcopy(body["policy"]) if role == "GA" else
                               {"kind": "raw_EMA_TrendConsensus" if role == "EMA" else "always_flat"},
                     "scaler": copy.deepcopy(body["scaler"]) if role == "GA" else None}
            self.models[object_sha(model)] = model
        self.binding = object_sha({"kernel_binding_sha256": self.binding, "role": role,
                                   "role_models": self.models, "profile": profile,
                                   "portfolio_contract_sha256": portfolio_sha,
                                   "implementation_sha256": implementation_sha})
        with localcontext() as ctx:
            ctx.prec = 50
            initial = dec(settings["initial_cash"]) / Decimal(8)
        self.state["sleeve_cash"] = {s: initial for s in rules}

    def sleeve_equity(self, symbol):
        pos = self.state["positions"][symbol]
        result = self.state["sleeve_cash"][symbol]
        if pos["quantity"]:
            result += pos["quantity"] * (self.state["marks"][symbol]["price"] - pos["entry"])
        return result

    def model_at(self, ts):
        matches = [sha for sha, model in self.models.items()
                   if model["origin_start_ms"] <= ts < model["origin_end_exclusive_ms"]]
        if len(matches) != 1:
            raise ValueError("missing or ambiguous origin model")
        return matches[0]

    def _decision(self, e):
        s, ts = self._symbol(e), e["ts"]
        sha = self.model_at(ts)
        if e.get("model_sha256") != sha or ts % TF or e.get("feature_cutoff_ms") != ts:
            raise ValueError("model/decision/closed-bar cutoff mismatch")
        features = e["features"]
        proposal = ema_proposal(features)
        model = self.models[sha]
        requested = overlay_target(features, proposal, model) if self.role == "GA" else proposal if self.role == "EMA" else 0
        target = 0 if self.state["risk_latched"] or self.sleeve_equity(s) <= 0 else requested
        decision = {"id": e["id"], "ts": ts, "symbol": s, "role": self.role,
                    "proposal": proposal, "requested_target": requested, "target": target,
                    "risk_exit": target != requested, "model_sha256": sha,
                    "feature_cutoff_ms": ts, "feature_values_sha256": object_sha(features),
                    "data_sha256": model["provenance"]["data_sha256"],
                    "execution_dataset_sha256": self.settings["dataset_sha256"],
                    "execution_binding_sha256": self.binding}
        self.state["decisions"][e["id"]] = decision
        self._row("decision", ts, **{k: v for k, v in decision.items() if k != "ts"})

    def _fill_risk(self, order, q, price):
        reason = super()._fill_risk(order, q, price)
        if reason or order["reduce_only"]:
            return reason
        s = order["symbol"]
        pos = self.state["positions"][s]["quantity"]
        mark = self.state["marks"][s]["price"]
        delta = q if order["side"] == "buy" else -q
        after = self.sleeve_equity(s) + delta * (mark - price) - q * price * dec(self.settings["taker_fee"])
        if after <= 0 or abs(pos + delta) * mark > after * dec(self.settings["max_gross_ratio"]):
            return "own_sleeve_exposure_budget"
        return None

    def _fill(self, order, q, price, ts, sid, exchange):
        before = self.state["cash"]
        super()._fill(order, q, price, ts, sid, exchange)
        self.state["sleeve_cash"][order["symbol"]] += self.state["cash"] - before

    def _funding(self, e):
        before = self.state["cash"]
        super()._funding(e)
        self.state["sleeve_cash"][e["symbol"]] += self.state["cash"] - before

    def plan(self, signal):
        """Called only after ALL same-time decisions, before any future book."""
        did, s, ts = signal["id"], signal["symbol"], signal["ts"]
        d = self.state["decisions"][did]
        reason, planned = "hold_or_flat", None
        with localcontext() as ctx:
            ctx.prec = 50
            if not self._fresh_marks(ts, s):
                reason = "missing_or_stale_mark"
            else:
                planned = sleeve_order(target=d["target"], quantity=self.state["positions"][s]["quantity"],
                                       equity=self.sleeve_equity(s), mark=self.state["marks"][s]["price"],
                                       protection_fraction=self.profile["protection_fraction"],
                                       taker_fee=self.settings["taker_fee"], rule=self.rules[s])
                if planned:
                    reason = planned["reason"] or "submit_IOC"
            d["controller_action"] = reason
            self._row("controller", ts, decision_id=did, symbol=s, action=reason,
                      sleeve_cash=self.state["sleeve_cash"][s], sleeve_equity=self.sleeve_equity(s),
                      position_quantity=self.state["positions"][s]["quantity"], model_sha256=d["model_sha256"])
        if planned is None or planned["reason"]:
            return None
        return {"id": "intent:" + did, "kind": "intent", "ts": ts, "symbol": s,
                "client_oid": "order:" + did, "decision_id": did, "model_sha256": d["model_sha256"],
                **{k: v for k, v in planned.items() if k != "reason"}}

    def portfolio_point(self, ts):
        with localcontext() as ctx:
            ctx.prec = 50
            equity = self._equity()
            sleeves = {s: self.sleeve_equity(s) for s in self.rules}
            # Decimal averages can leave sub-ULP residuals; never hide material drift.
            if abs(sum(self.state["sleeve_cash"].values(), ZERO) - self.state["cash"]) > Decimal("1e-36") or abs(sum(sleeves.values(), ZERO) - equity) > Decimal("1e-36"):
                raise ValueError("portfolio/sleeve cash reconciliation failure")
            return plain({"ts": ts, "cash": self.state["cash"], "equity": equity,
                          "sleeve_equity": sleeves, "gross_notional": self._gross(),
                          "risk_latched": self.state["risk_latched"]})

    def checkpoint(self):
        raise ValueError("portfolio restart requires the original immutable bundle, not a kernel checkpoint")

    @classmethod
    def restore(cls, *args, **kwargs):
        raise ValueError("portfolio restart requires replaying the original immutable bundle")


def _checked_events(events, symbols, max_input_events):
    if not isinstance(events, list) or not 0 < len(events) <= max_input_events:
        raise ValueError("explicit bounded nonempty market event list required")
    seen, unique, previous = {}, [], [-1, -1]
    for e in events:
        if not isinstance(e, dict):
            raise ValueError("event object required")
        eid = ident(e.get("id"))
        if len(eid) > 120 or eid.startswith(("intent:", "order:")):
            raise ValueError("reserved or oversized input ID")
        digest = object_sha(e)
        if eid in seen:
            if seen[eid] != digest:
                raise ValueError("input event ID conflict")
            continue
        seen[eid] = digest
        kind, ts = e.get("kind"), integer(e.get("ts"), "ts")
        if kind not in INPUT_PRIORITY:
            raise ValueError("only shared market/signal/clock input; no supplied orders")
        key = [ts, INPUT_PRIORITY[kind]]
        if key < previous:
            raise ValueError("input event ordering regressed")
        previous = key
        if kind != "clock" and e.get("symbol") not in symbols:
            raise ValueError("event outside full8 universe")
        if kind == "signal":
            if set(e) != {"id", "kind", "ts", "symbol", "feature_cutoff_ms", "features"}:
                raise ValueError("shared signal must not contain agent-specific directions or orders")
            if ts % TF or e["feature_cutoff_ms"] != ts:
                raise ValueError("signals require closed 4h boundaries")
            ema_proposal(e["features"])
        unique.append(copy.deepcopy(e))
    for _, group in groupby(unique, key=lambda e: e["ts"]):
        signals = [e for e in group if e["kind"] == "signal"]
        if signals and (len(signals) != 8 or {e["symbol"] for e in signals} != symbols):
            raise ValueError("complete exactly-once full8 signal grid required")
    if not any(e["kind"] == "signal" for e in unique):
        raise ValueError("at least one full8 decision boundary required")
    return unique


def run_comparison(bundle: dict, contract: dict, release: dict, implementation_sha256: str,
                   *, candidate_order=ROLES) -> dict:
    """Pure, bounded counterfactual comparison. Does not select/promote a winner."""
    validate_portfolio_contract(contract, release)
    if set(bundle) != {"schema_version", "settings", "rules", "models", "profile", "events", "implementation_sha256"} or bundle["schema_version"] != "exia.genetic_primus.portfolio_bundle/1":
        raise ValueError("unsupported portfolio bundle")
    if not is_sha(implementation_sha256) or bundle["implementation_sha256"] != implementation_sha256:
        raise ValueError("portfolio implementation hash mismatch")
    if len(candidate_order) != 3 or set(candidate_order) != set(ROLES):
        raise ValueError("all three registered candidates required")
    symbols = set(release["symbol_map"].values())
    settings, profile = bundle["settings"], bundle["profile"]
    if set(bundle["rules"]) != symbols:
        raise ValueError("eight instrument rule snapshots required; no implicit reweighting")
    if set(profile) != {"protection_fraction", "max_input_events", "feature_dataset_sha256"}:
        raise ValueError("explicit frozen execution/feature profile required")
    if not ZERO <= dec(profile["protection_fraction"]) < ONE or not is_sha(profile["feature_dataset_sha256"]):
        raise ValueError("invalid profile")
    if not 0 < integer(profile["max_input_events"], "max_input_events") <= 1024:
        raise ValueError("short portfolio replay limited to 1024 input events")
    if not 0 < integer(settings["order_ttl_ms"], "order_ttl_ms") < TF:
        raise ValueError("TTL must be shorter than the decision bar")
    if settings["contract_sha256"] != object_sha(release):
        raise ValueError("source release identity mismatch")
    if settings["dataset_sha256"] != object_sha({"rules": bundle["rules"], "events": bundle["events"]}):
        raise ValueError("common market/feature dataset hash mismatch")
    for k, v in (("capital_usdt", "initial_cash"), ("maximum_order_notional_usdt", "max_order_notional"),
                 ("maximum_loss_usdt", "max_loss"), ("maximum_drawdown_fraction", "max_drawdown")):
        value = release["risk_budget"][k]
        if value is not None and dec(value) != dec(settings[v]):
            raise ValueError("risk settings differ from declared release budget")
    for signed in bundle["models"]:
        body, p = signed["model"], signed["model"]["provenance"]
        if p["implementation_sha256"] != implementation_sha256:
            raise ValueError("source manifest implementation mismatch")
        rebuilt = bind_model(release, origin_start_ms=body["origin_start_ms"], scaler=body["scaler"],
                             data_sha256=p["data_sha256"], train_rows_sha256=p["train_rows_sha256"],
                             folds_sha256=p["folds_sha256"], resample_bank_sha256=p["resample_bank_sha256"],
                             audit_bank_sha256=p["audit_bank_sha256"], implementation_sha256=implementation_sha256)
        if rebuilt != signed:
            raise ValueError("invalid source model manifest")
    events = _checked_events(bundle["events"], symbols, profile["max_input_events"])
    if len(events) + sum(e["kind"] == "signal" for e in events) > integer(settings["max_events"], "max_events"):
        raise ValueError("kernel event budget must cover shared input plus all possible generated intents")
    agents = {role: _StrategyReplay(settings=settings, rules={s: bundle["rules"][s] for s in sorted(symbols)},
                                   models=bundle["models"], role=role, profile=profile,
                                   portfolio_sha=object_sha(contract), implementation_sha=implementation_sha256)
              for role in candidate_order}
    paths = {role: [] for role in ROLES}
    for ts, items in groupby(events, key=lambda e: e["ts"]):
        group = list(items)
        signals = sorted((e for e in group if e["kind"] == "signal"), key=lambda e: e["symbol"])
        for phase in (("funding", "mark"), ("signal",), ("book", "clock")):
            for e in (signals if phase == ("signal",) else [e for e in group if e["kind"] in phase]):
                for agent in agents.values():
                    if e["kind"] == "signal":
                        agent.ingest({**e, "kind": "decision", "model_sha256": agent.model_at(ts)})
                    else:
                        agent.ingest(e)
            if phase == ("signal",):
                for agent in agents.values():
                    for signal in signals:
                        intent = agent.plan(signal)
                        if intent:
                            agent.ingest(intent)
        for role, agent in agents.items():
            paths[role].append(agent.portfolio_point(ts))
    candidates = {}
    with localcontext() as ctx:
        ctx.prec = 50
        initial = dec(settings["initial_cash"])
        for role in ROLES:
            result = agents[role].result()
            rows = result["state"]["ledger"]
            candidates[role] = {
                "model_manifests": agents[role].models, "execution": result,
                "portfolio_path": paths[role], "portfolio_path_sha256": object_sha(paths[role]),
                "ending_equity": result["marked_equity"],
                "decision_count": len(result["state"]["decisions"]),
                "nonzero_target_decisions": sum(d["target"] != 0 for d in result["state"]["decisions"].values()),
                "marked_return_fraction": str((Decimal(result["marked_equity"]) - initial) / initial),
                "fills": sum(row["kind"] == "fill" for row in rows),
                "fill_notional": str(sum((Decimal(row["notional"]) for row in rows if row["kind"] == "fill"), ZERO)),
                "fees": str(sum((Decimal(row["fee"]) for row in rows if row["kind"] == "fill"), ZERO)),
                "funding": str(sum((Decimal(row["payment"]) for row in rows if row["kind"] == "funding"), ZERO)),
            }
        paired = [{"ts": point["ts"],
                   "GA_minus_EMA_equity": str(Decimal(point["equity"]) - Decimal(paths["EMA"][i]["equity"])),
                   "GA_minus_NoTrade_equity": str(Decimal(point["equity"]) - Decimal(paths["NoTrade"][i]["equity"]))}
                  for i, point in enumerate(paths["GA"])]
    body = {"schema_version": "exia.genetic_primus.portfolio_comparison/1",
            "portfolio_contract_sha256": object_sha(contract), "input_bundle_sha256": object_sha(bundle),
            "implementation_sha256": implementation_sha256, "common_dataset_sha256": settings["dataset_sha256"],
            "feature_dataset_sha256": profile["feature_dataset_sha256"],
            "input_events": len(bundle["events"]), "unique_input_events": len(events),
            "common_nonzero_ema_opportunities": sum(ema_proposal(e["features"]) != 0 for e in events if e["kind"] == "signal"),
            "decision_boundaries": sum(any(e["kind"] == "signal" for e in g) for _, g in groupby(events, key=lambda e: e["ts"])),
            "candidates": candidates, "paired_equity_path": paired,
            "within_replay_candidates_comparable": True, "historical_research_results_comparable": False,
            "evidence_kind": settings["evidence_kind"], "performance_gate_evaluation": "NOT_RUN",
            "qualification_eligible": False, "trading_authorized": False,
            "warnings": ["synthetic_or_uncalibrated_execution_not_performance_evidence",
                         "feature_causality_and_feed_completeness_require_independent_audit",
                         "virtual_sleeves_are_not_exchange_isolated_margin",
                         "no_intrabar_autonomous_risk_exit_or_liquidation_adapter",
                         "event_marks_are_not_independent_statistical_observations"]}
    return {**body, "result_body_sha256": object_sha(body)}
