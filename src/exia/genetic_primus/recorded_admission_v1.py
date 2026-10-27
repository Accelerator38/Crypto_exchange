"""Pure admission checks for bounded normalized recordings; NEVER run orders.

Checks integrity/coverage/declared lineage, not authenticity, feature arithmetic
or venue calibration. Does not read files, fetch feeds, train or run a replay.
"""
from __future__ import annotations

import copy

if __package__:
    from .execution_replay_v1 import Replay, dec, integer, round_step
    from .portfolio_replay_v1 import TF, _checked_events, ema_proposal, validate_portfolio_contract
    from .release_contract_v1 import bind_model, is_sha, object_sha
else:
    from execution_replay_v1 import Replay, dec, integer, round_step
    from portfolio_replay_v1 import TF, _checked_events, ema_proposal, validate_portfolio_contract
    from release_contract_v1 import bind_model, is_sha, object_sha


def expected_times(start: int, end: int, interval: int, anchor: int, *, limit: int = 1024) -> list[int]:
    """Exact half-open grid, bounded BEFORE allocating a list."""
    for value, name in ((start, "start"), (end, "end"), (interval, "interval"), (anchor, "anchor")):
        integer(value, name)
    if start >= end or interval <= 0:
        raise ValueError("invalid_time_grid")
    first = anchor + ((start - anchor + interval - 1) // interval) * interval
    count = max(0, (end - 1 - first) // interval + 1)
    if count > limit:
        raise ValueError("time_grid_budget_exceeded")
    return [first + i * interval for i in range(count)]


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _reference_sources(provenance, bundle, start, end, evidence):
    rules, settings = bundle["rules"], bundle["settings"]
    sources = provenance["rule_sources"]
    _require(set(sources) == set(rules), "rule_source_universe_mismatch")
    fields = ("price_tick", "quantity_step", "minimum_quantity", "minimum_notional", "valid_from_ms", "valid_until_exclusive_ms")
    for s, rule in rules.items():
        source = sources[s]
        _require(source.get("symbol") == s and source.get("venue") == "bitget" and
                 source.get("product") == "USDT-FUTURES" and source.get("evidence_kind") == evidence,
                 "rule_source_identity_mismatch:" + s)
        _require(object_sha(source) == rule["source_sha256"], "rule_source_hash_mismatch:" + s)
        _require(all(source.get(k) == rule[k] for k in fields), "rule_source_values_mismatch:" + s)
        _require(rule["valid_from_ms"] <= start and rule["valid_until_exclusive_ms"] >= end,
                 "rule_coverage_missing:" + s)
    fee = provenance["fee_source"]
    _require(fee.get("venue") == "bitget" and fee.get("product") == "USDT-FUTURES" and
             fee.get("evidence_kind") == evidence, "fee_source_identity_mismatch")
    _require(object_sha(fee) == settings["fee_schedule_sha256"], "fee_source_hash_mismatch")
    _require(fee.get("taker_fee") == settings["taker_fee"] and
             fee.get("valid_from_ms") == settings["fee_valid_from_ms"] and
             fee.get("valid_until_exclusive_ms") == settings["fee_valid_until_exclusive_ms"], "fee_source_values_mismatch")
    _require(fee["valid_from_ms"] <= start and fee["valid_until_exclusive_ms"] >= end, "fee_coverage_missing")


def _model_lineage(bundle, provenance, release, implementation_sha):
    records = provenance["scaler_lineage"]
    _require(isinstance(records, list) and len(records) == len(bundle["models"]), "scaler_lineage_count")
    index = {r["model_sha256"]: r for r in records}
    _require(len(index) == len(records), "duplicate_scaler_lineage")
    for signed in bundle["models"]:
        b, p = signed["model"], signed["model"]["provenance"]
        _require(p["implementation_sha256"] == implementation_sha, "source_model_implementation_mismatch")
        rebuilt = bind_model(release, origin_start_ms=b["origin_start_ms"], scaler=b["scaler"],
                             data_sha256=p["data_sha256"], train_rows_sha256=p["train_rows_sha256"],
                             folds_sha256=p["folds_sha256"], resample_bank_sha256=p["resample_bank_sha256"],
                             audit_bank_sha256=p["audit_bank_sha256"], implementation_sha256=implementation_sha)
        _require(rebuilt == signed, "source_model_manifest_mismatch")
        _require(signed["model_sha256"] in index, "scaler_lineage_missing")
        r = index[signed["model_sha256"]]
        _require(r.get("train_rows_sha256") == p["train_rows_sha256"] and
                 r.get("fit_values_sha256") == p["scaler_fit_values_sha256"] and
                 r.get("training_data_sha256") == p["data_sha256"] and
                 r.get("row_count") == b["scaler"]["fitted_rows"], "scaler_lineage_hash_or_count_mismatch")
        first = integer(r["first_feature_ms"], "first_feature_ms")
        last = integer(r["last_feature_ms"], "last_feature_ms")
        label_end = integer(r["max_label_end_ms"], "max_label_end_ms")
        fitted = integer(r["fitted_at_ms"], "fitted_at_ms")
        _require(b["scaler"]["train_start_ms"] <= first <= last < b["train_end_exclusive_ms"], "scaler_train_cutoff_leakage")
        _require(last <= label_end <= b["origin_start_ms"] and last <= fitted <= b["origin_start_ms"],
                 "scaler_label_or_availability_leakage")


def _feature_lineage(signals, provenance, bundle, release):
    feature = provenance["feature_lineage"]
    _require(object_sha(feature) == bundle["profile"]["feature_dataset_sha256"], "feature_dataset_hash_mismatch")
    _require(feature.get("feature_code_sha256") == release["source_pins"]["src/exia/genetic_primus/measurement_lab_v2.py"],
             "feature_code_identity_mismatch")
    rows = feature["records"]
    _require(isinstance(rows, list) and len(rows) == len(signals), "feature_lineage_count")
    index = {r["signal_id"]: r for r in rows}
    _require(len(index) == len(rows) and set(index) == {s["id"] for s in signals}, "feature_lineage_id_mismatch")
    for s in signals:
        row, ts = index[s["id"]], s["ts"]
        _require(row.get("values_sha256") == object_sha(s["features"]) and row.get("symbol") == s["symbol"],
                 "feature_values_or_symbol_mismatch")
        _require(row.get("feature_cutoff_ms") == ts and is_sha(row.get("source_bars_sha256")), "feature_cutoff_or_source_missing")
        closed = integer(row["max_source_close_ms"], "max_source_close_ms")
        available = integer(row["available_at_ms"], "available_at_ms")
        _require(closed <= available <= ts and ts - closed < TF, "future_or_stale_feature_lineage")


def _observations(events, bundle, start, end, policy):
    symbols = set(bundle["rules"])
    observations = {s: {"mark": [], "book": [], "funding": []} for s in symbols}
    snapshots = set()
    for e in events:
        kind = e["kind"]
        if kind not in ("mark", "book", "funding"):
            continue
        s, ts = e["symbol"], e["ts"]
        if kind == "funding":
            _require(e.get("settlement_ms") == ts and abs(dec(e["rate"])) < 1 and dec(e["settlement_mark"]) > 0,
                     "invalid_funding_event:" + s)
        else:
            exchange = integer(e.get("exchange_ms"), "exchange_ms")
            _require(exchange <= ts, "future_market_observation:" + s)
            seq = observations[s][kind]
            _require(not seq or exchange >= seq[-1]["exchange_ms"], "regressed_market_observation:" + s)
            if kind == "mark":
                _require(dec(e["price"]) > 0, "invalid_mark_price:" + s)
            else:
                _require(isinstance(e.get("snapshot_id"), str) and 0 < len(e["snapshot_id"]) <= 160, "invalid_snapshot_id")
                key = (s, e["snapshot_id"])
                _require(key not in snapshots, "duplicate_snapshot_requires_explicit_dedup:" + s)
                snapshots.add(key)
                sides = {}
                for side in ("bids", "asks"):
                    levels = e[side]
                    _require(isinstance(levels, list) and policy["minimum_book_levels"] <= len(levels) <= bundle["settings"]["max_levels"],
                             "insufficient_or_excessive_depth:" + s)
                    parsed = []
                    for level in levels:
                        _require(isinstance(level, list) and len(level) == 2, "invalid_book_level")
                        p, q = map(dec, level)
                        rule = bundle["rules"][s]
                        _require(p > 0 and q > 0 and p == round_step(p, rule["price_tick"]) and
                                 q == round_step(q, rule["quantity_step"]), "invalid_or_off_grid_depth:" + s)
                        parsed.append(p)
                    _require(all(a > b if side == "bids" else a < b for a, b in zip(parsed, parsed[1:])), "unordered_depth:" + s)
                    sides[side] = parsed
                _require(sides["bids"][0] < sides["asks"][0], "crossed_depth:" + s)
        observations[s][kind].append(e)
    for s in sorted(symbols):
        for kind in ("mark", "book"):
            _require(bool(observations[s][kind]), kind + "_feed_missing:" + s)
            for axis in ("ts", "exchange_ms"):
                times = [e[axis] for e in observations[s][kind]]
                gaps = [b - a for a, b in zip([start] + times, times + [end])]
                _require(max(gaps) <= policy["max_" + kind + "_gap_ms"], kind + "_coverage_gap:" + s)
    return observations


def validate_package(manifest, bundle, provenance, portfolio, release, implementation_sha):
    """Return a read-only admission result, never economic or release approval."""
    validate_portfolio_contract(portfolio, release)
    _require(manifest.get("schema_version") == "exia.genetic_primus.recorded_package/1", "unsupported_package")
    _require(provenance.get("schema_version") == "exia.genetic_primus.recorded_provenance/1", "unsupported_provenance")
    evidence = manifest.get("evidence_kind")
    _require(evidence in ("synthetic", "observed_unqualified"), "unsupported_evidence_kind")
    _require(set(bundle) == {"schema_version", "settings", "rules", "models", "profile", "events", "implementation_sha256"} and
             bundle["schema_version"] == "exia.genetic_primus.portfolio_bundle/1", "unsupported_bundle")
    _require(bundle["implementation_sha256"] == implementation_sha and
             manifest.get("portfolio_contract_sha256") == object_sha(portfolio), "portfolio_binding_mismatch")
    settings = bundle["settings"]
    _require(settings["evidence_kind"] == evidence and settings["contract_sha256"] == object_sha(release), "bundle_identity_mismatch")
    _require(settings["dataset_sha256"] == object_sha({"rules": bundle["rules"], "events": bundle["events"]}), "market_dataset_hash_mismatch")
    symbols = set(release["symbol_map"].values())
    _require(set(bundle["rules"]) == symbols, "full8_rules_required")
    profile = bundle["profile"]
    _require(set(profile) == {"protection_fraction", "max_input_events", "feature_dataset_sha256"}, "explicit_profile_required")
    _require(0 <= dec(profile["protection_fraction"]) < 1 and is_sha(profile["feature_dataset_sha256"]), "invalid_profile")
    _require(0 < integer(profile["max_input_events"], "max_input_events") <= 1024 and 0 < settings["order_ttl_ms"] < TF,
             "invalid_replay_budget_or_ttl")
    events = _checked_events(bundle["events"], symbols, profile["max_input_events"])
    _require(len(events) + sum(e["kind"] == "signal" for e in events) <= settings["max_events"], "insufficient_generated_intent_budget")
    # Constructor validation only: no ingest(), matching, orders or financial run.
    Replay(settings=settings, rules=bundle["rules"], models=bundle["models"])
    for k, v in (("capital_usdt", "initial_cash"), ("maximum_order_notional_usdt", "max_order_notional"),
                 ("maximum_loss_usdt", "max_loss"), ("maximum_drawdown_fraction", "max_drawdown")):
        value = release["risk_budget"][k]
        _require(value is None or dec(value) == dec(settings[v]), "release_risk_budget_mismatch")
    _model_lineage(bundle, provenance, release, implementation_sha)
    start, end = integer(manifest["start_ms"], "start_ms"), integer(manifest["end_exclusive_ms"], "end_exclusive_ms")
    grid = expected_times(start, end, TF, 0)
    _require(grid and all(start <= e["ts"] < end for e in events), "events_outside_capture_interval")
    signals = [e for e in events if e["kind"] == "signal"]
    _require({e["ts"] for e in signals} == set(grid), "missing_whole_4h_decision_boundary")
    _feature_lineage(signals, provenance, bundle, release)
    policy = manifest["coverage_policy"]
    _require(set(policy) == {"max_mark_gap_ms", "max_book_gap_ms", "minimum_book_levels", "max_clock_skew_ms"}, "explicit_coverage_policy_required")
    for k, v in policy.items():
        integer(v, k)
    _require(policy["max_mark_gap_ms"] > 0 and policy["max_book_gap_ms"] > 0 and
             0 < policy["minimum_book_levels"] <= settings["max_levels"], "invalid_coverage_policy")
    capture = provenance["capture"]
    _require(capture.get("venue") == "bitget" and capture.get("product") == "USDT-FUTURES" and
             capture.get("evidence_kind") == evidence and capture.get("timestamp_basis") == "UTC_epoch_ms" and
             is_sha(capture.get("recorder_code_sha256")) and isinstance(capture.get("session_id"), str) and
             0 < len(capture["session_id"]) <= 160, "capture_identity_missing_or_mixed")
    _require(integer(capture["max_clock_skew_ms"], "max_clock_skew_ms") <= policy["max_clock_skew_ms"], "capture_clock_budget_exceeded")
    _reference_sources(provenance, bundle, start, end, evidence)
    observations = _observations(events, bundle, start, end, policy)
    schedules = provenance["funding_schedule"]
    _require(set(schedules) == symbols, "full8_funding_schedule_required")
    for s in sorted(symbols):
        schedule = schedules[s]
        _require(schedule.get("evidence_kind") == evidence and is_sha(schedule.get("source_record_sha256")), "funding_schedule_source_missing")
        _require(integer(schedule["valid_from_ms"], "valid_from_ms") <= start and
                 integer(schedule["valid_until_exclusive_ms"], "valid_until_exclusive_ms") >= end, "funding_schedule_coverage_missing")
        expected = expected_times(start, end, schedule["interval_ms"], schedule["anchor_ms"])
        actual = [e["ts"] for e in observations[s]["funding"]]
        _require(sorted(actual) == expected, "funding_settlement_missing_duplicate_or_unexpected:" + s)
    for e in signals:
        s, ts = e["symbol"], e["ts"]
        origin = [m["model"] for m in bundle["models"] if m["model"]["origin_start_ms"] <= ts < m["model"]["origin_end_exclusive_ms"]]
        _require(len(origin) == 1, "missing_origin_model")
        marks = [m for m in observations[s]["mark"] if m["ts"] <= ts]
        _require(marks and ts - marks[-1]["exchange_ms"] <= settings["max_mark_age_ms"], "decision_mark_missing_or_stale:" + s)
        arrival = ts + settings["latency_ms"]
        expiry = min(ts + settings["order_ttl_ms"], origin[0]["origin_end_exclusive_ms"], end)
        books = [b for b in observations[s]["book"] if arrival <= b["ts"] < expiry and
                 arrival <= b["exchange_ms"] and b["ts"] - b["exchange_ms"] <= settings["max_quote_age_ms"]]
        _require(bool(books), "post_latency_book_coverage_missing:" + s)
    body = {"schema_version": "exia.genetic_primus.recorded_admission/1",
            "admitted_for_offline_replay": True, "qualification_eligible": False, "trading_authorized": False,
            "evidence_kind": evidence, "manifest_sha256": object_sha(manifest), "bundle_sha256": object_sha(bundle),
            "provenance_sha256": object_sha(provenance), "unique_events": len(events), "decision_boundaries": len(grid),
            "common_opportunities": sum(ema_proposal(e["features"]) != 0 for e in signals),
            "funding_settlements": sum(len(v["funding"]) for v in observations.values()),
            "source_authenticity_verified": False, "feature_computation_verified": False,
            "scaler_recomputed": False, "calibration_verified": False, "performance_gate_evaluation": "NOT_RUN",
            "limitations": ["lineage_timestamps_are_claims_cross_checked_not_raw_history_recomputed",
                            "coverage_thresholds_and_funding_schedule_are_supplied_not_venue_verified",
                            "admission_is_not_evidence_of_strategy_profitability"]}
    return {**body, "admission_body_sha256": object_sha(body)}


def admitted_bundle(manifest, bundle, provenance, portfolio, release, implementation_sha):
    report = validate_package(manifest, bundle, provenance, portfolio, release, implementation_sha)
    return copy.deepcopy(bundle), report
