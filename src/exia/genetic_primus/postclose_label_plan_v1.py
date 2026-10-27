"""Bounded price-free post-close clocks. NOT labels, fits or fill qualification.

Selected windows must eventually come from a verified depth/availability
adapter. Clocks alone do not establish liquidity, returns, or trading authority.
"""
from __future__ import annotations

from .unified_measurement_v1 import BAR_MS, HORIZON_MS, SAFETY, identity

OBS_KEYS = {"snapshot_id", "exchange_ms", "received_ms", "available_ms", "source_sha256", "snapshot_sha256"}
COMPONENTS = ("l2", "marks", "funding", "rules", "fees", "closed_features")


def integer(value):
    if type(value) is not int or value < 0:
        raise ValueError("nonnegative integer clock required")
    return value


def digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("lowercase SHA256 required")
    return value


def validate_contract(contract):
    if (contract["schema"] != "compact_postclose_label_plan/1" or contract["safety"] != SAFETY
            or any(contract["safety"].values()) or contract["ohlcv_to_l2_fallback"] is not False
            or contract["old_next_4h_open_labels_compatible"] is not False
            or contract["clock_resolution_is_execution_validation"] is not False):
        raise ValueError("price-free fail-closed engineering contract required")
    if (contract["bar_ms"] != BAR_MS or contract["deployment_horizon_ms"] != HORIZON_MS
            or contract["retrain_cadence_ms"] != HORIZON_MS or contract["purge_bars"] != 7
            or contract["train_end_offset_bars"] != 1 or contract["feature_min_history_bars"] != 49
            or contract["feature_state_requires_full_causal_prefix"] is not True):
        raise ValueError("three-day cadence and seven-bar purge required")
    for key in ("signal_compute_latency_ms", "order_latency_ms", "reference_wait_ms", "order_ttl_ms", "max_quote_age_ms"):
        if integer(contract[key]) > 60_000:
            raise ValueError("engineering timing bound exceeded")
    if (contract["reference_wait_ms"] == 0 or contract["max_quote_age_ms"] == 0
            or contract["order_ttl_ms"] <= contract["order_latency_ms"]
            or not 1 <= integer(contract["max_observations_per_endpoint"]) <= 512):
        raise ValueError("invalid deadline/freshness/observation bounds")
    digest(contract["engine_contract_file_sha256"])
    start, end = integer(contract["validation_start_ms"]), integer(contract["validation_end_ms_exclusive"])
    if start >= end or start % BAR_MS or end % BAR_MS:
        raise ValueError("UTC bar-aligned validation interval required")
    if len(contract["symbols"]) != 8 or len(set(contract["symbols"])) != 8:
        raise ValueError("registered full8 universe required")
    return identity(contract)


def training_context(origin_ms, contract):
    validate_contract(contract)
    origin = integer(origin_ms)
    if origin % BAR_MS or origin < 8 * BAR_MS:
        raise ValueError("bar-aligned mature origin required")
    train_end = origin - BAR_MS
    return {"origin_ms": origin, "train_end_ms": train_end,
            "label_cutoff_ms": train_end - 7 * BAR_MS, "purge_bars": 7}


def window_requests(decision_ms, feature_available_ms, contract):
    target_sha = validate_contract(contract)
    decision, available = integer(decision_ms), integer(feature_available_ms)
    if decision % BAR_MS or available < decision:
        raise ValueError("closed-bar availability cannot precede close")
    entry_ready = available + contract["signal_compute_latency_ms"]
    exit_decision = decision + HORIZON_MS
    exit_ready = exit_decision + contract["signal_compute_latency_ms"]
    if entry_ready >= exit_ready:
        raise ValueError("feature delay consumes the deployment horizon")
    def window(ready):
        return {"signal_available_ms": ready, "coverage_start_ms": max(0, ready - contract["max_quote_age_ms"]),
                "coverage_end_ms_exclusive": ready + contract["reference_wait_ms"] + contract["order_ttl_ms"]}
    return {"decision_ms": decision, "feature_available_ms": available, "scheduled_exit_decision_ms": exit_decision,
            "entry": window(entry_ready), "exit": window(exit_ready), "target_clock_sha256": target_sha}


def resolve_endpoint(signal_available_ms, window, contract):
    validate_contract(contract)
    ready = integer(signal_available_ms)
    start, end = integer(window["coverage_start_ms"]), integer(window["coverage_end_ms_exclusive"])
    observations = window["observations"]
    if start >= end or not isinstance(observations, list) or len(observations) > contract["max_observations_per_endpoint"]:
        raise ValueError("bounded ordered observation window required")
    previous, ids = -1, {}
    for obs in observations:
        if set(obs) != OBS_KEYS:
            raise ValueError("clock metadata only; no OHLCV/price fallback")
        exchange, received, available = (integer(obs[k]) for k in ("exchange_ms", "received_ms", "available_ms"))
        if not exchange <= received <= available or not start <= available < end or available < previous:
            raise ValueError("future/regressed/out-of-window observation")
        previous = available
        digest(obs["source_sha256"])
        digest(obs["snapshot_sha256"])
        if not isinstance(obs["snapshot_id"], str) or not 1 <= len(obs["snapshot_id"]) <= 128:
            raise ValueError("bounded snapshot identity required")
        fingerprint = (exchange, obs["source_sha256"], obs["snapshot_sha256"])
        if obs["snapshot_id"] in ids and ids[obs["snapshot_id"]] != fingerprint:
            raise ValueError("snapshot ID reused with changed evidence")
        ids[obs["snapshot_id"]] = fingerprint
    body = {"signal_available_ms": ready, "window_sha256": identity(window),
            "reference": None, "endpoint": None, "actual_fill_verified": False}
    if ready < start:
        return body | {"status": "MISSING_SIGNAL_COVERAGE"}
    deadline = ready + contract["reference_wait_ms"]
    fresh = lambda o: o["available_ms"] - o["exchange_ms"] <= contract["max_quote_age_ms"]
    reference = next((o for o in observations if ready <= o["available_ms"] < deadline
                      and o["exchange_ms"] >= ready and fresh(o)), None)
    if reference is None:
        return body | {"status": "EXPIRED_NO_REFERENCE" if end >= deadline else "PENDING_REFERENCE"}
    arrival = reference["available_ms"] + contract["order_latency_ms"]
    expires = reference["available_ms"] + contract["order_ttl_ms"]
    body.update(reference=reference, arrival_ms=arrival, expires_ms_exclusive=expires)
    endpoint = next((o for o in observations if o["snapshot_id"] != reference["snapshot_id"]
                     and arrival <= o["available_ms"] < expires and o["exchange_ms"] >= arrival and fresh(o)), None)
    if endpoint is None:
        return body | {"status": "EXPIRED_NO_POST_ARRIVAL" if end >= expires else "PENDING_POST_ARRIVAL"}
    return body | {"endpoint": endpoint, "status": "CLOCK_RESOLVED_NOT_FILL_QUALIFIED"}


def clock_pair(decision_ms, feature_available_ms, entry_window, exit_window, contract):
    request = window_requests(decision_ms, feature_available_ms, contract)
    entry = resolve_endpoint(request["entry"]["signal_available_ms"], entry_window, contract)
    exit_ = resolve_endpoint(request["exit"]["signal_available_ms"], exit_window, contract)
    resolved = all(x["status"] == "CLOCK_RESOLVED_NOT_FILL_QUALIFIED" for x in (entry, exit_))
    body = {"schema": "postclose_clock_pair/1", "request": request, "entry": entry, "exit": exit_,
            "clock_pair_resolved": resolved, "training_ready": False, "execution_qualified": False,
            "status": "CLOCKS_RESOLVED_EXECUTION_LABELS_MISSING" if resolved else "INCOMPLETE_CLOCKS",
            "clock_pair_available_ms": max(entry["endpoint"]["available_ms"], exit_["endpoint"]["available_ms"]) if resolved else None,
            "safety": SAFETY}
    return body | {"sha256": identity(body)}


def dependency_maturity(pair, dependency_available_ms, origin_ms, contract):
    """Time eligibility only; dependencies must later include every cost/fill source."""
    if pair["sha256"] != identity({k: v for k, v in pair.items() if k != "sha256"}):
        raise ValueError("clock pair integrity failure")
    if not pair["clock_pair_resolved"] or pair["request"]["target_clock_sha256"] != validate_contract(contract):
        raise ValueError("complete pair with matching target clock required")
    if not isinstance(dependency_available_ms, list) or not 1 <= len(dependency_available_ms) <= 512:
        raise ValueError("bounded explicit source-availability dependencies required")
    context = training_context(origin_ms, contract)
    available = max([pair["clock_pair_available_ms"]] + [integer(t) for t in dependency_available_ms])
    return context | {"all_declared_dependencies_available_ms": available,
                      "time_eligible": available <= context["label_cutoff_ms"],
                      "training_ready": False, "execution_labels_missing": True}


def bundle_declaration_issues(bundle, contract):
    """Structural preflight only: self-declared completeness is NOT qualification."""
    target_sha = validate_contract(contract)
    if bundle is None:
        return ["registered_execution_bundle_missing"]
    issues = []
    if bundle.get("schema") != "postclose_execution_bundle_declaration/1":
        issues.append("bundle_schema_mismatch")
    if bundle.get("target_clock_sha256") != target_sha:
        issues.append("target_clock_hash_mismatch")
    expected_end = contract["validation_end_ms_exclusive"] + contract["signal_compute_latency_ms"] + contract["reference_wait_ms"] + contract["order_ttl_ms"]
    components = bundle.get("components", {})
    if not isinstance(components, dict):
        raise ValueError("component map required")
    for name in COMPONENTS:
        records = components.get(name, [])
        if not isinstance(records, list) or len(records) > 128:
            raise ValueError("bounded component declarations required")
        for symbol in contract["symbols"]:
            candidates = [r for r in records if r.get("symbol") == symbol]
            if len(candidates) != 1:
                issues.append(name + ":" + symbol + ":missing_or_duplicate")
                continue
            row = candidates[0]
            required_start = contract["validation_start_ms"] - (
                contract["feature_min_history_bars"] * BAR_MS if name == "closed_features" else contract["max_quote_age_ms"])
            if (row.get("evidence_kind") != "observed" or row.get("quality_status") != "independently_verified"
                    or type(row.get("coverage_start_ms")) is not int or type(row.get("coverage_end_ms_exclusive")) is not int
                    or row["coverage_start_ms"] > required_start
                    or row["coverage_end_ms_exclusive"] < expected_end):
                issues.append(name + ":" + symbol + ":unverified_or_incomplete_coverage")
            try:
                digest(row.get("source_manifest_sha256"))
                digest(row.get("quality_receipt_sha256"))
            except ValueError:
                issues.append(name + ":" + symbol + ":missing_source_or_quality_hash")
    return issues
