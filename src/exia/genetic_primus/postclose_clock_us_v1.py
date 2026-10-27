"""Price-free microsecond clock diagnostics, never real availability/fill proof."""
from copy import deepcopy
import re

from .market_timebase_fragment_v1 import SAFETY, seal, verify
from .market_fragment_v1 import object_sha
from .postclose_label_plan_v1 import validate_contract as validate_parent

OBS_KEYS = {"snapshot_id", "exchange_us", "received_us", "available_us",
            "source_sha256", "snapshot_sha256"}
TIME_KEYS = ("bar", "deployment_horizon", "retrain_cadence", "signal_compute_latency",
             "order_latency", "reference_wait", "order_ttl", "max_quote_age")
BASE_FLAGS = {"actual_fill_verified": False, "strategy_availability_verified": False,
              "financial_replay_ready": False, "real_market_coverage_verified": False,
              "safety": SAFETY}


def epoch_us(value):
    if type(value) is not int or not 10**15 <= value < 10**16:
        raise ValueError("integer UTC microseconds required")
    return value


def digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid SHA256")
    return value


def lossless_ms(value):
    value = epoch_us(value)
    if value % 1000:
        raise ValueError("precision loss: microsecond timestamp is not millisecond aligned")
    return value // 1000


def lift_contract(parent, parent_file_sha256):
    """File hash must independently be checked by the caller; no I/O here."""
    body_sha = validate_parent(parent)
    return {"schema": "compact_postclose_microsecond_clock/1",
            "parent_file_sha256": digest(parent_file_sha256), "parent_body_sha256": body_sha,
            **{key + "_us": parent[key + "_ms"] * 1000 for key in TIME_KEYS},
            "max_observations_per_endpoint": min(256, parent["max_observations_per_endpoint"]),
            "purge_bars": parent["purge_bars"], "train_end_offset_bars": parent["train_end_offset_bars"],
            "tie_policy": "equal_signal_or_arrival_boundary_or_multiple_first_candidates_is_ambiguous",
            "legacy_bridge": "synthetic_only_and_lossless_integer_conversion",
            "recorded_availability": "unknown_never_inferred_from_provider_receipt",
            "resolver_scope": "synthetic_clock_fixtures_only",
            **deepcopy(BASE_FLAGS)}


def validate_contract(contract):
    if (not isinstance(contract, dict) or contract.get("schema") != "compact_postclose_microsecond_clock/1"
            or any(contract.get(key) is not False for key in BASE_FLAGS if key != "safety")
            or contract.get("safety") != SAFETY
            or any(value is not False for value in contract.get("safety", {}).values())
            or contract.get("resolver_scope") != "synthetic_clock_fixtures_only"
            or contract.get("recorded_availability") != "unknown_never_inferred_from_provider_receipt"
            or contract.get("legacy_bridge") != "synthetic_only_and_lossless_integer_conversion"
            or contract.get("tie_policy") != "equal_signal_or_arrival_boundary_or_multiple_first_candidates_is_ambiguous"):
        raise ValueError("fail-closed synthetic-only clock contract required")
    digest(contract.get("parent_file_sha256"))
    digest(contract.get("parent_body_sha256"))
    expected = {"bar_us": 14_400_000_000, "deployment_horizon_us": 259_200_000_000,
                "retrain_cadence_us": 259_200_000_000, "purge_bars": 7, "train_end_offset_bars": 1}
    if any(type(contract.get(key)) is not int or contract[key] != value for key, value in expected.items()):
        raise ValueError("registered three-day/seven-bar clock required")
    for key in TIME_KEYS[3:]:
        value = contract.get(key + "_us")
        if type(value) is not int or not 0 <= value <= 60_000_000 or value % 1000:
            raise ValueError("parent-derived bounded integer timing required")
    if (contract["reference_wait_us"] == 0 or contract["max_quote_age_us"] == 0
            or contract["order_ttl_us"] <= contract["order_latency_us"]
            or type(contract.get("max_observations_per_endpoint")) is not int
            or not 1 <= contract["max_observations_per_endpoint"] <= 256):
        raise ValueError("invalid timing/observation bounds")
    return object_sha(contract)


def window_requests_us(decision_us, feature_available_us, contract):
    contract_sha = validate_contract(contract)
    decision, available = epoch_us(decision_us), epoch_us(feature_available_us)
    if decision % contract["bar_us"] or available < decision:
        raise ValueError("bar-aligned close and causal feature availability required")
    exit_decision = epoch_us(decision + contract["deployment_horizon_us"])
    entry_ready = epoch_us(available + contract["signal_compute_latency_us"])
    exit_ready = epoch_us(exit_decision + contract["signal_compute_latency_us"])
    if entry_ready >= exit_ready:
        raise ValueError("feature delay consumes deployment horizon")
    def window(ready):
        return {"signal_available_us": ready,
                "coverage_start_us": epoch_us(ready - contract["max_quote_age_us"]),
                "coverage_end_us_exclusive": epoch_us(ready + contract["reference_wait_us"] + contract["order_ttl_us"])}
    return seal({"decision_us": decision, "feature_available_us": available,
                 "scheduled_exit_decision_us": exit_decision, "entry": window(entry_ready),
                 "exit": window(exit_ready), "contract_sha256": contract_sha, **BASE_FLAGS})


def adapt_recorded_books(events):
    """Preserve exact provider clocks/provenance, with available_us explicitly NULL."""
    if not isinstance(events, list) or not 1 <= len(events) <= 256:
        raise ValueError("bounded nonempty book prefix required")
    observations, links, previous = [], [], None
    for event in events:
        verify(event)
        if event.get("kind") != "engineering_book_fragment" or event.get("symbol") != "BTCUSDT":
            raise ValueError("only registered recorded BTC book fragments accepted")
        if event.get("strategy_availability_verified") is not False:
            raise ValueError("self-declared real availability not accepted")
        exchange, received = epoch_us(event["exchange_ts_us"]), epoch_us(event["received_ts_us"])
        if exchange > received or (previous is not None and received <= previous):
            raise ValueError("invalid/ambiguous provider book chronology")
        previous = received
        source = digest(event["source_prefix_sha256"])
        snapshot = digest(event["sha256"])
        snapshot_id = "book:" + snapshot
        row_end = event["source_rows_end_exclusive"]
        if type(row_end) is not int or not 1 <= row_end <= 4096:
            raise ValueError("invalid bounded source row ordinal")
        observations.append({"snapshot_id": snapshot_id, "exchange_us": exchange,
                             "received_us": received, "available_us": None,
                             "source_sha256": source, "snapshot_sha256": snapshot})
        links.append({"snapshot_id": snapshot_id, "source_observation_sha256": snapshot,
                      "source_message_rows_sha256": digest(event["source_message_rows_sha256"]),
                      "source_rows_end_exclusive": row_end})
    return seal({"schema": "recorded_unknown_availability_clock_adapter/1",
                 "clock_basis": "recorded_unknown", "observations": observations,
                 "provenance_links": links, "missing_strategy_availability_count": len(observations),
                 "provider_receipt_is_strategy_availability": False,
                 "timestamp_precision_loss_us": 0, **BASE_FLAGS})


def _validated_window(ready, window, contract):
    validate_contract(contract)
    ready = epoch_us(ready)
    if not isinstance(window, dict) or set(window) != {"coverage_start_us", "coverage_end_us_exclusive", "observations", "clock_basis"}:
        raise ValueError("price-free bounded clock window only")
    start, end = epoch_us(window["coverage_start_us"]), epoch_us(window["coverage_end_us_exclusive"])
    basis, observations = window["clock_basis"], window["observations"]
    if basis not in ("synthetic_fixture", "recorded_unknown") or start >= end:
        raise ValueError("invalid scope/coverage bounds")
    if not isinstance(observations, list) or len(observations) > contract["max_observations_per_endpoint"]:
        raise ValueError("observation budget exceeded")
    previous, ids, hashes = None, set(), set()
    for observation in observations:
        if not isinstance(observation, dict) or set(observation) != OBS_KEYS:
            raise ValueError("clock metadata only; no price/availability fallback")
        exchange, received = epoch_us(observation["exchange_us"]), epoch_us(observation["received_us"])
        snapshot_id = observation["snapshot_id"]
        if not isinstance(snapshot_id, str) or not 1 <= len(snapshot_id) <= 128 or snapshot_id in ids:
            raise ValueError("invalid/duplicate snapshot identity")
        ids.add(snapshot_id)
        digest(observation["source_sha256"])
        snapshot_sha = digest(observation["snapshot_sha256"])
        if snapshot_sha in hashes:
            raise ValueError("same snapshot evidence aliased as multiple observations")
        hashes.add(snapshot_sha)
        if exchange > received:
            raise ValueError("exchange later than receipt")
        available = observation["available_us"]
        if basis == "recorded_unknown":
            if available is not None:
                raise ValueError("unsupported self-declared recorded availability")
            clock = received
        else:
            clock = epoch_us(available)
            if clock < received:
                raise ValueError("availability before receipt")
        if not start <= clock < end or (previous is not None and clock < previous):
            raise ValueError("out-of-window/regressing observation clock")
        previous = clock
    return ready, start, end, basis, observations


def resolve_endpoint_us(signal_available_us, window, contract):
    ready, start, end, basis, observations = _validated_window(signal_available_us, window, contract)
    result = {"signal_available_us": ready, "window_sha256": object_sha(window),
              "contract_sha256": object_sha(contract), "clock_basis": basis,
              "reference": None, "endpoint": None, **deepcopy(BASE_FLAGS)}
    if basis == "recorded_unknown":
        return seal(result | {"status": "BLOCKED_UNKNOWN_STRATEGY_AVAILABILITY"})
    if not start <= ready < end:
        return seal(result | {"status": "MISSING_SIGNAL_COVERAGE"})
    deadline = epoch_us(ready + contract["reference_wait_us"])
    fresh = lambda o: o["available_us"] - o["exchange_us"] <= contract["max_quote_age_us"]
    candidates = [o for o in observations if ready <= o["available_us"] < deadline
                  and o["exchange_us"] >= ready and fresh(o)]
    if not candidates:
        return seal(result | {"status": "EXPIRED_NO_REFERENCE" if end >= deadline else "PENDING_REFERENCE"})
    reference = candidates[0]
    if reference["available_us"] == ready or reference["exchange_us"] == ready:
        return seal(result | {"status": "AMBIGUOUS_REFERENCE_BOUNDARY"})
    if sum(o["available_us"] == reference["available_us"] for o in observations) > 1:
        return seal(result | {"status": "AMBIGUOUS_REFERENCE_BATCH"})
    arrival = epoch_us(reference["available_us"] + contract["order_latency_us"])
    expires = epoch_us(reference["available_us"] + contract["order_ttl_us"])
    result.update(reference=deepcopy(reference), arrival_us=arrival, expires_us_exclusive=expires)
    candidates = [o for o in observations if o["snapshot_id"] != reference["snapshot_id"]
                  and o["snapshot_sha256"] != reference["snapshot_sha256"]
                  and arrival <= o["available_us"] < expires and o["exchange_us"] >= arrival and fresh(o)]
    if not candidates:
        return seal(result | {"status": "EXPIRED_NO_POST_ARRIVAL" if end >= expires else "PENDING_POST_ARRIVAL"})
    endpoint = candidates[0]
    if endpoint["available_us"] == arrival or endpoint["exchange_us"] == arrival:
        return seal(result | {"status": "AMBIGUOUS_POST_ARRIVAL_BOUNDARY"})
    if sum(o["available_us"] == endpoint["available_us"] for o in observations) > 1:
        return seal(result | {"status": "AMBIGUOUS_POST_ARRIVAL_BATCH"})
    return seal(result | {"endpoint": deepcopy(endpoint), "status": "CLOCK_RESOLVED_NOT_FILL_QUALIFIED"})


def lossless_legacy_window(signal_available_us, window, contract):
    ready, start, end, basis, observations = _validated_window(signal_available_us, window, contract)
    if basis != "synthetic_fixture":
        raise ValueError("recorded unknown availability cannot enter legacy resolver")
    if resolve_endpoint_us(ready, window, contract)["status"] != "CLOCK_RESOLVED_NOT_FILL_QUALIFIED":
        raise ValueError("unresolved/ambiguous synthetic clock cannot enter legacy resolver")
    # Do not round/floor/ceil even one timestamp or exclusive interval bound.
    translated = [{key: observation[key] for key in ("snapshot_id", "source_sha256", "snapshot_sha256")}
                  | {key.replace("_us", "_ms"): lossless_ms(observation[key])
                     for key in ("exchange_us", "received_us", "available_us")}
                  for observation in observations]
    return {"signal_available_ms": lossless_ms(ready),
            "window": {"coverage_start_ms": lossless_ms(start), "coverage_end_ms_exclusive": lossless_ms(end),
                       "observations": translated},
            "clock_basis": "synthetic_fixture", **deepcopy(BASE_FLAGS)}
