"""Pure metadata joins. No imports of research evaluators, data readers or runtime."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import re
from typing import Any


DAY_MS = 86_400_000
SHA = re.compile(r"[0-9a-f]{64}\Z")
SCHEMA = "exia.genetic_primus.result_context_links/1"
UNITS = {
    "interval_start_ms": "unix_ms", "interval_end_exclusive_ms": "unix_ms",
    "deployment_horizon_ms": "ms", "retrain_cadence_ms": "ms",
    "purge_bars": "bars", "bar_ms": "ms", "train_window_days": "days",
    "round_trip_cost_bps": "bps", "delay_bars": "bars", "accounting": "identifier",
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    # Exact legacy semantic-hash recipe, distinct from file byte hash.
    return sha256(canonical(value)).hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


@dataclass(frozen=True)
class Document:
    path: str
    byte_sha256: str
    size_bytes: int
    payload: dict


def document(path: str, raw: bytes, expected_byte_sha: str) -> Document:
    require(bool(SHA.fullmatch(expected_byte_sha)), "invalid expected byte hash")
    require(sha256(raw).hexdigest() == expected_byte_sha, "document byte hash mismatch")

    def unique(pairs):
        output = {}
        for key, value in pairs:
            require(key not in output, "ambiguous duplicate JSON key")
            output[key] = value
        return output

    payload = json.loads(raw, object_pairs_hook=unique)
    require(isinstance(payload, dict), "metadata object required")
    require(not (isinstance(payload.get("schema"), str) and isinstance(payload.get("schema_version"), str)
                 and payload["schema"] != payload["schema_version"]), "conflicting schemas")
    return Document(path, expected_byte_sha, len(raw), payload)


def schema(doc: Document, expected: str) -> None:
    require(doc.payload.get("schema", doc.payload.get("schema_version")) == expected, "unsupported sidecar schema")


def semantic_identity(payload: dict, field: str) -> str:
    claimed = payload.get(field)
    require(isinstance(claimed, str) and bool(SHA.fullmatch(claimed)), "missing semantic hash")
    require(digest({k: v for k, v in payload.items() if k != field}) == claimed, "semantic hash mismatch")
    return claimed


def bind(source: Document, field: str, target: Document, *, semantic_field: str | None = None) -> dict:
    identity = semantic_identity(target.payload, semantic_field) if semantic_field else target.byte_sha256
    require(source.payload.get(field) == identity, "sidecar binding mismatch: " + field)
    return {"source_path": source.path, "source_byte_sha256": source.byte_sha256,
            "source_field": field, "target_path": target.path, "target_byte_sha256": target.byte_sha256,
            "identity_kind": "semantic" if semantic_field else "bytes",
            "identity_field": semantic_field, "identity_sha256": identity}


def utc_ms(value: Any) -> int:
    require(isinstance(value, str), "UTC timestamp must be a string")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0, "explicit UTC required")
    require(stamp.microsecond % 1000 == 0, "sub-millisecond timestamp unsupported")
    delta = stamp - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return delta.days * DAY_MS + delta.seconds * 1000 + delta.microseconds // 1000


def integer(value: Any, *, positive: bool = False) -> int:
    require(type(value) is int and value >= (1 if positive else 0), "invalid nonnegative integer")
    return value


def number(value: Any) -> float:
    require(type(value) in (int, float), "numeric field required")
    try:
        valid = math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    require(valid, "invalid nonnegative finite number")
    return float(value)


@dataclass(frozen=True)
class Evidence:
    path: str
    byte_sha256: str
    field: str
    rule: str


@dataclass(frozen=True)
class Field:
    name: str
    value: int | float | str | None
    unit: str
    evidence: tuple[Evidence, ...]
    unknown_reason: str | None = None


def resolved(doc: Document, name: str, value: Any, source_field: str, rule: str = "explicit_declared_field") -> Field:
    require(name in UNITS, "unknown context field")
    if name == "round_trip_cost_bps":
        value = number(value)
    elif name != "accounting":
        value = integer(value)
    else:
        require(isinstance(value, str) and bool(value), "invalid accounting identifier")
    return Field(name, value, UNITS[name], (Evidence(doc.path, doc.byte_sha256, source_field, rule),))


def complete(fields: list[Field], *, fixed_baseline: bool = False) -> list[dict]:
    require(len({f.name for f in fields}) == len(fields), "duplicate context field")
    known = {f.name: f for f in fields}
    for name in UNITS:
        if name not in known:
            reason = "not_applicable_fixed_baseline" if fixed_baseline and name == "retrain_cadence_ms" else "not_proven_by_bound_sidecars"
            known[name] = Field(name, None, UNITS[name], (), reason)
    return [asdict(known[name]) for name in UNITS]


def evaluation_link(envelope: dict, evaluation: dict, fields: list[Field], *, fixed_baseline=False) -> dict:
    require(evaluation in envelope["evaluations"], "foreign evaluation")
    # Never override an existing known value with contradictory joined metadata.
    for field in fields:
        old = evaluation["context"].get(field.name)
        require(old is None or old == field.value, "joined context contradicts original catalog")
    return {"source_path": envelope["source"]["path"],
            "source_byte_sha256": envelope["source"]["byte_sha256"],
            "base_envelope_sha256": envelope["envelope_sha256"],
            "policy": evaluation["policy"], "scenario": evaluation["scenario"],
            "base_metrics_semantic_sha256": digest(evaluation["metrics"]),
            "fields": complete(fields, fixed_baseline=fixed_baseline),
            "comparison_authority": False, "independent_validation_verified": False}


def catalog_document(doc: Document, envelope: dict) -> None:
    source = envelope["source"]
    require(source["path"] == doc.path and source["byte_sha256"] == doc.byte_sha256
            and source["size_bytes"] == doc.size_bytes, "document/catalog identity mismatch")


def hash_claims(payload: dict, prefix: str, verified=()) -> list[dict]:
    output = []
    for key, value in sorted(payload.items()):
        if key.endswith("sha256") and isinstance(value, str) and SHA.fullmatch(value):
            output.append({"field": prefix + key, "sha256": value,
                           "verification": "semantic_payload_digest_verified" if key in verified else "reported_only"})
        elif key in ("sources_sha256", "preparation_implementation_sha256") and isinstance(value, dict):
            output.extend({"field": prefix + key + "." + role, "sha256": claimed, "verification": "reported_only"}
                          for role, claimed in sorted(value.items())
                          if isinstance(claimed, str) and SHA.fullmatch(claimed))
    return output


def reversal_family(final: Document, registration: Document, pilot: Document, model_draft: Document,
                    steps: list[Document], envelopes: dict[str, dict]) -> dict:
    catalog_document(final, envelopes[final.path])
    schema(final, "exia.genetic_primus.reversal_calibration_result/1")
    semantic_identity(final.payload, "result_sha256")
    schema(registration, "exia.genetic_primus.reversal_calibration_registration/1")
    schema(pilot, "exia.genetic_primus.reversal_calibration_pilot/1")
    schema(model_draft, "exia.genetic_primus.research_design_draft/1")
    proofs = [bind(final, "registration_sha256", registration, semantic_field="registration_sha256"),
              bind(registration, "pilot_draft_sha256", pilot), bind(registration, "model_draft_sha256", model_draft),
              bind(pilot, "model_draft_sha256", model_draft)]
    require(registration.payload["pilot_draft_path"] == pilot.path
            and registration.payload["model_draft_path"] == model_draft.path, "registration path mismatch")
    r, p, rolling = registration.payload, pilot.payload, model_draft.payload["rolling"]
    folds = r["folds"]
    require(isinstance(folds, list) and len(folds) == integer(r["origin_count"], positive=True) == len(steps), "origin coverage mismatch")
    require(digest(folds) == r["folds_sha256"] == p["folds_sha256"], "fold bank semantic hash mismatch")
    horizon = integer(rolling["deployment_horizon_days"], positive=True) * DAY_MS
    cadence = integer(rolling["retrain_cadence_days"], positive=True) * DAY_MS
    require(horizon == cadence == integer(p["origin_days"], positive=True) * DAY_MS, "horizon/cadence mismatch")
    purge = integer(p["purge_bars"], positive=True)
    require(purge == rolling["purge_bars"], "purge mismatch")
    start, end = integer(r["start_ms"]), integer(r["end_ms"])
    require(start < end and utc_ms(p["pilot_start_utc"]) == start and utc_ms(p["pilot_end_exclusive_utc"]) == end,
            "pilot interval mismatch")
    models, links, previous_sha, previous_state = [], [], None, None
    matched_decisions = 0
    for index, (fold, step) in enumerate(zip(folds, steps), 1):
        catalog_document(step, envelopes[step.path])
        value = step.payload
        step_sha = semantic_identity(value, "step_sha256")
        require(value["index"] == index and value["registration_sha256"] == r["registration_sha256"], "step registration/index mismatch")
        require(value["previous_step_sha256"] == previous_sha and value["previous_state_sha256"] == previous_state,
                "broken step/state chain")
        result = value["result"]
        require(result["schema"] == "exia.genetic_primus.reversal_rolling_step/1", "wrong origin schema")
        origin, stop = integer(fold["origin_ms"]), integer(fold["deployment_end_exclusive_ms"])
        require(origin == start + (index - 1) * cadence and stop == origin + horizon
                and result["origin_ms"] == origin and result["state"]["next_origin_ms"] == stop
                and value["input_end_ms"] == stop, "origin deployment interval mismatch")
        state_sha = semantic_identity(result["state"], "state_sha256")
        model = result["model"]
        model_sha = semantic_identity(model, "model_sha256")
        cutoff = integer(model["train_cutoff_exclusive_ms"])
        require(model["purge_bars"] == purge and cutoff == fold["train_cutoff_ms"] and cutoff < origin, "model cutoff mismatch")
        # Record cutoff; do not infer market bar duration from its numerical gap.
        decisions = result["decisions"]
        require(isinstance(decisions, list), "decision ledger required")
        for decision in decisions:
            require(decision["model_sha256"] == model_sha and decision["train_cutoff"] == cutoff,
                    "decision/model lineage mismatch")
            for key in ("folds_sha256", "weights_sha256", "scaler_sha256"):
                require(decision[key] == model[key], "decision identity mismatch: " + key)
        matched_decisions += len(decisions)
        models.append({"source_path": step.path, "source_byte_sha256": step.byte_sha256, "origin_ms": origin,
                       "model_sha256": model_sha, "model_payload_digest_verified": True,
                       "train_cutoff_exclusive_ms": cutoff, "decision_model_hashes_matched": len(decisions),
                       "hash_claims": hash_claims(model, "result.model.", ("model_sha256",))})
        env = envelopes[step.path]
        for evaluation in env["evaluations"]:
            scenario = evaluation["scenario"]
            declared, actual = p["execution_scenarios"][scenario], result["scenarios"][scenario]
            cost, delay = number(declared["round_trip_bps"]), integer(declared["delay_bars"])
            require(cost == actual["round_trip_cost_bps"] and delay == actual["delay_bars"], "origin scenario mismatch")
            fields = _reversal_fields(pilot, model_draft, horizon, cadence, purge, scenario)
            fields += [resolved(step, "interval_start_ms", origin, "result.origin_ms"),
                       resolved(step, "interval_end_exclusive_ms", stop, "result.state.next_origin_ms"),
                       resolved(step, "accounting", "continuous_nav_per_origin", "result.schema", "known_schema_adapter")]
            links.append(evaluation_link(env, evaluation, fields))
        previous_sha, previous_state = step_sha, state_sha
    require(end == folds[-1]["deployment_end_exclusive_ms"] and final.payload["last_step_sha256"] == previous_sha,
            "final chain endpoint mismatch")
    final_env = envelopes[final.path]
    for evaluation in final_env["evaluations"]:
        scenario = evaluation["scenario"]
        # Check all origins, not only the first, before propagating scenario context.
        require(len(final.payload["daily_returns"][scenario]) * DAY_MS == end - start, "final daily coverage mismatch")
        fields = _reversal_fields(pilot, model_draft, horizon, cadence, purge, scenario)
        fields += [resolved(registration, "interval_start_ms", start, "start_ms"),
                   resolved(registration, "interval_end_exclusive_ms", end, "end_ms", "exclusive_end_verified_by_fold_chain"),
                   resolved(final, "accounting", "continuous_nav_marked_and_hypothetical_terminal", "schema", "known_schema_adapter")]
        links.append(evaluation_link(final_env, evaluation, fields))
    return {"family": "reversal_calibration", "bindings": proofs, "evaluation_links": links,
            "models": models, "decision_model_hashes_matched": matched_decisions,
            "folds_semantic_sha256": r["folds_sha256"], "folds_payload_digest_verified": True,
            "identity_claims": hash_claims(r, "registration.", ("registration_sha256", "folds_sha256")),
            "raw_market_data_hashes_recomputed": False, "comparison_authority": False}


def _reversal_fields(pilot, draft, horizon, cadence, purge, scenario):
    execution = pilot.payload["execution_scenarios"][scenario]
    return [resolved(draft, "deployment_horizon_ms", horizon, "rolling.deployment_horizon_days", "declared_days_times_86400000"),
            resolved(draft, "retrain_cadence_ms", cadence, "rolling.retrain_cadence_days", "declared_days_and_verified_origin_chain"),
            resolved(pilot, "purge_bars", purge, "purge_bars"),
            resolved(draft, "train_window_days", draft.payload["rolling"]["train_window_days"], "rolling.train_window_days"),
            resolved(pilot, "round_trip_cost_bps", execution["round_trip_bps"], f"execution_scenarios.{scenario}.round_trip_bps"),
            resolved(pilot, "delay_bars", execution["delay_bars"], f"execution_scenarios.{scenario}.delay_bars")]


def linear_family(result: Document, config: Document, contract: Document, envelope: dict) -> dict:
    catalog_document(result, envelope)
    schema(result, "exia.genetic_primus.rolling_linear_matrix_result/1")
    schema(config, "exia.genetic_primus.rolling_linear_matrix/1")
    proofs = [bind(result, "config_sha256", config), bind(result, "source_contract_sha256", contract)]
    require(config.payload["source_contract_path"] == contract.path, "linear contract path mismatch")
    c, data = config.payload, result.payload
    horizon = integer(c["origin_days"], positive=True) * DAY_MS
    count = integer(c["training_origin_count"], positive=True)
    start = utc_ms(c["development_start_utc"])
    stop = start + count * horizon
    bar = integer(c["timeframe_minutes"], positive=True) * 60_000
    purge = integer(c["purge_bars"], positive=True)
    records, models = data["records"], []
    require(isinstance(records, list) and bool(records), "linear records missing")
    for arm, record in enumerate(records):
        origins = record["origin_models"]
        require(len(origins) == count, "linear origin count mismatch")
        for index, origin in enumerate(origins):
            origin_ms = start + index * horizon
            model = origin["model"]
            require(origin["origin_start"] == origin_ms and utc_ms(model["deploy_start_utc"]) == origin_ms,
                    "linear origin grid mismatch")
            model_sha = semantic_identity(model, "model_sha256")
            cutoff = utc_ms(model["train_cutoff_exclusive_utc"])
            require(model["purge_bars"] == purge and cutoff == origin_ms - purge * bar,
                    "linear purge/cutoff mismatch")
            require(integer(model["max_train_label_end_exclusive"]) <= cutoff, "reported label crosses cutoff")
            require(origin["decision"]["model_sha256"] == model_sha, "linear decision model mismatch")
            models.append({"arm": arm, "origin_ms": origin_ms, "model_sha256": model_sha,
                           "model_payload_digest_verified": True, "train_cutoff_exclusive_ms": cutoff,
                           "reported_label_end_respects_cutoff": True,
                           "decision_model_binding_scope": "parent_decision_batch_not_per_decision_field",
                           "decision_count": len(origin["decision"]["decisions"]),
                           "hash_claims": hash_claims(model, f"records[{arm}].origin_models[{index}].model.", ("model_sha256",))})
        require(all(len(e["daily_returns"]) == count * c["origin_days"] for e in record["evaluations"].values()),
                "linear daily coverage mismatch")
    links = []
    for evaluation in envelope["evaluations"]:
        scenario = evaluation["scenario"]
        require(scenario in ("normal", "stress", "gross_zero_cost"), "unknown linear cost scenario")
        cost = 0 if scenario == "gross_zero_cost" else c[scenario + "_round_trip_bps"]
        fields = [resolved(config, "interval_start_ms", start, "development_start_utc", "explicit_UTC_to_unix_ms"),
                  resolved(config, "interval_end_exclusive_ms", stop, "development_start_utc+training_origin_count*origin_days",
                           "derived_from_bound_config_verified_against_all_saved_origin_grids"),
                  resolved(config, "deployment_horizon_ms", horizon, "origin_days", "declared_days_times_86400000"),
                  resolved(config, "purge_bars", purge, "purge_bars"), resolved(config, "bar_ms", bar, "timeframe_minutes", "minutes_times_60000")]
        fields.append(resolved(result, "round_trip_cost_bps", cost, "records[*].evaluations.gross_zero_cost", "known_zero_cost_scenario")
                      if scenario == "gross_zero_cost" else resolved(config, "round_trip_cost_bps", cost, scenario + "_round_trip_bps"))
        fixed = evaluation["policy"] == "EMA_training_reference"
        if not fixed:
            fields.append(resolved(config, "retrain_cadence_ms", horizon, "origin_days", "verified_all_saved_model_origin_grids"))
        links.append(evaluation_link(envelope, evaluation, fields, fixed_baseline=fixed))
    return {"family": "rolling_linear_matrix", "bindings": proofs, "evaluation_links": links, "models": models,
            "identity_claims": hash_claims(data, "result."),
            "raw_market_data_hashes_recomputed": False, "comparison_authority": False}


def baseline_family(result: Document, config: Document, manifest: Document, envelope: dict) -> dict:
    catalog_document(result, envelope)
    schema(result, "exia.genetic_primus.baseline_economics_audit.result/1")
    schema(config, "exia.genetic_primus.baseline_economics_audit/1")
    proofs = [bind(result, "config_sha256", config), bind(result, "source_manifest_sha256", manifest),
              bind(config, "source_manifest_sha256", manifest)]
    c = config.payload
    require(c["source_manifest_path"] == manifest.path, "baseline manifest path mismatch")
    start, stop = utc_ms(c["period_start_utc"]), utc_ms(c["period_end_exclusive_utc"])
    require(start < stop and result.payload["interval_start"] == start and result.payload["interval_end_exclusive"] == stop,
            "baseline interval mismatch")
    links = []
    for evaluation in envelope["evaluations"]:
        scenario = evaluation["scenario"]
        cost = c["costs_round_trip_bps"][scenario]
        links.append(evaluation_link(envelope, evaluation, [
            resolved(config, "interval_start_ms", start, "period_start_utc", "explicit_UTC_to_unix_ms"),
            resolved(config, "interval_end_exclusive_ms", stop, "period_end_exclusive_utc", "explicit_UTC_to_unix_ms"),
            resolved(config, "round_trip_cost_bps", cost, f"costs_round_trip_bps.{scenario}")]))
    return {"family": "baseline_economics", "bindings": proofs, "evaluation_links": links, "models": [],
            "identity_claims": hash_claims(result.payload, "result."),
            "raw_market_data_hashes_recomputed": False, "comparison_authority": False}
