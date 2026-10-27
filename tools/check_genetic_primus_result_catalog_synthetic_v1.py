"""Short deterministic pure regressions; no pytest, asyncio, collector or data fit."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import runpy


def main():
    root = Path(__file__).resolve().parents[1]
    api = runpy.run_path(str(root / "src/exia/genetic_primus/result_catalog_v1.py"))
    names, units = api["MetricName"], api["Unit"]
    normalize = api["normalize_metric"]
    adapt, Source = api["adapt"], api["Source"]
    guard, validate = api["comparison_reasons"], api["validate_serialized"]
    checks = 0

    def check(condition):
        nonlocal checks
        assert condition
        checks += 1

    def rejects(call):
        try:
            call()
        except ValueError:
            return True
        return False

    check(normalize(names.NET, 1.25, units.PERCENT, "x").value == 0.0125)
    check(normalize(names.MEAN_TRADE, 125, units.BPS, "x").value == 0.0125)
    check(normalize(names.NET, 0.0125, units.FRACTION, "x").value == 0.0125)
    check(normalize(names.NET, 0, units.PERCENT, "x").value == 0)
    check(normalize(names.NET, None, units.PERCENT, "x").value is None)
    for invalid in (True, "0", float("nan"), float("inf"), 10 ** 1000):
        check(normalize(names.NET, invalid, units.FRACTION, "x").value is None)
    for invalid in (-1, 0.5, False):
        check(normalize(names.TRADES, invalid, units.COUNT, "x").value is None)
    check(normalize(names.DD, 101, units.PERCENT, "x").value is None)
    check(normalize(names.DD_CLOSE, -0.1, units.FRACTION, "x").value is None)
    check(rejects(lambda: normalize(names.NET, 3, units.COUNT, "x")))
    check(rejects(lambda: api["load_json_metadata"](b'{"net_return": 0, "net_return": 1}')))

    dummy_source = Source("synthetic/prospective/result.json", "a" * 64, 1, None)
    check(rejects(lambda: adapt({"schema": "a/1", "schema_version": "b/1"}, dummy_source)))
    unknown = adapt({"mean_bps": 2, "lcb_bps": 1}, dummy_source)
    daily = adapt({"mean_daily_bps": 2, "lcb_daily_bps": 1}, dummy_source)
    check(unknown.evidence_category == "unknown")  # filename is not evidence
    check(api["metric_value"](unknown.evaluations[0], names.TRADES) is None)
    check("metric_missing_or_incompatible_semantics" in guard(
        unknown, unknown.evaluations[0], daily, daily.evaluations[0], names.LCB_DAY))
    check("common_measurement_contract_not_verified" in guard(
        daily, daily.evaluations[0], daily, daily.evaluations[0], names.LCB_DAY))

    for filename, expected in (("scaler.json", "scaler"), ("model_manifest.json", "model_metadata"),
                               ("origin_01_started.json", "checkpoint_metadata"),
                               ("generation_000.json", "search_trace")):
        e = adapt({"net_return": 1}, replace(dummy_source, path="synthetic/" + filename))
        check(e.artifact_kind == expected and not e.evaluations)
    check(adapt({"independent_validation": True}, dummy_source).evidence_category == "claimed_independent_unverified")
    check(adapt({"promotion_authority": True, "orders_enabled": True}, dummy_source).orders_enabled is False)
    malformed = adapt({"policies": []}, replace(dummy_source,
                      reported_schema="exia.genetic_primus.baseline_economics_audit.result/1"))
    check(not malformed.evaluations)
    gate = adapt({}, replace(dummy_source, path="synthetic/promotion_gate.json"))
    check(gate.artifact_kind == "research_gate_decision" and not gate.evaluations)
    august = adapt({"data_already_revealed": True, "costs": {
        "normal": {"candidates": {"NoTrade": {"compounded_return": 0, "max_drawdown": 0, "trades": 0}}}}},
        replace(dummy_source, reported_schema="exia.genetic_primus.revealed_3d_summary/1"))
    check(august.evaluations[0].context.accounting == "geometric_chain_of_closed_origin_windows")
    check(api["metric_value"](august.evaluations[0], names.NET) == 0)
    origin = adapt({"registration_sha256": "b" * 64, "result": {
        "schema": "exia.genetic_primus.reversal_rolling_step/1", "origin_ms": 0,
        "evidence_class": "revealed_development", "state": {"next_origin_ms": 259200000},
        "model": {"model_sha256": "c" * 64},
        "scenarios": {"stress": {"origin_net_return": -0.01, "round_trip_cost_bps": 40, "delay_bars": 1}}}},
        replace(dummy_source, path="synthetic/origin_01.json"))
    check(origin.artifact_kind == "origin_step_result")
    check(origin.evaluations[0].context.interval_end_exclusive_ms == 259200000)
    check(origin.evaluations[0].context.delay_bars == 1)
    check(api["metric_value"](origin.evaluations[0], names.TRADES) is None)
    check(("result.model.model_sha256", "c" * 64) in origin.reported_hash_claims)
    check(origin.evidence_category == "reported_development_or_revealed")

    # Read only one already-audited JSON; no market parquet/series execution.
    relative = "Reports/Exia/Genetic_Primus/baseline_economics_audit_v1_20261002/result.json"
    raw = (root / relative).read_bytes()
    check(sha256(raw).hexdigest() == api["AUDITED_BATCH_SHA"])
    data = json.loads(raw)
    audited = adapt(data, Source(relative, sha256(raw).hexdigest(), len(raw), data["schema_version"]))
    normal = [e for e in audited.evaluations if e.scenario == "normal"]
    stress = [e for e in audited.evaluations if e.scenario == "stress"]
    check(len(normal) == len(stress) == 5)
    check(not guard(audited, normal[0], audited, normal[1], names.NET))
    check("cost_scenario_mismatch" in guard(audited, normal[0], audited, stress[0], names.NET))
    for field, changed in (("interval_start_ms", 0), ("round_trip_cost_bps", 99),
                           ("accounting", "other"), ("fingerprints", (("market_rows_sha256", "b" * 64),))):
        altered = replace(normal[1], context=replace(normal[1].context, **{field: changed}))
        changed_envelope = replace(audited, evaluations=(altered,))
        check(bool(guard(audited, normal[0], changed_envelope, altered, names.NET)))
    foreign = replace(audited, source=replace(audited.source, byte_sha256="c" * 64))
    check("common_measurement_contract_not_verified" in guard(audited, normal[0], foreign, normal[1], names.NET))
    serialized = json.loads(api["canonical_bytes"](api["envelope_dict"](audited)))
    validate(serialized)
    check(True)
    serialized["evaluations"][0]["metrics"][0]["value"] = 0.999
    check(rejects(lambda: validate(serialized)))
    print(f"catalog regressions: {checks} passed; offline, no training/runtime")


if __name__ == "__main__":
    main()
