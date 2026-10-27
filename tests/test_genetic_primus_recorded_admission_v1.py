"""Short synthetic admission tests. No pytest, collector, network or training."""
import copy
import hashlib
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = runpy.run_path(str(ROOT / "tools/import_genetic_primus_recorded_v1.py"))
PORTFOLIO, RELEASE, IDENTITY = API["context"]()
from recorded_admission_v1 import Replay, admitted_bundle, expected_times, validate_package
from release_contract_v1 import object_sha
from portfolio_replay_v1 import TF, run_comparison


def refresh(bundle, provenance):
    bundle["profile"]["feature_dataset_sha256"] = object_sha(provenance["feature_lineage"])
    bundle["settings"]["dataset_sha256"] = object_sha({"rules": bundle["rules"], "events": bundle["events"]})


def fixture():
    bundle = json.loads((ROOT / "Reports/Exia/Genetic_Primus/portfolio_replay_v1_20260910/synthetic_input.json").read_text())
    events, settings = bundle["events"], bundle["settings"]
    start, end = min(e["ts"] for e in events), max(e["ts"] for e in events) + 1
    rules = {}
    for s, r in bundle["rules"].items():
        source = {k: v for k, v in r.items() if k != "source_sha256"}
        source.update(symbol=s, venue="bitget", product="USDT-FUTURES", evidence_kind="synthetic")
        rules[s] = source
        r["source_sha256"] = object_sha(source)
    fee = {"venue": "bitget", "product": "USDT-FUTURES", "evidence_kind": "synthetic",
           "taker_fee": settings["taker_fee"], "valid_from_ms": settings["fee_valid_from_ms"],
           "valid_until_exclusive_ms": settings["fee_valid_until_exclusive_ms"]}
    settings["fee_schedule_sha256"] = object_sha(fee)
    scaler = []
    for m in bundle["models"]:
        body, p = m["model"], m["model"]["provenance"]
        last = body["train_end_exclusive_ms"] - TF
        scaler.append({"model_sha256": m["model_sha256"], "train_rows_sha256": p["train_rows_sha256"],
                       "fit_values_sha256": p["scaler_fit_values_sha256"], "training_data_sha256": p["data_sha256"],
                       "row_count": body["scaler"]["fitted_rows"], "first_feature_ms": last - 15 * TF,
                       "last_feature_ms": last, "max_label_end_ms": last + 7 * TF,
                       "fitted_at_ms": body["origin_start_ms"]})
    features = {"feature_code_sha256": RELEASE["source_pins"]["src/exia/genetic_primus/measurement_lab_v2.py"],
                "records": [{"signal_id": e["id"], "symbol": e["symbol"], "values_sha256": object_sha(e["features"]),
                             "feature_cutoff_ms": e["ts"], "max_source_close_ms": e["ts"], "available_at_ms": e["ts"],
                             "source_bars_sha256": object_sha("SYNTHETIC_NOT_RECOMPUTED_BARS")}
                            for e in events if e["kind"] == "signal"]}
    schedule = {s: {"interval_ms": 86400000, "anchor_ms": start + TF, "valid_from_ms": start,
                    "valid_until_exclusive_ms": end, "evidence_kind": "synthetic",
                    "source_record_sha256": object_sha("SYNTHETIC_FUNDING_SCHEDULE:" + s)} for s in rules}
    provenance = {"schema_version": "exia.genetic_primus.recorded_provenance/1",
                  "capture": {"venue": "bitget", "product": "USDT-FUTURES", "evidence_kind": "synthetic",
                              "timestamp_basis": "UTC_epoch_ms", "session_id": "SYNTHETIC_INTAKE_FIXTURE",
                              "recorder_code_sha256": object_sha("SYNTHETIC_RECORDER"), "max_clock_skew_ms": 0},
                  "feature_lineage": features, "scaler_lineage": scaler, "funding_schedule": schedule,
                  "rule_sources": rules, "fee_source": fee}
    manifest = {"schema_version": "exia.genetic_primus.recorded_package/1", "evidence_kind": "synthetic",
                "portfolio_contract_sha256": object_sha(PORTFOLIO), "start_ms": start, "end_exclusive_ms": end,
                "coverage_policy": {"max_mark_gap_ms": TF + 20, "max_book_gap_ms": TF + 20,
                                    "minimum_book_levels": 1, "max_clock_skew_ms": 0}}
    refresh(bundle, provenance)
    return manifest, bundle, provenance


def check(m, b, p):
    return validate_package(m, b, p, PORTFOLIO, RELEASE, IDENTITY["sha256"])


def fails(fn, message):
    try:
        fn()
    except ValueError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError("expected fail-closed rejection")


def test_expected_grid_is_half_open_and_bounded():
    assert expected_times(0, 20, 10, 0) == [0, 10]
    assert expected_times(1, 20, 10, 0) == [10]
    assert expected_times(1, 9, 10, 0) == []
    fails(lambda: expected_times(0, 1000000, 1, 0), "budget")
    fails(lambda: expected_times(0, 10, 0, 0), "invalid")


def test_valid_package_has_no_qualification_authority():
    r = check(*fixture())
    assert r["admitted_for_offline_replay"] and r["unique_events"] == 56
    assert r["decision_boundaries"] == 2 and r["funding_settlements"] == 8
    assert not any(r[k] for k in ("qualification_eligible", "trading_authorized", "source_authenticity_verified", "feature_computation_verified", "scaler_recomputed"))


def test_admission_never_calls_matcher():
    previous = Replay.ingest
    def forbidden(*args, **kwargs):
        raise AssertionError("admission attempted execution")
    Replay.ingest = forbidden
    try:
        check(*fixture())
    finally:
        Replay.ingest = previous


def test_missing_whole_bar_is_rejected():
    m, b, p = fixture()
    first = m["start_ms"]
    b["events"] = [e for e in b["events"] if e["kind"] != "signal" or e["ts"] == first]
    p["feature_lineage"]["records"] = [r for r in p["feature_lineage"]["records"] if r["feature_cutoff_ms"] == first]
    refresh(b, p)
    fails(lambda: check(m, b, p), "missing_whole_4h")


def test_missing_funding_is_not_zero_cost():
    m, b, p = fixture()
    idx = next(i for i, e in enumerate(b["events"]) if e["kind"] == "funding")
    b["events"].pop(idx)
    refresh(b, p)
    fails(lambda: check(m, b, p), "funding_settlement_missing")


def test_duplicated_funding_cannot_double_charge():
    m, b, p = fixture()
    idx = next(i for i, e in enumerate(b["events"]) if e["kind"] == "funding")
    b["events"].insert(idx, dict(b["events"][idx], id="duplicate_funding"))
    refresh(b, p)
    fails(lambda: check(m, b, p), "funding_settlement_missing_duplicate")


def test_missing_book_stream_and_post_latency_snapshot():
    m, b, p = fixture()
    s = next(iter(b["rules"]))
    b["events"] = [e for e in b["events"] if e["kind"] != "book" or e["symbol"] != s]
    refresh(b, p)
    fails(lambda: check(m, b, p), "book_feed_missing")
    m, b, p = fixture()
    for e in b["events"]:
        if e["kind"] == "book":
            e["exchange_ms"] -= 1
    refresh(b, p)
    fails(lambda: check(m, b, p), "post_latency_book")


def test_coverage_gap_and_stale_mark():
    m, b, p = fixture()
    m["coverage_policy"]["max_book_gap_ms"] = 1000
    fails(lambda: check(m, b, p), "book_coverage_gap")
    m, b, p = fixture()
    for e in b["events"]:
        if e["kind"] == "mark":
            e["exchange_ms"] -= 2000
    refresh(b, p)
    fails(lambda: check(m, b, p), "decision_mark_missing_or_stale")


def test_future_and_crossed_depth_rejected():
    m, b, p = fixture()
    e = next(e for e in b["events"] if e["kind"] == "book")
    e["exchange_ms"] = e["ts"] + 1
    refresh(b, p)
    fails(lambda: check(m, b, p), "future_market")
    m, b, p = fixture()
    next(e for e in b["events"] if e["kind"] == "book")["asks"] = [["99", "1"]]
    refresh(b, p)
    fails(lambda: check(m, b, p), "crossed_depth")


def test_feature_hash_and_future_availability():
    m, b, p = fixture()
    p["feature_lineage"]["records"][0]["values_sha256"] = "a" * 64
    refresh(b, p)
    fails(lambda: check(m, b, p), "feature_values")
    m, b, p = fixture()
    p["feature_lineage"]["records"][0]["available_at_ms"] += 1
    refresh(b, p)
    fails(lambda: check(m, b, p), "future_or_stale_feature")


def test_scaler_purge_and_label_leakage():
    m, b, p = fixture()
    p["scaler_lineage"][0]["last_feature_ms"] = b["models"][0]["model"]["train_end_exclusive_ms"]
    fails(lambda: check(m, b, p), "scaler_train_cutoff")
    m, b, p = fixture()
    p["scaler_lineage"][0]["max_label_end_ms"] = m["start_ms"] + 1
    fails(lambda: check(m, b, p), "scaler_label")


def test_rule_and_fee_payload_must_match_hashes():
    m, b, p = fixture()
    p["fee_source"]["taker_fee"] = "0"
    fails(lambda: check(m, b, p), "fee_source_hash")
    m, b, p = fixture()
    next(iter(p["rule_sources"].values()))["price_tick"] = "0.01"
    fails(lambda: check(m, b, p), "rule_source_hash")


def test_synthetic_cannot_be_relabelled_by_one_flag():
    m, b, p = fixture()
    m["evidence_kind"] = "observed_unqualified"
    fails(lambda: check(m, b, p), "bundle_identity")
    b["settings"]["evidence_kind"] = "observed_unqualified"
    fails(lambda: check(m, b, p), "capture_identity")


def test_clock_and_schedule_require_explicit_coverage():
    m, b, p = fixture()
    p["capture"]["max_clock_skew_ms"] = 1
    fails(lambda: check(m, b, p), "clock_budget")
    m, b, p = fixture()
    next(iter(p["funding_schedule"].values()))["valid_until_exclusive_ms"] -= 1
    fails(lambda: check(m, b, p), "funding_schedule_coverage")


def test_local_paths_and_strict_json():
    base = ROOT / "Reports/Exia/Genetic_Primus/recorded_admission_v1_20260911"
    for name in ("../secret.json", "E:/secret.json", "https://example.com/a.json", "/secret.json", "a\\b.json", "bars.parquet"):
        fails(lambda: API["reference_path"](base, name), "reference")
    assert API["reference_path"](base, "bundle.json") == base / "bundle.json"
    fails(lambda: API["strict_json"](b'{"a":1,"a":2}'), "duplicate_json")
    fails(lambda: API["strict_json"](b'{"a":NaN}'), "nonstandard")


def test_detached_export_and_compatible_portfolio():
    m, b, p = fixture()
    before = copy.deepcopy((m, b, p))
    out, report = admitted_bundle(m, b, p, PORTFOLIO, RELEASE, IDENTITY["sha256"])
    assert (m, b, p) == before and out == b and out is not b
    # Explicit tiny synthetic integration, separate from the read-only importer.
    r = run_comparison(out, PORTFOLIO, RELEASE, IDENTITY["sha256"])
    assert r["candidates"]["GA"]["ending_equity"] == r["candidates"]["EMA"]["ending_equity"]
    assert not r["qualification_eligible"] and report["admitted_for_offline_replay"]


def test_file_reader_checks_bytes_hashes_and_path_before_read():
    m, b, p = fixture()
    directory = ROOT / "Reports/Exia/Genetic_Primus/recorded_admission_v1_20260911"
    path = directory / "virtual_manifest.json"
    raw_b, raw_p = json.dumps(b).encode(), json.dumps(p).encode()
    m["artifacts"] = {"bundle": {"path": "b.json", "bytes": len(raw_b), "sha256": hashlib.sha256(raw_b).hexdigest()},
                      "provenance": {"path": "p.json", "bytes": len(raw_p), "sha256": hashlib.sha256(raw_p).hexdigest()}}
    virtual = {directory / "b.json": raw_b, directory / "p.json": raw_p}
    loader = API["load_package"]
    namespace = loader.__globals__
    previous = namespace["read_bounded"]
    def read_virtual(target, limit):
        if target == path:
            data = json.dumps(m).encode()
        else:
            assert target in virtual, "attempt to read outside declared package"
            data = virtual[target]
        if len(data) > limit:
            raise ValueError("file_missing_or_byte_budget_exceeded")
        return data
    namespace["read_bounded"] = read_virtual
    try:
        _, actual_b, actual_p, _ = loader(path)
        assert actual_b == b and actual_p == p
        m["artifacts"]["bundle"]["sha256"] = "f" * 64
        fails(lambda: loader(path), "hash_mismatch")
        m["artifacts"]["bundle"]["path"] = "../private.json"
        fails(lambda: loader(path), "reference_escape")
    finally:
        namespace["read_bounded"] = previous


def test_input_byte_budget_and_json_depth_fail_closed():
    fails(lambda: API["strict_json"](b'[' * 2000 + b'0' + b']' * 2000), "nesting")
    assert API["strict_json"](json.dumps({"text": '[' * 2000 + '\\"' + ']' * 2000}).encode())["text"].startswith('[')
    loader = API["load_package"]
    ns = loader.__globals__
    previous = ns["read_bounded"]
    manifest = {"artifacts": {"bundle": {"path": "b.json", "bytes": API["MAX_BYTES"] + 1, "sha256": "0" * 64},
                              "provenance": {"path": "p.json", "bytes": 1, "sha256": "0" * 64}}}
    ns["read_bounded"] = lambda path, limit: json.dumps(manifest).encode()
    try:
        fails(lambda: loader(ROOT / "virtual.json"), "package_byte_budget")
    finally:
        ns["read_bounded"] = previous


GROUPS = {
    "coverage": [test_expected_grid_is_half_open_and_bounded, test_valid_package_has_no_qualification_authority,
                 test_admission_never_calls_matcher, test_missing_whole_bar_is_rejected,
                 test_missing_funding_is_not_zero_cost, test_duplicated_funding_cannot_double_charge,
                 test_missing_book_stream_and_post_latency_snapshot, test_coverage_gap_and_stale_mark],
    "provenance": [test_future_and_crossed_depth_rejected, test_feature_hash_and_future_availability,
                   test_scaler_purge_and_label_leakage, test_rule_and_fee_payload_must_match_hashes,
                   test_synthetic_cannot_be_relabelled_by_one_flag, test_clock_and_schedule_require_explicit_coverage,
                   test_local_paths_and_strict_json, test_detached_export_and_compatible_portfolio,
                   test_file_reader_checks_bytes_hashes_and_path_before_read,
                   test_input_byte_budget_and_json_depth_fail_closed],
}
