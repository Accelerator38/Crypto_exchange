"""Single-file historical feature materialization, only after explicit scenario choice."""
from __future__ import annotations

from contextlib import closing
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
PURE = "src/exia/genetic_primus/market_feature_materialization_v1.py"
SELF = "tools/materialize_genetic_primus_market_features_v1.py"
DOC = "docs/GENETIC_PRIMUS_MARKET_FEATURE_MATERIALIZATION_V1.md"
PREVIOUS = "Reports/Exia/Genetic_Primus/offline_market_feature_adapter_v1_20261007"
CODE = "Reports/Exia/Genetic_Primus/market_feature_materialization_code_v1_20261007"
DEST = "Reports/Exia/Genetic_Primus/market_feature_materialization_v1_20261007"
KEYS = "Reports/Exia/Genetic_Primus/offline_input_verification_v2_20261007"
prior = runpy.run_path(str(ROOT / "tools/check_genetic_primus_offline_market_feature_adapter_v1.py"))
old, api, quality, adapter = prior["old"], prior["api"], prior["quality"], prior["pure"]
pure = runpy.run_path(str(ROOT / PURE))


def read(path):
    return quality["read"](path)


def emit(path, value):
    old["emit"](path, value, api)


def identity_data(relative, cap=4 * 1024 * 1024):
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT / DEST / "data")
    if path.suffix not in (".jsonl", ".json") or path.stat().st_size > cap:
        raise ValueError("derived data cap/path")
    content = path.read_bytes()
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def metadata():
    receipt = read(PREVIOUS + "/completion_receipt.json")
    bank = dict(receipt["inputs"])
    quality["unchanged"](bank)
    for name, expected in receipt["output_identities"].items():
        path = PREVIOUS + "/" + name
        if old["identity"](path) != expected:
            raise ValueError("previous identity changed")
        bank[path] = expected
    for path in (PREVIOUS + "/completion_receipt.json", PURE, SELF, DOC):
        bank[path] = old["identity"](path)
    protocol, manifest = read(prior["CONFIG"]), read(prior["MANIFEST"])
    adapter["protocol_gate"](protocol, manifest, bank[prior["MANIFEST"]]["sha256"])
    sources = []
    for index, source in enumerate(manifest["snapshot_files"], 1):
        audit = read(KEYS + f"/file_{index:02d}.json")
        if audit["source"] != source:
            raise ValueError("approved source/key audit mismatch")
        summary = audit["key_summary"]
        sources.append({"source": source, "first_timestamp_ms": summary["first_timestamp_ms"],
                        "last_timestamp_ms": summary["last_timestamp_ms"], "original_index": index})
    quality["unchanged"](bank)
    return bank, protocol, manifest, sources


def code_check():
    bank, protocol, manifest, sources = metadata()
    unset = pure["plan"](protocol, manifest["metadata"], sources, None, adapter)
    assert not unset["materialization_allowed"]
    counts = []
    for lag in (0, 1):
        candidate = pure["plan"](protocol, manifest["metadata"], sources, lag, adapter)
        assert len(candidate["jobs"]) == 9 and candidate["jobs"][0]["original_index"] == 4
        assert all(j["first_timestamp_ms"] <= candidate["cutoff_bar_start_ms"] for j in candidate["jobs"])
        counts.append({"hypothetical_lag_bars": lag, "selected_files": len(candidate["jobs"]),
                       "expected_included_rows": sum(j["expected_included_rows"] for j in candidate["jobs"]),
                       "feature_events": len(candidate["events"])})
    # Code wiring only: fabricated prices, no native decoder imported.
    b = adapter["BAR_MS"]
    fake_meta = {"coverage_start": 0, "coverage_end_exclusive": 120 * b,
                 "symbols": list(adapter["SYMBOLS"]), "timeframe_ms": b}
    new_state = lambda: adapter["MarketFeatureState"](fake_meta, "a" * 64, 0, 0, "b" * 64)
    scenario = {"events": [{"asof_ms": t * b, "kind": "fixture"} for t in (64, 82, 99, 100)],
                "ingestion_lag_ms": 0, "decision_asof_ms": 100 * b, "cutoff_bar_start_ms": 99 * b}
    bars = [prior["fake_bar"](i * b, i) for i in range(100)]
    whole = new_state()
    expected, cursor, included = pure["consume"](bars, whole, scenario, 0, adapter)
    first = new_state()
    head, middle_cursor, _ = pure["consume"](bars[:70], first, scenario, 0, adapter)
    restored = new_state()
    saved = api["strict_json"](api["canonical"](pure["checkpoint"](first, middle_cursor)))
    middle_cursor = pure["restore"](restored, saved, 4, adapter)
    tail, final_cursor, _ = pure["consume"](bars[70:], restored, scenario, middle_cursor, adapter)
    assert head + tail == expected and final_cursor == cursor == 4 and included == 800
    assert pure["checkpoint"](restored, cursor) == pure["checkpoint"](whole, cursor)
    if any(name.split(".")[0] in old["BLOCKED"] for name in sys.modules):
        raise ValueError("forbidden dependency loaded")
    quality["unchanged"](bank)
    result = {"schema": "exia.genetic_primus.market_materialization_code_check/1", "inputs": bank,
              "scenario_selected": False, "hypothetical_plan_counts": counts,
              "fabricated_partition_roundtrip_passed": True, "parquet_reads": 0, "fit_count": 0,
              "labels_materialized": False, "safety": protocol["safety"]}
    return result


def prepare(lag):
    if lag not in ("0", "1"):
        raise ValueError("explicit selected scenario required")
    bank, protocol, manifest, sources = metadata()
    scenario = pure["plan"](protocol, manifest["metadata"], sources, int(lag), adapter)
    preparation = {"schema": "exia.genetic_primus.market_feature_materialization_plan/1", "inputs": bank,
                   "scenario": scenario, "metadata": manifest["metadata"],
                   "manifest_sha256": protocol["source_manifest_byte_sha256"],
                   "authorization_basis": "human_requested_materialization_and_explicitly_selected_lag",
                   "selected_scenario_is_assumption_not_measured_evidence": True,
                   "machine_checkpoints_contain_historical_price_tail": True,
                   "safety": protocol["safety"]}
    old["local"](DEST + "/preparation.json").parent.mkdir(exist_ok=False)
    (ROOT / DEST / "data").mkdir(exist_ok=False)
    emit(DEST + "/preparation.json", preparation)
    print(f"PREPARED: lag_bars={lag}; selected files=9 of approved 11; labels/fit=0; no parquet read")


def decoded_bars(content, cutoff, expected_footer_rows, expected_decoder):
    pa, pq, decoder = old["import_decoder"]()
    if decoder != expected_decoder:
        raise ValueError("native decoder identity changed")
    reader = pa.BufferReader(content)
    pf = None
    try:
        pf = pq.ParquetFile(reader, pre_buffer=False, memory_map=False,
                            thrift_string_size_limit=1000000, thrift_container_size_limit=100000)
        if pf.metadata.num_rows != expected_footer_rows or not 0 < pf.metadata.num_rows <= 200000:
            raise ValueError("footer row count")
        if set(pf.schema_arrow.names) != adapter["ROW_KEYS"]:
            raise ValueError("physical source schema")
        for name in adapter["ROW_KEYS"]:
            dtype = pf.schema_arrow.field(name).type
            if name in ("timestamp", "source_rows"):
                valid = pa.types.is_integer(dtype)
            elif name == "symbol":
                valid = pa.types.is_string(dtype) or (pa.types.is_dictionary(dtype) and pa.types.is_string(dtype.value_type))
            else:
                valid = pa.types.is_integer(dtype) or pa.types.is_floating(dtype)
            if not valid:
                raise ValueError("physical source type")
        current = []
        last_stamp = -1
        for batch in pf.iter_batches(batch_size=4096, columns=sorted(adapter["ROW_KEYS"]),
                                     use_threads=False, use_pandas_metadata=False):
            stamps = batch.column(batch.schema.get_field_index("timestamp")).to_pylist()
            if not stamps:
                continue
            if any(type(t) is not int for t in stamps) or any(a > b for a, b in zip([last_stamp] + stamps[:-1], stamps)):
                raise ValueError("noncanonical source row order")
            last_stamp = stamps[-1]
            eligible = sum(t <= cutoff for t in stamps)
            if not eligible:
                break
            values = batch.slice(0, eligible).to_pydict()
            for i in range(eligible):
                row = {key: values[key][i] for key in adapter["ROW_KEYS"]}
                if current and row["timestamp"] != current[0]["timestamp"]:
                    if len(current) != 8:
                        raise ValueError("partial source bar")
                    yield current
                    current = []
                current.append(row)
            if eligible < len(stamps):
                break
        if current:
            if len(current) != 8:
                raise ValueError("partial final bar")
            yield current
    finally:
        if pf is not None:
            pf.close()
        reader.close()


def file_step(number):
    prep = read(DEST + "/preparation.json")
    quality["unchanged"](prep["inputs"])
    scenario = prep["scenario"]
    if type(number) is not int or not 1 <= number <= len(scenario["jobs"]):
        raise ValueError("single selected job index required")
    destination = DEST + f"/part_{number:02d}.json"
    data_path = DEST + f"/data/features_{number:02d}.jsonl"
    checkpoint_path = DEST + f"/data/checkpoint_{number:02d}.json"
    if any((ROOT / p).exists() for p in (destination, data_path, checkpoint_path)):
        raise ValueError("immutable or incomplete part exists; no blind retry")
    state = adapter["MarketFeatureState"](prep["metadata"], prep["manifest_sha256"],
        prep["metadata"]["coverage_start"], scenario["ingestion_lag_ms"], scenario["revision_policy_sha256"])
    parent_identity = None
    cursor = 0
    if number > 1:
        parent_path = DEST + f"/part_{number-1:02d}.json"
        parent_identity = old["identity"](parent_path)
        parent = read(parent_path)
        if parent["job"] != number - 1 or parent["source"] != scenario["jobs"][number - 2]["source"]:
            raise ValueError("parent job substitution")
        if identity_data(parent["checkpoint_path"]) != parent["checkpoint_identity"]:
            raise ValueError("checkpoint identity changed")
        saved = read(parent["checkpoint_path"])
        cursor = pure["restore"](state, saved, len(scenario["events"]), adapter)
    job = scenario["jobs"][number - 1]
    source = job["source"]
    path = old["local"](source["path"], parquet=True)
    if path.stat().st_size != source["size_bytes"] or path.stat().st_size > old["MAX_BYTES"]:
        raise ValueError("source size")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != source["sha256"]:
        raise ValueError("source SHA")
    audit = read(KEYS + f"/file_{job['original_index']:02d}.json")
    with closing(decoded_bars(content, scenario["cutoff_bar_start_ms"], audit["footer_rows"], audit["decoder"])) as bars:
        rows, cursor, included = pure["consume"](bars, state, scenario, cursor, adapter)
    if included != job["expected_included_rows"]:
        raise ValueError("included key count")
    if path.stat().st_size != len(content) or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("source changed during read")
    if parent_identity is not None and old["identity"](parent_path) != parent_identity:
        raise ValueError("parent changed")
    if any(name.split(".")[0] in old["BLOCKED"] for name in sys.modules):
        raise ValueError("forbidden dependency loaded")
    quality["unchanged"](prep["inputs"])
    body = b"".join(api["canonical"](row) + b"\n" for row in rows)
    if len(body) > 4 * 1024 * 1024:
        raise ValueError("feature artifact cap")
    with (ROOT / data_path).open("xb") as stream:
        stream.write(body)
    emit(checkpoint_path, pure["checkpoint"](state, cursor))
    summary = {"schema": "exia.genetic_primus.market_feature_part/1", "job": number,
               "source": source, "source_hash_verified_before_after": True, "included_rows": included,
               "feature_rows": len(rows), "state_bar_count": state.count, "event_cursor": cursor,
               "feature_path": data_path, "feature_identity": identity_data(data_path),
               "checkpoint_path": checkpoint_path, "checkpoint_identity": identity_data(checkpoint_path),
               "parent_identity": parent_identity, "labels_present": False, "fit_count": 0}
    emit(destination, summary)
    print(f"PART {number:02d}: included={included}; features={len(rows)}; source SHA=OK; labels/fit=0")


def finalize(verify=False):
    prep = read(DEST + "/preparation.json")
    quality["unchanged"](prep["inputs"])
    outputs = {"preparation.json": old["identity"](DEST + "/preparation.json")}
    included = features = 0
    for number, job in enumerate(prep["scenario"]["jobs"], 1):
        name = f"part_{number:02d}.json"
        part = read(DEST + "/" + name)
        if part["job"] != number or part["source"] != job["source"] or part["included_rows"] != job["expected_included_rows"]:
            raise ValueError("part source/count")
        if number > 1 and part["parent_identity"] != outputs[f"part_{number-1:02d}.json"]:
            raise ValueError("part identity chain")
        for path_key, id_key in (("feature_path", "feature_identity"), ("checkpoint_path", "checkpoint_identity")):
            if identity_data(part[path_key]) != part[id_key]:
                raise ValueError("derived artifact changed")
            outputs[part[path_key].removeprefix(DEST + "/")] = part[id_key]
        outputs[name] = old["identity"](DEST + "/" + name)
        included += part["included_rows"]
        features += part["feature_rows"]
    if part["event_cursor"] != len(prep["scenario"]["events"]) or features != len(prep["scenario"]["events"]) * 8:
        raise ValueError("incomplete feature event bank")
    result = {"schema": "exia.genetic_primus.market_feature_materialization_result/1",
              "selected_source_files": len(prep["scenario"]["jobs"]), "included_source_rows": included,
              "feature_rows": features, "scenario_sha256": adapter["sha"](prep["scenario"]),
              "labels_materialized": False, "candidate_rows_are_not_label_gated_training_rows": True,
              "fit_count": 0, "market_financial_evaluations": 0, "original_arrival_verified": False,
              "machine_checkpoints_contain_historical_price_tail": True, "safety": prep["safety"]}
    receipt = {"schema": "exia.genetic_primus.market_feature_materialization_receipt/1",
               "inputs": prep["inputs"], "output_identities": outputs, "safety": prep["safety"]}
    body = api["canonical"](result) + b"\n"
    receipt["output_identities"]["result.json"] = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    quality["unchanged"](prep["inputs"])
    if verify:
        if read(DEST + "/result.json") != result or read(DEST + "/completion_receipt.json") != receipt:
            raise ValueError("materialization receipt mismatch")
        print("VERIFIED: feature artifacts + receipt; no parquet reread; labels/fit=0")
    else:
        emit(DEST + "/result.json", result)
        emit(DEST + "/completion_receipt.json", receipt)
        print(f"FINALIZED: sources=9; included={included}; features={features}; labels/fit=0")


def main():
    args = sys.argv[1:]
    if args in (["--check-code"], ["--verify-code"]):
        result = code_check()
        body = api["canonical"](result) + b"\n"
        receipt = {"schema": "exia.genetic_primus.market_materialization_code_receipt/1", "inputs": result["inputs"],
                   "output_identities": {"result.json": {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}},
                   "safety": result["safety"]}
        if args == ["--check-code"]:
            old["local"](CODE + "/result.json").parent.mkdir(exist_ok=False)
            emit(CODE + "/result.json", result)
            emit(CODE + "/completion_receipt.json", receipt)
        elif read(CODE + "/result.json") != result or read(CODE + "/completion_receipt.json") != receipt:
            raise ValueError("code-check identity mismatch")
        print("CODE CHECK OK: partition/checkpoint; scenario unselected; parquet=0; labels/fit=0")
    elif len(args) == 2 and args[0] == "--prepare":
        prepare(args[1])
    elif len(args) == 2 and args[0] == "--file" and args[1].isdigit():
        file_step(int(args[1]))
    elif args in (["--finalize"], ["--verify"]):
        finalize(args == ["--verify"])
    else:
        raise ValueError("use --check-code/--verify-code, --prepare 0|1, --file N, --finalize/--verify")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("STOPPED: " + type(exc).__name__ + "; no fit/network")
        raise SystemExit(1)
