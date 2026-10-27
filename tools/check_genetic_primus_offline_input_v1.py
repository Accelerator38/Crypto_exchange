"""Explicitly authorized historical key audit, exactly one parquet per --file call."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = "Reports/Exia/Genetic_Primus/forward_preregistration_v1_20261007"
DESTINATION = "Reports/Exia/Genetic_Primus/offline_input_verification_v2_20261007"
MANIFEST = "Reports/Exia/Genetic_Primus/prospective_3d_v1_202609/data_manifest_origin_09.json"
MODULE = "src/exia/genetic_primus/offline_input_verification_v1.py"
API_MODULE = "src/exia/genetic_primus/research_execution_contract_v1.py"
BLOCKED = {"numpy", "pandas", "exia", "pytest", "websockets", "aiohttp", "httpx", "requests"}
MAX_BYTES = 8 * 1024 * 1024
PHASE = "arguments"
ASSERTIONS = 0


def check(value):
    global ASSERTIONS
    if not value:
        raise AssertionError("key audit regression")
    ASSERTIONS += 1


def rejected(call):
    try:
        call()
    except ValueError:
        check(True)
    else:
        raise AssertionError("bad input accepted")


def local(relative, parquet=False):
    if type(relative) is not str or ":" in relative or "\\" in relative:
        raise ValueError("relative approved path required")
    result = (ROOT / relative).resolve()
    result.relative_to(ROOT / "Retrodate" if parquet else ROOT)
    if result.suffix not in ((".parquet",) if parquet else (".py", ".json", ".md")):
        raise ValueError("unapproved file type")
    return result


def raw(relative):
    item = local(relative)
    if item.stat().st_size > 512_000:
        raise ValueError("metadata cap exceeded")
    return item.read_bytes()


def identity(relative):
    data = raw(relative)
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def unchanged(bank):
    if type(bank) is not dict or not 1 <= len(bank) <= 128:
        raise ValueError("bounded metadata/source bank required")
    for relative, expected in bank.items():
        if identity(relative) != expected:
            raise ValueError("source/metadata identity changed")


def emit(relative, value, api):
    with local(relative).open("xb") as stream:
        stream.write(api["canonical"](value) + b"\n")


def regress(module):
    b = module["BAR_MS"]
    meta = {"columns": ["timestamp", "symbol", "open", "high", "low", "close", "volume"],
            "coverage_start": 0, "coverage_end_exclusive": 3 * b, "row_count": 6,
            "symbols": ["A/USDT", "B/USDT"], "timeframe_ms": b, "timestamp_count": 3}
    rows = [(ts, symbol) for ts in (0, b, 2 * b) for symbol in meta["symbols"]]
    complete = module["key_summary"](rows, meta)
    check(complete["unique_keys"] == 6 and complete["key_checks_passed"] is True)
    check(complete["price_values_read"] is False and complete["fit_allowed"] is False)
    result = module["aggregate"]([complete], meta)
    check(result["key_structure_verified"] is True and result["complete_common_grid"] is True)
    for bad, field in (([(None, "A/USDT")], "null_timestamp"), ([(0, None)], "null_symbol"),
                       ([(True, "A/USDT")], "invalid_timestamp_type"), ([(0.0, "A/USDT")], "invalid_timestamp_type"),
                       ([(0, "Z/USDT")], "unknown_symbol"), ([(1, "A/USDT")], "off_grid_timestamp"),
                       ([(3 * b, "A/USDT")], "outside_manifest_coverage"),
                       ([(0, "A/USDT"), (0, "A/USDT")], "duplicate_keys")):
        value = module["key_summary"](bad, meta)
        check(value["counts"][field] == 1 and value["key_checks_passed"] is False)
    missing = module["key_summary"](rows[:-1], meta)
    result = module["aggregate"]([missing], meta)
    check(result["missing_keys_total"] == 1 and result["key_structure_verified"] is False)
    split = [module["key_summary"](rows[:2], meta), module["key_summary"](rows[2:], meta)]
    check(module["aggregate"](split, meta)["key_structure_verified"] is True)
    overlap = [complete, module["key_summary"](rows[:2], meta)]
    result = module["aggregate"](overlap, meta)
    check(result["cross_file_duplicate_keys"] == 2 and result["key_structure_verified"] is False)
    reversed_rows = module["key_summary"](list(reversed(rows)), meta)
    check(reversed_rows["counts"]["regressions_per_symbol"] == 4)
    check(reversed_rows["presence_hex"] == complete["presence_hex"])
    gap = module["key_summary"]([(0, "A/USDT"), (2 * b, "A/USDT")], meta)
    check(gap["per_symbol"]["A/USDT"]["internal_missing_bars"] == 1)
    for key, bad in (("timestamp_count", 4), ("timeframe_ms", 1), ("row_count", True),
                     ("coverage_start", 1), ("symbols", ["A/USDT", "A/USDT"])):
        changed = copy.deepcopy(meta)
        changed[key] = bad
        rejected(lambda: module["key_summary"]([], changed))
    for key, value in (("unique_keys", 5), ("presence_hex", {"A/USDT": "f", "B/USDT": "7"}),
                       ("presence_hex", {"A/USDT": "07", "B/USDT": "7"})):
        changed = copy.deepcopy(complete)
        changed[key] = value
        rejected(lambda: module["aggregate"]([changed], meta))
    rejected(lambda: module["key_summary"]([(0, "A/USDT", 123.0)], meta))
    core = meta["columns"]
    check(module["schema_layout"](core, core)["nonfeature_extra_columns"] == [])
    check(module["schema_layout"](core + ["source_rows"], core)["nonfeature_extra_columns"] == ["source_rows"])
    check(module["schema_layout"](list(reversed(core)), core)["normalized_core_columns"] == core)
    rejected(lambda: module["schema_layout"](core + ["future_return"], core))
    rejected(lambda: module["schema_layout"](core + ["timestamp"], core))
    rejected(lambda: module["schema_layout"](core[:-1], core))
    return ASSERTIONS


def prepare(api, module):
    previous = api["strict_json"](raw(PREVIOUS + "/completion_receipt.json"))
    bank = dict(previous["inputs"])
    unchanged(bank)
    for name in ("record.json", "verification.json", "summary.json"):
        if identity(PREVIOUS + "/" + name)["sha256"] != previous["output_byte_sha256"][name]:
            raise ValueError("predecessor output changed")
    additions = [MODULE, API_MODULE, MANIFEST, "tools/check_genetic_primus_offline_input_v1.py",
                 "docs/GENETIC_PRIMUS_OFFLINE_INPUT_VERIFICATION_V2.md", PREVIOUS + "/completion_receipt.json",
                 PREVIOUS + "/record.json", PREVIOUS + "/verification.json", PREVIOUS + "/summary.json"]
    additions += ["Reports/Exia/Genetic_Primus/offline_input_verification_v1_20261007/preparation.json",
                  "Reports/Exia/Genetic_Primus/offline_input_verification_v1_20261007/diagnostic_source_snapshot.json"]
    for relative in additions:
        value = identity(relative)
        if relative in bank and bank[relative] != value:
            raise ValueError("source identity collision")
        bank[relative] = value
    manifest = api["strict_json"](raw(MANIFEST))
    module["metadata"](manifest["metadata"])
    sources = manifest["snapshot_files"]
    if type(sources) is not list or len(sources) != 11 or len({s["path"] for s in sources}) != 11:
        raise ValueError("only eleven approved manifest entries allowed")
    for source in sources:
        api["keys"](source, ("path", "sha256", "size_bytes"))
        api["digest"](source["sha256"])
        api["integer"](source["size_bytes"], 8)
        if source["size_bytes"] > MAX_BYTES:
            raise ValueError("parquet byte cap exceeded")
        local(source["path"], parquet=True)
    direct = regress(module)
    inherited_safety = api["strict_json"](raw(PREVIOUS + "/record.json"))["record"]["safety"]
    api["keys"](inherited_safety, api["SAFETY"])
    if any(value is not False for value in inherited_safety.values()):
        raise ValueError("safety authority changed")
    unchanged(bank)
    value = {"schema": "exia.genetic_primus.offline_input_preparation/1",
             "authorization": "explicit_user_razreshayu_for_bounded_old_manifest_parquet_verification_only",
             "authorized_columns": ["timestamp", "symbol"], "price_values_decode_authorized_by_this_tool": False,
             "source_manifest": MANIFEST, "source_manifest_identity": identity(MANIFEST),
             "approved_sources": sources, "metadata": manifest["metadata"], "inputs": bank,
             "direct_assertions": direct, "safety": inherited_safety,
             "legacy_manifest_safety": manifest["safety"],
             "training_authorized": False, "network_authorized": False, "sealed_interval_read_authorized": False}
    (ROOT / DESTINATION).mkdir(exist_ok=False)
    emit(DESTINATION + "/preparation.json", value, api)
    print("PREPARED: " + str(direct) + " pure checks; 11 files; projection timestamp/symbol; fit=0")


def import_decoder():
    if any(name.split(".")[0] in BLOCKED for name in sys.modules):
        raise ValueError("prohibited module already loaded")
    attempts = set()

    class Guard:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in BLOCKED:
                attempts.add(fullname.split(".")[0])
                raise ModuleNotFoundError("blocked optional or trading import")
            return None

    def audit(event, args):
        if event.startswith(("socket.", "subprocess.", "os.spawn", "os.exec")) or event in ("os.system", "os.startfile"):
            raise PermissionError("network/process launch denied")

    sys.meta_path.insert(0, Guard())
    sys.addaudithook(audit)
    import pyarrow as pa
    import pyarrow.parquet as pq
    if any(name.split(".")[0] in BLOCKED for name in sys.modules):
        raise ValueError("prohibited module imported")
    pins = {}
    for role, name in (("package", "pyarrow"), ("lib", "pyarrow.lib"),
                       ("parquet_core", "pyarrow.parquet.core"), ("parquet_native", "pyarrow._parquet")):
        source = Path(sys.modules[name].__file__).resolve()
        source.relative_to((ROOT / ".venv/Lib/site-packages/pyarrow").resolve())
        size = source.stat().st_size
        if not 0 < size <= 128 * 1024 * 1024:
            raise ValueError("decoder file cap exceeded")
        pins[role] = {"path": str(source), "bytes": size, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    descriptor = {"name": "pyarrow", "version": pa.__version__, "files": pins,
                  "complete_native_dependency_closure_verified": False, "loaded_memory_bytes_attested": False,
                  "blocked_optional_import_attempts": sorted(attempts), "prohibited_modules_absent": True,
                  "buffered_input_only": True, "use_threads": False, "pre_buffer": False}
    return pa, pq, descriptor


def audit_file(index, preparation, api, module):
    global PHASE
    if type(index) is not int or not 1 <= index <= 11:
        raise ValueError("one approved index required")
    output = DESTINATION + "/file_" + format(index, "02d") + ".json"
    if local(output).exists():
        raise ValueError("published file audit cannot be overwritten")
    source = preparation["approved_sources"][index - 1]
    item = local(source["path"], parquet=True)
    PHASE = "file_identity"
    if item.stat().st_size != source["size_bytes"] or item.stat().st_size > MAX_BYTES:
        raise ValueError("source size mismatch")
    content = item.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != source["sha256"] or content[:4] != b"PAR1" or content[-4:] != b"PAR1":
        raise ValueError("source hash/header mismatch")
    PHASE = "decoder_import"
    pa, pq, decoder = import_decoder()
    PHASE = "schema"
    reader = pa.BufferReader(content)
    parquet = pq.ParquetFile(reader, pre_buffer=False, memory_map=False,
                             thrift_string_size_limit=1_000_000, thrift_container_size_limit=100_000)
    schema = parquet.schema_arrow
    layout = module["schema_layout"](schema.names, preparation["metadata"]["columns"])
    if not 0 < parquet.metadata.num_rows <= module["MAX_ROWS"]:
        raise ValueError("schema/row limit mismatch")
    if "source_rows" in layout["nonfeature_extra_columns"] and not pa.types.is_integer(schema.field("source_rows").type):
        raise ValueError("source_rows must be integer and excluded from decoded/model columns")
    if not 1 <= parquet.metadata.num_row_groups <= 64:
        raise ValueError("row-group limit exceeded")
    row_groups = parquet.metadata.num_row_groups
    if not pa.types.is_int64(schema.field("timestamp").type):
        raise ValueError("no automatic timestamp unit conversion allowed")
    symbol_type = schema.field("symbol").type
    if pa.types.is_dictionary(symbol_type):
        symbol_type = symbol_type.value_type
    if not (pa.types.is_string(symbol_type) or pa.types.is_large_string(symbol_type)):
        raise ValueError("symbol string basis required")
    for name in ("open", "high", "low", "close", "volume"):
        if not (pa.types.is_floating(schema.field(name).type) or pa.types.is_integer(schema.field(name).type)):
            raise ValueError("numeric OHLCV schema required")
    fields = {f.name: {"arrow_type": str(f.type), "nullable": f.nullable} for f in schema}
    PHASE = "key_projection"

    def rows():
        for batch in parquet.iter_batches(batch_size=4096, columns=["timestamp", "symbol"],
                                          use_threads=False, use_pandas_metadata=False):
            projected = batch.to_pydict()
            if set(projected) != {"timestamp", "symbol"}:
                raise ValueError("unexpected decoded column")
            yield from zip(projected["timestamp"], projected["symbol"], strict=True)

    keys = module["key_summary"](rows(), preparation["metadata"])
    if keys["counts"]["rows"] != parquet.metadata.num_rows:
        raise ValueError("footer/projection row count mismatch")
    parquet.close()
    reader.close()
    PHASE = "post_read_identity"
    if item.stat().st_size != source["size_bytes"] or hashlib.sha256(item.read_bytes()).hexdigest() != digest:
        raise ValueError("source changed during its audit")
    if any(name.split(".")[0] in BLOCKED for name in sys.modules):
        raise ValueError("forbidden module after decode")
    unchanged(preparation["inputs"])
    result = {"schema": "exia.genetic_primus.offline_file_audit/1", "index": index, "source": source,
              "source_byte_identity_verified_before_after_its_read": True, "schema_fields": fields, "schema_layout": layout,
              "schema_compatible": True, "footer_rows": keys["counts"]["rows"],
              "row_groups": row_groups, "key_summary": keys, "decoder": decoder,
              "decoder_identity_sha256": api["sha"](decoder), "projected_columns": ["timestamp", "symbol"],
              "price_values_decoded": False, "native_sandbox_proof": False, "fit_allowed": False,
              "network_used": False, "financial_evaluation_allowed": False}
    emit(output, result, api)
    print("FILE " + str(index) + "/11: " + ("PASS" if keys["key_checks_passed"] else "FINDINGS") +
          "; rows=" + str(keys["counts"]["rows"]) + "; symbols=" + str(keys["symbols_present"]) +
          "; duplicates=" + str(keys["counts"]["duplicate_keys"]) +
          "; internal_gaps=" + str(sum(v["internal_missing_bars"] for v in keys["per_symbol"].values())))
    if keys["key_checks_passed"] is not True:
        raise SystemExit(1)


def finalize(preparation, api, module, verify=False):
    outputs = {"preparation.json": identity(DESTINATION + "/preparation.json")}
    files = []
    files_with_extra = 0
    decoder_ids = set()
    for index, source in enumerate(preparation["approved_sources"], 1):
        name = "file_" + format(index, "02d") + ".json"
        outputs[name] = identity(DESTINATION + "/" + name)
        result = api["strict_json"](raw(DESTINATION + "/" + name))
        if result["index"] != index or api["canonical"](result["source"]) != api["canonical"](source):
            raise ValueError("file audit source substitution")
        if result["source_byte_identity_verified_before_after_its_read"] is not True or result["schema_compatible"] is not True:
            raise ValueError("unverified file")
        if result["projected_columns"] != ["timestamp", "symbol"] or result["price_values_decoded"] is not False:
            raise ValueError("column scope changed")
        layout = module["schema_layout"](result["schema_layout"]["physical_columns"], preparation["metadata"]["columns"])
        if api["canonical"](layout) != api["canonical"](result["schema_layout"]):
            raise ValueError("schema layout substitution")
        files_with_extra += bool(layout["nonfeature_extra_columns"])
        if result["decoder_identity_sha256"] != api["sha"](result["decoder"]):
            raise ValueError("decoder identity substitution")
        decoder_ids.add(result["decoder_identity_sha256"])
        files.append(result["key_summary"])
    if len(decoder_ids) != 1:
        raise ValueError("decoder changed between files")
    result = module["aggregate"](files, preparation["metadata"])
    body = {"schema": "exia.genetic_primus.offline_input_result/1", "key_structure": result,
            "approved_source_files": 11, "source_bytes": sum(s["size_bytes"] for s in preparation["approved_sources"]),
            "physical_vs_normalized_schema_checked": True, "files_with_nonfeature_source_rows": files_with_extra,
            "file_identities_verified_at_each_separate_read": True,
            "sources_rehashed_at_global_finalize": False, "global_stage_reads_only_file_audit_json": True,
            "data_file_read_authorized": True, "price_values_decoded": False, "model_fit_count": 0,
            "financial_evaluation_count": 0, "network_used": False,
            "arrival_revision_venue_product_provenance_verified": False, "ohlcv_price_quality_verified": False,
            "market_research_launch_ready": False, "real_money_ready": False,
            "safety": preparation["safety"]}
    body_bytes = api["canonical"](body) + b"\n"
    outputs["result.json"] = {"bytes": len(body_bytes), "sha256": hashlib.sha256(body_bytes).hexdigest()}
    unchanged(preparation["inputs"])
    receipt = {"schema": "exia.genetic_primus.offline_input_receipt/1", "inputs": preparation["inputs"],
               "inputs_verified_unchanged_before_and_after_each_stage": True,
               "data_identity_scope": "eleven_files_verified_in_separate_processes_not_rehashed_by_final_aggregation",
               "output_identities": outputs, "pure_assertions": preparation["direct_assertions"],
               "actual_key_grid_verified": result["key_structure_verified"],
               "fit_allowed": False, "market_research_launch_ready": False, "real_money_ready": False,
               "safety": preparation["safety"]}
    if verify:
        saved = api["strict_json"](raw(DESTINATION + "/completion_receipt.json"))
        if api["canonical"](saved) != api["canonical"](receipt) or raw(DESTINATION + "/result.json") != body_bytes:
            raise ValueError("receipt/result identity mismatch")
        print("VERIFIED: receipt + 13 outputs; no parquet reread; fit=0")
    else:
        emit(DESTINATION + "/result.json", body, api)
        emit(DESTINATION + "/completion_receipt.json", receipt, api)
        print("AGGREGATE: rows=" + str(result["total_rows"]) + "; timestamps=" + str(result["expected_timestamps"]) +
              "; cross_duplicates=" + str(result["cross_file_duplicate_keys"]) +
              "; missing_keys=" + str(result["missing_keys_total"]) +
              "; structural_pass=" + str(result["key_structure_verified"]) + "; fit=0")


def main():
    global PHASE
    if sys.executable.lower() != str(ROOT / ".venv/Scripts/python.exe").lower():
        raise ValueError("only workspace venv interpreter permitted")
    api = runpy.run_path(str(local(API_MODULE)))
    module = runpy.run_path(str(local(MODULE)))
    arguments = sys.argv[1:]
    if arguments == ["--prepare"]:
        PHASE = "prepare_metadata_and_pure_checks"
        prepare(api, module)
        return
    PHASE = "preparation_identity"
    preparation = api["strict_json"](raw(DESTINATION + "/preparation.json"))
    if preparation["authorized_columns"] != ["timestamp", "symbol"] or preparation["training_authorized"] is not False:
        raise ValueError("authorization scope mismatch")
    unchanged(preparation["inputs"])
    manifest = api["strict_json"](raw(MANIFEST))
    if preparation["source_manifest"] != MANIFEST or preparation["source_manifest_identity"] != identity(MANIFEST):
        raise ValueError("approved manifest identity substitution")
    if api["canonical"](preparation["approved_sources"]) != api["canonical"](manifest["snapshot_files"]):
        raise ValueError("approved file allowlist substitution")
    if api["canonical"](preparation["metadata"]) != api["canonical"](manifest["metadata"]):
        raise ValueError("reference metadata substitution")
    api["keys"](preparation["safety"], api["SAFETY"])
    if any(value is not False for value in preparation["safety"].values()) or preparation["network_authorized"] is not False:
        raise ValueError("authority substitution")
    if len(arguments) == 2 and arguments[0] == "--file" and arguments[1].isdigit():
        audit_file(int(arguments[1]), preparation, api, module)
    elif arguments in (["--finalize"], ["--verify"]):
        PHASE = "aggregate_metadata_only"
        finalize(preparation, api, module, verify=arguments == ["--verify"])
    else:
        raise ValueError("one action required: --prepare, --file N, --finalize, --verify")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAIL: " + type(exc).__name__ + "; phase=" + PHASE + "; no model fit")
        raise SystemExit(1)
