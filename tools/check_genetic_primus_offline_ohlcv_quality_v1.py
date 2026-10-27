"""One explicitly approved historical parquet per command; counter-only outputs."""
from __future__ import annotations

import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
OLD_TOOL = "tools/check_genetic_primus_offline_input_v1.py"
PURE = "src/exia/genetic_primus/offline_ohlcv_quality_v1.py"
SELF = "tools/check_genetic_primus_offline_ohlcv_quality_v1.py"
DOC = "docs/GENETIC_PRIMUS_OFFLINE_OHLCV_QUALITY_V1.md"
PREVIOUS = "Reports/Exia/Genetic_Primus/offline_input_verification_v2_20261007"
DEST = "Reports/Exia/Genetic_Primus/offline_ohlcv_quality_v1_20261007"

old = runpy.run_path(str(ROOT / OLD_TOOL))
api = runpy.run_path(str(ROOT / old["API_MODULE"]))
pure = runpy.run_path(str(ROOT / PURE))


def read(path):
    return api["strict_json"](old["raw"](path))


def unchanged(bank):
    if not bank or len(bank) > 192:
        raise ValueError("input bank cap")
    for path, expected in bank.items():
        if old["identity"](path) != expected:
            raise ValueError("input identity changed")


def emit(name, value):
    old["emit"](DEST + "/" + name, value, api)


def regressions():
    count = 0
    good = (2.0, 3.0, 1.0, 2.5, 4.0, 240)
    def check(ok):
        nonlocal count
        if not ok:
            raise AssertionError("quality regression")
        count += 1
    check(pure["summarize"]([good])["invalid_rows"] == 0)
    for field in range(5):
        for value, kind in ((None, "null"), (True, "non_numeric"),
                            ("x", "non_numeric"), (float("nan"), "nonfinite"),
                            (float("inf"), "nonfinite"), (-1, "domain_invalid")):
            row = list(good)
            row[field] = value
            summary = pure["summarize"]([row])
            check(summary["invalid_rows"] == 1 and
                  summary[pure["FIELDS"][field] + "_" + kind] == 1)
    for source in (None, True, 0, -1, 240.0, "240", 239, 241):
        check(pure["summarize"]([good[:5] + (source,)])["invalid_rows"] == 1)
    for row in ((2, 1, 1, 2, 4, 240), (2, 3, 2.5, 2, 4, 240)):
        check(pure["summarize"]([row])["ohlc_inconsistent_rows"] == 1)
    zero = pure["summarize"]([good[:4] + (0, 240)])
    check(zero["invalid_rows"] == 0 and zero["zero_volume_rows"] == 1)
    check(pure["summarize"]([])["rows"] == 0)
    check(pure["combine"]([zero, zero])["rows"] == 2)
    for call in (lambda: pure["summarize"]([(1,)]),
                 lambda: pure["combine"]([{}]),
                 lambda: pure["combine"]([{**pure["empty"](), "rows": True}])):
        try:
            call()
        except ValueError:
            check(True)
        else:
            raise AssertionError("malformed quality input accepted")
    return count


def sidecar_path(source):
    path = Path(source["path"])
    if path.name == "full8_240m.parquet":
        return str(path.with_suffix(".manifest.json")).replace("\\", "/")
    return str(path.with_name(path.stem.replace("_240m", "_manifest") + ".json")).replace("\\", "/")


def lineage(source, bank):
    path = sidecar_path(source)
    bank[path] = old["identity"](path)
    side = read(path)
    matches = [record for record in side["files"]
               if record["path"].replace("\\", "/") == source["path"]]
    if len(matches) != 1:
        raise ValueError("sidecar output binding missing")
    match = matches[0]
    if match["sha256"] != source["sha256"] or ("bytes" in match and match["bytes"] != source["size_bytes"]):
        raise ValueError("sidecar output identity mismatch")
    item = {"sidecar": path, "output_rows_claim": match["rows"],
            "sidecar_records_output_size": "bytes" in match,
            "source_rows_expected": 240, "bar_timestamp_semantics": "4h_bucket_start",
            "observed_arrival_verified": False, "raw_1m_content_verified": False,
            "venue_authenticity_verified": False}
    if "source_manifest" in side:
        if side["source_timeframe_minutes"] != 1 or side["timeframe"] != "240m":
            raise ValueError("unexpected older resampling contract")
        parent = side["source_manifest"].replace("\\", "/")
        bank[parent] = old["identity"](parent)
        panel = read(parent)
        if panel["dataset_sha256"] != side["source_dataset_sha256"]:
            raise ValueError("panel digest claim mismatch")
        item.update(source_manifest=parent, producer_identity_bound=False,
                    exchange_claim=panel.get("exchange"), market_type_claim=panel.get("market_type"))
        if "source_archive_manifest" in panel:
            archive = panel["source_archive_manifest"].replace("\\", "/")
            bank[archive] = old["identity"](archive)
            archive_meta = read(archive)
            if archive_meta["dataset_sha256"] != panel["source_archive_dataset_sha256"]:
                raise ValueError("archive digest claim mismatch")
            item["archive_manifest"] = archive
    else:
        if "builder_sha256" in side:
            producer, key = "tools/build_genetic_primus_v8_clean_ohlcv_parquet_v1.py", "builder_sha256"
        else:
            producer, key = "tools/fetch_genetic_primus_v8_clean_ohlcv_tail_v1.py", "code_sha256"
        bank[producer] = old["identity"](producer)
        if bank[producer]["sha256"] != side[key]:
            raise ValueError("producer source hash changed")
        item.update(producer=producer, producer_identity_bound=True,
                    chunks_referenced=len(side["chunks"]), chunks_read=False)
    return item


def prepare():
    receipt = read(PREVIOUS + "/completion_receipt.json")
    previous = read(PREVIOUS + "/preparation.json")
    if receipt["inputs"] != previous["inputs"] or not receipt["actual_key_grid_verified"]:
        raise ValueError("previous receipt/preparation mismatch")
    bank = dict(previous["inputs"])
    unchanged(bank)
    for name, expected in receipt["output_identities"].items():
        path = PREVIOUS + "/" + name
        if old["identity"](path) != expected:
            raise ValueError("previous output changed")
        bank[path] = expected
    for path in (PREVIOUS + "/completion_receipt.json", PURE, SELF, DOC,
                 "tools/build_exia_timeframe_archive_v1.py", "tools/run_exia_timebase_audit_v1.py"):
        bank[path] = old["identity"](path)
    sources = previous["approved_sources"]
    if len(sources) != 11 or any(previous["safety"].values()):
        raise ValueError("authorization/safety mismatch")
    lineages = [lineage(source, bank) for source in sources]
    assertions = regressions()
    unchanged(bank)
    old["local"](DEST + "/preparation.json").parent.mkdir(exist_ok=False)
    emit("preparation.json", {"schema": "offline-ohlcv-quality/1", "inputs": bank,
        "approved_sources": sources, "lineage": lineages, "safety": previous["safety"],
        "direct_assertions": assertions, "authorized_columns": list(pure["FIELDS"]),
        "price_output_allowed": False, "new_sealed_reads_allowed": False,
        "fit_count": 0, "financial_evaluation_count": 0})
    print(f"PREPARED: assertions={assertions}; inputs={len(bank)}; parquet reads=0; fit=0")


def file_check(index, preparation):
    if not 1 <= index <= 11:
        raise ValueError("file index 1..11 required")
    source = preparation["approved_sources"][index - 1]
    destination = old["local"](DEST + f"/file_{index:02d}.json")
    if destination.exists():
        raise ValueError("immutable output already exists")
    path = old["local"](source["path"], parquet=True)
    if path.stat().st_size != source["size_bytes"] or path.stat().st_size > old["MAX_BYTES"]:
        raise ValueError("parquet size mismatch")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != source["sha256"]:
        raise ValueError("parquet hash mismatch")
    pa, pq, descriptor = old["import_decoder"]()
    reader = pa.BufferReader(content)
    pf = None
    try:
        pf = pq.ParquetFile(reader, pre_buffer=False, memory_map=False,
                            thrift_string_size_limit=1000000, thrift_container_size_limit=100000)
        count = pf.metadata.num_rows
        if not 0 < count <= 200000:
            raise ValueError("row cap")
        for name in pure["FIELDS"]:
            dtype = pf.schema_arrow.field(name).type
            if name == "source_rows":
                valid = pa.types.is_integer(dtype)
            else:
                valid = pa.types.is_integer(dtype) or pa.types.is_floating(dtype)
            if not valid:
                raise ValueError("unexpected physical quality type")
        summary = pure["empty"]()
        for batch in pf.iter_batches(batch_size=4096, columns=list(pure["FIELDS"]),
                                     use_threads=False, use_pandas_metadata=False):
            values = batch.to_pydict()
            piece = pure["summarize"](zip(*(values[name] for name in pure["FIELDS"])))
            summary = pure["combine"]([summary, piece])
        if summary["rows"] != count or count != preparation["lineage"][index - 1]["output_rows_claim"]:
            raise ValueError("row count mismatch")
    finally:
        if pf is not None:
            pf.close()
        reader.close()
    if path.stat().st_size != len(content) or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("parquet changed during read")
    if any(name.split(".")[0] in old["BLOCKED"] for name in sys.modules):
        raise ValueError("forbidden package loaded")
    unchanged(preparation["inputs"])
    emit(f"file_{index:02d}.json", {"schema": "offline-ohlcv-quality-file/1", "index": index,
         "source": source, "summary": summary, "decoder": descriptor,
         "source_hash_verified_before_and_after": True, "price_values_retained": False})
    print(f"FILE {index:02d}: rows={count}; invalid={summary['invalid_rows']}; "
          f"zero-volume={summary['zero_volume_rows']}; hash=OK; fit=0")


def finalize(preparation, verify=False):
    outputs = {"preparation.json": old["identity"](DEST + "/preparation.json")}
    summaries, decoder = [], None
    for index, source in enumerate(preparation["approved_sources"], 1):
        name = f"file_{index:02d}.json"
        item = read(DEST + "/" + name)
        if item["index"] != index or item["source"] != source or not item["source_hash_verified_before_and_after"]:
            raise ValueError("per-file identity mismatch")
        current = item["decoder"]
        if decoder is None:
            decoder = current
        elif current != decoder:
            raise ValueError("decoder changed")
        summaries.append(item["summary"])
        outputs[name] = old["identity"](DEST + "/" + name)
    total = pure["combine"](summaries)
    result = {"schema": "offline-ohlcv-quality-result/1", "files": 11, "summary": total,
              "strict_quality_passed": total["invalid_rows"] == 0,
              "lineage": preparation["lineage"], "safety": preparation["safety"],
              "fit_count": 0, "financial_evaluation_count": 0,
              "market_research_launch_ready": False, "real_money_ready": False,
              "raw_1m_uniqueness_verified": False, "observed_arrival_verified": False,
              "source_identity_scope": "per_file_before_and_after_not_simultaneous_global_snapshot"}
    body = api["canonical"](result) + b"\n"
    outputs["result.json"] = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    receipt = {"schema": "offline-ohlcv-quality-receipt/1", "inputs": preparation["inputs"],
               "output_identities": outputs, "safety": preparation["safety"],
               "pure_assertions": preparation["direct_assertions"], "parquet_reread_on_finalize": False}
    unchanged(preparation["inputs"])
    if verify:
        if old["raw"](DEST + "/result.json") != body or read(DEST + "/completion_receipt.json") != receipt:
            raise ValueError("completion identity mismatch")
        print("VERIFIED: receipt + 13 outputs; parquet reads=0; fit=0")
    else:
        emit("result.json", result)
        emit("completion_receipt.json", receipt)
        print(f"FINALIZED: files=11; rows={total['rows']}; invalid={total['invalid_rows']}; fit=0")


def main():
    args = sys.argv[1:]
    if args == ["--prepare"]:
        prepare()
        return
    if args not in (["--finalize"], ["--verify"]) and not (len(args) == 2 and args[0] == "--file" and args[1].isdigit()):
        raise ValueError("use --prepare, --file N, --finalize or --verify")
    preparation = read(DEST + "/preparation.json")
    unchanged(preparation["inputs"])
    if args[0] == "--file":
        file_check(int(args[1]), preparation)
    else:
        finalize(preparation, args == ["--verify"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Never include native exception strings or decoded source values.
        print("STOPPED: " + type(exc).__name__ + "; no training/network")
        raise SystemExit(1)
