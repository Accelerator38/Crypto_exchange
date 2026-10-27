"""One bounded historical source per command, no fits/network/runtime."""
from __future__ import annotations

from contextlib import closing
from decimal import Decimal
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
PURE = "src/exia/genetic_primus/market_gross_label_materialization_v1.py"
SELF = "tools/materialize_genetic_primus_market_gross_labels_v1.py"
DOC = "docs/GENETIC_PRIMUS_MARKET_GROSS_LABELS_V1.md"
PREVIOUS = "Reports/Exia/Genetic_Primus/causal_market_label_plan_v1_20261007"
DEST = "Reports/Exia/Genetic_Primus/market_gross_labels_v1_20261007"
CODE = "Reports/Exia/Genetic_Primus/market_gross_label_code_v1_20261007"
checker = runpy.run_path(str(ROOT / "tools/check_genetic_primus_causal_market_label_plan_v1.py"))
features, schedule_api = checker["previous"], checker["pure"]
old, api, kernel = checker["old"], checker["api"], checker["kernel"]
pure = runpy.run_path(str(ROOT / PURE))
read = features["read"]


def data_identity(path, cap=4 * 1024 * 1024):
    item = (ROOT / path).resolve()
    item.relative_to(ROOT / DEST / "data")
    if item.suffix != ".jsonl" or item.stat().st_size > cap:
        raise ValueError("label artifact path/cap")
    data = item.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def emit_lines(path, rows, cap=4 * 1024 * 1024):
    data = b"".join(api["canonical"](row) + b"\n" for row in rows)
    if len(data) > cap:
        raise ValueError("derived label artifact cap")
    item = (ROOT / path).resolve()
    item.relative_to(ROOT / DEST / "data")
    with item.open("xb") as output:
        output.write(data)
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def metadata():
    features["finalize"](verify=True)
    prior_result = read(PREVIOUS + "/result.json")
    prior_receipt = read(PREVIOUS + "/completion_receipt.json")
    if old["identity"](PREVIOUS + "/result.json") != prior_receipt["output_identities"]["result.json"]:
        raise ValueError("prior schedule receipt")
    if prior_result["inputs"] != prior_receipt["inputs"]:
        raise ValueError("prior schedule bank")
    bank = dict(prior_result["inputs"])
    checker["unchanged"](bank)
    for path in (PURE, SELF, DOC, PREVIOUS + "/result.json", PREVIOUS + "/completion_receipt.json"):
        bank[path] = old["identity"](path)
    checker["unchanged"](bank)
    prep = read(features["DEST"] + "/preparation.json")
    protocol = read(checker["PROTOCOL"])
    schedule = schedule_api["build"](prep["scenario"]["events"], protocol["clock"],
                                      prep["metadata"], 0, 0, kernel)
    if schedule != prior_result["hypothetical_zero_delay_schedule"]:
        raise ValueError("reconstructed scheduling mismatch")
    return bank, prep, schedule


def code_check(verify):
    bank, prep, schedule = metadata()
    B = pure["BAR_MS"]
    source = {"path": "Retrodate/fabricated_only.parquet", "sha256": "a" * 64, "size_bytes": 1}
    def row(t, price):
        return {"timestamp": t, "symbol": "SYN_ONLY", "open": price, "high": price,
                "low": price, "close": price, "volume": 0, "source_rows": 240}
    feature = {"symbol": "SYN_ONLY", "decision_asof_ms": 9 * B, "feature_available_at_ms": 9 * B,
               "kind": "candidate_features_not_label_gated", "features": {"return_12": 0.0, "efficiency_12": 0.0},
               "proposal_long": False, "prefix_sha256": "b" * 64, "fit_allowed": False, "label_fields_present": False}
    timing = schedule_api["opportunity"](9 * B, 36 * B, 0, 0, 0, 100 * B, kernel)
    entry = pure["observation"](row(10 * B, 100), source, ("SYN_ONLY",))
    samples = [pure["sample"](feature, timing, entry,
               pure["observation"](row(28 * B, price), source, ("SYN_ONLY",)),
               {"fixture": True}, 0) for price in (100, 90, 110)]
    checker["check"]([Decimal(s["gross_open_return_decimal"]) for s in samples]
                       == [Decimal(0), Decimal("-0.1"), Decimal("0.1")])
    checker["check"](all(s["proposal_long"] is False and s["outcome_filter_applied"] is False for s in samples))
    wrong = {**timing, "entry_open_ms": 9 * B}
    checker["rejected"](lambda: pure["sample"](feature, wrong, entry, samples[0]["exit_observation"], {}, 0))
    checker["rejected"](lambda: pure["gross"](0, 100))
    checker["rejected"](lambda: pure["gross"](100, float("nan")))
    checker["rejected"](lambda: pure["observation"]({**row(10 * B, 100), "source_rows": 239}, source, ("SYN_ONLY",)))
    assert_guard()
    checker["unchanged"](bank)
    result = {"schema": "exia.genetic_primus.market_gross_label_code_result/1", "inputs": bank,
              "fabricated_regressions_passed": True, "eligible_events": len(schedule["eligible"]),
              "excluded_events": len(schedule["excluded"]), "parquet_reads": 0, "market_labels": 0,
              "fit_count": 0, "safety": prep["safety"]}
    finish_metadata(CODE, result, verify)
    print("CODE OK: zero/negative/positive retained; next-open/purge guards; parquet/market-labels/fit=0")


def finish_metadata(directory, result, verify):
    body = api["canonical"](result) + b"\n"
    receipt = {"inputs": result["inputs"], "result_identity": {
        "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}, "safety": result["safety"]}
    if verify:
        if read(directory + "/result.json") != result or read(directory + "/completion_receipt.json") != receipt:
            raise ValueError("code receipt mismatch")
        if old["identity"](directory + "/result.json") != receipt["result_identity"]:
            raise ValueError("code result byte SHA")
    else:
        (ROOT / directory).mkdir(exist_ok=False)
        old["emit"](directory + "/result.json", result, api)
        old["emit"](directory + "/completion_receipt.json", receipt, api)


def assert_guard(arrow=False):
    blocked = {"numpy", "pandas", "exia", "pytest", "websockets", "aiohttp", "httpx", "requests"}
    if not arrow:
        blocked.add("pyarrow")
    if any(name.split(".")[0] in blocked for name in sys.modules):
        raise ValueError("forbidden module imported")


def preparation_value(bank, feature_prep, schedule):
    if any(value is not False for value in feature_prep["safety"].values()):
        raise ValueError("safety authority changed")
    required = sorted({r[name] for r in schedule["eligible"] for name in ("entry_open_ms", "exit_open_ms")})
    jobs = []
    for number, job in enumerate(feature_prep["scenario"]["jobs"], 1):
        audit = read(features["KEYS"] + f"/file_{job['original_index']:02d}.json")
        summary = audit["key_summary"]
        needed = [t for t in required if summary["first_timestamp_ms"] <= t <= summary["last_timestamp_ms"]]
        jobs.append({**job, "label_job": number, "required_open_timestamps": needed,
                     "expected_open_records": len(needed) * 8})
    if sum(j["expected_open_records"] for j in jobs) != len(required) * 8:
        raise ValueError("label sources do not cover all scheduled dependencies")
    return {"schema": "exia.genetic_primus.market_gross_label_preparation/1", "inputs": bank,
            "schedule": schedule, "jobs": jobs, "required_open_timestamps": required,
            "ingestion_lag_ms": 0, "execution_delay_ms": 0,
            "authorization_basis": "human_confirmed_next_4h_open_after_instant_readiness_18_bar_exit",
            "raw_label_reads_authorized_for_this_diagnostic_step": True,
            "scenario_is_assumption_not_measured_execution": True,
            "reference_open_is_not_fill": True, "net_objective_allowed": False,
            "machine_artifacts_contain_historical_opens_and_gross_labels": True,
            "cutoff_bar_start_ms": feature_prep["scenario"]["cutoff_bar_start_ms"],
            "symbols": feature_prep["metadata"]["symbols"], "safety": feature_prep["safety"]}


def prepare():
    code_check(True)
    bank, feature_prep, schedule = metadata()
    prep = preparation_value(bank, feature_prep, schedule)
    checker["unchanged"](bank)
    (ROOT / DEST).mkdir(exist_ok=False)
    (ROOT / DEST / "data").mkdir(exist_ok=False)
    old["emit"](DEST + "/preparation.json", prep, api)
    print(f"PREPARED: delay=0; strict next-open; sources={len(prep['jobs'])}; eligible={len(schedule['eligible'])}; fit=0")


def load_prep():
    prep = read(DEST + "/preparation.json")
    checker["unchanged"](prep["inputs"])
    if (prep["execution_delay_ms"] != 0 or prep["ingestion_lag_ms"] != 0
            or prep["raw_label_reads_authorized_for_this_diagnostic_step"] is not True
            or prep["net_objective_allowed"] is not False
            or any(value is not False for value in prep["safety"].values())):
        raise ValueError("diagnostic authorization changed")
    bank, prior_prep, expected = metadata()
    if prep != preparation_value(bank, prior_prep, expected):
        raise ValueError("registered scheduling/source authorization changed")
    return prep


def file_step(number):
    prep = load_prep()
    prep_identity = old["identity"](DEST + "/preparation.json")
    if type(number) is not int or not 1 <= number <= len(prep["jobs"]):
        raise ValueError("single job index required")
    job = prep["jobs"][number - 1]
    part_path = DEST + f"/part_{number:02d}.json"
    open_path = DEST + f"/data/opens_{number:02d}.jsonl"
    if any((ROOT / p).exists() for p in (part_path, open_path)):
        raise ValueError("existing/incomplete job; no blind overwrite or retry")
    parent_path = DEST + f"/part_{number-1:02d}.json" if number > 1 else None
    parent_identity = old["identity"](parent_path) if parent_path else None
    if parent_path and read(parent_path)["job"] != number - 1:
        raise ValueError("job order")
    source = job["source"]
    path = old["local"](source["path"], parquet=True)
    if path.stat().st_size != source["size_bytes"] or path.stat().st_size > old["MAX_BYTES"]:
        raise ValueError("source size")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != source["sha256"]:
        raise ValueError("source SHA")
    audit = read(features["KEYS"] + f"/file_{job['original_index']:02d}.json")
    selected, seen, included = [], set(), 0
    requested = set(job["required_open_timestamps"])
    with closing(features["decoded_bars"](content, prep["cutoff_bar_start_ms"],
                                          audit["footer_rows"], audit["decoder"])) as bars:
        for bar in bars:
            if len(bar) != 8 or {row["symbol"] for row in bar} != set(prep["symbols"]):
                raise ValueError("incomplete label source panel")
            stamps = {row["timestamp"] for row in bar}
            if len(stamps) != 1:
                raise ValueError("mixed source bar timestamps")
            included += len(bar)
            if included > job["expected_included_rows"]:
                raise ValueError("source row cap")
            if next(iter(stamps)) in requested:
                for row in sorted(bar, key=lambda r: r["symbol"]):
                    key = (row["timestamp"], row["symbol"])
                    if key in seen:
                        raise ValueError("duplicate label observation")
                    seen.add(key)
                    selected.append(pure["observation"](row, source, prep["symbols"]))
    if included != job["expected_included_rows"] or len(selected) != job["expected_open_records"]:
        raise ValueError("label source dependency count")
    if path.stat().st_size != len(content) or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("source changed during read")
    if old["identity"](DEST + "/preparation.json") != prep_identity:
        raise ValueError("preparation changed")
    if parent_path and old["identity"](parent_path) != parent_identity:
        raise ValueError("parent changed")
    assert_guard(True)
    checker["unchanged"](prep["inputs"])
    data_sha = emit_lines(open_path, selected)
    part = {"job": number, "source": source, "included_rows": included,
            "open_records": len(selected), "open_path": open_path, "open_identity": data_sha,
            "parent_identity": parent_identity, "preparation_identity": prep_identity,
            "source_sha_before_after_verified": True, "fit_count": 0, "safety": prep["safety"]}
    old["emit"](part_path, part, api)
    print(f"PART {number:02d}: opens={len(selected)}; source SHA=OK; no labels yet; fit=0")


def lines(path, cap, derived=True):
    if derived:
        data_identity(path, cap)
    else:
        features["identity_data"](path, cap)
    with (ROOT / path).open("rb") as stream:
        for index, line in enumerate(stream, 1):
            if index > 6000 or not line.strip() or len(line) > 8192:
                raise ValueError("JSONL row count/size")
            yield index, api["strict_json"](line)


def assemble(prep):
    outputs = {"preparation.json": old["identity"](DEST + "/preparation.json")}
    opens = {}
    for number, job in enumerate(prep["jobs"], 1):
        name = f"part_{number:02d}.json"
        part = read(DEST + "/" + name)
        expected_parent = outputs[f"part_{number-1:02d}.json"] if number > 1 else None
        if (part["job"] != number or part["source"] != job["source"]
                or part["parent_identity"] != expected_parent
                or part["preparation_identity"] != outputs["preparation.json"]
                or part["included_rows"] != job["expected_included_rows"]
                or part["open_records"] != job["expected_open_records"]
                or data_identity(part["open_path"]) != part["open_identity"]):
            raise ValueError("open part identity/lineage")
        count = 0
        for _, row in lines(part["open_path"], 4 * 1024 * 1024):
            key = (row["timestamp"], row["symbol"])
            if key in opens or row["source"] != job["source"]:
                raise ValueError("duplicate/substituted open record")
            opens[key] = row
            count += 1
        if count != part["open_records"]:
            raise ValueError("open row count")
        outputs[name] = old["identity"](DEST + "/" + name)
        outputs[part["open_path"].removeprefix(DEST + "/")] = part["open_identity"]
    wanted = {(t, s) for t in prep["required_open_timestamps"] for s in prep["symbols"]}
    if set(opens) != wanted:
        raise ValueError("complete open dependency bank required")
    eligible = {r["decision_asof_ms"]: r for r in prep["schedule"]["eligible"]}
    samples, keys, feature_count = [], set(), 0
    prior_prep = read(features["DEST"] + "/preparation.json")
    for number, _ in enumerate(prior_prep["scenario"]["jobs"], 1):
        prior_part = read(features["DEST"] + f"/part_{number:02d}.json")
        feature_path = prior_part["feature_path"]
        if features["identity_data"](feature_path) != prior_part["feature_identity"]:
            raise ValueError("feature source identity")
        count = 0
        for line_no, feature in lines(feature_path, 4 * 1024 * 1024, False):
            count += 1
            if feature["kind"] != "candidate_features_not_label_gated" or feature["decision_asof_ms"] not in eligible:
                continue
            schedule = eligible[feature["decision_asof_ms"]]
            key = (feature["decision_asof_ms"], feature["symbol"])
            if key in keys or feature["symbol"] not in prep["symbols"]:
                raise ValueError("duplicate/substituted sample")
            keys.add(key)
            reference = {"path": feature_path, "line_1based": line_no, "file_identity": prior_part["feature_identity"]}
            samples.append(pure["sample"](feature, schedule,
                opens[(schedule["entry_open_ms"], feature["symbol"])],
                opens[(schedule["exit_open_ms"], feature["symbol"])], reference, 0))
        if count != prior_part["feature_rows"]:
            raise ValueError("feature row count")
        feature_count += count
    expected_keys = {(t, s) for t in eligible for s in prep["symbols"]}
    if keys != expected_keys or feature_count != 4592:
        raise ValueError("sample/feature bank completeness")
    samples.sort(key=lambda row: (row["decision_asof_ms"], row["symbol"]))
    return outputs, samples, len(opens)


def finalize(verify):
    prep = load_prep()
    outputs, samples, open_count = assemble(prep)
    signs = {"negative": 0, "zero": 0, "positive": 0}
    for row in samples:
        value = Decimal(row["gross_open_return_decimal"])
        signs["negative" if value < 0 else "positive" if value > 0 else "zero"] += 1
    labels_path = DEST + "/data/gross_label_samples.jsonl"
    body = b"".join(api["canonical"](row) + b"\n" for row in samples)
    label_identity = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    if len(body) > 16 * 1024 * 1024:
        raise ValueError("sample artifact cap")
    result = {"schema": "exia.genetic_primus.market_gross_labels_result/1", "sample_rows": len(samples),
              "eligible_events": len(prep["schedule"]["eligible"]), "excluded_events": len(prep["schedule"]["excluded"]),
              "open_dependency_rows": open_count, "outcome_counts_not_strategy_performance": signs,
              "label_path": labels_path, "label_identity": label_identity,
              "registered_ingestion_lag_ms": 0, "registered_execution_delay_ms": 0,
              "labels_are_gross_diagnostics_only": True, "outcome_filter_applied": False,
              "net_objective_allowed": False, "fit_count": 0, "model_financial_evaluations": 0,
              "original_arrival_or_execution_verified": False, "safety": prep["safety"]}
    outputs[labels_path.removeprefix(DEST + "/")] = label_identity
    result_body = api["canonical"](result) + b"\n"
    outputs["result.json"] = {"bytes": len(result_body), "sha256": hashlib.sha256(result_body).hexdigest()}
    receipt = {"schema": "exia.genetic_primus.market_gross_label_receipt/1", "inputs": prep["inputs"],
               "output_identities": outputs, "safety": prep["safety"]}
    assert_guard()
    checker["unchanged"](prep["inputs"])
    features["finalize"](verify=True)
    if old["identity"](DEST + "/preparation.json") != outputs["preparation.json"]:
        raise ValueError("preparation changed during assembly")
    for name, identity in outputs.items():
        if name in ("result.json", "data/gross_label_samples.jsonl"):
            continue
        current = data_identity(DEST + "/" + name) if name.startswith("data/") else old["identity"](DEST + "/" + name)
        if current != identity:
            raise ValueError("derived dependency changed during assembly")
    if verify:
        if (read(DEST + "/result.json") != result or read(DEST + "/completion_receipt.json") != receipt
                or old["identity"](DEST + "/result.json") != outputs["result.json"]
                or data_identity(labels_path, 16 * 1024 * 1024) != label_identity):
            raise ValueError("label receipt mismatch")
    else:
        if any((ROOT / path).exists() for path in (labels_path, DEST + "/result.json", DEST + "/completion_receipt.json")):
            raise ValueError("existing/incomplete finalization; no overwrite")
        emit_lines(labels_path, samples, 16 * 1024 * 1024)
        old["emit"](DEST + "/result.json", result, api)
        old["emit"](DEST + "/completion_receipt.json", receipt, api)
    print(f"{'VERIFIED' if verify else 'FINALIZED'}: samples={len(samples)}; opens={open_count}; gross only; fit=0")


if __name__ == "__main__":
    try:
        args = sys.argv[1:]
        if args == ["--check-code"]:
            code_check(False)
        elif args == ["--verify-code"]:
            code_check(True)
        elif args == ["--prepare"]:
            prepare()
        elif len(args) == 2 and args[0] == "--file":
            file_step(int(args[1]))
        elif args in (["--finalize"], ["--verify"]):
            finalize(args == ["--verify"])
        else:
            raise ValueError("use check/verify-code, prepare, file N, finalize or verify")
    except Exception as error:
        print("LABEL MATERIALIZATION FAILED: " + type(error).__name__ + ": " + str(error)[:160])
        raise SystemExit(1) from None
