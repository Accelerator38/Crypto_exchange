"""Bounded CLI for a fixed BTC market-only fragment. Never runs trading code."""
import argparse
import csv
import gzip
import hashlib
import json
import runpy
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "Reports/Exia/Genetic_Primus/recorded_archives_v1_20260914"
OUT = ROOT / "Reports/Exia/Genetic_Primus/market_fragment_v1_20260914"
CORE = ROOT / "src/exia/genetic_primus/market_fragment_v1.py"
TESTS = ROOT / "tests/test_genetic_primus_market_fragment_v1.py"
SOURCE_HASHES = {"l2": "26535dceb3da2d1ee86463b49c78cd0d907b01b470365e7fcf1a10760676b4a7",
                 "ticker": "47db0d0633045d6c3a646b6ed37571dfd65b7d882e92a24a60f47971272703c9"}
FINAL_HASHES = {"l2": "1509c4f49f8d5270ca5177b9ad03bde3c11538d04dc91607acfc1524b7c3cc75",
                "ticker": "bc0b81171f915b04ccfe6c0b8dd131770a8886a2150c5dcacb874477cb0c0580"}
NAMES = {"l2": "incremental_book_L2", "ticker": "derivative_ticker"}


def file_sha(path, limit=150 * 1024**2):
    deadline, count, h = time.monotonic() + 5, 0, hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            count += len(block)
            if count > limit or time.monotonic() > deadline:
                raise ValueError("hash byte/time budget exceeded")
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with path.open("rb") as stream:
        data = stream.read(2 * 1024**2 + 1)
    if len(data) > 2 * 1024**2:
        raise ValueError("JSON byte budget exceeded")
    return json.loads(data)


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def contract():
    return {"schema": "genetic_primus.market_fragment_plan/1", "start_us": 1733025595000000,
            "end_exclusive_us": 1733025605000000, "cutoff_us": 1733025600000000,
            "levels_per_side": 15, "book_max_age_us": 1000000, "mark_max_age_us": 5000000,
            "age_budgets_are_diagnostic_not_calibrated": True,
            "max_scan_bytes_after_seek": 64 * 1024**2, "max_extract_seconds": 12,
            "source_hashes": dict(SOURCE_HASHES),
            "sources": {k: {"path": str(Path("C:/Users/anton/Downloads") /
                              ("bitget-futures_" + NAMES[k] + "_2024-12-01_BTCUSDT.csv.gz")),
                             "sha256": sha} for k, sha in SOURCE_HASHES.items()},
            "code_sha256": {str(p.relative_to(ROOT)): file_sha(p, 1024**2)
                            for p in (CORE, TESTS, Path(__file__).resolve())},
            "safety": {k: False for k in ("network_allowed", "credentials_allowed", "orders_enabled",
                        "paper_allowed", "live_allowed", "promotion_authority", "runtime_authority")},
            "qualification_eligible": False, "trading_authorized": False}


def plan_and_core():
    plan = read_json(OUT / "plan.json")
    if plan != contract():
        raise ValueError("fixed plan/code identity mismatch")
    return plan, runpy.run_path(str(CORE))


def audit_states(kind):
    paths = sorted(AUDIT.glob(kind + "_coverage_*.json"))
    if not paths or len(paths) > 40 or file_sha(paths[-1], 65536) != FINAL_HASHES[kind]:
        raise ValueError("frozen coverage receipt mismatch")
    states, previous = [], None
    for n, path in enumerate(paths, 1):
        state = read_json(path)
        if (path.name != kind + "_coverage_" + str(n).zfill(3) + ".json" or
                state["previous_receipt_sha256"] != previous or
                state["source_sha256"] != SOURCE_HASHES[kind]):
            raise ValueError("broken frozen coverage chain")
        states.append(state)
        previous = file_sha(path, 65536)
    return states


def extract(kind):
    plan, core = plan_and_core()
    if (OUT / (kind + "_raw.json")).exists():
        raise FileExistsError("raw fragment already exists")
    source = Path(plan["sources"][kind]["path"])
    if file_sha(source) != SOURCE_HASHES[kind]:
        raise ValueError("source hash mismatch")
    states = audit_states(kind)
    earlier = [s for s in states if s["last_local_us"] < plan["start_us"]]
    offset = earlier[-1]["offset"] if earlier else 0
    raw_rows, first_offset, seen_end, scanned_rows = [], 0, False, 0
    started = time.monotonic()
    with gzip.open(source, "rb") as stream:
        header = next(csv.reader([stream.readline(4097).decode("ascii")]))
        if len(header) != len(set(header)) or not {"exchange", "symbol", "timestamp", "local_timestamp"} <= set(header):
            raise ValueError("invalid CSV header")
        if offset:
            stream.seek(offset)
        first_offset = stream.tell()
        previous_local = earlier[-1]["last_local_us"] if earlier else None
        while True:
            if time.monotonic() - started > plan["max_extract_seconds"]:
                raise TimeoutError("extraction time budget exceeded")
            raw = stream.readline(4097)
            if not raw or len(raw) > 4096 or not raw.endswith(b"\n"):
                raise ValueError("unexpected EOF/record length before right boundary")
            scanned_rows += 1
            if stream.tell() - first_offset > plan["max_scan_bytes_after_seek"]:
                raise ValueError("scan byte budget exceeded")
            values = next(csv.reader([raw.decode("ascii")], strict=True))
            if len(values) != len(header):
                raise ValueError("CSV width mismatch")
            row = dict(zip(header, values))
            local, _ = core["identity"](row)
            if previous_local is not None and local < previous_local:
                raise ValueError("receipt chronology regression")
            previous_local = local
            if local >= plan["end_exclusive_us"]:
                seen_end = True
                break
            if local >= plan["start_us"]:
                raw_rows.append(row)
                if len(raw_rows) > (8192 if kind == "l2" else 256):
                    raise ValueError("selected row budget exceeded")
        scanned_bytes = stream.tell() - first_offset
    if not seen_end or not raw_rows or file_sha(source) != SOURCE_HASHES[kind]:
        raise ValueError("incomplete selection or changed source")
    payload = {"schema": "genetic_primus.raw_market_fragment/1", "kind": kind,
               "plan_sha256": core["object_sha"](plan), "source_sha256": SOURCE_HASHES[kind],
               "source_offset": first_offset, "scanned_bytes": scanned_bytes,
               "scanned_rows": scanned_rows, "right_boundary_observed": seen_end,
               "rows": raw_rows, "safety": dict(core["SAFETY"]),
               "qualification_eligible": False, "trading_authorized": False}
    payload["payload_sha256"] = core["object_sha"](payload)
    save(kind + "_raw.json", payload)
    print(json.dumps({"kind": kind, "rows": len(raw_rows), "scanned_bytes": scanned_bytes,
                      "right_boundary_observed": True, "payload_sha256": payload["payload_sha256"]}))


def direct_tests():
    plan, core = plan_and_core()
    tests = runpy.run_path(str(TESTS))
    passed = []
    for name in sorted(tests):
        if name.startswith("test_") and callable(tests[name]):
            tests[name]()
            passed.append(name)
    if len(passed) != 12:
        raise ValueError("unexpected regression inventory")
    save("tests.json", {"passed": passed, "count": len(passed), "plan_sha256": core["object_sha"](plan),
                        "code_sha256": plan["code_sha256"], "safety": dict(core["SAFETY"]),
                        "qualification_eligible": False, "trading_authorized": False})
    print("direct deterministic regressions: 12 PASS")


def normalize():
    plan, core = plan_and_core()
    checked = read_json(OUT / "tests.json")
    if checked["plan_sha256"] != core["object_sha"](plan) or checked["count"] != 12:
        raise ValueError("regression receipt mismatch")
    fragments = {}
    for kind in NAMES:
        payload = read_json(OUT / (kind + "_raw.json"))
        sha = payload.pop("payload_sha256")
        if (sha != core["object_sha"](payload) or payload["plan_sha256"] != core["object_sha"](plan)
                or payload["source_sha256"] != SOURCE_HASHES[kind] or not payload["right_boundary_observed"]):
            raise ValueError("raw fragment identity mismatch")
        fragments[kind] = payload
    result = core["inspect_fragment"](fragments["l2"]["rows"], fragments["ticker"]["rows"], plan)
    save("fragment.json", result)
    view = result["asof"]
    print(json.dumps({"book_snapshots": len(result["book_snapshots"]), "marks": len(result["marks"]),
                      "causal_check": view["future_prefix_invariance"],
                      "book_receive_age_us": plan["cutoff_us"] - view["book"]["received_ts_us"],
                      "mark_receive_age_us": plan["cutoff_us"] - view["mark"]["received_ts_us"],
                      "fragment_sha256": result["fragment_sha256"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "tests", "extract-l2", "extract-ticker", "normalize"))
    stage = parser.parse_args().stage
    try:
        if stage == "plan":
            save("plan.json", contract()); print("fixed 10-second market-only plan saved; safety=false")
        elif stage == "tests":
            direct_tests()
        elif stage.startswith("extract-"):
            extract(stage.removeprefix("extract-"))
        else:
            normalize()
    except Exception as exc:
        print("FAIL_CLOSED " + type(exc).__name__ + ": " + str(exc)[:180])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
