"""Fixed September 22--27 intake; no model, credentials, or trading runtime.

One fetch invocation: one symbol/origin, <=6 public REST requests, no retries.
All artifacts are exclusive-create. Existing pages may only be verified/reused.
Plan, build and seal are offline. The previously frozen intake is not modified.
"""
import argparse
import hashlib
import json
import math
import runpy
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Retrodate/genetic_primus_v8_clean_ohlcv_tail_20260922_20260927"
LEGACY = ROOT / "tools/fetch_genetic_primus_v8_clean_ohlcv_backfill.py"
LEGACY_SHA = "22cee7f62136b70af1253cbf1c8519292b8f09ddb70aeefb3c18ef8c86dae89b"
CONTRACT = ROOT / "configs/genetic_primus_prospective_3d_v1.json"
CONTRACT_SHA = "0002201ea664d868a33cdc7e4c8ceb6630bf7a072fef8eccee86927ed4f6488f"
SYMBOLS = ("ADA", "BNB", "BTC", "DOGE", "ETH", "LINK", "SOL", "XRP")
BASE = "https://api.bitget.com/api/v3/market/"
SAFE = {k: False for k in ("credentials_allowed", "orders_enabled", "paper_allowed",
        "live_allowed", "promotion_authority", "runtime_authority", "network_allowed")}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    path = Path(path)
    require(path.stat().st_size <= 4194304, "JSON size budget exceeded")
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def validators():
    require(file_sha(LEGACY) == LEGACY_SHA, "frozen validator changed")
    # Only pure validation functions are called; no mutation of legacy globals.
    return runpy.run_path(str(LEGACY))


def bounds(origin):
    require(type(origin) is int and origin in (8, 9), "only origins 08/09 admitted")
    start = datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(days=3 * (origin - 1))
    return int(start.timestamp() * 1000), int((start + timedelta(days=3)).timestamp() * 1000)


def plan():
    require(file_sha(CONTRACT) == CONTRACT_SHA, "frozen research contract changed")
    require(file_sha(LEGACY) == LEGACY_SHA, "frozen validator changed")
    return {"schema": "v8-clean-ohlcv-tail/1", "code_sha256": file_sha(__file__),
            "validator_sha256": LEGACY_SHA, "contract_sha256": CONTRACT_SHA,
            "origins": [8, 9], "symbols": list(SYMBOLS), "interval": "1m",
            "origin_bounds": {str(i): list(bounds(i)) for i in (8, 9)},
            "max_requests_per_invocation": 6, "deadline_seconds": 20,
            "max_response_bytes": 262144, "retries": 0,
            "intake_network_scope": "explicitly authorized public Bitget OHLCV REST only",
            "endpoints": [BASE + "candles", BASE + "history-candles"],
            "primary_bounds": "(startTime,endTime]; request a-60000,b-60000 for [a,b)",
            "crosscheck": "last 100 minutes against exclusive-end history-candles",
            "model_safety": SAFE, "qualification_allowed": False,
            "training_allowed": False, "strategy_results_computed": False}


def frozen_plan():
    p = read_json(OUT / "plan.json")
    require(p == plan(), "intake plan changed")
    return p


def specs(origin, symbol):
    require(symbol in SYMBOLS, "symbol outside full8")
    start, end = bounds(origin)
    result = []
    for a in range(start, end, 1000 * 60000):
        b = min(a + 1000 * 60000, end)
        params = {"category": "USDT-FUTURES", "symbol": symbol + "USDT", "interval": "1m",
                  "type": "market", "startTime": str(a - 60000), "endTime": str(b - 60000),
                  "limit": str((b - a) // 60000)}
        result.append((BASE + "candles", params, a, b))
    a = end - 100 * 60000
    result.append((BASE + "history-candles",
                   {"category": "USDT-FUTURES", "symbol": symbol + "USDT", "interval": "1m",
                    "type": "market", "startTime": str(a), "endTime": str(end), "limit": "100"}, a, end))
    return result


def collect(origin, symbol, fetch=False):
    frozen = frozen_plan()
    start, end = bounds(origin)
    require(end <= int(datetime.now(timezone.utc).timestamp() * 1000), "origin still open")
    v = validators()
    plan_sha = object_sha(frozen)
    deadline, last_request, calls = time.monotonic() + 20, 0.0, 0
    rows, receipts = [], []
    for index, spec in enumerate(specs(origin, symbol)):
        endpoint, params, a, b = spec
        path = OUT / f"raw/origin_{origin:02d}_{symbol}_page_{index:02d}.json"
        if path.exists():
            record = read_json(path)
        else:
            require(fetch, "missing raw page; offline build cannot fetch")
            remaining = deadline - time.monotonic()
            require(remaining >= 1 and calls < 6, "request/time budget exhausted")
            delay = max(0, .11 - (time.monotonic() - last_request))
            if delay:
                time.sleep(delay)
            url = endpoint + "?" + urllib.parse.urlencode(params)
            last_request = time.monotonic()
            with urllib.request.urlopen(url, timeout=min(8, remaining)) as response:
                require(response.geturl() == url and response.status == 200, "HTTP status/redirect")
                raw = response.read(262145)
            calls += 1
            require(len(raw) <= 262144, "response size exceeded")
            record = {"endpoint": endpoint, "params": params, "plan_sha256": plan_sha,
                      "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                      "response_utf8": raw.decode("utf-8"), "response_sha256": hashlib.sha256(raw).hexdigest()}
            record["record_sha256"] = object_sha(record)
            v["verify_page"](record, spec, plan_sha)
            save(path, record)
        page = v["verify_page"](record, spec, plan_sha)
        receipts.append({"path": path.relative_to(ROOT).as_posix(), "sha256": file_sha(path)})
        if index < 5:
            rows.extend(page)
        else:
            require([[Decimal(x) for x in r] for r in rows[-100:]] ==
                    [[Decimal(x) for x in r] for r in page], "history crosscheck mismatch")
    rows = v["valid_rows"](rows, start, end)
    chunk = {"schema": "v8-clean-ohlcv-tail-chunk/1", "origin": origin, "symbol": symbol,
             "plan_sha256": plan_sha, "rows": len(rows), "start_ms": start,
             "end_exclusive_ms": end, "raw_pages": receipts, "history_tail_crosscheck_passed": True,
             "model_safety": SAFE, "qualification_allowed": False}
    path = OUT / f"chunks/origin_{origin:02d}_{symbol}.json"
    if path.exists():
        require(read_json(path) == chunk, "chunk manifest mismatch")
    else:
        require(fetch, "missing chunk manifest")
        save(path, chunk)
    return rows


def frames(origin):
    import pandas as pd
    start, end = bounds(origin)
    rows = []
    for symbol in SYMBOLS:
        rows.extend((int(r[0]), symbol + "/USDT", *map(float, r[1:6])) for r in collect(origin, symbol))
    minute = pd.DataFrame(rows, columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    minute = minute.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    minute["symbol"] = minute["symbol"].astype("category")
    for c in ("open", "high", "low", "close", "volume"):
        require(all(math.isfinite(x) for x in minute[c]), "nonfinite float conversion")
    work = minute.assign(timestamp=minute.timestamp // 14400000 * 14400000)
    four = work.groupby(["timestamp", "symbol"], observed=True, sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), source_rows=("close", "size")).reset_index()
    require(len(minute) == 34560 and len(four) == 144 and four.source_rows.eq(240).all(), "row coverage")
    for symbol in SYMBOLS:
        name = symbol + "/USDT"
        require(minute[minute.symbol == name].timestamp.tolist() == list(range(start, end, 60000)), "minute grid")
        require(four[four.symbol == name].timestamp.tolist() == list(range(start, end, 14400000)), "four-hour grid")
    return minute, four


def build(origin):
    import pyarrow as pa
    import pyarrow.parquet as pq
    minute, four = frames(origin)
    paths = [OUT / f"derived_v1/origin_{origin:02d}_{tf}.parquet" for tf in ("1m", "240m")]
    manifest_path = OUT / f"derived_v1/origin_{origin:02d}_manifest.json"
    require(not any(p.exists() for p in paths + [manifest_path]), "output already exists; never overwrite")
    files = []
    for path, frame in zip(paths, (minute, four)):
        table = pa.Table.from_pandas(frame, preserve_index=False)
        index = table.schema.get_field_index("symbol")
        table = table.set_column(index, "symbol", table.column(index).cast(pa.dictionary(pa.int8(), pa.string())))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            pq.write_table(table, stream, compression="snappy")
        restored = pq.read_table(path)
        require(table.equals(restored) and frame.equals(restored.to_pandas()), "Parquet roundtrip mismatch")
        files.append({"path": path.relative_to(ROOT).as_posix(), "sha256": file_sha(path), "rows": len(frame)})
    start, end = bounds(origin)
    chunks = [{"path": (OUT / f"chunks/origin_{origin:02d}_{s}.json").relative_to(ROOT).as_posix(),
               "sha256": file_sha(OUT / f"chunks/origin_{origin:02d}_{s}.json")} for s in SYMBOLS]
    result = {"schema": "v8-clean-ohlcv-tail-derived/1", "origin": origin,
              "plan_sha256": object_sha(frozen_plan()), "code_sha256": file_sha(__file__),
              "contract_sha256": CONTRACT_SHA, "files": files, "chunks": chunks,
              "start_ms": start, "end_exclusive_ms": end, "model_safety": SAFE,
              "qualification_allowed": False, "training_allowed": False,
              "strategy_results_computed": False, "arrow_and_pandas_roundtrip_verified": True}
    save(manifest_path, result)
    return {"origin": origin, "minute_rows": len(minute), "four_hour_rows": len(four)}


def seal():
    import pyarrow.parquet as pq
    frozen_plan()
    manifests, files = [], []
    raw_count = 0
    for origin in (8, 9):
        path = OUT / f"derived_v1/origin_{origin:02d}_manifest.json"
        m = read_json(path)
        require(m["plan_sha256"] == object_sha(frozen_plan()) and m["code_sha256"] == file_sha(__file__), "manifest pin")
        require(m["model_safety"] == SAFE and not m["qualification_allowed"], "manifest safety")
        expected, expected_four = frames(origin)
        for item in m["files"] + m["chunks"]:
            require(file_sha(ROOT / item["path"]) == item["sha256"], "derived/chunk hash mismatch")
        minute, four = [pq.read_table(ROOT / f["path"]).to_pandas() for f in m["files"]]
        require(minute.equals(expected) and four.equals(expected_four), "raw-to-Parquet mismatch")
        # Independent positional aggregation, not another groupby call.
        for symbol in SYMBOLS:
            part = minute[minute.symbol == symbol + "/USDT"].reset_index(drop=True)
            bars = four[four.symbol == symbol + "/USDT"].reset_index(drop=True)
            for i in range(18):
                block, bar = part.iloc[i * 240:(i + 1) * 240], bars.iloc[i]
                require(bar.open == block.open.iloc[0] and bar.high == max(block.high)
                        and bar.low == min(block.low) and bar.close == block.close.iloc[-1]
                        and math.isclose(bar.volume, math.fsum(block.volume), rel_tol=1e-12, abs_tol=1e-9),
                        "independent four-hour aggregation mismatch")
        raw_count += sum(len(read_json(ROOT / c["path"])["raw_pages"]) for c in m["chunks"])
        manifests.append({"path": path.relative_to(ROOT).as_posix(), "sha256": file_sha(path)})
        files.extend(m["files"])
    require(raw_count == 96, "raw receipt count")
    result = {"schema": "v8-clean-ohlcv-tail-seal/1", "plan_sha256": object_sha(frozen_plan()),
              "code_sha256": file_sha(__file__), "contract_sha256": CONTRACT_SHA,
              "manifests": manifests, "files": files, "minute_rows": 69120,
              "four_hour_rows": 288, "raw_receipts": raw_count, "crosschecks": 16,
              "model_safety": SAFE, "qualification_allowed": False,
              "strategy_results_computed": False, "independent_aggregation_verified": True}
    result["seal_sha256"] = object_sha(result)
    save(OUT / "dataset_seal_v1.json", result)
    return {k: result[k] for k in ("minute_rows", "four_hour_rows", "seal_sha256")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "fetch", "build", "seal"))
    parser.add_argument("--origin", type=int, choices=(8, 9))
    parser.add_argument("--symbol", choices=SYMBOLS)
    args = parser.parse_args()
    if args.stage == "plan":
        p = plan()
        save(OUT / "plan.json", p)
        result = {"plan_sha256": object_sha(p)}
    elif args.stage == "fetch":
        require(args.origin is not None and args.symbol is not None, "origin and symbol required")
        rows = collect(args.origin, args.symbol, fetch=True)
        result = {"origin": args.origin, "symbol": args.symbol, "rows": len(rows), "crosscheck": True}
    elif args.stage == "build":
        result = build(args.origin)
    else:
        result = seal()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
