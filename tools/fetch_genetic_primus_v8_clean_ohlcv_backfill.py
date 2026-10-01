"""Bounded public-REST OHLCV intake, separate from all trading/research runtimes.

One fetch command handles one fixed symbol/origin, <=6 requests, no automatic
retries or parallelism. Existing raw pages can be verified and reused after
interruption. Only NEW data/artifacts are written. No keys are read or used.
"""
import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Retrodate/genetic_primus_v8_clean_ohlcv_backfill_20260907_20260921_v2"
CONTRACT = ROOT / "configs/genetic_primus_prospective_3d_v1.json"
CONTRACT_SHA = "0002201ea664d868a33cdc7e4c8ceb6630bf7a072fef8eccee86927ed4f6488f"
SYMBOLS = ("ADA", "BNB", "BTC", "DOGE", "ETH", "LINK", "SOL", "XRP")
BASE = "https://api.bitget.com/api/v3/market/"
SAFE = {k: False for k in ("credentials_allowed", "orders_enabled", "paper_allowed",
                          "live_allowed", "promotion_authority", "runtime_authority")}


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with path.open("rb") as stream:
        raw = stream.read(4 * 1024**2 + 1)
    if len(raw) > 4 * 1024**2:
        raise ValueError("JSON size budget exceeded")
    return json.loads(raw)


def save(name, value):
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def bounds(origin):
    if origin not in range(3, 8):
        raise ValueError("origin outside fixed backlog")
    start = datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(days=3 * (origin - 1))
    return int(start.timestamp() * 1000), int((start + timedelta(days=3)).timestamp() * 1000)


def plan():
    if file_sha(CONTRACT) != CONTRACT_SHA:
        raise ValueError("frozen research contract changed")
    return {"schema": "genetic_primus.public_ohlcv_intake/1", "contract_sha256": CONTRACT_SHA,
            "downloader_sha256": file_sha(Path(__file__).resolve()),
            "symbols": list(SYMBOLS), "origins": {str(i): list(bounds(i)) for i in range(3, 8)},
            "category": "USDT-FUTURES", "interval": "1m", "type": "market",
            "primary_endpoint": BASE + "candles", "crosscheck_endpoint": BASE + "history-candles",
            "page_minutes": 1000, "history_tail_crosscheck_minutes": 100,
            "primary_bounds": "(startTime,endTime]; request start=a-1minute, end=b-1minute",
            "max_requests_per_command": 6, "max_command_network_seconds": 20,
            "max_response_bytes": 262144, "automatic_retries": 0, "parallelism": 1,
            "network_scope": "user_authorized_public_historical_GET_only",
            "evaluation_network_allowed": False, "training_invoked": False,
            "source_revision_policy": "immutable raw responses; mismatch fails closed",
            "safety": dict(SAFE), "qualification_eligible": False, "trading_authorized": False}


def frozen_plan():
    saved = read_json(OUT / "plan.json")
    if saved != plan():
        raise ValueError("intake plan/code mismatch")
    return saved


def valid_rows(rows, start, end):
    if not isinstance(rows, list) or len(rows) != (end - start) // 60000:
        raise ValueError("wrong candle count")
    checked = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 7 or not all(isinstance(x, str) for x in row):
            raise ValueError("expected seven string candle fields")
        if not row[0].isascii() or not row[0].isdigit() or len(row[0]) != 13:
            raise ValueError("invalid millisecond candle timestamp")
        numbers = []
        for raw in row[1:]:
            if len(raw) > 40 or not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", raw):
                raise ValueError("invalid nonnegative decimal")
            numbers.append(Decimal(raw))
        op, hi, lo, close, volume, quote_volume = numbers
        if min(op, hi, lo, close) <= 0 or not lo <= min(op, close) <= max(op, close) <= hi:
            raise ValueError("invalid OHLC bounds")
        checked.append(row)
    checked.sort(key=lambda r: int(r[0]))
    if [int(r[0]) for r in checked] != list(range(start, end, 60000)):
        raise ValueError("missing, duplicate, shifted or extra candles")
    return checked


def page_path(origin, symbol, index):
    return f"raw/origin_{origin:02d}_{symbol}_page_{index:02d}.json"


def specifications(origin, symbol):
    start, end = bounds(origin)
    spans = [(a, min(a + 1000 * 60000, end)) for a in range(start, end, 1000 * 60000)]
    specs = []
    for a, b in spans:
        params = {"category": "USDT-FUTURES", "symbol": symbol + "USDT", "interval": "1m",
                  "type": "market", "startTime": str(a - 60000), "endTime": str(b - 60000),
                  "limit": str((b - a) // 60000)}
        specs.append((BASE + "candles", params, a, b))
    # Unlike candles, history-candles has an EXCLUSIVE interval-aligned endTime.
    a = end - 100 * 60000
    params = {"category": "USDT-FUTURES", "symbol": symbol + "USDT", "interval": "1m",
              "type": "market", "startTime": str(a), "endTime": str(end), "limit": "100"}
    specs.append((BASE + "history-candles", params, a, end))
    return specs


def verify_page(record, spec, plan_sha):
    endpoint, params, start, end = spec
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    if (record["record_sha256"] != object_sha(body) or record["plan_sha256"] != plan_sha
            or record["endpoint"] != endpoint or record["params"] != params):
        raise ValueError("raw page provenance mismatch")
    raw = record["response_utf8"].encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != record["response_sha256"]:
        raise ValueError("raw response hash mismatch")
    payload = json.loads(raw)
    if payload.get("code") != "00000":
        raise ValueError("exchange rejected request")
    return valid_rows(payload.get("data"), start, end)


def fetch(origin, symbol):
    contract = frozen_plan()
    plan_sha, specs = object_sha(contract), specifications(origin, symbol)
    deadline, last_request, calls = time.monotonic() + 20, 0.0, 0
    rows, receipts, crosscheck = [], [], None
    for index, spec in enumerate(specs):
        endpoint, params, start, end = spec
        name = page_path(origin, symbol, index)
        if (OUT / name).exists():
            record = read_json(OUT / name)
            page = verify_page(record, spec, plan_sha)
        else:
            remaining = deadline - time.monotonic()
            if remaining < 1 or calls >= 6:
                raise TimeoutError("command budget reached; verified pages retained for resume")
            delay = max(0.0, 0.11 - (time.monotonic() - last_request))
            if delay:
                time.sleep(delay)
            url = endpoint + "?" + urllib.parse.urlencode(params)
            last_request = time.monotonic()
            with urllib.request.urlopen(url, timeout=min(8, remaining)) as response:
                if response.geturl() != url or response.status != 200:
                    raise ValueError("unexpected redirect or HTTP status")
                raw = response.read(262145)
            calls += 1
            if len(raw) > 262144:
                raise ValueError("response byte budget exceeded")
            record = {"endpoint": endpoint, "params": params, "plan_sha256": plan_sha,
                      "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                      "response_utf8": raw.decode("utf-8"), "response_sha256": hashlib.sha256(raw).hexdigest()}
            record["record_sha256"] = object_sha(record)
            page = verify_page(record, spec, plan_sha)
            save(name, record)
        receipts.append({"path": name, "sha256": file_sha(OUT / name)})
        if index == len(specs) - 1:
            crosscheck = page
        else:
            rows.extend(page)
    rows = valid_rows(rows, *bounds(origin))
    # Compare numeric values, not harmless differences in decimal formatting.
    canonical = lambda data: [[Decimal(x) for x in r] for r in data]
    if canonical(rows[-100:]) != canonical(crosscheck):
        raise ValueError("primary/history crosscheck differs; origin invalidated")
    manifest = {"schema": "genetic_primus.ohlcv_chunk/1", "origin": origin, "symbol": symbol,
                "plan_sha256": plan_sha, "rows": len(rows), "start_ms": bounds(origin)[0],
                "end_exclusive_ms": bounds(origin)[1], "raw_pages": receipts,
                "history_tail_crosscheck_passed": True, "safety": dict(SAFE),
                "qualification_eligible": False, "trading_authorized": False}
    name = f"chunks/origin_{origin:02d}_{symbol}.json"
    if (OUT / name).exists():
        if read_json(OUT / name) != manifest:
            raise ValueError("existing chunk differs; never overwrite")
    else:
        save(name, manifest)
    print(json.dumps({"origin": origin, "symbol": symbol, "rows": len(rows),
                      "new_requests": calls, "history_crosscheck": True}))


def chunk_rows(origin, symbol, expected_plan_sha):
    manifest = read_json(OUT / f"chunks/origin_{origin:02d}_{symbol}.json")
    if (manifest["plan_sha256"] != expected_plan_sha or manifest["origin"] != origin
            or manifest["symbol"] != symbol or not manifest["history_tail_crosscheck_passed"]
            or len(manifest["raw_pages"]) != 6):
        raise ValueError("chunk identity mismatch")
    data = []
    for index, spec in enumerate(specifications(origin, symbol)):
        name = page_path(origin, symbol, index)
        receipt = manifest["raw_pages"][index]
        if receipt["path"] != name or file_sha(OUT / name) != receipt["sha256"]:
            raise ValueError("raw page changed after commitment")
        page = verify_page(read_json(OUT / name), spec, expected_plan_sha)
        if index < 5:
            data.extend(page)
        elif [[Decimal(x) for x in r] for r in data[-100:]] != [[Decimal(x) for x in r] for r in page]:
            raise ValueError("history crosscheck changed")
    return valid_rows(data, *bounds(origin))


def build(origin):
    contract = frozen_plan()
    targets = [OUT / f"parts/origin_{origin:02d}_{tf}m.parquet" for tf in (1, 240)]
    if any(p.exists() for p in targets) or (OUT / f"origin_{origin:02d}_manifest.json").exists():
        raise FileExistsError("derived artifact exists; no overwrite")
    # Only this offline stage imports dataframe/parquet libraries; no model packages.
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = []
    plan_sha = object_sha(contract)
    for symbol in SYMBOLS:
        for r in chunk_rows(origin, symbol, plan_sha):
            records.append((int(r[0]), symbol + "/USDT", *(float(v) for v in r[1:6])))
    minute = pd.DataFrame(records, columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    minute = minute.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    minute["symbol"] = minute["symbol"].astype("category")
    grouped = minute.assign(timestamp=(minute.timestamp // 14400000) * 14400000).groupby(
        ["timestamp", "symbol"], observed=True, sort=True)
    bars = grouped.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                       close=("close", "last"), volume=("volume", "sum"), source_rows=("close", "size")).reset_index()
    if len(minute) != 34560 or len(bars) != 144 or not (bars.source_rows == 240).all():
        raise ValueError("unexpected full8/minute/4h grid")
    files = []
    for target, frame in zip(targets, (minute, bars)):
        target.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        with target.open("xb") as stream:
            pq.write_table(table, stream, compression="snappy")
        restored = pq.read_table(target)
        if not table.equals(restored):
            raise ValueError("Parquet roundtrip differs")
        files.append({"path": str(target.relative_to(ROOT)), "sha256": file_sha(target),
                      "bytes": target.stat().st_size, "rows": len(frame)})
    save(f"origin_{origin:02d}_manifest.json", {
        "schema": "genetic_primus.ohlcv_origin_dataset/1", "origin": origin,
        "plan_sha256": plan_sha, "contract_sha256": CONTRACT_SHA, "files": files,
        "chunks": [{"path": f"chunks/origin_{origin:02d}_{s}.json",
                    "sha256": file_sha(OUT / f"chunks/origin_{origin:02d}_{s}.json")} for s in SYMBOLS],
        "start_ms": bounds(origin)[0], "end_exclusive_ms": bounds(origin)[1],
        "safety": dict(SAFE), "training_invoked": False, "qualification_eligible": False,
        "trading_authorized": False, "strategy_results_computed": False})
    print(json.dumps({"origin": origin, "minute_rows": len(minute), "four_hour_rows": len(bars),
                      "parquet_roundtrip": True, "network_requests": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "fetch", "build"))
    parser.add_argument("--origin", type=int, choices=range(3, 8))
    parser.add_argument("--symbol", choices=SYMBOLS)
    args = parser.parse_args()
    try:
        if args.stage == "plan":
            save("plan.json", plan()); print("immutable intake plan saved")
        elif args.stage == "fetch":
            if args.origin is None or args.symbol is None:
                raise ValueError("fetch requires origin and symbol")
            fetch(args.origin, args.symbol)
        else:
            if args.origin is None:
                raise ValueError("build requires origin")
            build(args.origin)
    except Exception as exc:
        print("FAIL_CLOSED " + type(exc).__name__ + ": " + str(exc)[:160])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
