"""Fixed public OHLCV intake and offline seal for [2026-09-28, 2026-10-01) UTC.

The only network stage is fetch: one symbol, six bounded REST calls, no retries.
All outputs are exclusive-create and model execution is out of scope.
"""
import argparse
import hashlib
import json
import math
import runpy
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Retrodate/genetic_primus_v8_clean_ohlcv_final_20260928_20260930"
PRIOR = ROOT / "tools/fetch_genetic_primus_v8_clean_ohlcv_tail_v1.py"
PRIOR_SHA = "93774ec4b1e40857def7fd9096f6093735e12706c74df1fe1eac5fe9e64cb881"
CONTRACT = ROOT / "configs/genetic_primus_prospective_3d_v1.json"
CONTRACT_SHA = "0002201ea664d868a33cdc7e4c8ceb6630bf7a072fef8eccee86927ed4f6488f"
START, END = 1790553600000, 1790812800000
SYMBOLS = ("ADA", "BNB", "BTC", "DOGE", "ETH", "LINK", "SOL", "XRP")
BASE = "https://api.bitget.com/api/v3/market/"
SAFE = {k: False for k in ("credentials_allowed", "orders_enabled", "paper_allowed",
        "live_allowed", "promotion_authority", "runtime_authority", "network_allowed")}


def common():
    require(hashlib.sha256(PRIOR.read_bytes()).hexdigest() == PRIOR_SHA, "prior intake code changed")
    require(hashlib.sha256(CONTRACT.read_bytes()).hexdigest() == CONTRACT_SHA, "contract changed")
    return runpy.run_path(str(PRIOR))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def plan():
    c = common()
    require(END - START == 4320 * 60000, "origin duration")
    return {"schema": "v8-clean-ohlcv-final/1", "code_sha256": c["file_sha"](__file__),
            "prior_intake_sha256": PRIOR_SHA, "contract_sha256": CONTRACT_SHA,
            "origin": 10, "start_ms": START, "end_exclusive_ms": END,
            "symbols": list(SYMBOLS), "interval": "1m", "max_requests_per_invocation": 6,
            "deadline_seconds": 20, "max_response_bytes": 262144, "retries": 0,
            "intake_network_scope": "authorized public Bitget OHLCV REST only",
            "primary_bounds": "(startTime,endTime]; request a-60000,b-60000 for [a,b)",
            "crosscheck": "last 100 minutes against exclusive-end history-candles",
            "model_safety": SAFE, "qualification_allowed": False,
            "training_allowed": False, "strategy_results_computed": False}


def frozen_plan(c):
    p = c["read_json"](OUT / "plan.json")
    require(p == plan(), "plan changed")
    return p


def specs(symbol):
    require(symbol in SYMBOLS, "symbol outside full8")
    result = []
    for a in range(START, END, 1000 * 60000):
        b = min(a + 1000 * 60000, END)
        params = {"category": "USDT-FUTURES", "symbol": symbol + "USDT", "interval": "1m",
                  "type": "market", "startTime": str(a - 60000), "endTime": str(b - 60000),
                  "limit": str((b - a) // 60000)}
        result.append((BASE + "candles", params, a, b))
    a = END - 100 * 60000
    result.append((BASE + "history-candles",
                   {"category": "USDT-FUTURES", "symbol": symbol + "USDT", "interval": "1m",
                    "type": "market", "startTime": str(a), "endTime": str(END), "limit": "100"}, a, END))
    return result


def collect(symbol, fetch=False):
    c = common()
    p = frozen_plan(c)
    require(END <= int(datetime.now(timezone.utc).timestamp() * 1000), "origin still open")
    validator = c["validators"]()["verify_page"]
    valid_rows = c["validators"]()["valid_rows"]
    plan_sha = c["object_sha"](p)
    deadline, last_request, calls = time.monotonic() + 20, 0.0, 0
    rows, receipts = [], []
    for index, spec in enumerate(specs(symbol)):
        endpoint, params, _, _ = spec
        path = OUT / f"raw/origin_10_{symbol}_page_{index:02d}.json"
        if path.exists():
            record = c["read_json"](path)
        else:
            require(fetch, "offline stage missing raw page")
            remaining = deadline - time.monotonic()
            require(remaining >= 1 and calls < 6, "time/request budget exhausted")
            delay = max(0, .11 - (time.monotonic() - last_request))
            if delay:
                time.sleep(delay)
            url = endpoint + "?" + urllib.parse.urlencode(params)
            last_request = time.monotonic()
            with urllib.request.urlopen(url, timeout=min(8, remaining)) as response:
                require(response.geturl() == url and response.status == 200, "HTTP status/redirect")
                raw = response.read(262145)
            calls += 1
            require(len(raw) <= 262144, "response size budget")
            record = {"endpoint": endpoint, "params": params, "plan_sha256": plan_sha,
                      "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                      "response_utf8": raw.decode("utf-8"), "response_sha256": hashlib.sha256(raw).hexdigest()}
            record["record_sha256"] = c["object_sha"](record)
            validator(record, spec, plan_sha)
            c["save"](path, record)
        page = validator(record, spec, plan_sha)
        receipts.append({"path": path.relative_to(ROOT).as_posix(), "sha256": c["file_sha"](path)})
        if index < 5:
            rows.extend(page)
        else:
            require([[Decimal(x) for x in r] for r in rows[-100:]] ==
                    [[Decimal(x) for x in r] for r in page], "history crosscheck")
    rows = valid_rows(rows, START, END)
    chunk = {"schema": "v8-clean-ohlcv-final-chunk/1", "origin": 10, "symbol": symbol,
             "plan_sha256": plan_sha, "rows": len(rows), "start_ms": START,
             "end_exclusive_ms": END, "raw_pages": receipts, "history_tail_crosscheck_passed": True,
             "model_safety": SAFE, "qualification_allowed": False}
    path = OUT / f"chunks/origin_10_{symbol}.json"
    if path.exists():
        require(c["read_json"](path) == chunk, "chunk manifest changed")
    else:
        require(fetch, "missing chunk manifest")
        c["save"](path, chunk)
    return rows


def frames():
    import pandas as pd
    rows = []
    for symbol in SYMBOLS:
        rows.extend((int(r[0]), symbol + "/USDT", *map(float, r[1:6])) for r in collect(symbol))
    minute = pd.DataFrame(rows, columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    minute = minute.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    minute["symbol"] = minute["symbol"].astype("category")
    for field in ("open", "high", "low", "close", "volume"):
        require(all(math.isfinite(x) for x in minute[field]), "nonfinite float conversion")
    work = minute.assign(timestamp=minute.timestamp // 14400000 * 14400000)
    four = work.groupby(["timestamp", "symbol"], observed=True, sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), source_rows=("close", "size")).reset_index()
    require(len(minute) == 34560 and len(four) == 144 and four.source_rows.eq(240).all(), "row coverage")
    for symbol in SYMBOLS:
        name = symbol + "/USDT"
        require(minute[minute.symbol == name].timestamp.tolist() == list(range(START, END, 60000)), "minute grid")
        require(four[four.symbol == name].timestamp.tolist() == list(range(START, END, 14400000)), "four-hour grid")
    return minute, four


def build():
    import pyarrow as pa
    import pyarrow.parquet as pq
    c = common()
    minute, four = frames()
    paths = [OUT / f"derived_v1/origin_10_{tf}.parquet" for tf in ("1m", "240m")]
    manifest_path = OUT / "derived_v1/origin_10_manifest.json"
    require(not any(p.exists() for p in paths + [manifest_path]), "output exists; never overwrite")
    files = []
    for path, frame in zip(paths, (minute, four)):
        table = pa.Table.from_pandas(frame, preserve_index=False)
        index = table.schema.get_field_index("symbol")
        table = table.set_column(index, "symbol", table.column(index).cast(pa.dictionary(pa.int8(), pa.string())))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            pq.write_table(table, stream, compression="snappy")
        restored = pq.read_table(path)
        require(table.equals(restored) and frame.equals(restored.to_pandas()), "Parquet roundtrip")
        files.append({"path": path.relative_to(ROOT).as_posix(), "sha256": c["file_sha"](path), "rows": len(frame)})
    chunks = [{"path": (OUT / f"chunks/origin_10_{s}.json").relative_to(ROOT).as_posix(),
               "sha256": c["file_sha"](OUT / f"chunks/origin_10_{s}.json")} for s in SYMBOLS]
    result = {"schema": "v8-clean-ohlcv-final-derived/1", "origin": 10,
              "plan_sha256": c["object_sha"](frozen_plan(c)), "code_sha256": c["file_sha"](__file__),
              "contract_sha256": CONTRACT_SHA, "files": files, "chunks": chunks,
              "start_ms": START, "end_exclusive_ms": END, "model_safety": SAFE,
              "qualification_allowed": False, "training_allowed": False,
              "strategy_results_computed": False, "arrow_and_pandas_roundtrip_verified": True}
    c["save"](manifest_path, result)
    return {"minute_rows": len(minute), "four_hour_rows": len(four)}


def seal():
    import pyarrow.parquet as pq
    c = common()
    mpath = OUT / "derived_v1/origin_10_manifest.json"
    m = c["read_json"](mpath)
    require(m["plan_sha256"] == c["object_sha"](frozen_plan(c)) and m["code_sha256"] == c["file_sha"](__file__), "manifest pin")
    require(m["model_safety"] == SAFE and not m["qualification_allowed"], "safety")
    for item in m["files"] + m["chunks"]:
        require(c["file_sha"](ROOT / item["path"]) == item["sha256"], "derived/chunk hash")
    expected_minute, expected_four = frames()
    minute, four = [pq.read_table(ROOT / f["path"]).to_pandas() for f in m["files"]]
    require(minute.equals(expected_minute) and four.equals(expected_four), "raw-to-Parquet mismatch")
    for symbol in SYMBOLS:
        part = minute[minute.symbol == symbol + "/USDT"].reset_index(drop=True)
        bars = four[four.symbol == symbol + "/USDT"].reset_index(drop=True)
        for i in range(18):
            block, bar = part.iloc[i * 240:(i + 1) * 240], bars.iloc[i]
            require(bar.open == block.open.iloc[0] and bar.high == max(block.high)
                    and bar.low == min(block.low) and bar.close == block.close.iloc[-1]
                    and math.isclose(bar.volume, math.fsum(block.volume), rel_tol=1e-12, abs_tol=1e-9),
                    "independent four-hour aggregation")
    raw_count = sum(len(c["read_json"](ROOT / item["path"])["raw_pages"]) for item in m["chunks"])
    require(raw_count == 48, "raw receipt count")
    result = {"schema": "v8-clean-ohlcv-final-seal/1", "plan_sha256": c["object_sha"](frozen_plan(c)),
              "code_sha256": c["file_sha"](__file__), "contract_sha256": CONTRACT_SHA,
              "manifest": {"path": mpath.relative_to(ROOT).as_posix(), "sha256": c["file_sha"](mpath)},
              "files": m["files"], "minute_rows": 34560, "four_hour_rows": 144,
              "raw_receipts": raw_count, "crosschecks": 8, "model_safety": SAFE,
              "qualification_allowed": False, "strategy_results_computed": False,
              "independent_aggregation_verified": True}
    result["seal_sha256"] = c["object_sha"](result)
    c["save"](OUT / "dataset_seal_v1.json", result)
    return {"seal_sha256": result["seal_sha256"], "minute_rows": 34560, "four_hour_rows": 144}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "fetch", "build", "seal"))
    parser.add_argument("--symbol", choices=SYMBOLS)
    args = parser.parse_args()
    c = common()
    if args.stage == "plan":
        p = plan()
        c["save"](OUT / "plan.json", p)
        result = {"plan_sha256": c["object_sha"](p)}
    elif args.stage == "fetch":
        require(args.symbol is not None, "symbol required")
        result = {"symbol": args.symbol, "rows": len(collect(args.symbol, fetch=True)), "crosscheck": True}
    elif args.stage == "build":
        result = build()
    else:
        result = seal()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
