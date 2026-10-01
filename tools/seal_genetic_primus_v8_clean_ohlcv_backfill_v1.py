"""Offline-only independent aggregation/hash verification; no model imports."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Retrodate/genetic_primus_v8_clean_ohlcv_backfill_20260907_20260921_v2"
PINS = {
    "tools/fetch_genetic_primus_v8_clean_ohlcv_backfill.py": "22cee7f62136b70af1253cbf1c8519292b8f09ddb70aeefb3c18ef8c86dae89b",
    "tools/build_genetic_primus_v8_clean_ohlcv_parquet_v1.py": "b5efb0bbdab82589fba35b6e70b1401eaf885b7e62b31803f5cd6d717a18922f",
    "configs/genetic_primus_prospective_3d_v1.json": "0002201ea664d868a33cdc7e4c8ceb6630bf7a072fef8eccee86927ed4f6488f",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def read(path):
    with path.open("rb") as stream:
        value = stream.read(4 * 1024**2 + 1)
    if len(value) > 4 * 1024**2:
        raise ValueError("receipt too large")
    return json.loads(value)


def main():
    target = DATA / "dataset_seal_v1.json"
    if target.exists():
        raise FileExistsError("seal already exists; never overwrite")
    for path, expected in PINS.items():
        if sha(ROOT / path) != expected:
            raise ValueError("source/contract pin mismatch")
    import pyarrow.parquet as pq

    plan = read(DATA / "plan.json")
    plan_hash = object_sha(plan)
    symbols = [s + "/USDT" for s in plan["symbols"]]
    evidence, files, minute_total, bar_total, page_count = [], [], 0, 0, 0
    for origin in range(3, 8):
        path = DATA / "derived_v1" / f"origin_{origin:02d}_manifest.json"
        manifest = read(path)
        start, end = plan["origins"][str(origin)]
        if (manifest["plan_sha256"] != plan_hash or manifest["origin"] != origin
                or [manifest["start_ms"], manifest["end_exclusive_ms"]] != [start, end]
                or manifest["builder_sha256"] != PINS["tools/build_genetic_primus_v8_clean_ohlcv_parquet_v1.py"]):
            raise ValueError("origin identity mismatch")
        frames = []
        for tf, item in zip((1, 240), manifest["files"]):
            path = DATA / "derived_v1" / f"origin_{origin:02d}_{tf}m.parquet"
            if item["path"] != str(path.relative_to(ROOT)) or sha(path) != item["sha256"]:
                raise ValueError("derived file pin mismatch")
            frame = pq.read_table(path).to_pandas()
            if len(frame) != item["rows"] or set(frame.symbol.astype(str)) != set(symbols):
                raise ValueError("derived rows/universe mismatch")
            frames.append(frame)
            files.append(item)
        if len(frames) != 2 or (len(frames[0]), len(frames[1])) != (34560, 144):
            raise ValueError("incomplete derived files")
        minute, bars = frames
        for symbol in symbols:
            one = minute.loc[minute.symbol == symbol].sort_values("timestamp")
            four = bars.loc[bars.symbol == symbol].sort_values("timestamp")
            if one.timestamp.tolist() != list(range(start, end, 60000)):
                raise ValueError("minute grid mismatch")
            if four.timestamp.tolist() != list(range(start, end, 14400000)):
                raise ValueError("4h grid mismatch")
            for index, candle in enumerate(four.itertuples(index=False)):
                block = one.iloc[index * 240:(index + 1) * 240]
                expected = (block.open.iloc[0], max(block.high), min(block.low), block.close.iloc[-1])
                if (candle.open, candle.high, candle.low, candle.close) != expected or candle.source_rows != 240:
                    raise ValueError("independent OHLC aggregation differs")
                if not math.isclose(candle.volume, math.fsum(block.volume), rel_tol=1e-12, abs_tol=1e-9):
                    raise ValueError("independent volume aggregation differs")
        if len(manifest["chunks"]) != 8:
            raise ValueError("wrong number of chunks")
        for symbol, chunk_ref in zip(plan["symbols"], manifest["chunks"]):
            cp = DATA / "chunks" / f"origin_{origin:02d}_{symbol}.json"
            if chunk_ref["path"] != str(cp.relative_to(ROOT)) or sha(cp) != chunk_ref["sha256"]:
                raise ValueError("chunk pin mismatch")
            chunk = read(cp)
            if chunk["plan_sha256"] != plan_hash or chunk["rows"] != 4320 or not chunk["history_tail_crosscheck_passed"]:
                raise ValueError("unverified chunk")
            if len(chunk["raw_pages"]) != 6:
                raise ValueError("wrong number of raw pages")
            for index, raw_ref in enumerate(chunk["raw_pages"]):
                name = f"raw/origin_{origin:02d}_{symbol}_page_{index:02d}.json"
                if raw_ref["path"] != name or sha(DATA / name) != raw_ref["sha256"]:
                    raise ValueError("raw file pin mismatch")
                record = read(DATA / name)
                claimed = record.pop("record_sha256")
                if object_sha(record) != claimed or record["plan_sha256"] != plan_hash:
                    raise ValueError("raw page body/plan mismatch")
                if hashlib.sha256(record["response_utf8"].encode()).hexdigest() != record["response_sha256"]:
                    raise ValueError("raw response hash mismatch")
                page_count += 1
        evidence.append({"path": str((DATA / "derived_v1" / f"origin_{origin:02d}_manifest.json").relative_to(ROOT)),
                         "sha256": sha(DATA / "derived_v1" / f"origin_{origin:02d}_manifest.json")})
        minute_total += len(minute)
        bar_total += len(bars)
    if (minute_total, bar_total, page_count) != (172800, 720, 240):
        raise ValueError("total evidence inventory mismatch")
    result = {"schema": "genetic_primus.ohlcv_backfill_seal/1", "plan_sha256": plan_hash,
              "source_pins": PINS, "checker_sha256": sha(Path(__file__)),
              "origins": evidence, "files": files, "minute_rows": minute_total,
              "four_hour_rows": bar_total, "raw_pages_verified": page_count,
              "history_tail_checks": 40, "independent_aggregation_verified": True,
              "full8_grid_verified": True, "safety": plan["safety"], "network_allowed": False,
              "strategy_results_computed": False, "qualification_eligible": False,
              "trading_authorized": False}
    result["seal_sha256"] = object_sha(result)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"minute_rows": minute_total, "four_hour_rows": bar_total,
                      "raw_pages": page_count, "seal_sha256": result["seal_sha256"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAIL_CLOSED " + type(exc).__name__ + ": " + str(exc)[:160])
        raise SystemExit(1)
