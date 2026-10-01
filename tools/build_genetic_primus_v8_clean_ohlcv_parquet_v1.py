"""Offline-only canonical Parquet build from pinned public OHLCV intake pages.

Does not call the intake fetch/build entrypoints. Keeps the earlier rejected
roundtrip artifact untouched. Explicit Arrow string dictionary matches the
existing historical dataset and survives Parquet roundtrip without coercion.
"""
import argparse
import hashlib
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTAKE = ROOT / "tools/fetch_genetic_primus_v8_clean_ohlcv_backfill.py"
INTAKE_SHA = "22cee7f62136b70af1253cbf1c8519292b8f09ddb70aeefb3c18ef8c86dae89b"


def expected_rows(minutes, symbols):
    if type(minutes) is not int or minutes != 4320 or type(symbols) is not int or symbols != 8:
        raise ValueError("only fixed three-day full8 blocks allowed")
    return minutes * symbols, minutes // 240 * symbols


def build(origin):
    if hashlib.sha256(INTAKE.read_bytes()).hexdigest() != INTAKE_SHA:
        raise ValueError("pinned intake implementation changed")
    intake = runpy.run_path(str(INTAKE))
    plan = intake["frozen_plan"]()
    plan_sha = intake["object_sha"](plan)
    out = intake["OUT"] / "derived_v1"
    targets = [out / f"origin_{origin:02d}_{tf}m.parquet" for tf in (1, 240)]
    manifest_path = out / f"origin_{origin:02d}_manifest.json"
    if any(p.exists() for p in [*targets, manifest_path]):
        raise FileExistsError("derived artifact exists; never overwrite")
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = []
    for symbol in intake["SYMBOLS"]:
        for row in intake["chunk_rows"](origin, symbol, plan_sha):
            records.append((int(row[0]), symbol + "/USDT", *(float(v) for v in row[1:6])))
    minute = pd.DataFrame(records, columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    minute = minute.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    minute["symbol"] = minute["symbol"].astype("category")
    grouped = minute.assign(timestamp=minute.timestamp // 14400000 * 14400000).groupby(
        ["timestamp", "symbol"], observed=True, sort=True)
    bars = grouped.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                       close=("close", "last"), volume=("volume", "sum"), source_rows=("close", "size")).reset_index()
    if (len(minute), len(bars)) != expected_rows(4320, 8) or not (bars.source_rows == 240).all():
        raise ValueError("incomplete full8 grids")
    start, end = intake["bounds"](origin)
    for frame, step in ((minute, 60000), (bars, 14400000)):
        for symbol in intake["SYMBOLS"]:
            stamps = frame.loc[frame.symbol == symbol + "/USDT", "timestamp"].tolist()
            if stamps != list(range(start, end, step)):
                raise ValueError("incorrect timestamp grid")
    files = []
    out.mkdir(parents=True, exist_ok=True)
    for target, frame in zip(targets, (minute, bars)):
        table = pa.Table.from_pandas(frame, preserve_index=False)
        index = table.schema.get_field_index("symbol")
        table = table.set_column(index, "symbol", table.column(index).cast(pa.dictionary(pa.int8(), pa.string())))
        with target.open("xb") as stream:
            pq.write_table(table, stream, compression="snappy")
        restored = pq.read_table(target)
        if not table.equals(restored) or not frame.equals(restored.to_pandas()):
            raise ValueError("Arrow or pandas roundtrip differs")
        files.append({"path": str(target.relative_to(ROOT)), "sha256": intake["file_sha"](target),
                      "bytes": target.stat().st_size, "rows": len(frame)})
    result = {"schema": "genetic_primus.ohlcv_origin_dataset/2", "origin": origin,
              "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "intake_sha256": INTAKE_SHA, "plan_sha256": plan_sha,
              "contract_sha256": intake["CONTRACT_SHA"], "files": files,
              "chunks": [{"path": str((intake["OUT"] / f"chunks/origin_{origin:02d}_{s}.json").relative_to(ROOT)),
                          "sha256": intake["file_sha"](intake["OUT"] / f"chunks/origin_{origin:02d}_{s}.json")}
                         for s in intake["SYMBOLS"]],
              "start_ms": start, "end_exclusive_ms": end, "safety": dict(intake["SAFE"]),
              "network_allowed": False, "training_invoked": False, "strategy_results_computed": False,
              "qualification_eligible": False, "trading_authorized": False,
              "canonical_symbol_type": "dictionary<int8,string>",
              "arrow_and_pandas_roundtrip_verified": True, "full_grid_verified": True}
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"origin": origin, "minute_rows": len(minute), "four_hour_rows": len(bars),
                      "roundtrip": True, "network_requests": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", type=int, choices=range(3, 8), required=True)
    args = parser.parse_args()
    try:
        build(args.origin)
    except Exception as exc:
        print("FAIL_CLOSED " + type(exc).__name__ + ": " + str(exc)[:160])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
