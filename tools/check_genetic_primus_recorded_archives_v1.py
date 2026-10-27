"""Bounded, sequential audit of three user-supplied research archives only.

No project imports, network, collector, training, or execution replay. Sources
are read-only. Receipts are new files, never replacements of frozen evidence.
"""
import argparse
import csv
import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("C:/Users/anton/Downloads")
OUT = Path("E:/Crypto_exchange/Reports/Exia/Genetic_Primus/recorded_archives_v1_20260914")
KINDS = {"l2": "incremental_book_L2", "ticker": "derivative_ticker", "trades": "trades"}
HEADERS = {
    "l2": "exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount",
    "ticker": "exchange,symbol,timestamp,local_timestamp,funding_timestamp,funding_rate,predicted_funding_rate,open_interest,last_price,index_price,mark_price",
    "trades": "exchange,symbol,timestamp,local_timestamp,id,side,price,amount",
}
DAY_START = 1733011200000000
DAY_END = DAY_START + 86400 * 1000000
MAX_COMPRESSED = 150 * 1024**2
MAX_EXPANDED = 2 * 1024**3
CHUNK_BYTES = 64 * 1024**2


def parse_row(raw, kind):
    if not raw.endswith(b"\n") or len(raw) > 4096:
        raise ValueError("unterminated or overlong CSV record")
    values = next(csv.reader([raw.decode("ascii")], strict=True))
    keys = HEADERS[kind].split(",")
    if len(values) != len(keys) or values[:2] != ["bitget-futures", "BTCUSDT"]:
        raise ValueError("unexpected CSV schema or instrument")
    if not all(v.isdigit() and len(v) == 16 for v in values[2:4]):
        raise ValueError("expected integer microsecond timestamps")
    return dict(zip(keys, values))


def utc(value):
    return datetime.fromtimestamp(value / 1000000, timezone.utc).isoformat()


def sha_file(path, deadline):
    if not 0 < path.stat().st_size <= MAX_COMPRESSED:
        raise ValueError("compressed size outside limit")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            if time.monotonic() > deadline:
                raise TimeoutError("hash time budget exceeded")
            h.update(block)
    return h.hexdigest()


def save_new(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    value["qualification_eligible"] = False
    value["trading_authorized"] = False
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def integrity(path, kind):
    target = OUT / (kind + "_integrity.json")
    if target.exists():
        raise FileExistsError("integrity receipt already exists")
    deadline = time.monotonic() + 20
    before = path.stat()
    sha = sha_file(path, deadline)
    expanded = 0
    # Reading through EOF verifies gzip CRC/trailers, including concatenated members.
    with gzip.open(path, "rb") as stream:
        while block := stream.read(256 * 1024):
            expanded += len(block)
            if expanded > MAX_EXPANDED or time.monotonic() > deadline:
                raise ValueError("expanded-size or time budget exceeded")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("source changed during integrity audit")
    if sha_file(path, deadline) != sha:
        raise ValueError("source hash changed during integrity audit")
    result = {
        "schema": "genetic_primus.archive_integrity/1", "source": str(path),
        "source_sha256": sha, "compressed_bytes": before.st_size,
        "expanded_bytes": expanded, "gzip_crc_eof_verified": True,
        "source_authenticity_verified": False, "coverage_verified": False,
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    save_new(target.name, result)
    print(json.dumps({"file": kind, "crc_eof": True, "compressed": before.st_size,
                      "expanded": expanded, "sha256": sha}))


def coverage(path, kind):
    started = time.monotonic()
    receipt = json.loads((OUT / (kind + "_integrity.json")).read_text("utf-8"))
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if code_sha != receipt["auditor_sha256"]:
        raise ValueError("auditor differs from integrity receipt")
    sha = sha_file(path, started + 20)
    if sha != receipt["source_sha256"]:
        raise ValueError("source differs from integrity receipt")
    prior = sorted(OUT.glob(kind + "_coverage_*.json"))
    if len(prior) > 40:
        raise ValueError("too many audit segments")
    if prior:
        state = json.loads(prior[-1].read_text("utf-8"))
        if state["source_sha256"] != sha or state["auditor_sha256"] != code_sha:
            raise ValueError("checkpoint identity mismatch")
        if state["complete"]:
            print(kind + " coverage already complete")
            return
    else:
        state = {"schema": "genetic_primus.archive_coverage/1", "source_sha256": sha,
                 "auditor_sha256": code_sha, "source": str(path), "offset": 0,
                 "rows": 0, "complete": False, "local_regressions": 0,
                 "exchange_regressions": 0, "local_outside_day": 0,
                 "negative_receive_lag": 0, "max_local_gap_us": 0,
                 "snapshot_rows": 0, "missing_mark": 0, "missing_funding_rate": 0,
                 "funding_timestamps": [], "previous_receipt_sha256": None}
    state["previous_receipt_sha256"] = (
        hashlib.sha256(prior[-1].read_bytes()).hexdigest() if prior else None)
    first_offset = state["offset"]
    with gzip.open(path, "rb") as stream:
        header = stream.readline(4097).decode("ascii").rstrip("\r\n")
        if header != HEADERS[kind]:
            raise ValueError("unexpected CSV header")
        if first_offset:
            stream.seek(first_offset)
        else:
            state["offset"] = stream.tell()
            first_offset = state["offset"]
        while state["offset"] - first_offset < CHUNK_BYTES:
            if state["rows"] % 4096 == 0 and time.monotonic() - started > 12:
                break
            raw = stream.readline(4097)
            if not raw:
                state["complete"] = True
                break
            row = parse_row(raw, kind)
            exchange, local = int(row["timestamp"]), int(row["local_timestamp"])
            if state["rows"]:
                state["local_regressions"] += local < state["last_local_us"]
                state["exchange_regressions"] += exchange < state["last_exchange_us"]
                state["max_local_gap_us"] = max(state["max_local_gap_us"], local - state["last_local_us"])
            else:
                state.update(min_local_us=local, max_local_us=local,
                             min_exchange_us=exchange, max_exchange_us=exchange)
            for label, value in (("local", local), ("exchange", exchange)):
                state["min_" + label + "_us"] = min(state["min_" + label + "_us"], value)
                state["max_" + label + "_us"] = max(state["max_" + label + "_us"], value)
                state["last_" + label + "_us"] = value
            state["local_outside_day"] += not DAY_START <= local < DAY_END
            state["negative_receive_lag"] += local < exchange
            if kind == "l2":
                if row["is_snapshot"] not in ("true", "false") or row["side"] not in ("ask", "bid"):
                    raise ValueError("unexpected L2 flag or side")
                state["snapshot_rows"] += row["is_snapshot"] == "true"
            if kind == "ticker":
                state["missing_mark"] += not row["mark_price"]
                state["missing_funding_rate"] += not row["funding_rate"]
                funding = row["funding_timestamp"]
                if funding and funding not in state["funding_timestamps"]:
                    if not funding.isdigit() or len(funding) != 16 or len(state["funding_timestamps"]) >= 32:
                        raise ValueError("unexpected funding timestamp")
                    state["funding_timestamps"].append(funding)
            state["rows"] += 1
            state["offset"] = stream.tell()
            if state["offset"] > receipt["expanded_bytes"]:
                raise ValueError("expanded size differs from integrity receipt")
    if state["offset"] == first_offset and not state["complete"]:
        raise TimeoutError("no scan progress within time budget")
    if state["complete"] and state["offset"] != receipt["expanded_bytes"]:
        raise ValueError("EOF differs from integrity receipt")
    # Recheck the immutable input before writing a successful progress receipt.
    if sha_file(path, time.monotonic() + 5) != sha:
        raise ValueError("source changed during scan")
    state["local_utc_range"] = [utc(state["min_local_us"]), utc(state["max_local_us"])]
    state["exchange_utc_range"] = [utc(state["min_exchange_us"]), utc(state["max_exchange_us"])]
    state["content_semantics_verified"] = False
    state["funding_settlements_verified"] = False
    save_new(kind + "_coverage_" + str(len(prior) + 1).zfill(3) + ".json", state)
    print(json.dumps({"file": kind, "segment": len(prior) + 1, "rows": state["rows"],
                      "percent": round(state["offset"] / receipt["expanded_bytes"] * 100, 1),
                      "complete": state["complete"], "local_utc": state["local_utc_range"],
                      "max_gap_s": state["max_local_gap_us"] / 1000000}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=KINDS)
    parser.add_argument("stage", choices=("integrity", "coverage"))
    args = parser.parse_args()
    source = BASE / ("bitget-futures_" + KINDS[args.kind] + "_2024-12-01_BTCUSDT.csv.gz")
    try:
        {"integrity": integrity, "coverage": coverage}[args.stage](source, args.kind)
    except Exception as exc:
        print("FAIL_CLOSED " + type(exc).__name__ + ": " + str(exc)[:180])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
