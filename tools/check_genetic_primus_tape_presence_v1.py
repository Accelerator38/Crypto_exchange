"""Read-only filename/first-row gate for frozen train execution-event tapes."""

from __future__ import annotations

import csv
import gzip
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.prospective_contract_v1 import file_sha256
from exia.genetic_primus.rolling_linear_retrain_v1 import DT

SOURCE = ROOT / "Reports/Exia/Genetic_Primus/rolling_linear_matrix_v1_20261002/result.json"
SOURCE_SHA = "99418ebacb37ebc66ed139746543215809464612b169af8af6a33514b46a44ad"
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/tape_presence_v1_20261002/result.json"
DOWNLOADS = Path(r"C:\Users\anton\Downloads")
STREAMS = ("book_ticker", "incremental_book_L2", "trades")
FILE_RE = re.compile(
    r"bitget-futures_(book_ticker|incremental_book_L2|trades|derivative_ticker)_"
    r"(\d{4}-\d{2}-\d{2})_([A-Z0-9]+)\.csv\.gz$"
)


def _day(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).date().isoformat()


def _tape_name(stream: str, day: str, symbol: str) -> str:
    return f"bitget-futures_{stream}_{day}_{symbol.replace('/', '')}.csv.gz"


def _events(source: dict, policy: str, delay_bars: int) -> list[dict]:
    if delay_bars not in (0, 1):
        raise ValueError("only registered whole-bar delays allowed")
    if policy == "EMA_TrendConsensus":
        record = next(r for r in source["records"]
                      if r["feature_set"] == "two" and r["variant"] == "uniform")
    elif policy == "linear_four_uniform":
        record = next(r for r in source["records"]
                      if r["feature_set"] == "four" and r["variant"] == "uniform")
    else:
        raise ValueError("policy outside frozen active set")
    ledger = record["origin_models"]
    if len(ledger) != 20:
        raise ValueError("not the registered train origin count")
    symbols = tuple(ledger[0]["decision"]["decisions"])
    prior = {symbol: 0 for symbol in symbols}
    events = []
    for item in ledger:
        origin = int(item["origin_start"])
        decisions = item["decision"]["decisions"]
        if set(decisions) != set(symbols):
            raise ValueError("incomplete frozen decision universe")
        for symbol in symbols:
            decision = decisions[symbol]
            target = (int(decision["predicted_gross_3d"] is not None)
                      if policy == "EMA_TrendConsensus" else int(decision["target"]))
            if target != prior[symbol]:
                timestamp = origin + delay_bars * DT
                events.append({"policy": policy, "delay_bars": delay_bars,
                               "timestamp_utc": datetime.fromtimestamp(
                                   timestamp / 1000, timezone.utc).isoformat(),
                               "day_utc": _day(timestamp), "symbol": symbol,
                               "action": "enter" if target else "exit"})
            prior[symbol] = target
    if any(prior.values()):
        raise ValueError("terminal trade would need special close-tape treatment")
    return events


def _first_row_metadata(path: Path, filename_day: str) -> dict:
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            first = next(reader, None)
        if (first is None or not {"exchange", "symbol", "timestamp", "local_timestamp"}
                .issubset(fields)):
            raise ValueError("missing first row or required columns")
        local_us = int(first["local_timestamp"])
        exchange_us = int(first["timestamp"])
        local_day = datetime.fromtimestamp(local_us / 1_000_000, timezone.utc).date().isoformat()
        return {"name": path.name, "bytes": path.stat().st_size,
                "first_exchange_timestamp_us": exchange_us,
                "first_local_timestamp_us": local_us,
                "first_local_day_utc": local_day,
                "first_local_day_matches_filename": local_day == filename_day,
                "first_exchange_timestamp_not_after_local": exchange_us <= local_us,
                "header_columns": fields,
                "scope": "header_and_first_record_only_not_full_file_integrity"}
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        return {"name": path.name, "bytes": path.stat().st_size,
                "first_row_error": type(exc).__name__,
                "scope": "header_and_first_record_only_not_full_file_integrity"}


def _evaluate() -> dict:
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("project Python required")
    if file_sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("frozen source result changed")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if (source["selected_candidate"] is not None
            or source["validation_result"] is not None
            or source["final_policy"] != "NoTrade"):
        raise ValueError("unexpected source attempt")
    all_events = [_events(source, policy, delay)
                  for policy in ("EMA_TrendConsensus", "linear_four_uniform")
                  for delay in (0, 1)]
    flat_events = [event for group in all_events for event in group]
    required_names = sorted({
        _tape_name(stream, event["day_utc"], event["symbol"])
        for event in flat_events for stream in STREAMS})
    if not required_names or any("2026-06" not in n and "2026-07" not in n
                                 for n in required_names):
        raise AssertionError("required tape request escaped disclosed train interval")
    available = []
    for path in sorted(DOWNLOADS.glob("bitget-futures_*.csv.gz")):
        match = FILE_RE.fullmatch(path.name)
        if match:
            available.append(_first_row_metadata(path, match.group(2)))
    present = {item["name"] for item in available}
    matches = sorted(set(required_names) & present)
    return {"schema_version": "exia.genetic_primus.tape_presence/1",
            "evidence_class": "read_only_local_data_availability_not_execution_validation",
            "source_result_sha256": SOURCE_SHA,
            "script_sha256": file_sha256(Path(__file__)),
            "requirements": {
                "policy_delay_event_counts": {
                    f"{policy}:{delay}": len(_events(source, policy, delay))
                    for policy in ("EMA_TrendConsensus", "linear_four_uniform")
                    for delay in (0, 1)},
                "required_streams": STREAMS,
                "required_unique_daily_files": len(required_names),
                "required_names": required_names,
                "matching_local_files": matches,
                "missing_unique_daily_files": len(required_names) - len(matches),
            },
            "available_local_archives": available,
            "full_file_integrity_checked_this_run": False,
            "price_or_fill_quality_checked": False,
            "tape_presence_complete": len(matches) == len(required_names),
            "execution_ready": False,
            "reason": "matching_tape_files_absent_or_full_integrity_and_asof_depth_unverified",
            "new_model_fits": 0, "validation_labels_used": False,
            "network_allowed": False, "promotion_authority": False}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("immutable tape-presence result already exists")
    result = _evaluate()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "TAPE_PRESENCE_CHECK_COMPLETE",
                      "required": result["requirements"]["required_unique_daily_files"],
                      "matching": len(result["requirements"]["matching_local_files"]),
                      "execution_ready": result["execution_ready"]},
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
