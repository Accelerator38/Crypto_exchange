from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import sha256_file
from .contracts_v5 import V5_SAFETY_KEYS


ZERO_SHA256 = "0" * 64
TAPE_SCHEMA = "exia.genetic_primus_liquidation_tape.v8"


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def record_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "record_sha256"}
    return hashlib.sha256(canonical_json(unsigned).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class PrimusV8Contract:
    source_path: Path
    payload: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return sha256_file(self.source_path)

    @property
    def stream(self) -> Mapping[str, Any]:
        return self.payload["liquidation_stream"]

    def validate(self, root: Path) -> None:
        if self.payload.get("schema_version") != "exia.genetic_primus_prospective_contract.v8":
            raise ValueError("unexpected Genetic_Primus V8 schema")
        if self.payload.get("agent_id") != "Genetic_Primus_V8" or self.payload.get("immutable_after_freeze") is not True:
            raise ValueError("invalid Genetic_Primus V8 identity or mutability")
        safety = self.payload.get("safety", {})
        if set(safety) != V5_SAFETY_KEYS or any(value is not False for value in safety.values()):
            raise ValueError("V8 contract must be fail-closed")
        source = self.payload["source_hypothesis"]
        for path_key, sha_key in (
            ("v7_event_manifest", "v7_event_manifest_sha256"),
            ("v7_posthoc_registry", "v7_posthoc_registry_sha256"),
        ):
            if sha256_file(root / str(source[path_key])) != source[sha_key]:
                raise ValueError(f"V8 source hypothesis SHA mismatch: {path_key}")
        if source.get("hypothesis_id") != "link_isolated_liquidation_reversal_prospective":
            raise ValueError("V8 hypothesis changed")
        if source.get("symbol") != "LINK/USDT" or source.get("parameter_tuning_allowed") is not False:
            raise ValueError("V8 must remain an untuned LINK replication")
        stream = self.stream
        if stream.get("url") != "wss://dstream.binance.com/ws/!forceOrder@arr":
            raise ValueError("V8 liquidation endpoint changed")
        if int(stream.get("required_symbol_type", 0)) != 2 or len(stream.get("symbols", {})) != 8:
            raise ValueError("V8 requires full8 COIN-M filtering")
        if stream.get("credentials_allowed") is not False or stream.get("orders_allowed") is not False:
            raise ValueError("V8 collector must be public read-only")
        event = self.payload["frozen_event"]
        expected = {
            "trailing_window_bars": 180,
            "minimum_trailing_bars": 90,
            "symbol_total_quantile": 0.95,
            "symbol_return_quantile": 0.75,
            "symbol_side_share_minimum": 0.75,
            "event_cooldown_bars": 3,
            "decision_timeframe_minutes": 240,
            "holding_bars": 3,
            "stress_round_trip_cost_bps": 24.0,
        }
        if any(event.get(key) != value for key, value in expected.items()):
            raise ValueError("V8 frozen LINK event changed")
        gate = self.payload["prospective_gate"]
        if int(gate.get("minimum_events", 0)) != 60 or float(gate.get("minimum_source_coverage", 0.0)) != 0.98:
            raise ValueError("V8 prospective gate changed")
        policy = self.payload["root_policy"]
        if policy.get("resume_existing_root") is not False or policy.get("stitch_roots") is not False:
            raise ValueError("V8 roots may not be resumed or stitched")


def load_v8_contract(path: Path, *, root: Path) -> PrimusV8Contract:
    contract = PrimusV8Contract(path.resolve(), json.loads(path.read_text(encoding="utf-8")))
    contract.validate(root)
    return contract


def decode_force_order(raw_message: str | bytes, contract: PrimusV8Contract) -> tuple[list[dict[str, Any]], list[str]]:
    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8")
    try:
        message = json.loads(raw_message)
    except json.JSONDecodeError:
        return [], ["invalid_json"]
    if isinstance(message, Mapping) and "data" in message:
        message = message["data"]
    messages = message if isinstance(message, list) else [message]
    events: list[dict[str, Any]] = []
    quality: list[str] = []
    mappings = contract.stream["symbols"]
    for item in messages:
        if not isinstance(item, Mapping):
            quality.append("invalid_message_type")
            continue
        if item.get("e") != "forceOrder":
            quality.append("unexpected_event_type")
            continue
        symbol_type = int(item.get("st") or 0)
        if symbol_type != int(contract.stream["required_symbol_type"]):
            continue
        order = item.get("o")
        if not isinstance(order, Mapping):
            quality.append("missing_order")
            continue
        source_symbol = str(order.get("s") or "").upper()
        mapping = mappings.get(source_symbol)
        if mapping is None:
            continue
        try:
            event_time = int(item["E"])
            trade_time = int(order["T"])
            original_quantity = float(order["q"])
            average_price = float(order["ap"])
            accumulated_quantity = float(order["z"])
        except (KeyError, TypeError, ValueError):
            quality.append("invalid_order_numeric")
            continue
        side = str(order.get("S") or "")
        if event_time <= 0 or trade_time <= 0 or side not in {"BUY", "SELL"} or accumulated_quantity < 0 or average_price < 0:
            quality.append("invalid_order_values")
            continue
        contract_size = float(mapping["contract_size_usd"])
        events.append(
            {
                "event_time_ms": event_time,
                "trade_time_ms": trade_time,
                "source_symbol": source_symbol,
                "target_symbol": mapping["target_symbol"],
                "symbol_type": symbol_type,
                "side": side,
                "order_type": str(order.get("o") or ""),
                "time_in_force": str(order.get("f") or ""),
                "order_status": str(order.get("X") or ""),
                "original_quantity": original_quantity,
                "average_price": average_price,
                "accumulated_fill_quantity": accumulated_quantity,
                "contract_size_usd": contract_size,
                "liquidation_notional_usd": accumulated_quantity * contract_size,
                "raw": dict(item),
            }
        )
    events.sort(key=lambda row: (row["event_time_ms"], row["source_symbol"], row["trade_time_ms"], row["side"]))
    return events, quality


class LiquidationTape:
    def __init__(self, path: Path, *, contract_sha256: str, collector_fingerprint_sha256: str, run_id: str) -> None:
        self.path = path.resolve()
        self.contract_sha256 = contract_sha256
        self.collector_fingerprint_sha256 = collector_fingerprint_sha256
        self.run_id = run_id
        self.sequence = 0
        self.previous_sha256 = ZERO_SHA256
        if self.path.exists() and self.path.stat().st_size:
            raise FileExistsError("V8 root already contains a tape; create a new root")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, *, record_type: str, observed_at_ms: int, payload: Mapping[str, Any]) -> dict[str, Any]:
        record = {
            "schema_version": TAPE_SCHEMA,
            "run_id": self.run_id,
            "contract_sha256": self.contract_sha256,
            "collector_fingerprint_sha256": self.collector_fingerprint_sha256,
            "sequence": self.sequence,
            "previous_record_sha256": self.previous_sha256,
            "record_type": record_type,
            "observed_at_ms": int(observed_at_ms),
            "payload": dict(payload),
            "orders_enabled": False,
            "promotion_authority": False,
        }
        record["record_sha256"] = record_sha256(record)
        with self.path.open("a", encoding="ascii", newline="\n") as handle:
            handle.write(canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.sequence += 1
        self.previous_sha256 = record["record_sha256"]
        return record


def validate_tape(path: Path, *, contract_sha256: str | None = None) -> dict[str, Any]:
    records = []
    previous = ZERO_SHA256
    run_id = None
    fingerprint = None
    previous_observed = -1
    with path.open("r", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or not line.strip():
                raise ValueError(f"V8 tape has a partial or blank line: {line_number}")
            record = json.loads(line)
            if record.get("schema_version") != TAPE_SCHEMA:
                raise ValueError("V8 tape schema mismatch")
            if int(record.get("sequence", -1)) != line_number - 1:
                raise ValueError("V8 tape sequence mismatch")
            if record.get("previous_record_sha256") != previous:
                raise ValueError("V8 tape hash chain is broken")
            if record_sha256(record) != record.get("record_sha256"):
                raise ValueError("V8 tape record hash mismatch")
            if record.get("orders_enabled") is not False or record.get("promotion_authority") is not False:
                raise ValueError("V8 tape is not fail-closed")
            if contract_sha256 and record.get("contract_sha256") != contract_sha256:
                raise ValueError("V8 tape contract mismatch")
            run_id = run_id or record.get("run_id")
            fingerprint = fingerprint or record.get("collector_fingerprint_sha256")
            if record.get("run_id") != run_id or record.get("collector_fingerprint_sha256") != fingerprint:
                raise ValueError("V8 tape identity changed")
            observed = int(record["observed_at_ms"])
            if observed < previous_observed:
                raise ValueError("V8 tape observation time regressed")
            previous_observed = observed
            previous = record["record_sha256"]
            records.append(record)
    if not records:
        raise ValueError("V8 tape is empty")
    event_records = [record for record in records if record["record_type"] == "liquidation"]
    return {
        "run_id": run_id,
        "records": len(records),
        "liquidation_events": len(event_records),
        "link_events": sum(record["payload"].get("target_symbol") == "LINK/USDT" for record in event_records),
        "first_observed_at_ms": int(records[0]["observed_at_ms"]),
        "last_observed_at_ms": int(records[-1]["observed_at_ms"]),
        "last_record_sha256": previous,
        "collector_fingerprint_sha256": fingerprint,
        "contract_sha256": records[0]["contract_sha256"],
        "orders_enabled": False,
        "promotion_authority": False,
    }
