from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


GENESIS_HASH = "0" * 64
REQUIRED_SYMBOL_TYPE = 2
ALLOWED_SYMBOLS = frozenset(
    {
        "ADAUSD_PERP",
        "BNBUSD_PERP",
        "BTCUSD_PERP",
        "DOGEUSD_PERP",
        "ETHUSD_PERP",
        "LINKUSD_PERP",
        "SOLUSD_PERP",
        "XRPUSD_PERP",
    }
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")


@dataclass(frozen=True, slots=True)
class LiquidationEvent:
    symbol: str
    symbol_type: int
    side: str
    event_time_ms: int
    order_time_ms: int
    quantity_contracts: float
    average_price: float
    status: str


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path, *, root: Path) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if contract.get("schema") != "genetic_primus.v8.clean.contract/2":
        raise ValueError("unexpected V8 clean contract schema")
    if contract.get("strategy_id") != "Genetic_Primus_V8_Clean":
        raise ValueError("unexpected V8 clean strategy id")
    safety = contract.get("safety")
    if not isinstance(safety, dict) or any(value is not False for value in safety.values()):
        raise PermissionError("V8 clean safety must remain fail-closed")
    hypothesis = contract.get("source_hypothesis")
    if not isinstance(hypothesis, dict):
        raise ValueError("missing V7 source hypothesis")
    expected = (
        ("v7_contract", "v7_contract_sha256"),
        ("v7_manifest", "v7_manifest_sha256"),
        ("v7_posthoc_registry", "v7_posthoc_registry_sha256"),
    )
    for path_key, hash_key in expected:
        source_path = root / str(hypothesis.get(path_key, ""))
        if _sha256_file(source_path) != hypothesis.get(hash_key):
            raise ValueError(f"V7 source hash mismatch: {path_key}")
    if (
        hypothesis.get("hypothesis_id") != "link_isolated_liquidation_reversal_prospective"
        or hypothesis.get("symbol") != "LINK/USDT"
        or hypothesis.get("parameter_tuning_allowed") is not False
    ):
        raise ValueError("V8 clean may only replicate the frozen V7 LINK hypothesis")
    frozen_event = contract.get("frozen_event")
    expected_event = {
        "decision_timeframe_minutes": 240,
        "decision_at": "bar_close",
        "entry_at": "next_bar_open",
        "holding_bars": 3,
        "direction": "opposite dominant forced-flow direction",
        "trailing_window_bars": 180,
        "minimum_trailing_bars": 90,
        "symbol_total_quantile": 0.95,
        "symbol_return_quantile": 0.75,
        "symbol_side_share_minimum": 0.75,
        "event_cooldown_bars": 3,
        "stress_round_trip_cost_bps": 24.0,
    }
    if frozen_event != expected_event:
        raise ValueError("V8 clean frozen V7 event parameters changed")
    gate = contract.get("prospective_gate")
    if not isinstance(gate, dict):
        raise ValueError("missing V8 clean prospective gate")
    if (
        gate.get("minimum_calendar_days") != 180
        or gate.get("maximum_calendar_days") != 365
        or gate.get("minimum_source_coverage") != 0.98
        or gate.get("minimum_events") != 60
        or gate.get("minimum_positive_folds") != 3
    ):
        raise ValueError("V8 clean prospective gate changed")
    source = contract.get("source")
    if not isinstance(source, dict):
        raise ValueError("missing public liquidation source")
    if source.get("required_symbol_type") != REQUIRED_SYMBOL_TYPE:
        raise ValueError("V8 clean requires Binance COIN-M symbol_type=2")
    if frozenset(source.get("allowed_symbols", ())) != ALLOWED_SYMBOLS:
        raise ValueError("V8 clean full8 symbol set changed")
    if source.get("raw_payload_retained") is not True:
        raise ValueError("V8 clean must retain raw source payloads")
    policy = contract.get("root_policy")
    if not isinstance(policy, dict) or policy.get("resume") is not False or policy.get("stitch") is not False:
        raise ValueError("V8 clean roots may not be resumed or stitched")
    return contract


def parse_force_order(
    message: str | bytes,
    *,
    required_symbol_type: int = REQUIRED_SYMBOL_TYPE,
    allowed_symbols: frozenset[str] = ALLOWED_SYMBOLS,
) -> tuple[LiquidationEvent, ...]:
    """Parse only explicitly tagged Binance COIN-M force-order messages."""
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    payload = json.loads(message)
    if isinstance(payload, Mapping) and "data" in payload:
        payload = payload["data"]
    items: Iterable[Any] = payload if isinstance(payload, list) else (payload,)
    events: list[LiquidationEvent] = []
    for item in items:
        if not isinstance(item, dict) or item.get("e") != "forceOrder":
            continue
        try:
            symbol_type = int(item.get("st"))
        except (TypeError, ValueError):
            continue
        if symbol_type != required_symbol_type:
            continue
        order = item.get("o")
        if not isinstance(order, dict) or order.get("s") not in allowed_symbols:
            continue
        side = str(order.get("S", ""))
        if side not in {"BUY", "SELL"}:
            raise ValueError("force-order side must be BUY or SELL")
        events.append(
            LiquidationEvent(
                symbol=str(order["s"]),
                symbol_type=symbol_type,
                side=side,
                event_time_ms=int(item["E"]),
                order_time_ms=int(order["T"]),
                quantity_contracts=float(order["q"]),
                average_price=float(order["ap"]),
                status=str(order["X"]),
            )
        )
    return tuple(events)


class HashChainTape:
    """Append-only JSONL tape whose records commit to all earlier records."""

    def __init__(self, path: Path, *, root_id: str, contract_sha256: str) -> None:
        self.path = Path(path)
        self.root_id = root_id
        self.contract_sha256 = contract_sha256
        self.previous_hash = GENESIS_HASH
        if self.path.exists():
            raise FileExistsError(f"refusing to append to existing root: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, *, kind: str, observed_at_ms: int, payload: dict[str, Any]) -> str:
        body = {
            "schema": "genetic_primus.v8.clean.tape/2",
            "root_id": self.root_id,
            "contract_sha256": self.contract_sha256,
            "kind": kind,
            "observed_at_ms": int(observed_at_ms),
            "payload": payload,
            "previous_sha256": self.previous_hash,
        }
        record_hash = hashlib.sha256(_canonical(body)).hexdigest()
        record = {**body, "record_sha256": record_hash}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.previous_hash = record_hash
        return record_hash

    def append_liquidation(self, event: LiquidationEvent, *, observed_at_ms: int) -> str:
        return self.append(kind="liquidation", observed_at_ms=observed_at_ms, payload=asdict(event))


def verify_tape(path: Path, *, contract_sha256: str) -> dict[str, Any]:
    previous = GENESIS_HASH
    count = 0
    root_id: str | None = None
    last_kind: str | None = None
    session_payload: dict[str, Any] | None = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"partial record at line {line_number}")
            record = json.loads(line)
            claimed = record.pop("record_sha256")
            if record.get("schema") != "genetic_primus.v8.clean.tape/2":
                raise ValueError(f"schema mismatch at line {line_number}")
            if root_id is None:
                root_id = str(record.get("root_id"))
                if record.get("kind") != "session_start":
                    raise ValueError("first tape record must be session_start")
                if not isinstance(record.get("payload"), dict):
                    raise ValueError("session_start payload must be an object")
                session_payload = record["payload"]
            elif record.get("root_id") != root_id:
                raise ValueError(f"root mismatch at line {line_number}")
            if record.get("contract_sha256") != contract_sha256:
                raise ValueError(f"contract mismatch at line {line_number}")
            if record.get("previous_sha256") != previous:
                raise ValueError(f"chain mismatch at line {line_number}")
            actual = hashlib.sha256(_canonical(record)).hexdigest()
            if actual != claimed:
                raise ValueError(f"record hash mismatch at line {line_number}")
            previous = claimed
            last_kind = str(record.get("kind"))
            count += 1
    if count == 0:
        raise ValueError("empty tape")
    return {
        "records": count,
        "root_id": root_id,
        "last_kind": last_kind,
        "session_payload": session_payload,
        "last_record_sha256": previous,
        "valid": True,
    }
