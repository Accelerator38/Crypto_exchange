from __future__ import annotations

import json
from pathlib import Path

import pytest

from exia.genetic_primus_v8_clean.core import HashChainTape, parse_force_order, verify_tape


CONTRACT_HASH = "a" * 64


def test_parse_link_force_order() -> None:
    raw = json.dumps(
        {
            "e": "forceOrder",
            "E": 1_700_000_000_100,
            "st": 2,
            "o": {
                "s": "LINKUSD_PERP",
                "S": "SELL",
                "T": 1_700_000_000_000,
                "q": "15",
                "ap": "14.25",
                "X": "FILLED",
            },
        }
    )
    events = parse_force_order(raw)
    assert len(events) == 1
    assert events[0].symbol == "LINKUSD_PERP"
    assert events[0].symbol_type == 2
    assert events[0].side == "SELL"
    assert events[0].quantity_contracts == 15.0


def test_parse_rejects_missing_or_wrong_symbol_type() -> None:
    order = {
        "e": "forceOrder",
        "E": 1_700_000_000_100,
        "o": {
            "s": "LINKUSD_PERP",
            "S": "SELL",
            "T": 1_700_000_000_000,
            "q": "15",
            "ap": "14.25",
            "X": "FILLED",
        },
    }
    assert parse_force_order(json.dumps(order)) == ()
    assert parse_force_order(json.dumps({**order, "st": 1})) == ()
    wrapped = {"stream": "!forceOrder@arr", "data": {**order, "st": 2}}
    assert len(parse_force_order(json.dumps(wrapped))) == 1


def test_tape_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "root" / "liquidation_tape.jsonl"
    tape = HashChainTape(path, root_id="root_test", contract_sha256=CONTRACT_HASH)
    tape.append(kind="session_start", observed_at_ms=10, payload={"connected": False})
    tape.append(kind="session_stop", observed_at_ms=20, payload={"reason": "test"})
    result = verify_tape(path, contract_sha256=CONTRACT_HASH)
    assert result["valid"] is True
    assert result["records"] == 2


def test_tape_rejects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "root" / "liquidation_tape.jsonl"
    tape = HashChainTape(path, root_id="root_test", contract_sha256=CONTRACT_HASH)
    tape.append(kind="session_start", observed_at_ms=10, payload={"connected": False})
    record = json.loads(path.read_text(encoding="utf-8"))
    record["payload"]["connected"] = True
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="record hash mismatch"):
        verify_tape(path, contract_sha256=CONTRACT_HASH)


def test_existing_tape_cannot_be_resumed(tmp_path: Path) -> None:
    path = tmp_path / "liquidation_tape.jsonl"
    path.write_text("occupied\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        HashChainTape(path, root_id="new_root", contract_sha256=CONTRACT_HASH)
