from __future__ import annotations

import json
from pathlib import Path

import pytest

from exia.genetic_primus.prospective_v8 import LiquidationTape, decode_force_order, load_v8_contract, validate_tape


ROOT = Path(__file__).resolve().parents[1]


def test_v8_contract_is_fail_closed_and_pins_v7() -> None:
    contract = load_v8_contract(ROOT / "configs/genetic_primus_v8_prospective_link_contract.json", root=ROOT)
    assert contract.payload["source_hypothesis"]["symbol"] == "LINK/USDT"
    assert contract.payload["safety"]["orders_enabled"] is False
    assert contract.payload["root_policy"]["stitch_roots"] is False


def test_v8_decoder_filters_um_and_normalizes_coinm() -> None:
    contract = load_v8_contract(ROOT / "configs/genetic_primus_v8_prospective_link_contract.json", root=ROOT)
    order = {"s": "LINKUSD_PERP", "S": "SELL", "o": "LIMIT", "f": "IOC", "q": "7", "ap": "12.5", "X": "FILLED", "z": "6", "T": 1001}
    events, quality = decode_force_order(json.dumps({"e": "forceOrder", "E": 1002, "o": order, "st": 2}), contract)
    assert quality == []
    assert events[0]["target_symbol"] == "LINK/USDT"
    assert events[0]["liquidation_notional_usd"] == 60
    ignored, quality = decode_force_order(json.dumps({"e": "forceOrder", "E": 1002, "o": order, "st": 1}), contract)
    assert ignored == [] and quality == []


def test_v8_tape_is_hash_chained_and_tamper_evident(tmp_path: Path) -> None:
    path = tmp_path / "tape.jsonl"
    tape = LiquidationTape(path, contract_sha256="a" * 64, collector_fingerprint_sha256="b" * 64, run_id="root-1")
    tape.append(record_type="session_start", observed_at_ms=1000, payload={"connected": False})
    tape.append(record_type="heartbeat", observed_at_ms=2000, payload={"connected": True})
    assert validate_tape(path, contract_sha256="a" * 64)["records"] == 2
    path.write_text(path.read_text(encoding="ascii").replace('"connected":true', '"connected":false'), encoding="ascii")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_tape(path, contract_sha256="a" * 64)
