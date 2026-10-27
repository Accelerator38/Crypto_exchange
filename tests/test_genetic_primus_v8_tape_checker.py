from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from exia.genetic_primus.prospective_v8 import LiquidationTape, load_v8_contract


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/check_genetic_primus_v8_liquidation_tape.py"
CONTRACT = ROOT / "configs/genetic_primus_v8_prospective_link_contract.json"


def _module():
    spec = importlib.util.spec_from_file_location("primus_v8_tape_checker_test", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v8_checker_accepts_completed_bounded_root(tmp_path: Path) -> None:
    contract = load_v8_contract(CONTRACT, root=ROOT)
    tape = LiquidationTape(
        tmp_path / "liquidation_tape.jsonl",
        contract_sha256=contract.sha256,
        collector_fingerprint_sha256="b" * 64,
        run_id="bounded-root",
    )
    tape.append(record_type="session_start", observed_at_ms=1000, payload={"connected": False})
    tape.append(record_type="session_stop", observed_at_ms=2000, payload={"connected": False})
    (tmp_path / "collector_status.json").write_text(
        json.dumps(
            {
                "run_id": "bounded-root",
                "pid": os.getpid(),
                "run_state": "completed",
                "connected": False,
                "updated_at": datetime.now(UTC).isoformat(),
                "contract_sha256": contract.sha256,
                "connections": 1,
                "liquidation_events": 0,
                "link_events": 0,
                "quality_events": 0,
                "connected_seconds": 1,
                "uptime_seconds": 1,
                "orders_enabled": False,
                "promotion_authority": False,
            }
        ),
        encoding="utf-8",
    )
    assert _module().check(tmp_path, CONTRACT)["status"] == "COMPLETED_BOUNDED_ROOT"
