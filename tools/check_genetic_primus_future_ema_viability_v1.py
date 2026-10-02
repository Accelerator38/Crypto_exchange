"""Static, no-data preflight for the registered future EMA viability screen."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.measurement_lab_v2 import array_sha, block_bank
from exia.genetic_primus.prospective_contract_v1 import file_sha256

CONFIG = ROOT / "configs/genetic_primus_future_ema_viability_v1.json"
CONFIG_SHA = "11b83d2e3ecbdda572307dd902b73c9c2d8ae108499be85e22b988f193b227b6"
DAY_MS = 86_400_000
DT = 14_400_000


def _ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError("UTC timestamp required")
    return int(parsed.timestamp() * 1000)


def validate_contract(c: dict, known_manifest: dict, contract: dict) -> dict:
    if (c["schema_version"] != "exia.genetic_primus.future_ema_viability/1"
            or c["status"] != "REGISTERED_AWAITING_NEW_OFFLINE_DATA"
            or c["evidence_class"] != "future_only_single_final_analysis_not_yet_observed"
            or c["policy_fits_or_parameter_search"] != 0
            or c["origin_days"] != 3 or c["origin_count"] != 60
            or c["timeframe_minutes"] != 240
            or c["minimum_distinct_signal_origins"] != 20
            or c["maximum_drawdown"] != 0.10
            or c["single_origin_positive_pnl_concentration_limit"] != 0.40
            or c["evaluation_scenarios"] != [
                {"name": "normal", "delay_bars": 0, "round_trip_cost_bps": 16.0},
                {"name": "execution_stress", "delay_bars": 1, "round_trip_cost_bps": 40.0}]
            or c["automatic_evaluation"] or c["automatic_promotion"]
            or c["full_tape_execution_verified"]
            or c["safety"] != {key: False for key in (
                "credentials_allowed", "orders_enabled", "paper_allowed", "live_allowed",
                "promotion_authority", "runtime_authority", "network_allowed")}):
        raise ValueError("future-only objective or safety changed")
    known_end = _ms(c["known_revealed_end_exclusive_utc"])
    segment_start = _ms(c["offline_future_segment_start_utc"])
    first = _ms(c["future_start_utc"])
    end = _ms(c["future_end_exclusive_utc"])
    if (known_end != 1_790_553_600_000 or segment_start != known_end
            or first != _ms("2026-10-05T00:00:00Z")
            or end != _ms("2027-04-03T00:00:00Z")
            or first <= known_end or end - first != 60 * 3 * DAY_MS
            or (end - segment_start) % DT
            or c["required_new_rows"] != ((end - segment_start) // DT) * 8):
        raise ValueError("future interval, origin grid or required rows changed")
    metadata = known_manifest["metadata"]
    # The manifest records its state *at creation*, before later research readouts.
    if (known_manifest["data_already_revealed"] is not False
            or known_manifest["snapshot_sha256"] != c["known_revealed_snapshot_sha256"]
            or metadata["coverage_end_exclusive"] != known_end
            or metadata["timeframe_ms"] != DT
            or metadata["symbols"] != c["universe"]
            or contract["universe"] != c["universe"]):
        raise ValueError("known revealed data/contract mismatch")
    for relative, expected in c["frozen_code_sha256"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or file_sha256(path) != expected:
            raise ValueError(f"frozen code changed: {relative}")
    bank = c["bootstrap"]
    if (bank["daily_observations"] != 180 or bank["samples"] != 4096
            or bank["block_days"] != 5 or bank["alpha"] != 0.05):
        raise ValueError("bootstrap design changed")
    for label in ("primary", "audit"):
        generated = block_bank(bank["daily_observations"], bank["samples"],
                               bank["block_days"], bank[f"{label}_seed"])
        if array_sha(generated) != bank[f"{label}_bank_sha256"]:
            raise ValueError(f"{label} resample bank changed")
    return {"status": "REGISTERED_AWAITING_NEW_OFFLINE_DATA",
            "known_revealed_end_exclusive_utc": c["known_revealed_end_exclusive_utc"],
            "future_start_utc": c["future_start_utc"],
            "future_end_exclusive_utc": c["future_end_exclusive_utc"],
            "future_origin_count": c["origin_count"],
            "required_new_rows": c["required_new_rows"],
            "outcome_metrics_computed": False,
            "network_used": False, "promotion_authority": False}


def main() -> None:
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("project Python required")
    if file_sha256(CONFIG) != CONFIG_SHA:
        raise ValueError("registered future-only contract changed")
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest_path = ROOT / c["known_revealed_manifest_path"]
    contract_path = ROOT / c["source_contract_path"]
    if (file_sha256(manifest_path) != c["known_revealed_manifest_sha256"]
            or file_sha256(contract_path) != c["source_contract_sha256"]):
        raise ValueError("known source lineage changed")
    result = validate_contract(
        c, json.loads(manifest_path.read_text(encoding="utf-8")),
        json.loads(contract_path.read_text(encoding="utf-8")))
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
