from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import sha256_file
from .contracts_v5 import V5_SAFETY_KEYS


V7_EVENT_IDS = (
    "cross_market_cascade_reversal",
    "cross_market_cascade_continuation",
    "isolated_symbol_cascade_reversal",
)


@dataclass(frozen=True)
class PrimusV7Contract:
    source_path: Path
    payload: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return sha256_file(self.source_path)

    @property
    def liquidation_source(self) -> Mapping[str, Any]:
        return self.payload["liquidation_source"]

    @property
    def feasibility(self) -> Mapping[str, Any]:
        return self.payload["source_feasibility"]

    @property
    def folds(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.payload["walk_forward"])

    def validate(self, root: Path) -> None:
        if self.payload.get("schema_version") != "exia.genetic_primus_experiment_contract.v7":
            raise ValueError("unexpected Genetic_Primus V7 schema")
        if self.payload.get("agent_id") != "Genetic_Primus_V7" or self.payload.get("immutable_after_freeze") is not True:
            raise ValueError("invalid Genetic_Primus V7 identity or mutability")
        safety = self.payload.get("safety", {})
        if set(safety) != V5_SAFETY_KEYS or any(value is not False for value in safety.values()):
            raise ValueError("V7 contract must be fail-closed")
        source = self.liquidation_source
        if source.get("source_family") != "cross_venue_forced_liquidations":
            raise ValueError("unexpected V7 source family")
        if source.get("start_date") != "2023-06-25" or source.get("end_date") != "2024-10-14":
            raise ValueError("V7 liquidation window changed")
        symbols = source.get("symbols", {})
        if len(symbols) != 8 or set(source.get("required_anchor_symbols", [])) != {"BTC/USDT", "ETH/USDT"}:
            raise ValueError("V7 requires full8 mapping and BTC/ETH anchors")
        if source.get("missing_daily_archive_policy") != "missing_not_zero":
            raise ValueError("V7 may not zero-impute missing archives")
        if source.get("exact_duplicate_policy") != "drop_and_count" or source.get("conflicting_duplicate_policy") != "reject":
            raise ValueError("V7 duplicate policy changed")
        if source.get("official_checksum_required") is not True or source.get("raw_archive_required") is not True:
            raise ValueError("V7 requires checksummed retained raw data")
        execution = self.payload["execution_source"]
        manifest_path = root / str(execution["manifest"])
        if sha256_file(manifest_path) != execution["manifest_sha256"]:
            raise ValueError("V7 execution manifest SHA mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_sha256") != execution.get("dataset_sha256"):
            raise ValueError("V7 execution dataset SHA mismatch")
        if int(execution.get("holding_bars", 0)) != 3 or execution.get("cross_fold_positions_allowed") is not False:
            raise ValueError("V7 execution lifecycle changed")
        if [event.get("event_id") for event in self.feasibility.get("events", [])] != list(V7_EVENT_IDS):
            raise ValueError("V7 preregistered events changed")
        if float(self.feasibility.get("stress_round_trip_cost_bps", 0.0)) != 24.0:
            raise ValueError("V7 stress cost changed")
        if int(self.feasibility.get("minimum_events_per_fold", 0)) != 30:
            raise ValueError("V7 incidence gate changed")
        if [fold.get("name") for fold in self.folds] != ["WF1", "WF2", "WF3", "WF4", "WF5"]:
            raise ValueError("V7 walk-forward folds changed")
        evolution = self.payload["evolution"]
        if evolution.get("extension_allowed") is not False or evolution.get("fallback_selection_allowed") is not False:
            raise ValueError("V7 search cannot be extended or use fallback selection")


def load_v7_contract(path: Path, *, root: Path) -> PrimusV7Contract:
    contract = PrimusV7Contract(path.resolve(), json.loads(path.read_text(encoding="utf-8")))
    contract.validate(root)
    return contract
