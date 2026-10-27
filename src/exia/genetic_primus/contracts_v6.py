from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import sha256_file
from .contracts_v5 import V5_EVENT_IDS, V5_SAFETY_KEYS


@dataclass(frozen=True)
class PrimusV6Contract:
    source_path: Path
    payload: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return sha256_file(self.source_path)

    @property
    def flow_source(self) -> Mapping[str, Any]:
        return self.payload["flow_source"]

    @property
    def feasibility(self) -> Mapping[str, Any]:
        return self.payload["source_feasibility"]

    @property
    def folds(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.payload["walk_forward"])

    def validate(self, root: Path) -> None:
        if self.payload.get("schema_version") != "exia.genetic_primus_experiment_contract.v6":
            raise ValueError("unexpected Genetic_Primus V6 schema")
        if self.payload.get("agent_id") != "Genetic_Primus_V6" or self.payload.get("immutable_after_freeze") is not True:
            raise ValueError("invalid Genetic_Primus V6 identity or mutability")
        safety = self.payload.get("safety", {})
        if set(safety) != V5_SAFETY_KEYS or any(value is not False for value in safety.values()):
            raise ValueError("V6 contract must be fail-closed")
        base = self.payload["base_derivatives_source"]
        base_manifest_path = root / str(base["manifest"])
        if sha256_file(base_manifest_path) != base["manifest_sha256"]:
            raise ValueError("V6 base derivatives manifest SHA mismatch")
        base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
        if base_manifest.get("dataset_sha256") != base["dataset_sha256"]:
            raise ValueError("V6 base derivatives dataset SHA mismatch")
        flow = self.flow_source
        if flow.get("source_family") != "cross_exchange_oi_kline_flow":
            raise ValueError("unexpected V6 source family")
        if flow.get("start_month") != "2022-01" or flow.get("end_month") != "2024-12":
            raise ValueError("V6 flow source window changed")
        if int(flow.get("timeframe_minutes", 0)) != 240 or len(flow.get("symbols", {})) != 8:
            raise ValueError("V6 requires full8 four-hour flow")
        if flow.get("derived_flow_formula") != "2 * taker_buy_quote_volume / quote_volume - 1":
            raise ValueError("V6 taker-flow formula changed")
        if flow.get("official_checksum_required") is not True or flow.get("raw_archive_required") is not True:
            raise ValueError("V6 requires checksummed retained raw flow")
        execution = self.payload["execution_source"]
        if int(execution.get("holding_bars", 0)) != 6 or execution.get("cross_fold_positions_allowed") is not False:
            raise ValueError("V6 execution lifecycle changed")
        execution_manifest = json.loads((root / str(execution["manifest"])).read_text(encoding="utf-8"))
        if execution_manifest.get("dataset_sha256") != execution.get("dataset_sha256"):
            raise ValueError("V6 execution dataset SHA mismatch")
        if [item.get("event_id") for item in self.feasibility.get("events", [])] != list(V5_EVENT_IDS):
            raise ValueError("V6 preregistered events changed")
        if float(self.feasibility.get("stress_round_trip_cost_bps", 0.0)) != 24.0:
            raise ValueError("V6 stress cost changed")
        if int(self.feasibility.get("minimum_events_per_fold", 0)) != 30:
            raise ValueError("V6 incidence gate changed")
        if [item.get("name") for item in self.folds] != ["WF1", "WF2", "WF3", "WF4"]:
            raise ValueError("V6 walk-forward folds changed")
        evolution = self.payload["evolution"]
        if list(evolution.get("screen_seeds", [])) != [919, 1020, 1121]:
            raise ValueError("V6 screen seeds changed")
        if evolution.get("extension_allowed") is not False or evolution.get("fallback_selection_allowed") is not False:
            raise ValueError("V6 search cannot be extended or use fallback selection")


def load_v6_contract(path: Path, *, root: Path) -> PrimusV6Contract:
    contract = PrimusV6Contract(path.resolve(), json.loads(path.read_text(encoding="utf-8")))
    contract.validate(root)
    return contract
