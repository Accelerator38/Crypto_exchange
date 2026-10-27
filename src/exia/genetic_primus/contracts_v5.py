from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

from .contracts import sha256_file


V5_SCHEMA = "exia.genetic_primus_experiment_contract.v5"
V5_SAFETY_KEYS = {
    "validation_opened",
    "oos_opened",
    "paper_allowed",
    "live_allowed",
    "orders_enabled",
    "promotion_authority",
}
V5_EVENT_IDS = (
    "oi_taker_price_continuation",
    "crowding_unwind_reversal",
    "oi_price_divergence_reversal",
)


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError(f"V5 timestamp must have an explicit timezone: {value}")
    return parsed


@dataclass(frozen=True)
class PrimusV5Contract:
    source_path: Path
    payload: Mapping[str, Any]

    @property
    def source(self) -> Mapping[str, Any]:
        return self.payload["source"]

    @property
    def execution_source(self) -> Mapping[str, Any]:
        return self.payload["execution_source"]

    @property
    def feasibility(self) -> Mapping[str, Any]:
        return self.payload["source_feasibility"]

    @property
    def folds(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.payload["walk_forward"])

    @property
    def sha256(self) -> str:
        return sha256_file(self.source_path)

    def validate(self, root: Path) -> None:
        if self.payload.get("schema_version") != V5_SCHEMA:
            raise ValueError("unexpected Genetic_Primus V5 contract schema")
        if self.payload.get("agent_id") != "Genetic_Primus_V5":
            raise ValueError("unexpected Genetic_Primus V5 agent id")
        if self.payload.get("immutable_after_freeze") is not True:
            raise ValueError("V5 contract must be immutable after freeze")
        safety = self.payload.get("safety", {})
        if set(safety) != V5_SAFETY_KEYS or any(value is not False for value in safety.values()):
            raise ValueError("V5 contract must be fail-closed")
        source = self.source
        if source.get("source_family") != "cross_exchange_derivatives_metrics":
            raise ValueError("V5 requires a new cross-exchange derivatives family")
        if int(source.get("source_interval_minutes", 0)) != 5:
            raise ValueError("V5 derivatives source must be five-minute point-in-time data")
        if int(source.get("decision_timeframe_minutes", 0)) != 240:
            raise ValueError("V5 decisions must remain on four-hour bars")
        if len(source.get("symbols", {})) != 8 or int(source.get("minimum_symbols", 0)) != 8:
            raise ValueError("V5 source must cover the fixed full8 universe")
        if date.fromisoformat(str(source["start_date"])) != date(2022, 1, 1):
            raise ValueError("V5 source start date changed")
        if date.fromisoformat(str(source["end_date"])) != date(2024, 12, 31):
            raise ValueError("V5 source end date changed")
        if source.get("point_in_time_required") is not True:
            raise ValueError("V5 source must be point-in-time")
        if source.get("official_checksum_required") is not True or source.get("raw_archive_required") is not True:
            raise ValueError("V5 source must retain checksummed raw provenance")
        execution = self.execution_source
        if int(execution.get("timeframe_minutes", 0)) != 240:
            raise ValueError("V5 execution timeframe changed")
        if int(execution.get("holding_bars", 0)) != 6:
            raise ValueError("V5 maximum/fixed holding must be six bars")
        if execution.get("cross_fold_positions_allowed") is not False:
            raise ValueError("V5 cannot carry positions across folds")
        manifest_path = root / str(execution["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_sha256") != execution.get("dataset_sha256"):
            raise ValueError("V5 Bitget execution dataset SHA mismatch")
        feasibility = self.feasibility
        if [item.get("event_id") for item in feasibility.get("events", [])] != list(V5_EVENT_IDS):
            raise ValueError("V5 feasibility events changed after preregistration")
        if float(feasibility.get("stress_round_trip_cost_bps", 0.0)) != 24.0:
            raise ValueError("V5 source gate stress cost must be 24 bps")
        if int(feasibility.get("minimum_events_per_fold", 0)) != 30:
            raise ValueError("V5 per-fold incidence gate changed")
        folds = list(self.folds)
        if [fold.get("name") for fold in folds] != ["WF1", "WF2", "WF3", "WF4"]:
            raise ValueError("V5 walk-forward folds changed")
        for fold in folds:
            train_start = _utc(str(fold["train_start"]))
            train_end = _utc(str(fold["train_end"]))
            evaluation_start = _utc(str(fold["evaluation_start"]))
            evaluation_end = _utc(str(fold["evaluation_end"]))
            if not train_start < train_end <= evaluation_start < evaluation_end:
                raise ValueError(f"invalid V5 walk-forward fold: {fold['name']}")
        evolution = self.payload["evolution"]
        if list(evolution.get("screen_seeds", [])) != [515, 616, 717]:
            raise ValueError("V5 seed screen changed")
        if int(evolution.get("required_passing_seeds", 0)) != 2:
            raise ValueError("V5 requires two passing seeds")
        if evolution.get("extension_allowed") is not False or evolution.get("fallback_selection_allowed") is not False:
            raise ValueError("V5 search budget cannot be extended or use fallback selection")


def load_v5_contract(path: Path, *, root: Path) -> PrimusV5Contract:
    contract = PrimusV5Contract(path.resolve(), json.loads(path.read_text(encoding="utf-8")))
    contract.validate(root)
    return contract
