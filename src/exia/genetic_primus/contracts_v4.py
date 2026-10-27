from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import FEATURE_NAMES


@dataclass(frozen=True)
class PrimusV4Config:
    source_path: Path
    payload: Mapping[str, Any]

    @property
    def dataset(self) -> Mapping[str, Any]:
        return self.payload["dataset"]

    @property
    def folds(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.payload["folds"])

    @property
    def architectures(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.payload["architectures"])

    @property
    def costs(self) -> Mapping[str, float]:
        return self.payload["costs_bps"]

    @property
    def fitness(self) -> Mapping[str, Any]:
        return self.payload["fitness"]

    def validate(self, root: Path) -> None:
        if self.payload.get("schema_version") != "exia.genetic_primus.v4":
            raise ValueError("unexpected Genetic_Primus V4 schema")
        if self.payload.get("agent_id") != "Genetic_Primus_V4":
            raise ValueError("unexpected V4 agent id")
        safety = self.payload.get("safety", {})
        if set(safety) != {"paper_allowed", "live_allowed", "orders_enabled", "promotion_authority"}:
            raise ValueError("invalid V4 safety keys")
        if any(value is not False for value in safety.values()):
            raise ValueError("V4 safety flags must all be false")
        dataset = self.dataset
        if int(dataset.get("timeframe_minutes", 0)) != 240 or len(dataset.get("symbols", [])) != 8:
            raise ValueError("V4 requires Bitget full8 at 4h")
        manifest = json.loads((root / str(dataset["manifest"])).read_text(encoding="utf-8"))
        if manifest.get("dataset_sha256") != dataset.get("dataset_sha256"):
            raise ValueError("V4 dataset SHA mismatch")
        if manifest.get("validation", {}).get("passed") is not True:
            raise ValueError("V4 dataset manifest is not valid")
        folds = list(self.folds)
        expected_names = ["2022_H1", "2022_H2", "2023_H1", "2023_H2", "2024_H1", "2024_H2"]
        if [fold.get("name") for fold in folds] != expected_names:
            raise ValueError("V4 must define six fixed half-year folds")
        if folds[0]["start"] != "2022-01-01T00:00:00Z" or folds[-1]["end"] != "2025-01-01T00:00:00Z":
            raise ValueError("V4 research folds must cover 2022-2024")
        for first, second in zip(folds[:-1], folds[1:], strict=True):
            if first["end"] != second["start"]:
                raise ValueError("V4 folds must be contiguous")
        windows = self.payload["windows"]
        if windows != {
            "research": {"start": "2022-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z"},
            "validation": {"start": "2025-01-01T00:00:00Z", "end": "2026-01-01T00:00:00Z"},
            "oos": {"start": "2026-01-01T00:00:00Z", "end": "2026-07-15T00:00:00Z"},
        }:
            raise ValueError("V4 windows are not sealed")
        architectures = {item["id"]: item["hidden"] for item in self.architectures}
        if architectures != {"linear": [], "compact_8": [8]}:
            raise ValueError("V4 architecture screen permits only linear and one 8-unit hidden layer")
        if any(int(item.get("inputs", 0)) != len(FEATURE_NAMES) or int(item.get("outputs", 0)) != 3 for item in self.architectures):
            raise ValueError("V4 architecture IO does not match feature/action contract")
        if list(self.payload["screen"]["seeds"]) != [141, 242, 343]:
            raise ValueError("V4 screen seeds are not preregistered")
        if float(self.costs["base_round_trip"]) != 16.0 or float(self.costs["stress_round_trip"]) != 24.0:
            raise ValueError("V4 costs must be fixed to 16/24 bps")


def load_v4_config(path: Path, *, root: Path) -> PrimusV4Config:
    config = PrimusV4Config(path.resolve(), json.loads(path.read_text(encoding="utf-8")))
    config.validate(root)
    return config
