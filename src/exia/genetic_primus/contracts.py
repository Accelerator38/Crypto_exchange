from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SPLIT_NAMES = ("development", "validation", "oos", "sanity")
MARKET_STATES = (
    "trend_up",
    "trend_down",
    "quiet_range",
    "volatile_range",
    "transition",
)
STATIC_FEATURE_NAMES = (
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "ema_gap_12_48",
    "ema_slope_12_3",
    "atr_14_pct",
    "realized_vol_6",
    "realized_vol_ratio_6_24",
    "efficiency_12",
    "range_position_24",
    "donchian_position_20",
    "volume_z_24",
    "volume_trend_6_24",
    "candle_body_pct",
    "wick_asymmetry",
    "market_return_3",
    "market_return_12",
    "market_breadth_3",
    "market_breadth_12",
    "market_dispersion_3",
    "market_vol_percentile",
    "transition_score",
    "relative_momentum_rank",
    "residual_return_3",
)
DYNAMIC_FEATURE_NAMES = (
    "position_direction",
    "unrealized_net_bps",
    "holding_bars",
    "mfe_bps",
    "mae_bps",
)
FEATURE_NAMES = STATIC_FEATURE_NAMES + DYNAMIC_FEATURE_NAMES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


@dataclass(frozen=True)
class PrimusConfig:
    source_path: Path
    payload: Mapping[str, Any]

    SCHEMA_VERSION = "exia.genetic_primus.v1"

    @property
    def dataset(self) -> Mapping[str, Any]:
        return self.payload["dataset"]

    @property
    def windows(self) -> Mapping[str, Mapping[str, str]]:
        return {item["name"]: item for item in self.payload["windows"]}

    @property
    def network(self) -> Mapping[str, Any]:
        return self.payload["network"]

    @property
    def labels(self) -> Mapping[str, Any]:
        return self.payload["labels"]

    @property
    def costs(self) -> Mapping[str, float]:
        return self.payload["costs_bps"]

    @property
    def fitness(self) -> Mapping[str, Any]:
        return self.payload["fitness"]

    @property
    def evolution(self) -> Mapping[str, Any]:
        return self.payload["evolution"]

    def validate(self, *, root: Path | None = None) -> None:
        expected = {
            "schema_version",
            "agent_id",
            "description",
            "dataset",
            "windows",
            "network",
            "labels",
            "costs_bps",
            "fitness",
            "evolution",
            "safety",
        }
        _exact_keys(self.payload, expected, "config")
        if self.payload["schema_version"] != self.SCHEMA_VERSION:
            raise ValueError("unsupported Genetic_Primus schema")
        if self.payload["agent_id"] != "Genetic_Primus":
            raise ValueError("agent_id must be Genetic_Primus")
        safety = self.payload["safety"]
        _exact_keys(
            safety,
            {"paper_allowed", "live_allowed", "orders_enabled", "promotion_authority"},
            "safety",
        )
        if any(value is not False for value in safety.values()):
            raise ValueError("Genetic_Primus V1 safety flags must all be false")

        dataset = self.dataset
        _exact_keys(
            dataset,
            {"manifest", "dataset_sha256", "timeframe_minutes", "symbols"},
            "dataset",
        )
        if int(dataset["timeframe_minutes"]) != 240:
            raise ValueError("Genetic_Primus V1 requires a fixed 4h timeframe")
        symbols = tuple(str(value) for value in dataset["symbols"])
        if len(symbols) != 8 or len(set(symbols)) != 8:
            raise ValueError("Genetic_Primus V1 requires eight unique symbols")

        windows = self.payload["windows"]
        if [item.get("name") for item in windows] != list(SPLIT_NAMES):
            raise ValueError("windows must be development, validation, oos, sanity")
        previous_end = None
        for item in windows:
            if set(item) != {"name", "start", "end"}:
                raise ValueError("window keys must be name/start/end")
            if item["start"] >= item["end"]:
                raise ValueError(f"invalid window: {item['name']}")
            if previous_end is not None and item["start"] != previous_end:
                raise ValueError("windows must be contiguous and non-overlapping")
            previous_end = item["end"]

        network = self.network
        _exact_keys(network, {"inputs", "hidden", "outputs", "activation"}, "network")
        if int(network["inputs"]) != len(FEATURE_NAMES):
            raise ValueError("network input count does not match feature contract")
        if list(network["hidden"]) != [32, 16] or int(network["outputs"]) != 3:
            raise ValueError("network topology must be 30->32->16->3")
        if network["activation"] != "tanh":
            raise ValueError("network activation must be tanh")

        labels = self.labels
        _exact_keys(labels, {"horizons_bars", "minimum_edge_bps", "minimum_votes"}, "labels")
        if list(labels["horizons_bars"]) != [1, 3, 6]:
            raise ValueError("label horizons must be 1/3/6 bars")
        if int(labels["minimum_votes"]) not in (2, 3):
            raise ValueError("minimum_votes must be 2 or 3")

        costs = self.costs
        _exact_keys(costs, {"base_round_trip", "stress_round_trip"}, "costs_bps")
        if float(costs["base_round_trip"]) <= 0:
            raise ValueError("base cost must be positive")
        if float(costs["stress_round_trip"]) < float(costs["base_round_trip"]):
            raise ValueError("stress cost cannot be below base cost")

        fitness = self.fitness
        _exact_keys(
            fitness,
            {
                "period",
                "weights",
                "bootstrap_samples",
                "bootstrap_alpha",
                "minimum_period_trades",
                "minimum_split_trades",
                "minimum_slice_trades",
                "minimum_supported_period_rate",
                "max_drawdown",
                "max_direction_share",
                "max_state_share",
                "max_symbol_share",
            },
            "fitness",
        )
        if fitness["period"] != "calendar_month":
            raise ValueError("fitness period must be calendar_month")
        weights = fitness["weights"]
        _exact_keys(weights, {"mean", "median", "lcb"}, "fitness.weights")
        if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
            raise ValueError("fitness weights must sum to 1")

        evolution = self.evolution
        _exact_keys(
            evolution,
            {
                "seed",
                "workers",
                "population_size",
                "generations",
                "elite_count",
                "tournament_size",
                "crossover_rate",
                "mutation_rate",
                "mutation_sigma",
            },
            "evolution",
        )
        if int(evolution["elite_count"]) >= int(evolution["population_size"]):
            raise ValueError("elite_count must be below population_size")

        if root is not None:
            manifest_path = root / str(dataset["manifest"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("dataset_sha256") != dataset["dataset_sha256"]:
                raise ValueError("dataset SHA mismatch")
            if manifest.get("validation", {}).get("passed") is not True:
                raise ValueError("dataset validation did not pass")
            if any(
                manifest.get(key) is not False
                for key in ("paper_allowed", "live_allowed", "orders_enabled", "promotion_authority")
            ):
                raise ValueError("source manifest has unsafe authority flags")


def load_primus_config(path: Path, *, root: Path | None = None) -> PrimusConfig:
    config = PrimusConfig(path.resolve(), json.loads(path.read_text(encoding="utf-8")))
    config.validate(root=root)
    return config
