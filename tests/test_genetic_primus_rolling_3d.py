from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from exia.genetic_primus.contracts import FEATURE_NAMES, STATIC_FEATURE_NAMES
from exia.genetic_primus.features import RobustScaler
from exia.genetic_primus.policy_v4 import V4NetworkShape
from exia.genetic_primus.rolling import (
    RollingProtocol,
    RollingOrigin,
    TrainedOriginModel,
    build_rolling_origins,
    evaluate_rolling,
    load_rolling_protocol,
    model_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/genetic_primus_rolling_3d_v1.json"


def test_repository_protocol_builds_non_overlapping_three_day_origins() -> None:
    protocol = load_rolling_protocol(PROTOCOL_PATH)
    origins = build_rolling_origins(protocol)
    assert len(origins) == 121
    assert all(origin.deployment_end - origin.deployment_start == 3 * 86_400_000 for origin in origins)
    assert all(first.deployment_end == second.deployment_start for first, second in zip(origins[:-1], origins[1:], strict=True))


def _toy_protocol() -> RollingProtocol:
    payload = {
        "timeframe_minutes": 240,
        "training_start": "2024-01-01T00:00:00Z",
        "validation_start": "2024-01-22T00:00:00Z",
        "validation_end": "2024-01-31T00:00:00Z",
        "deployment_horizon_days": 3,
        "retrain_cadence_days": 3,
        "label_purge_bars": 7,
    }
    return RollingProtocol(Path("toy"), payload, "toy-protocol-sha")


def _toy_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = pd.date_range("2024-01-01", "2024-01-31", freq="4h", inclusive="left", tz="UTC")
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(("BTC/USDT", "LINK/USDT")):
        for index, timestamp in enumerate(timestamps):
            close = 100.0 + symbol_index * 10.0 + index * 0.05
            row: dict[str, object] = {
                "timestamp": int(timestamp.timestamp() * 1000),
                "symbol": symbol,
                "open": close - 0.02,
                "high": close + 0.10,
                "low": close - 0.10,
                "close": close,
                "market_state": "quiet_range",
                "feature_supported": True,
            }
            for feature_index, feature in enumerate(STATIC_FEATURE_NAMES):
                row[feature] = index * (feature_index + 1) * 0.001 + symbol_index
            rows.append(row)
    features = pd.DataFrame(rows).sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)
    labels = features.loc[:, ["timestamp", "symbol"]].copy()
    labels["label_supported"] = True
    labels["target_position"] = np.int8(0)
    return features, labels


def test_rolling_refits_scaler_and_pins_every_decision_to_model_sha() -> None:
    features, labels = _toy_frames()
    shape = V4NetworkShape(inputs=len(FEATURE_NAMES), hidden=(), outputs=3)
    medians: list[float] = []

    def trainer(context):
        medians.append(float(context.scaler.median[0]))
        return TrainedOriginModel(
            genome=np.zeros(shape.genome_size, dtype=np.float32),
            shape=shape,
            trainer_id="toy-trainer/1",
            metadata={"origin_index": context.origin.index},
        )

    result = evaluate_rolling(
        features=features,
        labels=labels,
        protocol=_toy_protocol(),
        trainer=trainer,
        round_trip_cost_bps=16.0,
        max_origins=2,
    )
    assert len(result.origins) == 2
    assert len(medians) == 2 and medians[0] != medians[1]
    assert result.models["scaler_sha256"].nunique() == 2
    assert result.models["model_sha256"].nunique() == 2
    assert not result.decisions.empty
    assert not result.decisions["model_sha256"].isna().any()
    assert result.decisions["model_sha256"].nunique() == 2
    assert set(result.decisions["model_sha256"]) == set(result.models["model_sha256"])
    assert (result.decisions["purge_cutoff"] == result.decisions["train_end"] - 7 * 4 * 60 * 60 * 1000).all()
    assert len(result.benchmarks) == 11
    assert set(result.benchmarks["agent_id"]) == {
        "GeneticPrimusChallenger",
        "GeneticPrimusActive",
        "GeneticPrimusIncumbent",
        "NoTrade",
        "Momentum_return_1",
        "MeanReversion_return_1",
    }
    no_trade = result.benchmarks.loc[result.benchmarks["agent_id"].eq("NoTrade")]
    assert (no_trade["net_return"] == 0.0).all()
    assert {"candidate_delta_bps", "candidate_outperformed", "return_rank"} <= set(
        result.benchmarks.columns
    )
    incumbent = result.benchmarks.loc[
        result.benchmarks["agent_id"].eq("GeneticPrimusIncumbent")
    ]
    assert incumbent.iloc[0]["policy_sha256"] == result.models.iloc[0]["model_sha256"]


def test_stress_benchmarks_keep_their_own_cost_scope() -> None:
    features, labels = _toy_frames()
    shape = V4NetworkShape(inputs=len(FEATURE_NAMES), hidden=(), outputs=3)

    def trainer(context):
        return TrainedOriginModel(
            genome=np.zeros(shape.genome_size, dtype=np.float32),
            shape=shape,
            trainer_id="toy-stress-trainer/1",
            metadata={"origin_index": context.origin.index},
        )

    result = evaluate_rolling(
        features=features,
        labels=labels,
        protocol=_toy_protocol(),
        trainer=trainer,
        round_trip_cost_bps=16.0,
        stress_round_trip_cost_bps=24.0,
        max_origins=1,
    )
    stress = result.benchmarks.loc[result.benchmarks["agent_id"].str.contains("Stress")]
    assert len(stress) == 2
    assert set(stress["round_trip_cost_bps"]) == {24.0}


def test_rolling_passes_frozen_ensemble_to_next_origin_as_incumbent() -> None:
    features, labels = _toy_frames()
    shape = V4NetworkShape(inputs=len(FEATURE_NAMES), hidden=(), outputs=3)
    base = np.zeros(shape.genome_size, dtype=np.float32)
    members = (base.copy(), base.copy(), base.copy())
    incumbent_sizes: list[int] = []
    incumbent_shas: list[str | None] = []

    def trainer(context):
        incumbent_sizes.append(len(context.incumbent_genomes))
        incumbent_shas.append(context.incumbent_model_sha256)
        return TrainedOriginModel(
            genome=members[0],
            shape=shape,
            trainer_id="toy-ensemble-trainer/1",
            metadata={"origin_index": context.origin.index},
            ensemble_genomes=members,
        )

    result = evaluate_rolling(
        features=features,
        labels=labels,
        protocol=_toy_protocol(),
        trainer=trainer,
        round_trip_cost_bps=16.0,
        max_origins=2,
    )
    assert incumbent_sizes == [0, 3]
    assert incumbent_shas == [None, result.models.iloc[0]["model_sha256"]]
    assert result.models["ensemble_size"].tolist() == [3, 3]


def test_model_sha_binds_every_ensemble_member() -> None:
    shape = V4NetworkShape(inputs=len(FEATURE_NAMES), hidden=(), outputs=3)
    base = np.zeros(shape.genome_size, dtype=np.float32)
    changed = base.copy()
    changed[0] = 0.1
    common = {
        "genome": base,
        "shape": shape,
        "trainer_id": "sha-test/1",
        "metadata": {},
    }
    first = TrainedOriginModel(**common, ensemble_genomes=(base, base.copy(), base.copy()))
    second = TrainedOriginModel(**common, ensemble_genomes=(base, base.copy(), changed))
    scaler = RobustScaler(np.zeros(25), np.ones(25))
    origin = RollingOrigin(0, 0, 10, 3, 10, 20)
    assert model_sha256(first, scaler, origin, "protocol") != model_sha256(
        second, scaler, origin, "protocol"
    )


def test_failed_promotion_retains_frozen_incumbent_sha_and_scaler() -> None:
    features, labels = _toy_frames()
    shape = V4NetworkShape(inputs=len(FEATURE_NAMES), hidden=(), outputs=3)

    def trainer(context):
        genome = np.zeros(shape.genome_size, dtype=np.float32)
        genome[-1 if context.origin.index == 0 else -3] = 1.0
        return TrainedOriginModel(
            genome=genome,
            shape=shape,
            trainer_id="promotion-retention-test/1",
            metadata={"origin_index": context.origin.index},
            promotion_approved=context.origin.index == 0,
        )

    result = evaluate_rolling(
        features=features,
        labels=labels,
        protocol=_toy_protocol(),
        trainer=trainer,
        round_trip_cost_bps=16.0,
        max_origins=2,
    )
    first, second = result.models.iloc[0], result.models.iloc[1]
    assert bool(first["promotion_approved"])
    assert not bool(second["promotion_approved"])
    assert second["model_sha256"] == first["model_sha256"]
    assert second["scaler_sha256"] == first["scaler_sha256"]
    assert second["challenger_model_sha256"] != second["model_sha256"]
    second_decisions = result.decisions.loc[
        result.decisions["origin_timestamp"].eq(result.origins.iloc[1]["deployment_start"])
    ]
    assert set(second_decisions["model_sha256"]) == {first["model_sha256"]}
