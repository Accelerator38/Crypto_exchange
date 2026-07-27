from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .strategy_candidate_registry import compute_candidate_profile_sha256


DEVELOPMENT_CANDIDATE_SCHEMA_VERSION = (
    "panteon.strategy_development_candidate.v1"
)


class DevelopmentCandidateError(ValueError):
    pass


@dataclass(frozen=True)
class DevelopmentCandidateContract:
    path: Path
    payload: dict[str, Any]

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> "DevelopmentCandidateContract":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DevelopmentCandidateError(
                f"cannot load development candidate: {source}"
            ) from exc
        return cls(
            path=source.resolve(),
            payload=validate_development_candidate(payload),
        )


def validate_development_candidate(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    row = _json_copy(payload)
    expected = {
        "schema_version",
        "created_at",
        "exchange",
        "dataset",
        "protocol",
        "candidate",
        "operational_candidate_id",
        "orders_enabled",
        "promotion_authority",
    }
    if set(row) != expected:
        raise DevelopmentCandidateError("development contract keys are invalid")
    if row["schema_version"] != DEVELOPMENT_CANDIDATE_SCHEMA_VERSION:
        raise DevelopmentCandidateError("unsupported development contract schema")
    _timestamp(row["created_at"])
    if row["exchange"] != "BITGET":
        raise DevelopmentCandidateError("development exchange must be BITGET")
    if row["operational_candidate_id"] is not None:
        raise DevelopmentCandidateError(
            "development contract cannot set an operational candidate"
        )
    _no_authority(row, "development contract")

    dataset = row["dataset"]
    if not isinstance(dataset, Mapping):
        raise DevelopmentCandidateError("development dataset is invalid")
    if set(dataset) != {
        "dataset_id",
        "data_dir",
        "dataset_sha256",
        "integrity_manifest_sha256",
        "symbols",
    }:
        raise DevelopmentCandidateError("development dataset keys are invalid")
    symbols = tuple(str(item) for item in dataset["symbols"])
    if symbols != ("BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "BNB", "LINK"):
        raise DevelopmentCandidateError("development dataset must pin full8")
    for key in ("dataset_sha256", "integrity_manifest_sha256"):
        if len(str(dataset[key])) != 64:
            raise DevelopmentCandidateError("development dataset SHA is invalid")

    protocol = row["protocol"]
    if not isinstance(protocol, Mapping):
        raise DevelopmentCandidateError("development protocol is invalid")
    if set(protocol) != {
        "stage",
        "development_start_at",
        "development_end_at",
        "sealed_windows",
        "signal_timing",
        "retuning_policy",
        "costs",
        "portfolio",
        "gates",
    }:
        raise DevelopmentCandidateError("development protocol keys are invalid")
    if protocol["stage"] != "development_only":
        raise DevelopmentCandidateError("only development stage may be opened")
    start = _timestamp(protocol["development_start_at"])
    end = _timestamp(protocol["development_end_at"])
    if end <= start or end.year > 2023:
        raise DevelopmentCandidateError("development window opens sealed data")
    if protocol["sealed_windows"] != ["validation", "oos", "sanity"]:
        raise DevelopmentCandidateError("validation/OOS/sanity must stay sealed")
    if (
        protocol["signal_timing"]
        != "closed_4h_bar_signal_next_4h_bar_open_fill"
    ):
        raise DevelopmentCandidateError("development signal timing is invalid")
    if (
        protocol["retuning_policy"]
        != "new_candidate_id_required_after_development_result"
    ):
        raise DevelopmentCandidateError("development retuning policy is invalid")

    costs = protocol["costs"]
    cost_floor = sum(
        float(costs[key])
        for key in (
            "round_trip_fee_bps",
            "slippage_bps",
            "safety_buffer_bps",
        )
    )
    if cost_floor <= 0.0 or float(costs["cost_stress_multiplier"]) < 1.0:
        raise DevelopmentCandidateError("development costs are invalid")

    candidate = row["candidate"]
    if not isinstance(candidate, Mapping):
        raise DevelopmentCandidateError("development candidate is invalid")
    candidate_id = str(candidate.get("candidate_id") or "")
    portfolio = protocol["portfolio"]
    expected_positions = (
        2
        if candidate_id == "market_neutral_relative_momentum_4h_v1"
        else 1
    )
    if float(portfolio["notional_per_trade_usd"]) <= 0.0 or int(
        portfolio["max_open_positions"]
    ) != expected_positions:
        raise DevelopmentCandidateError("development portfolio is invalid")
    gates = protocol["gates"]
    if not _gates_are_strict(
        gates,
        paired=candidate_id == "market_neutral_relative_momentum_4h_v1",
    ):
        raise DevelopmentCandidateError("development gates are too weak")

    if candidate_id not in {
        "cross_sectional_trend_4h_v1",
        "market_neutral_relative_momentum_4h_v1",
    }:
        raise DevelopmentCandidateError("unexpected development candidate ID")
    if candidate["status"] != "pre_registered_development_only":
        raise DevelopmentCandidateError("development candidate status is invalid")
    event = candidate["event_contract"]
    low_turnover_cost_aware = (
        int(event["aggregate_bars"]) == 4
        and int(event["decision_every_bars"]) == 6
    )
    if candidate_id == "cross_sectional_trend_4h_v1":
        low_turnover_cost_aware = (
            low_turnover_cost_aware
            and float(event["min_momentum_bps"]) >= cost_floor * 4.0
        )
    else:
        low_turnover_cost_aware = (
            low_turnover_cost_aware
            and float(event["min_cross_sectional_spread_bps"])
            >= cost_floor * 8.0
            and candidate["hypothesis"]["portfolio_unit"]
            == "synchronized_pair"
            and int(candidate["selection_contract"]["required_legs"]) == 2
            and candidate["selection_contract"]["equal_notional_legs"] is True
            and int(candidate["selection_contract"]["max_open_positions"]) == 2
            and candidate["exit_contract"]["execution"]
            == "both_legs_next_bar_open"
        )
    if not low_turnover_cost_aware:
        raise DevelopmentCandidateError(
            "development candidate is not low-turnover cost-aware"
        )
    if (
        candidate["profile_sha256"]
        != compute_candidate_profile_sha256(candidate)
    ):
        raise DevelopmentCandidateError("development profile SHA mismatch")
    _no_authority(candidate, "development candidate")
    if (
        candidate["runtime_actor_created"] is not False
        or candidate["paper_allowed"] is not False
        or candidate["live_allowed"] is not False
    ):
        raise DevelopmentCandidateError(
            "development candidate has runtime authority"
        )
    return row


def _gates_are_strict(gates: Mapping[str, Any], *, paired: bool) -> bool:
    common = (
        int(gates["min_filled_orders"]) >= (40 if paired else 20)
        and int(gates["min_closed_trades"]) >= 10
        and float(gates["min_mean_net_bps"]) >= 0.0
        and float(gates["min_lcb_95_net_bps"]) >= 0.0
        and gates["cost_stress_hard_fail"] is True
        and gates["require_baseline_improvement"] is True
    )
    if paired:
        return (
            common
            and gates["pair_collapse_hard_fail"] is True
            and gates["paired_fill_atomicity_hard_fail"] is True
        )
    return (
        common
        and int(gates["min_direction_trades"]) >= 10
        and gates["direction_collapse_hard_fail"] is True
    )


def _no_authority(row: Mapping[str, Any], label: str) -> None:
    if row["orders_enabled"] is not False or row["promotion_authority"] is not False:
        raise DevelopmentCandidateError(f"{label} has trading authority")


def _timestamp(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise DevelopmentCandidateError("invalid development timestamp") from exc
    if parsed.tzinfo is None:
        raise DevelopmentCandidateError(
            "development timestamp must be timezone-aware"
        )
    return parsed


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise DevelopmentCandidateError(
            "development contract must be finite JSON"
        ) from exc
