from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


STRATEGY_CANDIDATE_REGISTRY_SCHEMA_VERSION = (
    "panteon.strategy_candidate_registry.v1"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_READY = "pre_registered"
_BLOCKED = "blocked_missing_data"
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "created_at",
        "exchange",
        "dataset",
        "common_evaluation",
        "operational_candidate_id",
        "candidates",
        "orders_enabled",
        "promotion_authority",
    }
)
_DATASET_KEYS = frozenset(
    {
        "dataset_id",
        "data_dir",
        "dataset_sha256",
        "integrity_manifest_sha256",
        "start_at",
        "end_at",
        "timeframe",
        "symbols",
        "fields",
    }
)
_CANDIDATE_KEYS = frozenset(
    {
        "candidate_id",
        "family_id",
        "status",
        "hypothesis",
        "data_requirements",
        "event_contract",
        "exit_contract",
        "selection_contract",
        "profile_sha256",
        "evaluation_state",
        "historical_evaluation_allowed",
        "runtime_actor_created",
        "paper_allowed",
        "live_allowed",
        "orders_enabled",
        "promotion_authority",
    }
)
_DATA_REQUIREMENT_KEYS = frozenset(
    {"required_fields", "missing_fields", "availability"}
)


class StrategyCandidateRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class StrategyCandidateRegistry:
    path: Path
    payload: dict[str, Any]

    @classmethod
    def from_json(cls, path: str | Path) -> "StrategyCandidateRegistry":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StrategyCandidateRegistryError(
                f"cannot load strategy candidate registry: {source}"
            ) from exc
        return cls(
            path=source.resolve(),
            payload=validate_strategy_candidate_registry(payload),
        )

    def candidate(self, candidate_id: str) -> dict[str, Any]:
        target = str(candidate_id)
        for row in self.payload["candidates"]:
            if row["candidate_id"] == target:
                return dict(row)
        raise StrategyCandidateRegistryError(f"unknown candidate: {target}")

    @property
    def ready_candidate_ids(self) -> tuple[str, ...]:
        return tuple(
            row["candidate_id"]
            for row in self.payload["candidates"]
            if row["status"] == _READY
        )


def compute_candidate_profile_sha256(candidate: Mapping[str, Any]) -> str:
    requirements = candidate.get("data_requirements")
    if not isinstance(requirements, Mapping):
        raise StrategyCandidateRegistryError(
            "candidate data requirements are invalid"
        )
    payload = {
        "candidate_id": candidate.get("candidate_id"),
        "family_id": candidate.get("family_id"),
        "hypothesis": candidate.get("hypothesis"),
        "required_fields": requirements.get("required_fields"),
        "event_contract": candidate.get("event_contract"),
        "exit_contract": candidate.get("exit_contract"),
        "selection_contract": candidate.get("selection_contract"),
    }
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_strategy_candidate_registry(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    row = _json_copy(payload)
    _exact_keys(row, _TOP_LEVEL_KEYS, "strategy candidate registry")
    if row["schema_version"] != STRATEGY_CANDIDATE_REGISTRY_SCHEMA_VERSION:
        raise StrategyCandidateRegistryError(
            "unsupported strategy candidate registry schema"
        )
    _parse_datetime(row["created_at"], "registry created_at")
    if row["exchange"] != "BITGET":
        raise StrategyCandidateRegistryError("candidate exchange must be BITGET")
    _validate_no_authority(row, "strategy candidate registry")
    if row["operational_candidate_id"] is not None:
        raise StrategyCandidateRegistryError(
            "pre-registration cannot set an operational candidate"
        )

    dataset = row["dataset"]
    if not isinstance(dataset, Mapping):
        raise StrategyCandidateRegistryError("candidate dataset is invalid")
    _exact_keys(dataset, _DATASET_KEYS, "candidate dataset")
    _relative_path(dataset["data_dir"], "candidate data directory")
    for key in ("dataset_sha256", "integrity_manifest_sha256"):
        _sha256(dataset[key], f"candidate dataset {key}")
    start = _parse_datetime(dataset["start_at"], "dataset start_at")
    end = _parse_datetime(dataset["end_at"], "dataset end_at")
    if end <= start:
        raise StrategyCandidateRegistryError("candidate dataset window is invalid")
    if dataset["timeframe"] != "1h":
        raise StrategyCandidateRegistryError("candidate dataset must use 1h bars")
    symbols = _string_list(dataset["symbols"], "candidate symbols")
    fields = set(_string_list(dataset["fields"], "candidate dataset fields"))
    if len(symbols) != 8:
        raise StrategyCandidateRegistryError("candidate dataset must pin full8")
    if not {"timestamp", "open", "high", "low", "close", "volume"}.issubset(
        fields
    ):
        raise StrategyCandidateRegistryError("candidate OHLCV fields are incomplete")

    _validate_common_evaluation(row["common_evaluation"])
    candidates = row["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise StrategyCandidateRegistryError(
            "P2 registry must contain exactly three candidates"
        )

    candidate_ids: set[str] = set()
    ready = 0
    blocked = 0
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise StrategyCandidateRegistryError("candidate must be an object")
        _exact_keys(candidate, _CANDIDATE_KEYS, "strategy candidate")
        candidate_id = _identifier(candidate["candidate_id"], "candidate ID")
        _identifier(candidate["family_id"], "candidate family ID")
        if candidate_id in candidate_ids:
            raise StrategyCandidateRegistryError(
                f"duplicate candidate ID: {candidate_id}"
            )
        candidate_ids.add(candidate_id)
        requirements = candidate["data_requirements"]
        if not isinstance(requirements, Mapping):
            raise StrategyCandidateRegistryError(
                f"candidate {candidate_id} data requirements are invalid"
            )
        _exact_keys(
            requirements,
            _DATA_REQUIREMENT_KEYS,
            "candidate data requirements",
        )
        required_fields = set(
            _string_list(
                requirements["required_fields"],
                f"candidate {candidate_id} required fields",
            )
        )
        missing_fields = set(
            str(item) for item in requirements["missing_fields"]
        )
        actual_missing = required_fields - fields
        if missing_fields != actual_missing:
            raise StrategyCandidateRegistryError(
                f"candidate {candidate_id} missing fields do not match dataset"
            )
        if "open_interest" in required_fields:
            raise StrategyCandidateRegistryError(
                f"candidate {candidate_id} cannot depend on open interest"
            )

        status = candidate["status"]
        if status == _READY:
            ready += 1
            if missing_fields or requirements["availability"] != "available":
                raise StrategyCandidateRegistryError(
                    f"ready candidate {candidate_id} has missing data"
                )
            if candidate["historical_evaluation_allowed"] is not True:
                raise StrategyCandidateRegistryError(
                    f"ready candidate {candidate_id} cannot be evaluated"
                )
        elif status == _BLOCKED:
            blocked += 1
            if not missing_fields or requirements["availability"] != "missing":
                raise StrategyCandidateRegistryError(
                    f"blocked candidate {candidate_id} has no data blocker"
                )
            if candidate["historical_evaluation_allowed"] is not False:
                raise StrategyCandidateRegistryError(
                    f"blocked candidate {candidate_id} allows evaluation"
                )
        else:
            raise StrategyCandidateRegistryError(
                f"candidate {candidate_id} status is invalid"
            )

        if candidate["evaluation_state"] != "not_started":
            raise StrategyCandidateRegistryError(
                f"candidate {candidate_id} was already evaluated"
            )
        if not all(
            isinstance(candidate[key], Mapping) and candidate[key]
            for key in (
                "hypothesis",
                "event_contract",
                "exit_contract",
                "selection_contract",
            )
        ):
            raise StrategyCandidateRegistryError(
                f"candidate {candidate_id} contract is incomplete"
            )
        expected_sha = compute_candidate_profile_sha256(candidate)
        if str(candidate["profile_sha256"]).lower() != expected_sha:
            raise StrategyCandidateRegistryError(
                f"candidate {candidate_id} profile SHA-256 mismatch"
            )
        if (
            candidate["runtime_actor_created"] is not False
            or candidate["paper_allowed"] is not False
            or candidate["live_allowed"] is not False
        ):
            raise StrategyCandidateRegistryError(
                f"candidate {candidate_id} has runtime authority"
            )
        _validate_no_authority(candidate, f"candidate {candidate_id}")

    if ready != 2 or blocked != 1:
        raise StrategyCandidateRegistryError(
            "P2 registry must have two ready and one data-blocked candidate"
        )
    return row


def _validate_common_evaluation(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise StrategyCandidateRegistryError("common evaluation is invalid")
    required = {
        "signal_timing",
        "splits",
        "purge_hours",
        "costs",
        "portfolio",
        "gates",
        "retuning_policy",
    }
    if set(value) != required:
        raise StrategyCandidateRegistryError("common evaluation keys are invalid")
    if value["signal_timing"] != "closed_bar_signal_next_bar_open_fill":
        raise StrategyCandidateRegistryError("candidate signal timing is invalid")
    splits = value["splits"]
    if not isinstance(splits, list) or [item.get("name") for item in splits] != [
        "development",
        "validation",
        "oos",
        "sanity",
    ]:
        raise StrategyCandidateRegistryError("candidate splits are invalid")
    previous_end: datetime | None = None
    for split in splits:
        start = _parse_datetime(split.get("start_at"), "split start_at")
        end = _parse_datetime(split.get("end_at"), "split end_at")
        if end <= start or (previous_end is not None and start <= previous_end):
            raise StrategyCandidateRegistryError(
                "candidate splits overlap or are unordered"
            )
        previous_end = end
    if int(value["purge_hours"]) < 48:
        raise StrategyCandidateRegistryError("candidate purge is too short")
    costs = value["costs"]
    if not isinstance(costs, Mapping):
        raise StrategyCandidateRegistryError("candidate costs are invalid")
    required_move = (
        float(costs.get("round_trip_fee_bps", 0.0))
        + float(costs.get("slippage_bps", 0.0))
        + float(costs.get("safety_buffer_bps", 0.0))
    )
    if required_move <= 0.0 or float(costs.get("cost_stress_multiplier", 0.0)) < 1.0:
        raise StrategyCandidateRegistryError("candidate costs are not conservative")
    gates = value["gates"]
    if (
        int(gates.get("min_total_closed_trades", 0)) < 50
        or int(gates.get("min_window_fills", 0)) < 20
        or int(gates.get("min_window_closed_trades", 0)) < 10
        or float(gates.get("min_costed_expectancy_bps", -1.0)) < 0.0
        or float(gates.get("min_lcb_95_bps", -1.0)) < 0.0
        or gates.get("direction_collapse_hard_fail") is not True
        or gates.get("regime_collapse_hard_fail") is not True
        or gates.get("root_collapse_hard_fail") is not True
        or gates.get("require_baseline_improvement") is not True
    ):
        raise StrategyCandidateRegistryError("candidate gates are too weak")
    if value["retuning_policy"] != "new_candidate_id_required_after_any_oos_change":
        raise StrategyCandidateRegistryError("candidate retuning policy is invalid")


def _validate_no_authority(row: Mapping[str, Any], label: str) -> None:
    if row["orders_enabled"] is not False or row["promotion_authority"] is not False:
        raise StrategyCandidateRegistryError(f"{label} can authorize trading")


def _parse_datetime(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise StrategyCandidateRegistryError(f"{label} is invalid") from exc
    if parsed.tzinfo is None:
        raise StrategyCandidateRegistryError(f"{label} must be timezone-aware")
    return parsed


def _identifier(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", text):
        raise StrategyCandidateRegistryError(f"{label} is invalid")
    return text


def _relative_path(value: Any, label: str) -> None:
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise StrategyCandidateRegistryError(f"{label} is invalid")


def _sha256(value: Any, label: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise StrategyCandidateRegistryError(f"{label} is invalid")
    return text


def _string_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise StrategyCandidateRegistryError(f"{label} is invalid")
    rows = tuple(str(item).strip() for item in value)
    if any(not item for item in rows) or len(set(rows)) != len(rows):
        raise StrategyCandidateRegistryError(f"{label} is invalid")
    return rows


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise StrategyCandidateRegistryError(
            f"{label} keys mismatch; missing={missing}, extra={extra}"
        )


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise StrategyCandidateRegistryError(
            "strategy candidate registry must be finite JSON"
        ) from exc
