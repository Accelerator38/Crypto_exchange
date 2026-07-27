from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


EXPERIMENT_REGISTRY_SCHEMA_VERSION = "panteon.strategy_experiment_registry.v1"
TERMINAL_STATUS = "terminal_rejected"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FAMILY_KEYS = frozenset(
    {
        "family_id",
        "status",
        "reason",
        "retry_policy",
        "forbidden_variants",
        "continuation_allowed",
        "profile_registration_allowed",
        "orders_enabled",
        "promotion_authority",
    }
)
_EXPERIMENT_KEYS = frozenset(
    {
        "experiment_id",
        "family_id",
        "profile_id",
        "status",
        "verdict_reason",
        "retry_policy",
        "metrics",
        "artifacts",
        "continuation_allowed",
        "profile_registration_allowed",
        "runtime_actor_created",
        "orders_enabled",
        "promotion_authority",
    }
)
_ARTIFACT_KEYS = frozenset({"kind", "path", "sha256"})


class ExperimentRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class StrategyExperimentRegistry:
    path: Path
    payload: dict[str, Any]

    @classmethod
    def from_json(cls, path: str | Path) -> "StrategyExperimentRegistry":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentRegistryError(
                f"cannot load experiment registry: {source}"
            ) from exc
        return cls(path=source.resolve(), payload=validate_experiment_registry(payload))

    @property
    def operational_candidate_id(self) -> str | None:
        value = self.payload["operational_candidate_id"]
        return str(value) if value is not None else None

    def experiment(self, experiment_id: str) -> dict[str, Any]:
        target = str(experiment_id)
        for row in self.payload["experiments"]:
            if row["experiment_id"] == target:
                return dict(row)
        raise ExperimentRegistryError(f"unknown experiment: {target}")

    def family_is_terminal(self, family_id: str) -> bool:
        target = str(family_id)
        return any(
            row["family_id"] == target and row["status"] == TERMINAL_STATUS
            for row in self.payload["family_verdicts"]
        )


def validate_experiment_registry(payload: Mapping[str, Any]) -> dict[str, Any]:
    row = _json_copy(payload)
    expected_keys = frozenset(
        {
            "schema_version",
            "updated_at",
            "exchange",
            "operational_candidate_id",
            "orders_enabled",
            "promotion_authority",
            "family_verdicts",
            "experiments",
        }
    )
    _exact_keys(row, expected_keys, "experiment registry")
    if row["schema_version"] != EXPERIMENT_REGISTRY_SCHEMA_VERSION:
        raise ExperimentRegistryError("unsupported experiment registry schema")
    if row["exchange"] != "BITGET":
        raise ExperimentRegistryError("experiment registry exchange must be BITGET")
    if row["orders_enabled"] is not False or row["promotion_authority"] is not False:
        raise ExperimentRegistryError("experiment registry cannot authorize trading")

    family_rows = row["family_verdicts"]
    experiment_rows = row["experiments"]
    if not isinstance(family_rows, list) or not family_rows:
        raise ExperimentRegistryError("experiment registry has no family verdicts")
    if not isinstance(experiment_rows, list) or not experiment_rows:
        raise ExperimentRegistryError("experiment registry has no experiments")

    family_ids: set[str] = set()
    for family in family_rows:
        if not isinstance(family, Mapping):
            raise ExperimentRegistryError("family verdict must be an object")
        _exact_keys(family, _FAMILY_KEYS, "family verdict")
        family_id = _identifier(family["family_id"], "family ID")
        if family_id in family_ids:
            raise ExperimentRegistryError(f"duplicate family ID: {family_id}")
        family_ids.add(family_id)
        if family["status"] != TERMINAL_STATUS:
            raise ExperimentRegistryError("registered family verdict must be terminal")
        if family["retry_policy"] not in {
            "forbidden",
            "materially_different_event_contract_only",
        }:
            raise ExperimentRegistryError("family retry policy is invalid")
        forbidden = family["forbidden_variants"]
        if not isinstance(forbidden, list) or not forbidden:
            raise ExperimentRegistryError("family verdict must name forbidden variants")
        _validate_safety_flags(family, "family verdict")

    experiment_ids: set[str] = set()
    for experiment in experiment_rows:
        if not isinstance(experiment, Mapping):
            raise ExperimentRegistryError("experiment must be an object")
        _exact_keys(experiment, _EXPERIMENT_KEYS, "experiment")
        experiment_id = _identifier(experiment["experiment_id"], "experiment ID")
        if experiment_id in experiment_ids:
            raise ExperimentRegistryError(
                f"duplicate experiment ID: {experiment_id}"
            )
        experiment_ids.add(experiment_id)
        if experiment["family_id"] not in family_ids:
            raise ExperimentRegistryError(
                f"experiment {experiment_id} references unknown family"
            )
        if experiment["status"] != TERMINAL_STATUS:
            raise ExperimentRegistryError(
                f"experiment {experiment_id} is not terminal"
            )
        if experiment["retry_policy"] not in {
            "forbidden",
            "materially_different_event_contract_only",
        }:
            raise ExperimentRegistryError(
                f"experiment {experiment_id} retry policy is invalid"
            )
        _validate_safety_flags(experiment, f"experiment {experiment_id}")
        if experiment["runtime_actor_created"] is not False:
            raise ExperimentRegistryError(
                f"experiment {experiment_id} cannot create a runtime actor"
            )
        if not isinstance(experiment["metrics"], Mapping):
            raise ExperimentRegistryError(
                f"experiment {experiment_id} metrics are invalid"
            )
        artifacts = experiment["artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            raise ExperimentRegistryError(
                f"experiment {experiment_id} has no evidence artifacts"
            )
        for artifact in artifacts:
            _validate_artifact(artifact, experiment_id)

    candidate_id = row["operational_candidate_id"]
    if candidate_id is not None:
        if candidate_id not in experiment_ids:
            raise ExperimentRegistryError("operational candidate is unknown")
        candidate = next(
            item for item in experiment_rows if item["experiment_id"] == candidate_id
        )
        if candidate["status"] == TERMINAL_STATUS:
            raise ExperimentRegistryError(
                "terminal experiment cannot be an operational candidate"
            )
    return row


def _validate_safety_flags(row: Mapping[str, Any], label: str) -> None:
    if (
        row["continuation_allowed"] is not False
        or row["profile_registration_allowed"] is not False
        or row["orders_enabled"] is not False
        or row["promotion_authority"] is not False
    ):
        raise ExperimentRegistryError(f"{label} safety flags are invalid")


def _validate_artifact(artifact: Any, experiment_id: str) -> None:
    if not isinstance(artifact, Mapping):
        raise ExperimentRegistryError(
            f"experiment {experiment_id} artifact must be an object"
        )
    _exact_keys(artifact, _ARTIFACT_KEYS, "experiment artifact")
    path = PurePosixPath(str(artifact["path"]))
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise ExperimentRegistryError(
            f"experiment {experiment_id} artifact path is invalid"
        )
    sha256 = str(artifact["sha256"]).lower()
    if not _SHA256_RE.fullmatch(sha256):
        raise ExperimentRegistryError(
            f"experiment {experiment_id} artifact SHA-256 is invalid"
        )


def _identifier(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", text):
        raise ExperimentRegistryError(f"{label} is invalid")
    return text


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise ExperimentRegistryError(
            f"{label} keys mismatch; missing={missing}, extra={extra}"
        )


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ExperimentRegistryError(
            "experiment registry must be finite JSON"
        ) from exc


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
