from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")
MARKET_STATES = frozenset({"TREND_UP", "TREND_DOWN", "RANGE", "UNSAFE"})
CANDIDATE_STATUSES = frozenset(
    {"DEVELOPMENT", "LOCKED", "REJECTED", "PROMOTABLE"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _identifier(value: Any, *, label: str) -> str:
    text = str(value or "")
    if not IDENTIFIER_PATTERN.fullmatch(text):
        raise ValueError(f"{label} must match {IDENTIFIER_PATTERN.pattern}")
    return text


def _sha(value: Any, *, label: str) -> str:
    text = str(value or "")
    if not SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _git_revision(value: Any, *, label: str) -> str:
    text = str(value or "")
    if not GIT_REVISION_PATTERN.fullmatch(text):
        raise ValueError(f"{label} must be a 40-character Git revision")
    return text


def _positive_number(value: Any, *, label: str) -> float:
    number = float(value)
    if number <= 0:
        raise ValueError(f"{label} must be positive")
    return number


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    version: str
    status: str
    hypothesis: str
    strategy: Mapping[str, Any]
    taxonomy: Mapping[str, Any]
    dataset: Mapping[str, Any]
    splits: Mapping[str, str]
    allowed_market_states: tuple[str, ...]
    parameters: Mapping[str, Any]
    costs: Mapping[str, float]
    risk: Mapping[str, Any]
    baseline_id: str
    trial_family: str
    source: Mapping[str, Any]

    SCHEMA_VERSION = "exia.candidate_spec.v1"
    TOP_LEVEL_KEYS = {
        "schema_version",
        "candidate_id",
        "family",
        "version",
        "status",
        "hypothesis",
        "strategy",
        "taxonomy",
        "dataset",
        "splits",
        "allowed_market_states",
        "parameters",
        "costs",
        "risk",
        "baseline_id",
        "trial_family",
        "paper_allowed",
        "live_allowed",
        "orders_enabled",
        "promotion_authority",
    }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CandidateSpec:
        _exact_keys(payload, cls.TOP_LEVEL_KEYS, label="candidate")
        if payload["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("unsupported candidate schema_version")

        candidate_id = _identifier(payload["candidate_id"], label="candidate_id")
        family = _identifier(payload["family"], label="family")
        trial_family = _identifier(payload["trial_family"], label="trial_family")
        baseline_id = _identifier(payload["baseline_id"], label="baseline_id")
        version = str(payload["version"] or "")
        if not version:
            raise ValueError("version must be non-empty")
        status = str(payload["status"] or "")
        if status not in CANDIDATE_STATUSES:
            raise ValueError(f"invalid candidate status: {status}")
        hypothesis = str(payload["hypothesis"] or "").strip()
        if not hypothesis:
            raise ValueError("hypothesis must be non-empty")

        strategy = _mapping(payload["strategy"], label="strategy")
        _exact_keys(
            strategy,
            {"source", "class_name", "source_sha256"},
            label="strategy",
        )
        _sha(strategy["source_sha256"], label="strategy.source_sha256")
        if not str(strategy["source"] or "") or not str(
            strategy["class_name"] or ""
        ):
            raise ValueError("strategy source and class_name must be non-empty")

        taxonomy = _mapping(payload["taxonomy"], label="taxonomy")
        _exact_keys(
            taxonomy,
            {"version", "source", "source_sha256"},
            label="taxonomy",
        )
        _sha(taxonomy["source_sha256"], label="taxonomy.source_sha256")

        dataset = _mapping(payload["dataset"], label="dataset")
        _exact_keys(
            dataset,
            {
                "dataset_id",
                "dataset_sha256",
                "manifest",
                "timeframe",
                "symbols",
            },
            label="dataset",
        )
        _sha(dataset["dataset_sha256"], label="dataset.dataset_sha256")
        symbols = dataset["symbols"]
        if not isinstance(symbols, list) or not symbols or len(set(symbols)) != len(symbols):
            raise ValueError("dataset.symbols must be a non-empty unique list")
        if dataset["timeframe"] != "1h":
            raise ValueError("Exia v1 supports only the 1h timeframe")

        splits = _mapping(payload["splits"], label="splits")
        _exact_keys(
            splits,
            {"development", "validation", "oos", "sanity"},
            label="splits",
        )
        if any(not str(value or "") for value in splits.values()):
            raise ValueError("all split timeranges must be non-empty")

        states = payload["allowed_market_states"]
        if not isinstance(states, list) or not states:
            raise ValueError("allowed_market_states must be a non-empty list")
        if len(set(states)) != len(states) or not set(states) <= MARKET_STATES:
            raise ValueError("allowed_market_states contains invalid or duplicate values")

        parameters = _mapping(payload["parameters"], label="parameters")
        costs = _mapping(payload["costs"], label="costs")
        _exact_keys(
            costs,
            {"base_fee_per_fill", "stress_fee_per_fill"},
            label="costs",
        )
        base_fee = _positive_number(costs["base_fee_per_fill"], label="base fee")
        stress_fee = _positive_number(
            costs["stress_fee_per_fill"], label="stress fee"
        )
        if stress_fee < base_fee:
            raise ValueError("stress fee cannot be below base fee")

        risk = _mapping(payload["risk"], label="risk")
        _exact_keys(
            risk,
            {
                "max_open_trades",
                "stake_amount_usdt",
                "leverage",
                "daily_loss_limit_usdt",
                "expiry",
            },
            label="risk",
        )
        if int(risk["max_open_trades"]) != 1:
            raise ValueError("Exia v1 requires max_open_trades=1")
        if float(risk["leverage"]) != 1.0:
            raise ValueError("Exia v1 requires leverage=1")
        _positive_number(risk["stake_amount_usdt"], label="stake_amount_usdt")
        _positive_number(
            risk["daily_loss_limit_usdt"], label="daily_loss_limit_usdt"
        )
        if risk["expiry"] is not None and not str(risk["expiry"]).strip():
            raise ValueError("risk.expiry must be null or a non-empty timestamp")

        safety = (
            "paper_allowed",
            "live_allowed",
            "orders_enabled",
            "promotion_authority",
        )
        if any(payload[name] is not False for name in safety):
            raise ValueError("candidate research safety flags must all be false")

        return cls(
            candidate_id=candidate_id,
            family=family,
            version=version,
            status=status,
            hypothesis=hypothesis,
            strategy=dict(strategy),
            taxonomy=dict(taxonomy),
            dataset=dict(dataset),
            splits={str(key): str(value) for key, value in splits.items()},
            allowed_market_states=tuple(str(value) for value in states),
            parameters=dict(parameters),
            costs={
                "base_fee_per_fill": base_fee,
                "stress_fee_per_fill": stress_fee,
            },
            risk=dict(risk),
            baseline_id=baseline_id,
            trial_family=trial_family,
            source=dict(payload),
        )

    def verify_files(self, root: Path) -> None:
        for label, item in (("strategy", self.strategy), ("taxonomy", self.taxonomy)):
            path = root / str(item["source"])
            if not path.is_file():
                raise ValueError(f"{label} source does not exist: {path}")
            actual = sha256_file(path)
            if actual != item["source_sha256"]:
                raise ValueError(f"{label} source SHA mismatch")

        manifest_path = root / str(self.dataset["manifest"])
        if not manifest_path.is_file():
            raise ValueError(f"dataset manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_sha256") != self.dataset["dataset_sha256"]:
            raise ValueError("dataset SHA mismatch")


@dataclass(frozen=True)
class ExperimentManifest:
    experiment_id: str
    candidate_id: str
    candidate_spec_sha256: str
    strategy_sha256: str
    taxonomy_sha256: str
    dataset_sha256: str
    freqtrade_revision: str
    experiment_code_sha256: str
    runtime_config_sha256: str
    historical_config_sha256: str
    timeranges: Mapping[str, str]
    costs: Mapping[str, float]
    command: tuple[str, ...]

    SCHEMA_VERSION = "exia.experiment_manifest.v1"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ExperimentManifest:
        expected = {
            "schema_version",
            "experiment_id",
            "candidate_id",
            "candidate_spec_sha256",
            "strategy_sha256",
            "taxonomy_sha256",
            "dataset_sha256",
            "freqtrade_revision",
            "experiment_code_sha256",
            "runtime_config_sha256",
            "historical_config_sha256",
            "timeranges",
            "costs",
            "command",
            "orders_enabled",
            "promotion_authority",
        }
        _exact_keys(payload, expected, label="experiment")
        if payload["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("unsupported experiment schema_version")
        if payload["orders_enabled"] is not False or payload[
            "promotion_authority"
        ] is not False:
            raise ValueError("experiment cannot authorize orders or promotion")
        timeranges = _mapping(payload["timeranges"], label="timeranges")
        costs = _mapping(payload["costs"], label="costs")
        if not timeranges or any(not str(value or "") for value in timeranges.values()):
            raise ValueError("experiment timeranges must be non-empty")
        _exact_keys(costs, {"base", "stress"}, label="experiment costs")
        base_cost = _positive_number(costs["base"], label="experiment base cost")
        stress_cost = _positive_number(
            costs["stress"], label="experiment stress cost"
        )
        if stress_cost < base_cost:
            raise ValueError("experiment stress cost cannot be below base cost")
        command = payload["command"]
        if not isinstance(command, list) or not command:
            raise ValueError("experiment command must be a non-empty list")
        return cls(
            experiment_id=_identifier(payload["experiment_id"], label="experiment_id"),
            candidate_id=_identifier(payload["candidate_id"], label="candidate_id"),
            candidate_spec_sha256=_sha(
                payload["candidate_spec_sha256"], label="candidate_spec_sha256"
            ),
            strategy_sha256=_sha(payload["strategy_sha256"], label="strategy_sha256"),
            taxonomy_sha256=_sha(payload["taxonomy_sha256"], label="taxonomy_sha256"),
            dataset_sha256=_sha(payload["dataset_sha256"], label="dataset_sha256"),
            freqtrade_revision=_git_revision(
                payload["freqtrade_revision"], label="freqtrade_revision"
            ),
            experiment_code_sha256=_sha(
                payload["experiment_code_sha256"],
                label="experiment_code_sha256",
            ),
            runtime_config_sha256=_sha(
                payload["runtime_config_sha256"],
                label="runtime_config_sha256",
            ),
            historical_config_sha256=_sha(
                payload["historical_config_sha256"],
                label="historical_config_sha256",
            ),
            timeranges={str(key): str(value) for key, value in timeranges.items()},
            costs={"base": base_cost, "stress": stress_cost},
            command=tuple(str(value) for value in command),
        )


def load_candidate_spec(
    path: Path,
    *,
    root: Path | None = None,
    verify_files: bool = True,
) -> CandidateSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    spec = CandidateSpec.from_mapping(payload)
    if verify_files:
        spec.verify_files(root or path.resolve().parents[2])
    return spec
