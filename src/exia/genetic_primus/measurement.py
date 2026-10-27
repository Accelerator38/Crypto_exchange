from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MeasurementProtocol:
    source_path: Path
    payload: Mapping[str, Any]
    sha256: str

    @property
    def cutoff_ms(self) -> int:
        return int(pd.Timestamp(self.payload["training_cutoff"]).timestamp() * 1000)

    @property
    def purge_bars(self) -> int:
        return int(self.payload["label_purge_bars"])

    @property
    def purge_cutoff_ms(self) -> int:
        bar_ms = int(self.payload["timeframe_minutes"]) * 60 * 1000
        return self.cutoff_ms - self.purge_bars * bar_ms

    @property
    def audit_top_k(self) -> int:
        return int(self.payload["audit_bootstrap"]["top_k"])


def load_measurement_protocol(path: Path) -> MeasurementProtocol:
    source = path.resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != "exia.genetic_primus.measurement.3d/1":
        raise ValueError("unexpected Genetic Primus measurement schema")
    if payload.get("protocol_id") != "genetic_primus_3d_measurement_v1":
        raise ValueError("unexpected Genetic Primus measurement protocol id")
    if int(payload.get("timeframe_minutes", 0)) != 240:
        raise ValueError("3d measurement requires 4h bars")
    if int(payload.get("deployment_horizon_days", 0)) != 3:
        raise ValueError("deployment horizon must be exactly 3 days")
    if int(payload.get("label_purge_bars", 0)) != 7:
        raise ValueError("1/3/6 forward labels require a seven-bar purge")
    selection = payload.get("selection_bootstrap", {})
    audit = payload.get("audit_bootstrap", {})
    if int(selection.get("seed", 0)) == int(audit.get("seed", 0)):
        raise ValueError("selection and audit bootstrap seeds must be independent")
    if int(selection.get("samples", 0)) < 100 or int(audit.get("samples", 0)) < 1000:
        raise ValueError("bootstrap banks are undersized")
    if int(selection.get("block_days", 0)) != 7 or int(audit.get("block_days", 0)) != 7:
        raise ValueError("measurement bootstrap block must be seven days")
    if int(audit.get("top_k", 0)) < 2:
        raise ValueError("audit must cover multiple finalists")
    safety = payload.get("safety", {})
    expected = {"paper_allowed", "live_allowed", "orders_enabled", "promotion_authority"}
    if set(safety) != expected or any(safety[key] is not False for key in expected):
        raise PermissionError("measurement protocol must remain fail-closed")
    return MeasurementProtocol(source, payload, hashlib.sha256(raw).hexdigest())


def label_purge_mask(timestamps: pd.Series, protocol: MeasurementProtocol) -> pd.Series:
    values = pd.to_numeric(timestamps, errors="raise").astype("int64")
    return values < protocol.purge_cutoff_ms


@dataclass(frozen=True)
class BootstrapBank:
    bank_id: str
    seed: int
    samples: int
    block_days: int
    cutoff_ms: int
    folds_sha256: str
    scope_timestamps: Mapping[str, np.ndarray]
    scope_indices: Mapping[str, np.ndarray]
    content_sha256: str

    def indices_for(self, scope: str, timestamps: np.ndarray) -> np.ndarray:
        expected = self.scope_timestamps.get(scope)
        if expected is None:
            raise KeyError(f"bootstrap bank has no scope: {scope}")
        actual = np.asarray(timestamps, dtype=np.int64)
        if not np.array_equal(expected, actual):
            raise ValueError(f"bootstrap scope timestamps mismatch: {scope}")
        return self.scope_indices[scope]

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": "exia.genetic_primus.bootstrap_bank/1",
            "bank_id": self.bank_id,
            "seed": self.seed,
            "samples": self.samples,
            "block_days": self.block_days,
            "cutoff_ms": self.cutoff_ms,
            "folds_sha256": self.folds_sha256,
            "scope_lengths": {
                scope: int(len(values)) for scope, values in sorted(self.scope_timestamps.items())
            },
            "content_sha256": self.content_sha256,
        }


def folds_sha256(folds: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256([dict(item) for item in folds])


def _resample_indices(*, length: int, samples: int, block_days: int, seed: int) -> np.ndarray:
    if length < block_days:
        raise ValueError("bootstrap scope is shorter than one block")
    rng = np.random.default_rng(seed)
    starts = np.arange(0, length - block_days + 1, dtype=np.int64)
    blocks_needed = int(np.ceil(length / block_days))
    selected = rng.choice(starts, size=(samples, blocks_needed), replace=True)
    offsets = np.arange(block_days, dtype=np.int64)[None, None, :]
    return (selected[:, :, None] + offsets).reshape(samples, -1)[:, :length]


def _content_sha(
    metadata: Mapping[str, Any],
    timestamps: Mapping[str, np.ndarray],
    indices: Mapping[str, np.ndarray],
) -> str:
    digest = hashlib.sha256(canonical_sha256(metadata).encode("ascii"))
    for scope in sorted(timestamps):
        digest.update(scope.encode("utf-8"))
        digest.update(np.asarray(timestamps[scope], dtype="<i8").tobytes())
        digest.update(np.asarray(indices[scope], dtype="<i8").tobytes())
    return digest.hexdigest()


def build_bootstrap_bank(
    frame: pd.DataFrame,
    *,
    start_timestamp: int,
    cutoff_timestamp: int,
    folds: Sequence[Mapping[str, Any]],
    bank_id: str,
    seed: int,
    samples: int,
    block_days: int,
) -> BootstrapBank:
    if cutoff_timestamp <= start_timestamp:
        raise ValueError("measurement cutoff must follow the training start")
    local = frame.loc[
        frame["timestamp"].ge(start_timestamp) & frame["timestamp"].lt(cutoff_timestamp),
        ["timestamp", "symbol"],
    ].copy()
    if local.empty:
        raise ValueError("cannot build a bootstrap bank from an empty frame")
    next_timestamps: list[np.ndarray] = []
    for _, group in local.groupby("symbol", sort=True, observed=True):
        values = group.sort_values("timestamp", kind="stable")["timestamp"].to_numpy(np.int64)
        if len(values) > 1:
            next_timestamps.append(values[1:])
    if not next_timestamps:
        raise ValueError("bootstrap bank has no executable timestamps")
    executable = np.unique(np.concatenate(next_timestamps))
    days = (
        pd.to_datetime(executable, unit="ms", utc=True)
        .floor("1D")
        .map(lambda value: int(value.timestamp() * 1000))
        .to_numpy(np.int64)
    )
    days = np.unique(days)
    scope_timestamps: dict[str, np.ndarray] = {"overall": days}
    for fold in folds:
        fold_start = int(pd.Timestamp(fold["start"]).timestamp() * 1000)
        fold_end = int(pd.Timestamp(fold["end"]).timestamp() * 1000)
        scope_timestamps[str(fold["name"])] = days[(days >= fold_start) & (days < fold_end)]
    scope_indices: dict[str, np.ndarray] = {}
    for ordinal, scope in enumerate(sorted(scope_timestamps)):
        scope_indices[scope] = _resample_indices(
            length=len(scope_timestamps[scope]),
            samples=samples,
            block_days=block_days,
            seed=int(seed) + ordinal * 1_000_003,
        )
    fold_sha = folds_sha256(folds)
    base = {
        "bank_id": bank_id,
        "seed": int(seed),
        "samples": int(samples),
        "block_days": int(block_days),
        "cutoff_ms": int(cutoff_timestamp),
        "folds_sha256": fold_sha,
    }
    content_sha = _content_sha(base, scope_timestamps, scope_indices)
    return BootstrapBank(
        bank_id=bank_id,
        seed=int(seed),
        samples=int(samples),
        block_days=int(block_days),
        cutoff_ms=int(cutoff_timestamp),
        folds_sha256=fold_sha,
        scope_timestamps=scope_timestamps,
        scope_indices=scope_indices,
        content_sha256=content_sha,
    )


def save_bootstrap_bank(path: Path, bank: BootstrapBank) -> None:
    arrays: dict[str, np.ndarray] = {}
    for ordinal, scope in enumerate(sorted(bank.scope_timestamps)):
        arrays[f"timestamps_{ordinal}"] = bank.scope_timestamps[scope]
        arrays[f"indices_{ordinal}"] = bank.scope_indices[scope]
    arrays["metadata_utf8"] = np.frombuffer(
        json.dumps(bank.metadata(), ensure_ascii=False, sort_keys=True).encode("utf-8"),
        dtype=np.uint8,
    )
    arrays["scope_names_utf8"] = np.frombuffer(
        json.dumps(sorted(bank.scope_timestamps)).encode("utf-8"), dtype=np.uint8
    )
    np.savez_compressed(path, **arrays)
