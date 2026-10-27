from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


DAY_MS = 86_400_000


@dataclass(frozen=True)
class TemporalValidationSplit:
    inner_start: int
    inner_end: int
    validation_start: int
    validation_end: int
    audit_start: int
    audit_end: int

    def to_mapping(self) -> dict[str, int]:
        return {
            "inner_start": self.inner_start,
            "inner_end": self.inner_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "audit_start": self.audit_start,
            "audit_end": self.audit_end,
        }


def build_temporal_validation_split(
    *,
    train_start: int,
    train_end: int,
    timeframe_ms: int,
    config: Mapping[str, object],
) -> TemporalValidationSplit:
    purge_ms = int(config["boundary_purge_bars"]) * int(timeframe_ms)
    validation_ms = int(config["validation_days"]) * DAY_MS
    audit_ms = int(config["audit_days"]) * DAY_MS
    audit_end = int(train_end)
    audit_start = audit_end - audit_ms
    validation_end = audit_start - purge_ms
    validation_start = validation_end - validation_ms
    inner_end = validation_start - purge_ms
    split = TemporalValidationSplit(
        inner_start=int(train_start),
        inner_end=inner_end,
        validation_start=validation_start,
        validation_end=validation_end,
        audit_start=audit_start,
        audit_end=audit_end,
    )
    if not (
        split.inner_start < split.inner_end < split.validation_start
        < split.validation_end < split.audit_start < split.audit_end
    ):
        raise ValueError("temporal validation split is not chronological")
    if split.validation_start - split.inner_end != purge_ms:
        raise ValueError("inner/validation purge mismatch")
    if split.audit_start - split.validation_end != purge_ms:
        raise ValueError("validation/audit purge mismatch")
    return split
