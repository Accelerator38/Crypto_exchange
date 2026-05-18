"""Allocation actionability diagnostics derived from Panteon trading logs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_FIELD_RE = re.compile(r"\b([A-Za-z_]+)=([^\s]+)")


@dataclass(frozen=True)
class LeaderActionability:
    label: str
    bars: int = 0
    raw_zero_bars: int = 0
    filled_zero_bars: int = 0
    raw_signals: int = 0
    signals: int = 0
    filled: int = 0

    @property
    def raw_zero_share_pct(self) -> float:
        return _share(self.raw_zero_bars, self.bars)

    @property
    def filled_zero_share_pct(self) -> float:
        return _share(self.filled_zero_bars, self.bars)


@dataclass(frozen=True)
class AllocationDiagnostics:
    bars: int
    no_trade_bars: int
    raw_zero_bars: int
    filled_zero_bars: int
    by_leader: dict[str, LeaderActionability] = field(default_factory=dict)

    @property
    def no_trade_share_pct(self) -> float:
        return _share(self.no_trade_bars, self.bars)

    @property
    def raw_zero_share_pct(self) -> float:
        return _share(self.raw_zero_bars, self.bars)

    @property
    def filled_zero_share_pct(self) -> float:
        return _share(self.filled_zero_bars, self.bars)


def analyze_trading_log(path: str | Path) -> AllocationDiagnostics:
    leader_rows: dict[str, dict[str, int]] = {}
    bars = no_trade = raw_zero = filled_zero = 0
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if " bar=" not in line:
            continue
        fields = dict(_FIELD_RE.findall(line))
        leader = fields.get("leader")
        if not leader:
            continue
        raw = _int(fields.get("raw_signals"))
        signals = _int(fields.get("signals"))
        filled = _int(fields.get("filled"))
        bars += 1
        no_trade += int(leader == "NoTrade")
        raw_zero += int(raw == 0)
        filled_zero += int(filled == 0)
        row = leader_rows.setdefault(
            leader,
            {
                "bars": 0,
                "raw_zero_bars": 0,
                "filled_zero_bars": 0,
                "raw_signals": 0,
                "signals": 0,
                "filled": 0,
            },
        )
        row["bars"] += 1
        row["raw_zero_bars"] += int(raw == 0)
        row["filled_zero_bars"] += int(filled == 0)
        row["raw_signals"] += raw
        row["signals"] += signals
        row["filled"] += filled
    return AllocationDiagnostics(
        bars=bars,
        no_trade_bars=no_trade,
        raw_zero_bars=raw_zero,
        filled_zero_bars=filled_zero,
        by_leader={
            label: LeaderActionability(label=label, **values)
            for label, values in leader_rows.items()
        },
    )


def allocation_diagnostics_to_dict(report: AllocationDiagnostics) -> dict[str, Any]:
    return {
        "bars": int(report.bars),
        "no_trade_bars": int(report.no_trade_bars),
        "no_trade_share_pct": report.no_trade_share_pct,
        "raw_zero_bars": int(report.raw_zero_bars),
        "raw_zero_share_pct": report.raw_zero_share_pct,
        "filled_zero_bars": int(report.filled_zero_bars),
        "filled_zero_share_pct": report.filled_zero_share_pct,
        "leaders": {
            label: {
                "bars": int(row.bars),
                "bar_share_pct": _share(row.bars, report.bars),
                "raw_zero_bars": int(row.raw_zero_bars),
                "raw_zero_share_pct": row.raw_zero_share_pct,
                "filled_zero_bars": int(row.filled_zero_bars),
                "filled_zero_share_pct": row.filled_zero_share_pct,
                "raw_signals": int(row.raw_signals),
                "signals": int(row.signals),
                "filled": int(row.filled),
            }
            for label, row in sorted(
                report.by_leader.items(),
                key=lambda item: (-item[1].bars, item[0]),
            )
        },
    }


def write_allocation_diagnostics(
    output_dir: str | Path,
    report: AllocationDiagnostics,
) -> Path:
    path = Path(output_dir) / "allocation_diagnostics.json"
    path.write_text(
        json.dumps(allocation_diagnostics_to_dict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _share(part: int, total: int) -> float:
    return 100.0 * float(part) / float(total) if total else 0.0


def _int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
