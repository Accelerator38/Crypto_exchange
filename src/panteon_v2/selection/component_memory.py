"""Causal prior-bar component statistics for Flash routing."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ComponentStat:
    actor_label: str
    symbol: str
    regime: str
    action: str
    bar: int
    closed_trades: int
    expectancy: float
    pnl_lcb: float | None = None


class ComponentMemory:
    def __init__(self, rows: Iterable[ComponentStat | Mapping[str, Any]] = ()) -> None:
        self._rows = tuple(_coerce_stat(row) for row in rows)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ComponentMemory":
        rows: list[ComponentStat] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, Mapping):
                    rows.append(_coerce_stat(payload))
        return cls(rows)

    def best_prior(
        self,
        actor_label: str,
        *,
        symbol: str,
        regime: str,
        action: str,
        bar: int,
        min_closed_trades: int = 0,
        min_expectancy: float | None = None,
    ) -> ComponentStat | None:
        label = _norm_text(actor_label)
        symbol_key = _norm_symbol(symbol)
        regime_key = _norm_text(regime)
        action_key = _norm_text(action)
        current_bar = int(bar)
        min_closed = max(0, int(min_closed_trades or 0))
        matches = [
            row
            for row in self._rows
            if _norm_text(row.actor_label) == label
            and _matches_symbol(row.symbol, symbol_key)
            and _matches_text(row.regime, regime_key)
            and _matches_text(row.action, action_key)
            and int(row.bar) < current_bar
            and int(row.closed_trades) >= min_closed
            and (
                min_expectancy is None
                or float(row.expectancy) > float(min_expectancy)
            )
        ]
        if not matches:
            return None
        return max(
            matches,
            key=lambda row: (
                _context_score(
                    row,
                    symbol_key=symbol_key,
                    regime_key=regime_key,
                    action_key=action_key,
                ),
                int(row.closed_trades),
                -1.0e18 if row.pnl_lcb is None else float(row.pnl_lcb),
                float(row.expectancy),
                int(row.bar),
            ),
        )


def _coerce_stat(row: ComponentStat | Mapping[str, Any]) -> ComponentStat:
    if isinstance(row, ComponentStat):
        return row
    closed = _int_value(row.get("closed_trades"), 0)
    expectancy = _float_value(
        row.get("expectancy", row.get("pnl_per_trade", row.get("pnl_per_trade_pct"))),
        0.0,
    )
    return ComponentStat(
        actor_label=str(row.get("actor_label", row.get("label", row.get("actor", ""))) or ""),
        symbol=str(row.get("symbol", "") or ""),
        regime=str(row.get("regime", "") or ""),
        action=str(row.get("action", "") or ""),
        bar=_int_value(row.get("bar"), 0),
        closed_trades=closed,
        expectancy=expectancy,
        pnl_lcb=_optional_float(row.get("pnl_lcb", row.get("pnl_per_trade_lcb"))),
    )


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _norm_symbol(value: Any) -> str:
    text = str(value or "").strip().upper().replace("-", "/").replace("_", "/")
    if text == "*":
        return "*"
    if "/" not in text and text.endswith("USDT"):
        return f"{text[:-4]}/USDT"
    return text


def _is_wildcard_text(value: Any) -> bool:
    return _norm_text(value) in {"", "*", "all", "any"}


def _matches_text(candidate: Any, requested: str) -> bool:
    return _is_wildcard_text(candidate) or _norm_text(candidate) == requested


def _matches_symbol(candidate: Any, requested: str) -> bool:
    return _is_wildcard_text(candidate) or _norm_symbol(candidate) == requested


def _context_score(
    row: ComponentStat,
    *,
    symbol_key: str,
    regime_key: str,
    action_key: str,
) -> int:
    score = 0
    if _norm_symbol(row.symbol) == symbol_key:
        score += 4
    if _norm_text(row.regime) == regime_key:
        score += 2
    if _norm_text(row.action) == action_key:
        score += 1
    return score


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _float_value(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
