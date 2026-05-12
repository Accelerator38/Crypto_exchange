"""Migration: v1 persisted state → v2 PerformanceMemory snapshot.

v1 хранил _regime_memory как `Panteon._regime_memory[regime][label]`
с полями ema_score, samples, wins, losses, pnl_pct и т.п.

В v2 PerformanceMemory имеет другую структуру (per-(label, regime)
LabelRegimeState с returns[], equity, etc). Полная миграция невозможна
без пересчёта по trades, но мы можем восстановить ключевые поля:
  closed_trades, wins, losses, pnl_pct (best-effort).

После миграции v2 продолжит копить статистику с того места где v1
остановился — холодного старта нет.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..domain.types import Regime
from ..memory import PerformanceMemory


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MigrationReport:
    """Что мигрировано."""

    source_path:    str
    n_labels:       int
    n_regime_pairs: int
    warnings:       List[str]


# ────────────────────────────────────────────────────────────────────


def migrate_v1_regime_memory(
    v1_memory_dict: dict,
    perf:           PerformanceMemory,
    *,
    source_path:    str = "",
) -> MigrationReport:
    """Импортирует v1 _regime_memory структуру в v2 PerformanceMemory.

    v1_memory_dict ожидается формы:
        {
            "bullish": {
                "FundingArb": {"samples": 12, "wins": 3, "losses": 9,
                                "pnl_pct": -2.5, "ema_score": -0.5, ...},
                "LiveAfterShock": {...},
                ...
            },
            "bearish": {...}, ...
        }

    Best-effort: v2 будет иметь приближённую картину. Уточнение —
    через replay реальных trades после миграции.
    """
    warnings: List[str] = []
    n_labels = 0
    n_pairs = 0

    if not isinstance(v1_memory_dict, dict):
        warnings.append("v1_memory_dict is not a dict; nothing to migrate")
        return MigrationReport(source_path, 0, 0, warnings)

    snapshot = perf.snapshot()
    state_dict = snapshot.get("state", {})

    seen_labels = set()
    for regime_str, by_label in v1_memory_dict.items():
        if not isinstance(by_label, dict):
            warnings.append(f"regime {regime_str!r}: skipped (not a dict)")
            continue
        try:
            regime = Regime.from_string(str(regime_str))
        except Exception:
            warnings.append(f"unknown regime {regime_str!r}; skipping")
            continue

        for label, raw in by_label.items():
            if not isinstance(raw, dict):
                continue
            label = str(label).replace("V_", "")
            seen_labels.add(label)
            n_pairs += 1

            # Извлекаем поля best-effort
            samples = int(raw.get("samples") or raw.get("closed_trades") or 0)
            wins = int(raw.get("wins") or 0)
            losses = int(raw.get("losses") or 0)
            pnl_pct = float(raw.get("pnl_pct") or raw.get("avg_pnl") or 0.0)

            key = f"{label}|{regime.label}"
            state_dict[key] = {
                "closed_trades": samples,
                "entries":       samples,   # best-effort
                "signals":       samples,
                "wins":          wins,
                "losses":        max(losses, 0),
                "pnl_pct":       pnl_pct,
                "returns":       [],        # без истории returns Sharpe не воссоздать
                "equity":        1.0 + pnl_pct / 100.0,
                "peak":          max(1.0, 1.0 + pnl_pct / 100.0),
                "max_dd_pct":    float(raw.get("max_dd_pct") or 0.0),
            }

    snapshot["state"] = state_dict
    perf.restore(snapshot)
    n_labels = len(seen_labels)

    if not n_labels:
        warnings.append("No labels migrated — v1_memory_dict was empty/invalid")

    return MigrationReport(
        source_path=source_path,
        n_labels=n_labels,
        n_regime_pairs=n_pairs,
        warnings=warnings,
    )


def migrate_from_v1_memory_file(
    path: str,
    perf: PerformanceMemory,
) -> MigrationReport:
    """Загрузка из v1 memory.json и применение."""
    if not os.path.exists(path):
        return MigrationReport(
            source_path=path, n_labels=0, n_regime_pairs=0,
            warnings=[f"file not found: {path}"],
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return MigrationReport(
            source_path=path, n_labels=0, n_regime_pairs=0,
            warnings=[f"failed to read JSON: {exc}"],
        )

    # v1 хранит _regime_memory под разными ключами в разных версиях
    if "_regime_memory" in data:
        regime_memory = data["_regime_memory"]
    elif "regime_memory" in data:
        regime_memory = data["regime_memory"]
    elif isinstance(data, dict) and any(
        isinstance(v, dict) for v in data.values()
    ):
        regime_memory = data
    else:
        return MigrationReport(
            source_path=path, n_labels=0, n_regime_pairs=0,
            warnings=["could not find _regime_memory key in JSON"],
        )

    return migrate_v1_regime_memory(regime_memory, perf, source_path=path)


# ────────────────────────────────────────────────────────────────────
# Persistence v2 → disk
# ────────────────────────────────────────────────────────────────────


def save_v2_snapshot(
    perf:        PerformanceMemory,
    path:        str,
) -> None:
    """Сохранить PerformanceMemory snapshot в JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(perf.snapshot(), f, indent=2)


def load_v2_snapshot(
    perf:        PerformanceMemory,
    path:        str,
) -> bool:
    """Загрузить PerformanceMemory snapshot из JSON. True если успешно."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            snap = json.load(f)
        perf.restore(snap)
        return True
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("load_v2_snapshot failed (%s): %s", path, exc)
        return False
