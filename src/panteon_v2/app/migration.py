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
from typing import Dict, List, Mapping, Optional, Sequence

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


_V1_PLAYER_LABELS = {
    "V_Panteon_shadow": "DefaultEnsemble",
    "V_PanteonTrendResearch": "TrendResearch",
    "V_PanteonMeanRevResearch": "MeanRevResearch",
    "V_PanteonDefensiveResearch": "DefensiveResearch",
    "V_PlayerBomberman": "BombermanStrong",
}


def _normalize_v1_label(label: object) -> str:
    raw = str(label)
    mapped = _V1_PLAYER_LABELS.get(raw)
    if mapped:
        return mapped
    if raw.startswith("V_"):
        return raw[2:]
    return raw


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
            label = _normalize_v1_label(label)
            seen_labels.add(label)
            n_pairs += 1

            # Извлекаем поля best-effort
            samples = int(raw.get("samples") or raw.get("closed_trades") or 0)
            wins = int(raw.get("wins") or 0)
            losses = int(raw.get("losses") or 0)
            samples = max(samples, wins + losses)
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
    elif "player_regime_memory" in data:
        regime_memory = data["player_regime_memory"]
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


def migrate_from_v1_memory_files(
    paths: Sequence[str],
    perf: PerformanceMemory,
) -> MigrationReport:
    """Load and merge several v1 memory files into one PerformanceMemory."""
    warnings: List[str] = []
    n_pairs = 0
    source_paths: List[str] = []
    labels_before = set(perf.all_labels())

    for path in paths:
        report = migrate_from_v1_memory_file(path, perf)
        source_paths.append(report.source_path)
        n_pairs += report.n_regime_pairs
        warnings.extend(report.warnings)

    labels_after = set(perf.all_labels())
    n_labels = len(labels_after - labels_before)
    if n_pairs and n_labels == 0:
        n_labels = len(labels_after)
    return MigrationReport(
        source_path=";".join(source_paths),
        n_labels=n_labels,
        n_regime_pairs=n_pairs,
        warnings=warnings,
    )


# ────────────────────────────────────────────────────────────────────
# Persistence v2 → disk
# ────────────────────────────────────────────────────────────────────


def save_v2_snapshot(
    perf:        PerformanceMemory,
    path:        str,
    *,
    real_perf:   Optional[PerformanceMemory] = None,
    order_ledger = None,
    position_tracker = None,
    shadow_positions: Optional[Mapping[str, Sequence[Mapping[str, object]]]] = None,
) -> None:
    """Сохранить PerformanceMemory snapshot в JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    snapshot = perf.snapshot()
    if real_perf is not None:
        snapshot["_real_memory"] = real_perf.snapshot()
    if order_ledger is not None and hasattr(order_ledger, "snapshot"):
        snapshot["_order_ledger"] = order_ledger.snapshot()
    if position_tracker is not None and hasattr(position_tracker, "snapshot"):
        snapshot["_position_tracker"] = position_tracker.snapshot()
    if shadow_positions is not None:
        snapshot["_shadow_player_positions"] = _json_safe_shadow_positions(
            shadow_positions
        )
    # Атомарная запись: пишем во временный файл, fsync, затем os.replace.
    # Это предотвращает потерю всей памяти при креше/убийстве процесса
    # посреди json.dump (см. аудит C1). os.replace атомарен и на Windows,
    # и на POSIX. Перед заменой сохраняем предыдущий валидный файл в .bak.
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, ValueError):
            # fsync может быть недоступен на некоторых ФС/платформах — не фатально.
            pass
    if os.path.exists(path):
        try:
            os.replace(path, f"{path}.bak")
        except OSError:
            # Не блокируем сохранение, если .bak обновить не удалось.
            pass
    os.replace(tmp_path, path)


def load_v2_snapshot(
    perf:        PerformanceMemory,
    path:        str,
    *,
    real_perf:   Optional[PerformanceMemory] = None,
    order_ledger = None,
    position_tracker = None,
    shadow_positions_target: Optional[dict] = None,
) -> bool:
    """Загрузить PerformanceMemory snapshot из JSON. True если успешно.

    Если основной файл повреждён (например, креш во время старой не-атомарной
    записи), пытаемся восстановиться из ``<path>.bak``.
    """
    snap = _load_snapshot_json(path)
    if snap is None:
        return False
    try:
        perf.restore(snap)
        if real_perf is not None and isinstance(snap.get("_real_memory"), dict):
            real_perf.restore(snap["_real_memory"])
        if order_ledger is not None and hasattr(order_ledger, "restore"):
            order_ledger.restore(snap.get("_order_ledger") or {})
        if position_tracker is not None and hasattr(position_tracker, "restore"):
            position_tracker.restore(snap.get("_position_tracker") or {})
        if shadow_positions_target is not None:
            shadow_positions_target.clear()
            shadow_positions_target.update(
                _restore_shadow_positions(snap.get("_shadow_player_positions") or {})
            )
        return True
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("load_v2_snapshot failed (%s): %s", path, exc)
        return False


def _load_snapshot_json(path: str) -> Optional[dict]:
    """Прочитать snapshot JSON, при повреждении основного файла — из ``.bak``.

    Возвращает dict либо None, если ни основной, ни backup-файл недоступны/валидны.
    """
    candidates = []
    if os.path.exists(path):
        candidates.append(path)
    bak = f"{path}.bak"
    if os.path.exists(bak):
        candidates.append(bak)
    for candidate in candidates:
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                snap = json.load(f)
            if candidate != path:
                log.warning(
                    "load_v2_snapshot: primary %s unusable, recovered from backup %s",
                    path,
                    candidate,
                )
            return snap
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("load_v2_snapshot: cannot read %s: %s", candidate, exc)
            continue
    return None


def _json_safe_shadow_positions(
    positions: Mapping[str, Sequence[Mapping[str, object]]],
) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for label, payloads in dict(positions or {}).items():
        rows: List[dict] = []
        for payload in tuple(payloads or ()):
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
        out[str(label)] = rows
    return out


def _restore_shadow_positions(raw: object) -> Dict[str, tuple]:
    out: Dict[str, tuple] = {}
    if not isinstance(raw, dict):
        return out
    for label, payloads in raw.items():
        rows: List[dict] = []
        if isinstance(payloads, (list, tuple)):
            for payload in payloads:
                if isinstance(payload, dict):
                    rows.append(dict(payload))
        out[str(label)] = tuple(rows)
    return out
