"""Цветовая палитра дашбордов.

Каноничные значения, единые для всех панелей. Любой backend (matplotlib,
HTML, etc.) использует эти hex-коды.

КЛЮЧЕВОЕ ПРАВИЛО: карантинные участники ВСЕГДА серые. Не зелёные,
не красные. Это — единственный способ устранить визуальное противоречие
из v1 P28.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DashboardPalette:
    """Каноничные цвета. Frozen — никаких мутаций."""

    # ── Фоны ─────────────────────────────────────────────────────────
    bg:      str = "#0D1117"  # основной фон
    mid:     str = "#161B22"  # вторичный (карточки)
    grid:    str = "#21262D"  # сетка

    # ── Тексты ───────────────────────────────────────────────────────
    text:    str = "#E6EDF3"  # белый — основной текст
    muted:   str = "#8B949E"  # серый — приглушённый текст
    accent:  str = "#D29922"  # жёлтый — выделение/REAL

    # ── PnL индикаторы ───────────────────────────────────────────────
    positive: str = "#3FB950"  # зелёный — прибыль
    negative: str = "#F85149"  # красный — убыток
    neutral:  str = "#58A6FF"  # синий — нейтрально / inform
    quarantine: str = "#8B949E"  # серый — карантин (NEVER green/red!)

    def for_pnl(self, pnl: float, *, is_quarantined: bool) -> str:
        """Главная функция расцветки.

        Гарантирует: карантинный → серый, всегда. Это единственное
        место, где принимается это решение.
        """
        if is_quarantined:
            return self.quarantine
        if pnl > 0:
            return self.positive
        if pnl < 0:
            return self.negative
        return self.muted


# Дефолтная палитра — для использования по всему пакету.
DEFAULT_PALETTE = DashboardPalette()
