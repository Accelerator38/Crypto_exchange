"""VotingPolicy — стратегии агрегации сигналов агентов в финальный набор.

3 политики:
  • WeightedConsensus — основная: weighted vote, агрегируем по знаку.
  • StrongConsensus   — открываем только при единогласии всех агентов.
  • RiskParity        — веса ~ 1/vol; для дайверсификации вместо производительности.

Каждая политика — pure: на вход берёт votes + thresholds + регим, на выход
List[Action] для каждого sym. Без состояния, deterministic.

ThresholdProfile — числовые пороги открытия/закрытия. В v1 их было ~6 в
разных классах. В v2 — один immutable объект.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Protocol, Sequence

from ..domain.types import Action, MarketSnapshot, Regime


# ────────────────────────────────────────────────────────────────────
# Threshold profile
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ThresholdProfile:
    """Параметры порогов для voting policy.

    Заменяет в v1 разрозненные `OPEN_SINGLE_THRESHOLD`, `OPEN_MULTI_THRESHOLD`,
    `CLOSE_SINGLE_THRESHOLD`, `CLOSE_MULTI_THRESHOLD`, `CLOSE_STRONG_THRESHOLD`,
    `MIN_OPEN_SINGLE_THRESHOLD`, `MIN_OPEN_MULTI_THRESHOLD` — теперь один
    объект с явными именами.
    """

    open_single:        float = 0.34   # порог для открытия одним агентом
    open_multi:         float = 0.26   # порог при ≥2 голосующих
    open_floor:         float = 0.20   # абсолютный минимум
    close_single:       float = 0.30
    close_multi:        float = 0.22
    close_strong:       float = 0.40   # для close при сильном консенсусе

    # Регулировки от noise/regime — мультипликатор к основным порогам
    noisy_market_open_bonus:  float = 0.04
    noisy_market_close_bonus: float = 0.02

    def __post_init__(self) -> None:
        if self.open_floor > self.open_single:
            raise ValueError(
                f"open_floor ({self.open_floor}) > open_single ({self.open_single})"
            )
        if self.open_multi > self.open_single:
            # open_multi обычно ниже open_single (легче открыть при консенсусе)
            raise ValueError(
                f"open_multi ({self.open_multi}) > open_single ({self.open_single}); "
                "expected open_multi ≤ open_single"
            )


# ────────────────────────────────────────────────────────────────────
# Protocol
# ────────────────────────────────────────────────────────────────────


# Тип "votes по агенту": agent_label → {sym → Action}
AgentVotes = Mapping[str, Mapping[str, Action]]


class VotingPolicy(Protocol):
    """Интерфейс политики голосования.

    Возвращает map sym → Action — агрегированное решение по каждому символу.
    Результат deterministic при одном и том же входе.
    """

    label: str

    def aggregate(
        self,
        votes: AgentVotes,
        weights: Mapping[str, float],
        thresholds: ThresholdProfile,
        market: MarketSnapshot,
    ) -> Dict[str, Action]:
        ...


# ────────────────────────────────────────────────────────────────────
# WeightedConsensus
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WeightedConsensus:
    """Default-политика: считаем взвешенное «направленное» голосование.

    Алгоритм для каждого sym:
      • директивный score = Σ weight[a] * direction(action[a])
      • direction: long_open=+1, short_open=−1, close=0, hold=0
      • если |score| ≥ open_threshold → открываем full
      • если |score| ≥ open_floor    → открываем half
      • если есть голос за close (взвешенный ≥ close_threshold) → закрываем

    Параметры:
      • открытие требует чистого консенсуса (нет противоположных votes)
      • close — отдельный канал, требует взвешенный support
    """

    label: str = "WeightedConsensus"

    def aggregate(
        self,
        votes: AgentVotes,
        weights: Mapping[str, float],
        thresholds: ThresholdProfile,
        market: MarketSnapshot,
    ) -> Dict[str, Action]:
        out: Dict[str, Action] = {}
        if not votes:
            return out

        n_agents = len(votes)

        for sym in market.prices:
            score_long_open = 0.0
            score_short_open = 0.0
            score_close = 0.0
            agents_voting = 0
            for label, agent_votes in votes.items():
                weight = float(weights.get(label, 0.0))
                if weight <= 0:
                    continue
                action = agent_votes.get(sym, Action.HOLD)
                if action.is_long_open:
                    score_long_open += weight
                    agents_voting += 1
                elif action.is_short_open:
                    score_short_open += weight
                    agents_voting += 1
                elif action.is_close:
                    score_close += weight

            # Открытие
            open_thr = (
                thresholds.open_multi if agents_voting >= 2
                else thresholds.open_single
            )
            open_thr = max(thresholds.open_floor, open_thr)

            net = score_long_open - score_short_open
            if abs(net) >= open_thr:
                full = abs(net) >= max(open_thr, thresholds.open_single)
                if net > 0:
                    out[sym] = Action.FUT_LONG_FULL if full else Action.FUT_LONG_HALF
                else:
                    out[sym] = Action.FUT_SHORT_FULL if full else Action.FUT_SHORT_HALF
                continue

            # Если не открыли — может быть close
            close_thr = (
                thresholds.close_multi if n_agents >= 2
                else thresholds.close_single
            )
            if score_close >= close_thr:
                out[sym] = Action.FUT_CLOSE_ALL

        return out


# ────────────────────────────────────────────────────────────────────
# StrongConsensus
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StrongConsensus:
    """Открытие только при единогласии (как PlayerBomberman v1).

    Алгоритм:
      • для open: ВСЕ голосующие агенты с weight > 0 должны вернуть один
        и тот же long-open или short-open action
      • для close: ХОТЯ БЫ ОДИН close-голос
    """

    label: str = "StrongConsensus"

    def aggregate(
        self,
        votes: AgentVotes,
        weights: Mapping[str, float],
        thresholds: ThresholdProfile,
        market: MarketSnapshot,
    ) -> Dict[str, Action]:
        out: Dict[str, Action] = {}
        if not votes:
            return out

        # Активные агенты с положительным весом
        active_labels = [
            lbl for lbl in votes
            if float(weights.get(lbl, 0.0)) > 0
        ]
        if not active_labels:
            return out

        for sym in market.prices:
            actions = [votes[lbl].get(sym, Action.HOLD) for lbl in active_labels]

            # Close: любой голос за close
            if any(a.is_close for a in actions):
                out[sym] = Action.FUT_CLOSE_ALL
                continue

            # Open: все одного направления (single non-hold direction)
            opens = [a for a in actions if a.is_open]
            if not opens or len(opens) != len(active_labels):
                # Не все проголосовали за open или были holds — пропускаем
                continue

            # Все голосовали open. Проверяем что направления одинаковые
            if all(a.is_long_open for a in opens):
                # Используем самый "слабый" среди голосующих (half если
                # хоть один сказал half).
                full = all(a.is_fraction_full for a in opens)
                out[sym] = Action.FUT_LONG_FULL if full else Action.FUT_LONG_HALF
            elif all(a.is_short_open for a in opens):
                full = all(a.is_fraction_full for a in opens)
                out[sym] = Action.FUT_SHORT_FULL if full else Action.FUT_SHORT_HALF
            # Иначе — конфликт направлений → HOLD

        return out


# ────────────────────────────────────────────────────────────────────
# RiskParity
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RiskParity:
    """Risk-parity: weight = 1/vol(equity).

    На текущей фазе принимает уже подсчитанные ``volatilities`` (через
    конструктор), но имеет тот же интерфейс aggregate. В будущем можно
    расширить чтобы считать vol на лету по equity_curve.

    По умолчанию ведёт себя как WeightedConsensus с равными весами,
    что является фолбэком если volatilities пуст.
    """

    label: str = "RiskParity"
    volatilities: Mapping[str, float] = field(default_factory=dict)

    def aggregate(
        self,
        votes: AgentVotes,
        weights: Mapping[str, float],     # игнорируется, считаются заново
        thresholds: ThresholdProfile,
        market: MarketSnapshot,
    ) -> Dict[str, Action]:
        # Считаем веса 1/vol, нормализуем
        rp_weights = self._compute_weights(votes.keys())
        # Делегируем основной механизм WeightedConsensus
        return WeightedConsensus().aggregate(votes, rp_weights, thresholds, market)

    def _compute_weights(self, labels: Sequence[str]) -> Dict[str, float]:
        labels = list(labels)
        if not labels:
            return {}
        if not self.volatilities:
            equal = 1.0 / len(labels)
            return {lbl: equal for lbl in labels}
        # 1/vol для тех у кого есть, fallback к среднему для остальных
        inv = {}
        for lbl in labels:
            v = float(self.volatilities.get(lbl, 0.0) or 0.0)
            if v > 0:
                inv[lbl] = 1.0 / v
        if not inv:
            equal = 1.0 / len(labels)
            return {lbl: equal for lbl in labels}
        avg_inv = sum(inv.values()) / len(inv)
        for lbl in labels:
            if lbl not in inv:
                inv[lbl] = avg_inv
        total = sum(inv.values())
        return {lbl: w / total for lbl, w in inv.items()}
