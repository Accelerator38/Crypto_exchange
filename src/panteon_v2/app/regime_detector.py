"""Market-regime detection for live v2 feeds.

The detector owns the conversion from raw prices into canonical v2 regimes.
It keeps global regime compatibility, but also classifies every symbol so Flash
can score BTC, ETH, SOL, etc. against their own local market state.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from statistics import median
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..domain.types import Regime


_DEFAULT_HORIZONS_SEC = (60, 180, 720, 3600)
_TREND_REGIMES = {Regime.BULLISH, Regime.BEARISH, Regime.CHOPPY_UP, Regime.CHOPPY_DOWN}
_DEFAULT_HYSTERESIS_BARS = 3
_BUILTIN_REGIME_THRESHOLDS: Mapping[str, Mapping[str, float]] = {
    "BITGET:BTC": {
        "trend_threshold": 0.0020,
        "low_volatility": 0.0025,
        "choppy_volatility": 0.0055,
    },
    "BITGET:ETH": {
        "trend_threshold": 0.0028,
        "low_volatility": 0.0035,
        "choppy_volatility": 0.0075,
    },
    "BITGET:SOL": {
        "trend_threshold": 0.0040,
        "low_volatility": 0.0045,
        "choppy_volatility": 0.0100,
    },
    "MEXC:BTC": {
        "trend_threshold": 0.0025,
        "low_volatility": 0.0030,
        "choppy_volatility": 0.0065,
    },
    "MEXC:NEAR": {
        "trend_threshold": 0.0045,
        "low_volatility": 0.0060,
        "choppy_volatility": 0.0140,
    },
    "MEXC:LAB": {
        "trend_threshold": 0.0060,
        "low_volatility": 0.0080,
        "choppy_volatility": 0.0180,
    },
}


class PriceRegimeDetector:
    """Deterministic multi-symbol regime detector with local symbol regimes."""

    def __init__(
        self,
        *,
        exchange_name: str = "",
        lookback: int = 3,
        long_lookback: int = 12,
        bullish_return: float = 0.010,
        bearish_return: float = -0.010,
        crash_return: float = -0.060,
        anchor_symbols: Tuple[str, ...] = ("BTC", "ETH"),
        anchor_weight: float = 3.0,
        min_crash_breadth: float = 0.50,
        hysteresis_bars: int = _DEFAULT_HYSTERESIS_BARS,
        max_history: int = 128,
        poll_interval_sec: float = 5.0,
        horizons_sec: Sequence[int] = _DEFAULT_HORIZONS_SEC,
        min_history_bars: int = 2,
        low_volatility: float = 0.0030,
        choppy_volatility: float = 0.0060,
        choppy_noise_ratio: float = 0.45,
        choppy_volatility_percentile: float = 0.80,
        regime_thresholds: Mapping[str, Mapping[str, float]] | None = None,
        use_builtin_regime_thresholds: bool = True,
        min_directional_consensus: float = 0.50,
    ) -> None:
        self.exchange_name = str(exchange_name or "").strip().upper()
        self.lookback = max(2, int(lookback or 2))
        self.long_lookback = max(self.lookback, int(long_lookback or self.lookback))
        self.bullish_return = float(bullish_return)
        self.bearish_return = float(bearish_return)
        self.crash_return = float(crash_return)
        self.anchor_symbols = tuple(str(sym).upper() for sym in anchor_symbols)
        self.anchor_weight = max(1.0, float(anchor_weight or 1.0))
        self.min_crash_breadth = max(0.0, min(1.0, float(min_crash_breadth)))
        self.hysteresis_bars = max(1, int(hysteresis_bars or 1))
        self.poll_interval_sec = max(1e-6, float(poll_interval_sec or 5.0))
        self.horizons_sec = tuple(int(item) for item in (horizons_sec or _DEFAULT_HORIZONS_SEC))
        self.min_history_bars = max(2, int(min_history_bars or 2))
        self.low_volatility = max(0.0, float(low_volatility))
        self.choppy_volatility = max(self.low_volatility, float(choppy_volatility))
        self.choppy_noise_ratio = max(0.0, float(choppy_noise_ratio))
        self.choppy_volatility_percentile = max(
            0.0,
            min(1.0, float(choppy_volatility_percentile)),
        )
        # #4: порог согласия символов для директивного режима (vs mixed_rotational).
        # Ниже 0.50 → меньше mixed_rotational, больше bullish/bearish/choppy.
        self.min_directional_consensus = max(0.0, min(1.0, float(min_directional_consensus)))
        threshold_source: Dict[str, Mapping[str, float]] = {}
        if use_builtin_regime_thresholds:
            threshold_source.update(_BUILTIN_REGIME_THRESHOLDS)
        threshold_source.update(dict(regime_thresholds or {}))
        self._threshold_profiles = _normalize_threshold_profiles(threshold_source)

        horizon_bars = [self._horizon_to_bars(item) for item in self.horizons_sec]
        legacy_max = max(self.long_lookback, int(max_history))
        history_len = max(legacy_max, max(horizon_bars, default=self.long_lookback), 2)
        self._history_len = history_len
        self._history: Deque[Dict[str, float]] = deque(maxlen=history_len)
        self._current = Regime.NEUTRAL
        self._confidence = 0.0
        self._symbol_regimes: Dict[str, Regime] = {}
        self._symbol_stats: Dict[str, dict] = {}
        self._pending: Optional[Regime] = None
        self._pending_count = 0
        self._symbol_pending: Dict[str, Regime] = {}
        self._symbol_pending_count: Dict[str, int] = {}
        self._volatility_history: Dict[str, Deque[float]] = {}
        self._volume_history: Dict[str, Deque[float]] = {}

    @property
    def current(self) -> Regime:
        return self._current

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def symbol_regimes(self) -> Dict[str, Regime]:
        return dict(self._symbol_regimes)

    @property
    def symbol_stats(self) -> Dict[str, dict]:
        return {symbol: dict(stats) for symbol, stats in self._symbol_stats.items()}

    def update(
        self,
        prices: Mapping[str, float],
        *,
        volumes: Mapping[str, float] | None = None,
        funding: Mapping[str, float] | None = None,
    ) -> Regime:
        clean = self._clean_prices(prices)
        if not clean:
            self._confidence = 0.0
            self._symbol_regimes = {}
            self._symbol_stats = {}
            return self._current
        clean_volumes = self._clean_metric_map(volumes, symbols=clean.keys())
        clean_funding = self._clean_metric_map(funding, symbols=clean.keys())

        self._history.append(clean)
        if len(self._history) < 2:
            self._current = Regime.NEUTRAL
            self._confidence = 0.0
            self._symbol_regimes = {sym: self._current for sym in clean}
            self._symbol_stats = {}
            return self._current

        stats_by_symbol = {
            symbol: self._decorate_symbol_stats(
                symbol,
                self._symbol_window_stats(symbol),
                volumes=clean_volumes,
                funding=clean_funding,
            )
            for symbol in clean
        }
        proposed_symbol_regimes = {
            symbol: self._classify_symbol(stats)
            for symbol, stats in stats_by_symbol.items()
        }
        symbol_regimes = {
            symbol: self._apply_symbol_hysteresis(symbol, regime)
            for symbol, regime in proposed_symbol_regimes.items()
        }
        self._prune_symbol_hysteresis(clean.keys())
        proposed, confidence = self._classify_global(symbol_regimes, stats_by_symbol)
        self._confidence = confidence
        self._current = self._apply_hysteresis(proposed)
        self._symbol_regimes = symbol_regimes
        self._symbol_stats = {
            symbol: self._stats_payload(
                stats,
                symbol_regimes.get(symbol, self._current),
            )
            for symbol, stats in stats_by_symbol.items()
        }
        return self._current

    def seed(self, rows: Iterable[Mapping[str, float]], *, limit: Optional[int] = None) -> Regime:
        selected = list(rows)
        if limit is not None and int(limit) > 0:
            selected = selected[-int(limit):]
        regime = self._current
        for row in selected:
            regime = self.update(row)
        return regime

    def _classify_symbol(self, stats: Mapping[str, object]) -> Regime:
        if int(stats.get("available_bars", 0) or 0) < self.min_history_bars:
            return Regime.NEUTRAL
        crash_return = float(stats.get("crash_return", self.crash_return) or self.crash_return)
        short_crash = float(stats.get("short_return", 0.0) or 0.0) <= crash_return
        medium_crash = float(stats.get("medium_return", 0.0) or 0.0) <= crash_return
        if short_crash or medium_crash:
            return Regime.CRASH

        trend = float(stats.get("trend", 0.0) or 0.0)
        vol = float(stats.get("volatility", 0.0) or 0.0)
        vol_percentile = float(stats.get("volatility_percentile", 0.0) or 0.0)
        abs_trend = abs(trend)
        threshold = float(stats.get("trend_threshold", self._trend_threshold()) or self._trend_threshold())
        low_volatility = float(stats.get("low_volatility", self.low_volatility) or self.low_volatility)
        choppy_volatility = float(
            stats.get("choppy_volatility", self.choppy_volatility)
            or self.choppy_volatility
        )
        choppy_noise_ratio = float(
            stats.get("choppy_noise_ratio", self.choppy_noise_ratio)
            or self.choppy_noise_ratio
        )
        if abs_trend < threshold:
            return Regime.RANGE_LOW_VOL if vol <= low_volatility else Regime.NEUTRAL

        noisy = (
            vol >= choppy_volatility
            or vol >= abs_trend * choppy_noise_ratio
            or (
                vol_percentile >= self.choppy_volatility_percentile
                and vol > low_volatility
            )
        )
        if trend > 0:
            return Regime.CHOPPY_UP if noisy else Regime.BULLISH
        return Regime.CHOPPY_DOWN if noisy else Regime.BEARISH

    def _classify_global(
        self,
        symbol_regimes: Mapping[str, Regime],
        stats_by_symbol: Mapping[str, Mapping[str, object]],
    ) -> tuple[Regime, float]:
        regimes = list(symbol_regimes.values())
        if not regimes:
            return Regime.NEUTRAL, 0.0
        crash_breadth = regimes.count(Regime.CRASH) / len(regimes)
        worst_anchor = min(
            (
                float(stats_by_symbol[sym].get("short_return", 0.0) or 0.0)
                for sym in self.anchor_symbols
                if sym in stats_by_symbol
            ),
            default=1.0,
        )
        if crash_breadth >= self.min_crash_breadth or worst_anchor <= self.crash_return:
            return Regime.CRASH, max(crash_breadth, 0.7)

        positives = {Regime.BULLISH, Regime.CHOPPY_UP}
        negatives = {Regime.BEARISH, Regime.CHOPPY_DOWN}
        n_pos = sum(1 for item in regimes if item in positives)
        n_neg = sum(1 for item in regimes if item in negatives)
        if n_pos and n_neg:
            return Regime.MIXED_ROTATIONAL, self._agreement_confidence(max(n_pos, n_neg), len(regimes))
        if n_pos:
            if n_pos / len(regimes) >= self.min_directional_consensus:
                return self._dominant_trend_regime(regimes, positives), self._agreement_confidence(n_pos, len(regimes))
            return Regime.MIXED_ROTATIONAL, self._agreement_confidence(n_pos, len(regimes))
        if n_neg:
            if n_neg / len(regimes) >= self.min_directional_consensus:
                return self._dominant_trend_regime(regimes, negatives), self._agreement_confidence(n_neg, len(regimes))
            return Regime.MIXED_ROTATIONAL, self._agreement_confidence(n_neg, len(regimes))

        counts = Counter(regimes)
        if counts.get(Regime.RANGE_LOW_VOL, 0) >= max(1, len(regimes) // 2):
            return Regime.RANGE_LOW_VOL, self._agreement_confidence(counts[Regime.RANGE_LOW_VOL], len(regimes))
        return Regime.NEUTRAL, 0.3

    @staticmethod
    def _dominant_trend_regime(regimes: Sequence[Regime], allowed: set[Regime]) -> Regime:
        counts = Counter(item for item in regimes if item in allowed)
        if not counts:
            return Regime.NEUTRAL
        return counts.most_common(1)[0][0]

    @staticmethod
    def _agreement_confidence(count: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return max(0.1, min(1.0, count / total))

    def _symbol_window_stats(self, symbol: str) -> Dict[str, object]:
        rows = list(self._history)
        available = sum(1 for row in rows if symbol in row)
        stats: List[dict] = []
        for horizon_sec, horizon in self._horizon_pairs():
            values = [
                float(row[symbol])
                for row in rows[-min(horizon, len(rows)):]
                if symbol in row and float(row[symbol]) > 0.0
            ]
            if len(values) < 2:
                continue
            stats.append({
                "bars": len(values),
                "horizon_sec": horizon_sec,
                "return": (values[-1] - values[0]) / values[0],
                "slope": _normalized_slope(values),
                "volatility": _one_step_volatility(values),
            })
        if not stats:
            return {
                "available_bars": available,
                "trend": 0.0,
                "volatility": 0.0,
                "short_return": 0.0,
                "medium_return": 0.0,
                "micro_trend": 0.0,
                "macro_trend": 0.0,
                "horizons": [],
            }
        weights = _linear_weights(len(stats))
        returns = [float(item["return"]) for item in stats]
        slopes = [float(item["slope"]) for item in stats]
        vols = [float(item["volatility"]) for item in stats]
        weighted_return = _weighted_average(returns, weights)
        weighted_slope = _weighted_average(slopes, weights)
        trend = (weighted_slope * 0.60) + (weighted_return * 0.40)
        micro_stats = [
            item for item in stats
            if int(item.get("horizon_sec", 0) or 0) <= 180
        ]
        macro_stats = [
            item for item in stats
            if int(item.get("horizon_sec", 0) or 0) >= 720
        ]
        micro_trend = _trend_from_window_stats(micro_stats or stats[:1])
        macro_trend = _trend_from_window_stats(macro_stats or stats[-1:])
        return {
            "available_bars": available,
            "trend": trend,
            "weighted_return": weighted_return,
            "weighted_slope": weighted_slope,
            "volatility": _weighted_average(vols, weights),
            "short_return": returns[0],
            "medium_return": returns[min(1, len(returns) - 1)],
            "micro_trend": micro_trend,
            "macro_trend": macro_trend,
            "horizons": stats,
        }

    def _decorate_symbol_stats(
        self,
        symbol: str,
        stats: Mapping[str, object],
        *,
        volumes: Mapping[str, float],
        funding: Mapping[str, float],
    ) -> Dict[str, object]:
        clean_symbol = str(symbol or "").upper()
        out = dict(stats)
        profile = self._thresholds_for_symbol(clean_symbol)
        out.update(profile)
        volatility = max(0.0, float(out.get("volatility", 0.0) or 0.0))
        vol_history = self._volatility_history.setdefault(
            clean_symbol,
            deque(maxlen=self._history_len),
        )
        vol_history.append(volatility)
        out["volatility_percentile"] = _percentile_rank(vol_history, volatility)

        volume_value = volumes.get(clean_symbol)
        volume_history = self._volume_history.setdefault(
            clean_symbol,
            deque(maxlen=self._history_len),
        )
        if volume_value is not None:
            volume_history.append(max(0.0, float(volume_value)))
        out["liquidity"] = 0.0 if volume_value is None else max(0.0, float(volume_value))
        out["liquidity_percentile"] = (
            0.5
            if volume_value is None
            else _percentile_rank(volume_history, max(0.0, float(volume_value)))
        )
        out["funding_rate"] = float(funding.get(clean_symbol, 0.0) or 0.0)

        threshold = float(out.get("trend_threshold", self._trend_threshold()) or self._trend_threshold())
        out["micro_direction"] = _trend_direction(
            float(out.get("micro_trend", 0.0) or 0.0),
            threshold,
        )
        out["macro_direction"] = _trend_direction(
            float(out.get("macro_trend", 0.0) or 0.0),
            threshold,
        )
        out["micro_macro_alignment"] = _micro_macro_alignment(
            str(out["micro_direction"]),
            str(out["macro_direction"]),
        )
        return out

    def _apply_hysteresis(self, proposed: Regime) -> Regime:
        if proposed == self._current:
            self._pending = None
            self._pending_count = 0
            return self._current
        if self._current == Regime.NEUTRAL:
            self._pending = None
            self._pending_count = 0
            return proposed
        if proposed == Regime.CRASH:
            self._pending = None
            self._pending_count = 0
            return proposed
        if self._pending == proposed:
            self._pending_count += 1
        else:
            self._pending = proposed
            self._pending_count = 1
        if self._pending_count >= self.hysteresis_bars:
            self._pending = None
            self._pending_count = 0
            return proposed
        return self._current

    def _apply_symbol_hysteresis(self, symbol: str, proposed: Regime) -> Regime:
        clean_symbol = str(symbol or "").upper()
        current = self._symbol_regimes.get(clean_symbol)
        if current is None:
            self._symbol_pending.pop(clean_symbol, None)
            self._symbol_pending_count.pop(clean_symbol, None)
            return proposed
        if current == Regime.NEUTRAL:
            self._symbol_pending.pop(clean_symbol, None)
            self._symbol_pending_count.pop(clean_symbol, None)
            return proposed
        if proposed == current:
            self._symbol_pending.pop(clean_symbol, None)
            self._symbol_pending_count.pop(clean_symbol, None)
            return current
        if proposed == Regime.CRASH:
            self._symbol_pending.pop(clean_symbol, None)
            self._symbol_pending_count.pop(clean_symbol, None)
            return proposed
        if self._symbol_pending.get(clean_symbol) == proposed:
            self._symbol_pending_count[clean_symbol] = (
                self._symbol_pending_count.get(clean_symbol, 0) + 1
            )
        else:
            self._symbol_pending[clean_symbol] = proposed
            self._symbol_pending_count[clean_symbol] = 1
        if self._symbol_pending_count.get(clean_symbol, 0) >= self.hysteresis_bars:
            self._symbol_pending.pop(clean_symbol, None)
            self._symbol_pending_count.pop(clean_symbol, None)
            return proposed
        return current

    def _prune_symbol_hysteresis(self, symbols: Iterable[str]) -> None:
        active = {str(symbol or "").upper() for symbol in symbols}
        for key in list(self._symbol_pending):
            if key not in active:
                self._symbol_pending.pop(key, None)
                self._symbol_pending_count.pop(key, None)

    def _thresholds_for_symbol(self, symbol: str) -> Dict[str, float]:
        clean_symbol = str(symbol or "").upper()
        base = {
            "trend_threshold": self._trend_threshold(),
            "low_volatility": self.low_volatility,
            "choppy_volatility": self.choppy_volatility,
            "choppy_noise_ratio": self.choppy_noise_ratio,
            "crash_return": self.crash_return,
        }
        keys = ["*", "DEFAULT"]
        if self.exchange_name:
            keys.append(f"{self.exchange_name}:*")
        keys.append(clean_symbol)
        if self.exchange_name and clean_symbol:
            keys.append(f"{self.exchange_name}:{clean_symbol}")
        for key in keys:
            profile = self._threshold_profiles.get(key)
            if profile:
                base.update(profile)
        return base

    def _horizon_bars(self) -> List[int]:
        bars = [bars for _, bars in self._horizon_pairs(include_legacy=False)]
        bars.extend([self.lookback, self.long_lookback])
        return sorted({max(2, int(item)) for item in bars})

    def _horizon_pairs(self, *, include_legacy: bool = True) -> List[Tuple[int, int]]:
        pairs = [
            (int(seconds), self._horizon_to_bars(int(seconds)))
            for seconds in self.horizons_sec
        ]
        if include_legacy:
            pairs.extend([
                (int(round(self.lookback * self.poll_interval_sec)), self.lookback),
                (int(round(self.long_lookback * self.poll_interval_sec)), self.long_lookback),
            ])
        dedup: Dict[int, int] = {}
        for seconds, bars in pairs:
            clean_bars = max(2, int(bars))
            dedup[clean_bars] = max(int(seconds), int(round(clean_bars * self.poll_interval_sec)))
        return sorted((seconds, bars) for bars, seconds in dedup.items())

    def _horizon_to_bars(self, seconds: int) -> int:
        return max(2, int(round(float(seconds) / self.poll_interval_sec)))

    def _trend_threshold(self) -> float:
        legacy_threshold = max(abs(self.bullish_return), abs(self.bearish_return), 1e-12)
        return max(legacy_threshold * 0.60, 0.0025)

    @staticmethod
    def _clean_prices(prices: Mapping[str, float]) -> Dict[str, float]:
        clean: Dict[str, float] = {}
        for sym, price in (prices or {}).items():
            try:
                value = float(price)
            except (TypeError, ValueError):
                continue
            if value > 0:
                clean[str(sym).upper()] = value
        return clean

    @staticmethod
    def _clean_metric_map(
        values: Mapping[str, float] | None,
        *,
        symbols: Iterable[str],
    ) -> Dict[str, float]:
        allowed = {str(symbol or "").upper() for symbol in symbols}
        clean: Dict[str, float] = {}
        for sym, raw in (values or {}).items():
            clean_symbol = str(sym or "").upper()
            if clean_symbol not in allowed:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                clean[clean_symbol] = value
        return clean

    @staticmethod
    def _stats_payload(stats: Mapping[str, object], regime: Regime) -> dict:
        carry_risk = _carry_risk(
            regime,
            float(stats.get("funding_rate", 0.0) or 0.0),
        )
        risk_pressure = _risk_pressure(
            volatility_percentile=float(stats.get("volatility_percentile", 0.0) or 0.0),
            liquidity_percentile=float(stats.get("liquidity_percentile", 0.5) or 0.5),
            carry_risk=carry_risk,
        )
        trend_threshold = float(stats.get("trend_threshold", 0.0) or 0.0)
        trend = float(stats.get("trend", 0.0) or 0.0)
        confidence = _regime_confidence(
            regime,
            trend=trend,
            threshold=trend_threshold,
            volatility=float(stats.get("volatility", 0.0) or 0.0),
            low_volatility=float(stats.get("low_volatility", 0.0) or 0.0),
        )
        return {
            "regime": regime.label,
            "regime_confidence": confidence,
            "trend": trend,
            "weighted_return": float(stats.get("weighted_return", 0.0) or 0.0),
            "weighted_slope": float(stats.get("weighted_slope", 0.0) or 0.0),
            "volatility": float(stats.get("volatility", 0.0) or 0.0),
            "volatility_percentile": float(
                stats.get("volatility_percentile", 0.0) or 0.0
            ),
            "liquidity": float(stats.get("liquidity", 0.0) or 0.0),
            "liquidity_percentile": float(
                stats.get("liquidity_percentile", 0.5) or 0.5
            ),
            "funding_rate": float(stats.get("funding_rate", 0.0) or 0.0),
            "carry_risk": carry_risk,
            "risk_pressure": risk_pressure,
            "short_return": float(stats.get("short_return", 0.0) or 0.0),
            "medium_return": float(stats.get("medium_return", 0.0) or 0.0),
            "micro_trend": float(stats.get("micro_trend", 0.0) or 0.0),
            "macro_trend": float(stats.get("macro_trend", 0.0) or 0.0),
            "micro_direction": str(stats.get("micro_direction", "flat") or "flat"),
            "macro_direction": str(stats.get("macro_direction", "flat") or "flat"),
            "micro_macro_alignment": str(
                stats.get("micro_macro_alignment", "flat") or "flat"
            ),
            "trend_threshold": trend_threshold,
            "low_volatility": float(stats.get("low_volatility", 0.0) or 0.0),
            "choppy_volatility": float(stats.get("choppy_volatility", 0.0) or 0.0),
            "available_bars": int(stats.get("available_bars", 0) or 0),
        }


def _normalize_threshold_profiles(
    raw_profiles: Mapping[str, Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    profiles: Dict[str, Dict[str, float]] = {}
    allowed = {
        "trend_threshold",
        "low_volatility",
        "choppy_volatility",
        "choppy_noise_ratio",
        "crash_return",
    }
    for raw_key, raw_values in (raw_profiles or {}).items():
        key = str(raw_key or "").strip().upper()
        if not key or not isinstance(raw_values, Mapping):
            continue
        clean: Dict[str, float] = {}
        for raw_name, raw_value in raw_values.items():
            name = str(raw_name or "").strip().lower()
            if name not in allowed:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                clean[name] = value
        if clean:
            profiles[key] = clean
    return profiles


def _trend_from_window_stats(stats: Sequence[Mapping[str, object]]) -> float:
    if not stats:
        return 0.0
    weights = _linear_weights(len(stats))
    trends = [
        (float(item.get("slope", 0.0) or 0.0) * 0.60)
        + (float(item.get("return", 0.0) or 0.0) * 0.40)
        for item in stats
    ]
    return _weighted_average(trends, weights)


def _percentile_rank(history: Iterable[float], value: float) -> float:
    values = [float(item) for item in history if math.isfinite(float(item))]
    if not values:
        return 0.5
    current = float(value)
    below_or_equal = sum(1 for item in values if item <= current)
    return max(0.0, min(1.0, below_or_equal / len(values)))


def _trend_direction(value: float, threshold: float) -> str:
    threshold = max(1e-12, float(threshold))
    parsed = float(value)
    if parsed >= threshold:
        return "up"
    if parsed <= -threshold:
        return "down"
    return "flat"


def _micro_macro_alignment(micro: str, macro: str) -> str:
    micro = str(micro or "flat").lower()
    macro = str(macro or "flat").lower()
    if micro in {"up", "down"} and macro in {"up", "down"}:
        return "aligned" if micro == macro else "conflict"
    if micro in {"up", "down"}:
        return "micro_only"
    if macro in {"up", "down"}:
        return "macro_only"
    return "flat"


def _carry_risk(regime: Regime, funding_rate: float) -> str:
    funding = float(funding_rate)
    if abs(funding) <= 1e-12:
        return "neutral"
    if regime in {Regime.BEARISH, Regime.CHOPPY_DOWN, Regime.CRASH}:
        return "short_receives_funding" if funding > 0.0 else "short_pays_funding"
    if regime in {Regime.BULLISH, Regime.CHOPPY_UP}:
        return "long_receives_funding" if funding < 0.0 else "long_pays_funding"
    return "neutral"


def _risk_pressure(
    *,
    volatility_percentile: float,
    liquidity_percentile: float,
    carry_risk: str,
) -> float:
    pressure = 1.0
    vol_pct = max(0.0, min(1.0, float(volatility_percentile)))
    liq_pct = max(0.0, min(1.0, float(liquidity_percentile)))
    if vol_pct >= 0.80:
        pressure += (vol_pct - 0.80) * 1.25
    if liq_pct <= 0.25:
        pressure += (0.25 - liq_pct) * 1.20
    if carry_risk in {"short_pays_funding", "long_pays_funding"}:
        pressure += 0.35
    elif carry_risk in {"short_receives_funding", "long_receives_funding"}:
        pressure -= 0.10
    return max(0.50, round(float(pressure), 6))


def _regime_confidence(
    regime: Regime,
    *,
    trend: float,
    threshold: float,
    volatility: float,
    low_volatility: float,
) -> float:
    if regime == Regime.CRASH:
        return 1.0
    threshold = max(1e-12, float(threshold))
    if regime in _TREND_REGIMES:
        return max(0.1, min(1.0, abs(float(trend)) / (threshold * 2.0)))
    if regime == Regime.RANGE_LOW_VOL:
        low_vol = max(1e-12, float(low_volatility))
        return max(0.1, min(1.0, 1.0 - min(1.0, float(volatility) / low_vol) * 0.5))
    return 0.3


def _normalized_slope(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    denom = sum((idx - x_mean) ** 2 for idx in range(n))
    if denom <= 1e-12 or y_mean <= 0.0:
        return 0.0
    slope_per_bar = sum((idx - x_mean) * (value - y_mean) for idx, value in enumerate(values)) / denom
    return slope_per_bar * (n - 1) / y_mean


def _one_step_volatility(values: Sequence[float]) -> float:
    returns = []
    for prev, current in zip(values, values[1:]):
        if prev > 0.0:
            returns.append((current - prev) / prev)
    if len(returns) < 2:
        return 0.0
    avg = sum(returns) / len(returns)
    variance = sum((item - avg) ** 2 for item in returns) / (len(returns) - 1)
    return math.sqrt(max(0.0, variance))


def _linear_weights(count: int) -> List[float]:
    if count <= 0:
        return []
    if count == 1:
        return [1.0]
    weights = [1.0 / (idx + 1) for idx in range(count)]
    total = sum(weights)
    return [item / total for item in weights]


def _weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values:
        return 0.0
    if not weights or len(weights) != len(values):
        return sum(float(item) for item in values) / len(values)
    return sum(float(value) * float(weight) for value, weight in zip(values, weights))
