from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import MARKET_STATES, STATIC_FEATURE_NAMES


EPSILON = 1e-12


def _rolling_percentile(values: pd.Series, window: int = 180) -> pd.Series:
    minimum = max(30, window // 3)
    return values.rolling(window, min_periods=minimum).apply(
        lambda item: float(np.mean(item <= item[-1])), raw=True
    )


def _per_symbol_features(group: pd.DataFrame) -> pd.DataFrame:
    source = group.sort_values("timestamp", kind="stable").copy()
    close = source["close"]
    returns = close.pct_change()
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            source["high"] - source["low"],
            (source["high"] - previous_close).abs(),
            (source["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_48 = close.ewm(span=48, adjust=False).mean()
    abs_path = close.diff().abs().rolling(12, min_periods=12).sum()
    rolling_low_24 = source["low"].rolling(24, min_periods=24).min()
    rolling_high_24 = source["high"].rolling(24, min_periods=24).max()
    previous_low_20 = source["low"].shift(1).rolling(20, min_periods=20).min()
    previous_high_20 = source["high"].shift(1).rolling(20, min_periods=20).max()
    volume_mean_24 = source["volume"].rolling(24, min_periods=24).mean()
    volume_std_24 = source["volume"].rolling(24, min_periods=24).std(ddof=0)
    upper_wick = source["high"] - source[["open", "close"]].max(axis=1)
    lower_wick = source[["open", "close"]].min(axis=1) - source["low"]

    source["return_1"] = returns
    for horizon in (3, 6, 12):
        source[f"return_{horizon}"] = close.pct_change(horizon)
    source["ema_gap_12_48"] = ema_12 / ema_48.replace(0.0, np.nan) - 1.0
    source["ema_slope_12_3"] = ema_12 / ema_12.shift(3).replace(0.0, np.nan) - 1.0
    source["atr_14_pct"] = true_range.rolling(14, min_periods=14).mean() / close
    source["realized_vol_6"] = returns.rolling(6, min_periods=6).std(ddof=0)
    vol_24 = returns.rolling(24, min_periods=24).std(ddof=0)
    source["realized_vol_ratio_6_24"] = source["realized_vol_6"] / vol_24.replace(0.0, np.nan)
    source["efficiency_12"] = (close - close.shift(12)).abs() / abs_path.replace(0.0, np.nan)
    source["range_position_24"] = (
        2.0 * (close - rolling_low_24) / (rolling_high_24 - rolling_low_24).replace(0.0, np.nan) - 1.0
    )
    source["donchian_position_20"] = (
        2.0 * (close - previous_low_20) / (previous_high_20 - previous_low_20).replace(0.0, np.nan) - 1.0
    )
    source["volume_z_24"] = (source["volume"] - volume_mean_24) / volume_std_24.replace(0.0, np.nan)
    source["volume_trend_6_24"] = np.log(
        source["volume"].rolling(6, min_periods=6).mean()
        / volume_mean_24.replace(0.0, np.nan)
    )
    source["candle_body_pct"] = (source["close"] - source["open"]) / source["open"].replace(0.0, np.nan)
    source["wick_asymmetry"] = (lower_wick - upper_wick) / true_range.replace(0.0, np.nan)
    return source


def build_feature_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Build close-time features using current and earlier 4h bars only."""
    frames: list[pd.DataFrame] = []
    for symbol, group in panel.groupby("symbol", sort=True, observed=True):
        featured = _per_symbol_features(group)
        featured["symbol"] = str(symbol)
        frames.append(featured)
    if not frames:
        raise ValueError("cannot build Genetic_Primus features from an empty panel")
    local = pd.concat(frames, ignore_index=True)
    local["symbol"] = local["symbol"].astype(str)
    local = local.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)

    pivot_close = local.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    ret_3 = pivot_close.pct_change(3)
    ret_12 = pivot_close.pct_change(12)
    market = pd.DataFrame(index=pivot_close.index)
    market["market_return_3"] = ret_3.median(axis=1)
    market["market_return_12"] = ret_12.median(axis=1)
    market["market_breadth_3"] = (ret_3 > 0.0).mean(axis=1) * 2.0 - 1.0
    market["market_breadth_12"] = (ret_12 > 0.0).mean(axis=1) * 2.0 - 1.0
    market["market_dispersion_3"] = ret_3.std(axis=1, ddof=0)
    market_vol = pivot_close.pct_change().median(axis=1).rolling(6, min_periods=6).std(ddof=0)
    market["market_vol_percentile"] = _rolling_percentile(market_vol)
    trend_scale = market_vol.rolling(24, min_periods=12).mean().replace(0.0, np.nan)
    trend_score = market["market_return_12"] / (trend_scale * np.sqrt(12.0))
    sign_change = np.sign(market["market_return_3"]) != np.sign(market["market_return_12"].shift(2))
    market["transition_score"] = (
        (market["market_return_3"] - market["market_return_12"].shift(2)).abs()
        / trend_scale.replace(0.0, np.nan)
    ).clip(0.0, 8.0)
    state = pd.Series("volatile_range", index=market.index, dtype="object")
    state.loc[market["market_vol_percentile"] < 0.35] = "quiet_range"
    state.loc[trend_score >= 1.0] = "trend_up"
    state.loc[trend_score <= -1.0] = "trend_down"
    state.loc[sign_change & (market["transition_score"] >= 1.5)] = "transition"
    market["market_state"] = state

    local = local.merge(market.reset_index(), on="timestamp", how="left", validate="many_to_one")
    local["relative_momentum_rank"] = (
        local.groupby("timestamp", observed=True)["return_12"].rank(pct=True, method="average") * 2.0 - 1.0
    )
    local["residual_return_3"] = local["return_3"] - local["market_return_3"]
    if not set(local["market_state"].dropna().unique()) <= set(MARKET_STATES):
        raise ValueError("feature builder emitted an unknown market state")
    local.loc[:, list(STATIC_FEATURE_NAMES)] = local.loc[:, list(STATIC_FEATURE_NAMES)].replace(
        [np.inf, -np.inf], np.nan
    )
    local["feature_supported"] = local.loc[:, list(STATIC_FEATURE_NAMES)].notna().all(axis=1)
    return local


@dataclass(frozen=True)
class RobustScaler:
    median: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "RobustScaler":
        values = frame.loc[:, list(STATIC_FEATURE_NAMES)].to_numpy(dtype=np.float64)
        median = np.nanmedian(values, axis=0)
        q25 = np.nanquantile(values, 0.25, axis=0)
        q75 = np.nanquantile(values, 0.75, axis=0)
        scale = q75 - q25
        scale[~np.isfinite(scale) | (scale < EPSILON)] = 1.0
        median[~np.isfinite(median)] = 0.0
        return cls(median=median, scale=scale)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, list(STATIC_FEATURE_NAMES)].to_numpy(dtype=np.float64)
        normalized = (values - self.median) / self.scale
        normalized[~np.isfinite(normalized)] = 0.0
        return np.clip(normalized, -8.0, 8.0).astype(np.float32)

    def to_mapping(self) -> dict[str, list[float]]:
        return {"median": self.median.tolist(), "scale": self.scale.tolist()}
