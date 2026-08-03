# Current model regime audit

Generated: `2026-08-03T11:14:17.887061+00:00`

All rankings use the common stress cost of 16 bps round trip. Regimes are attributed from the last closed BTC 1h candle before entry and never alter signals.

## Scope and regime contract

- Uniformly ranked models: `8` of `10` (flat and long-only are reference baselines).
- `bullish`: BTC close > EMA72 > EMA336 and seven-day momentum > 0.
- `bearish`: BTC close < EMA72 < EMA336 and seven-day momentum < 0.
- `volatile_mixed`: no directional trend and ATR14/close is at or above the trailing 180-day 75th percentile.
- `range_low_vol`: no directional trend and ATR14/close is at or below the trailing 180-day 25th percentile.
- `neutral`: the remaining non-trending observations.
- Attribution completeness: every closed trade has exactly one known regime.

| Window | Timerange | Role |
|---|---|---|
| development | 20220101-20240101 | hypothesis construction only |
| validation | 20240104-20250101 | first held-out check |
| oos | 20250104-20260101 | independent out-of-sample check |
| sanity | 20260104-20260715 | most recent temporal stability check |

## Verdict

- Confirmed specializations: `0`.
- Promising but unconfirmed: `1`.
- Positive mean without robust LCB is not treated as edge.
- Orders enabled: `false`; promotion authority: `false`.

## Best available model per regime

| Regime | Model | Holdout trades | Stress mean bps | LCB bps | Adjusted LCB bps | Positive windows | Status |
|---|---|---:|---:|---:|---:|---:|---|
| bullish | regime_pullback_12_48_v1 | 3583 | -10.58 | -14.70 | -17.08 | 0/3 | FAILED_OR_INSUFFICIENT |
| bearish | candle_momentum_3_v1 | 13189 | -15.31 | -18.03 | -19.60 | 0/3 | FAILED_OR_INSUFFICIENT |
| volatile_mixed | regime_pullback_12_48_v1 | 980 | -6.31 | -15.62 | -20.98 | 1/3 | FAILED_OR_INSUFFICIENT |
| range_low_vol | candle_momentum_3_v1 | 4895 | -13.12 | -16.23 | -18.02 | 0/3 | FAILED_OR_INSUFFICIENT |
| neutral | ema_trend_12_48_v1 | 991 | 57.51 | 10.24 | -17.02 | 2/3 | PROMISING_UNCONFIRMED |

## Positive-mean slices

| Model | Regime | Trades | Stress mean bps | LCB bps | Adjusted LCB bps | Status |
|---|---|---:|---:|---:|---:|---|
| ema_trend_12_48_v1 | neutral | 991 | 57.51 | 10.24 | -17.02 | PROMISING_UNCONFIRMED |
| donchian_55_v1 | neutral | 605 | 38.98 | -2.29 | -26.08 | POSITIVE_MEAN_ONLY |
| ema_trend_24_96_v1 | range_low_vol | 275 | 31.83 | -35.06 | -73.62 | POSITIVE_MEAN_ONLY |
| donchian_55_v1 | range_low_vol | 323 | 25.07 | -25.01 | -53.89 | POSITIVE_MEAN_ONLY |
| ema_trend_24_96_v1 | neutral | 625 | 16.64 | -34.77 | -64.41 | POSITIVE_MEAN_ONLY |
| vol_compression_breakout_v1 | neutral | 290 | 14.87 | -13.49 | -29.84 | POSITIVE_MEAN_ONLY |
| ema_trend_12_48_v1 | range_low_vol | 502 | 13.99 | -22.36 | -43.32 | POSITIVE_MEAN_ONLY |
| ema_trend_24_96_v1 | bullish | 586 | 12.27 | -65.74 | -110.73 | POSITIVE_MEAN_ONLY |
| donchian_55_v1 | bullish | 687 | 11.97 | -35.11 | -62.26 | POSITIVE_MEAN_ONLY |
| donchian_20_v1 | neutral | 1365 | 0.73 | -17.03 | -27.27 | POSITIVE_MEAN_ONLY |

## Nearest candidate: EMA 12/48 in neutral

The aggregate holdout LCB is positive before multiple-testing correction, but performance decays across time and the latest window is negative.

| Window | Trades | Stress mean bps | LCB bps | Adjusted LCB bps |
|---|---:|---:|---:|---:|
| validation | 379 | 140.04 | 28.64 | -35.60 |
| oos | 370 | 13.52 | -34.51 | -62.20 |
| sanity | 242 | -4.50 | -42.19 | -63.92 |
| combined holdout | 991 | 57.51 | 10.24 | -17.02 |

## Development-only false positive

`mean_reversion_24_v1 / volatile_mixed` passed even the adjusted LCB on development, then failed every later LCB and became negative in sanity. It is retained as a concrete overfitting warning.

| Window | Trades | Stress mean bps | LCB bps | Adjusted LCB bps |
|---|---:|---:|---:|---:|
| development | 307 | 50.29 | 23.68 | 8.33 |
| validation | 218 | 9.36 | -41.44 | -70.74 |
| oos | 235 | 10.53 | -41.22 | -71.06 |
| sanity | 104 | -43.59 | -106.93 | -143.46 |

## Retained non-uniform evidence

These rows are diagnostic only. Their data, cost contracts, and regime definitions are not uniform enough to merge into the ranking above.

| Artifact | Slice | Trades | Mean bps | LCB bps | Verdict |
|---|---|---:|---:|---:|---|
| LongHorizonTrendStrategyV1 | bullish / sanity | 17 | -148.82 | -270.63 | TERMINAL_REJECTED_RETROSPECTIVE |
| regime_pullback_hourly_v1 | bearish | 565 | -1.83 | -21.45 | TERMINAL_REJECTED |
| ohlcv_compression_transition_hourly_v1 | compression_transition | 960 | -3.92 | -14.57 | TERMINAL_REJECTED |
| CarryFlow range-transition breakout | range_low_vol transition | 8 | 11.74 | -15.82 | ACTIVATION_OR_EXPECTANCY_FAILURE |

## Evidence limitations

- Named Pantheon/Genetics agents have no retained promotable checkpoints or common costed regime ledgers.
- CarryFlow and funding candidates have insufficient or zero trades and are excluded from ranking.
- Trade-level LCB assumes independent observations; overlapping symbols can make it optimistic.
- This is retrospective model selection and cannot authorize paper/live trading.
