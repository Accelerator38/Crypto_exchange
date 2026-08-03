# LongHorizonTrendStrategyV1 regime attribution

This is a post-trade diagnostic. Regimes did not filter or alter entries.
Each trade is attributed using the last closed BTC 1h candle before entry.

## Regime definition

- `bullish`: BTC close > EMA72 > EMA336 and 7-day momentum > 0.
- `bearish`: BTC close < EMA72 < EMA336 and 7-day momentum < 0.
- `volatile_mixed`: no directional trend and ATR14/close is at or above its trailing 180-day 75th percentile.
- `range_low_vol`: no directional trend and ATR14/close is at or below its trailing 180-day 25th percentile.
- `neutral`: remaining observations.

## Results

| Window | Regime | Trades | Win % | Mean base bps | LCB base bps | Mean stress bps | LCB stress bps | Regime gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| development | bullish | 84 | 41.7 | 146.87 | -129.11 | 138.73 | -137.04 | FAIL |
| development | bearish | 68 | 25.0 | -11.74 | -208.79 | -19.72 | -216.92 | FAIL |
| development | volatile_mixed | 13 | 7.7 | -403.87 | -651.18 | -411.68 | -658.77 | FAIL |
| development | range_low_vol | 25 | 48.0 | 172.54 | -35.54 | 164.65 | -43.51 | FAIL |
| development | neutral | 36 | 33.3 | -66.53 | -279.81 | -74.46 | -287.77 | FAIL |
| validation | bullish | 54 | 37.0 | 97.89 | -125.23 | 89.81 | -133.12 | FAIL |
| validation | bearish | 38 | 26.3 | -38.19 | -233.93 | -46.30 | -242.07 | FAIL |
| validation | volatile_mixed | 5 | 20.0 | 1798.71 | -2313.61 | 1789.00 | -2320.18 | FAIL |
| validation | range_low_vol | 6 | 33.3 | 115.86 | -356.28 | 107.58 | -364.30 | FAIL |
| validation | neutral | 13 | 30.8 | 229.62 | -464.01 | 221.45 | -471.69 | FAIL |
| oos | bullish | 28 | 53.6 | 156.69 | -171.06 | 148.57 | -178.92 | FAIL |
| oos | bearish | 45 | 28.9 | -89.39 | -214.14 | -97.44 | -222.29 | FAIL |
| oos | volatile_mixed | 6 | 33.3 | 66.54 | -499.46 | 58.63 | -507.81 | FAIL |
| oos | range_low_vol | 3 | 66.7 | 197.40 | -175.03 | 189.25 | -182.88 | FAIL |
| oos | neutral | 14 | 50.0 | 270.23 | -86.59 | 262.26 | -94.58 | FAIL |
| sanity | bullish | 17 | 17.6 | -140.93 | -262.85 | -148.82 | -270.63 | FAIL |
| sanity | bearish | 28 | 28.6 | 43.29 | -246.55 | 35.31 | -254.76 | FAIL |
| sanity | volatile_mixed | 2 | 50.0 | 5.68 | -97.62 | -2.32 | -105.70 | FAIL |
| sanity | range_low_vol | 3 | 33.3 | 297.10 | -481.85 | 288.85 | -489.48 | FAIL |
| sanity | neutral | 6 | 50.0 | 24.63 | -295.96 | 16.42 | -304.08 | FAIL |

A regime passes only with at least 10 trades and positive mean and LCB under both base and stress costs.
No regime result grants paper/live authority.
