# Exia market mode foundation v1

Generated: `2026-08-04T15:17:40.567108+00:00`

This report validates state coverage only. It contains no alpha claim and cannot authorize paper/live.

## Contract

- `TREND_UP`: close > EMA24 > EMA96 and positive 24h momentum.
- `TREND_DOWN`: close < EMA24 < EMA96 and negative 24h momentum.
- `RANGE`: valid non-shock observation without an aligned trend.
- `UNSAFE`: incomplete warm-up, invalid cadence/data, or volatility shock.

Status: `FOUNDATION_READY_FOR_CANDIDATE_DESIGN`.

## Window distribution

| Window | Rows | Up % | Down % | Range % | Unsafe % | Median episode h |
|---|---:|---:|---:|---:|---:|---:|
| development | 140160 | 24.4 | 27.5 | 39.6 | 8.5 | 4.0 |
| validation | 69696 | 28.0 | 24.6 | 40.2 | 7.2 | 3.0 |
| oos | 69504 | 25.7 | 27.4 | 42.3 | 4.6 | 3.0 |
| sanity | 36864 | 22.9 | 31.3 | 41.6 | 4.2 | 3.0 |

## Symbol distribution

| Symbol | Rows | Up % | Down % | Range % | Unsafe % | Shocks |
|---|---:|---:|---:|---:|---:|---:|
| ADA/USDT | 39744 | 22.7 | 30.3 | 40.2 | 6.8 | 1964 |
| BNB/USDT | 39744 | 28.2 | 23.8 | 41.0 | 7.1 | 2058 |
| BTC/USDT | 39744 | 27.7 | 25.1 | 40.1 | 7.2 | 2106 |
| DOGE/USDT | 39744 | 22.5 | 29.1 | 41.6 | 6.8 | 1943 |
| ETH/USDT | 39744 | 26.7 | 25.4 | 40.8 | 7.0 | 2040 |
| LINK/USDT | 39744 | 25.8 | 27.7 | 39.9 | 6.6 | 1866 |
| SOL/USDT | 39744 | 26.1 | 28.2 | 39.1 | 6.5 | 1846 |
| XRP/USDT | 39744 | 23.7 | 27.9 | 41.5 | 6.9 | 2001 |

## Safety

- Strategy entries are hard-coded to zero.
- `orders_enabled=false`; `promotion_authority=false`.
- The next permitted step is candidate design, not paper/live.
