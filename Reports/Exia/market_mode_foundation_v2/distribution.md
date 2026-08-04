# Exia market mode foundation v2

Generated: `2026-08-04T18:06:16.058175+00:00`

This report validates state coverage and restart parity only. It contains no alpha claim and cannot authorize paper/live.

## Contract

- Four local states: `TREND_UP`, `TREND_DOWN`, `RANGE`, `UNSAFE`.
- Volatility threshold uses a fixed 720-hour causal window.
- Runtime requests 999 startup bars; state age is capped at 168 bars.
- Entries remain hard-coded to zero.

Status: `FOUNDATION_READY_FOR_CANDIDATE_DESIGN`.

## Window distribution

| Window | Rows | Up % | Down % | Range % | Unsafe % | Median episode h |
|---|---:|---:|---:|---:|---:|---:|
| development | 140160 | 23.6 | 26.7 | 39.4 | 10.3 | 4.0 |
| validation | 69696 | 27.9 | 24.6 | 40.9 | 6.6 | 3.0 |
| oos | 69504 | 24.9 | 27.3 | 42.0 | 5.8 | 3.0 |
| sanity | 36864 | 22.5 | 30.2 | 41.0 | 6.2 | 3.0 |

## Restart parity

| Symbol | Checkpoints | Mode mismatches | Previous-mode mismatches | State-age mismatches | Result |
|---|---:|---:|---:|---:|---|
| ADA/USDT | 55 | 0 | 0 | 0 | PASS |
| BNB/USDT | 55 | 0 | 0 | 0 | PASS |
| BTC/USDT | 55 | 0 | 0 | 0 | PASS |
| DOGE/USDT | 55 | 0 | 0 | 0 | PASS |
| ETH/USDT | 55 | 0 | 0 | 0 | PASS |
| LINK/USDT | 55 | 0 | 0 | 0 | PASS |
| SOL/USDT | 55 | 0 | 0 | 0 | PASS |
| XRP/USDT | 55 | 0 | 0 | 0 | PASS |

## Safety

- `paper_allowed=false`; `live_allowed=false`.
- `orders_enabled=false`; `promotion_authority=false`.
- A passing foundation permits new candidate design only.
