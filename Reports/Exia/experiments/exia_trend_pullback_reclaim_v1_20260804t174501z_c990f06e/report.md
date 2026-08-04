# Exia experiment: exia_trend_pullback_reclaim_v1_20260804t174501z_c990f06e

Candidate: `exia_trend_pullback_reclaim_v1` version `1.0.0`.
Status: `DEVELOPMENT_SCORED`.

This is an offline Freqtrade experiment. It cannot authorize paper/live.

## Runs

| Window | Cost | Trades | Fills | Mean bps | Block LCB | Drawdown |
|---|---|---:|---:|---:|---:|---:|
| development | base | 398 | 796 | 28.88 | -13.81 | 0.0049 |
| development | stress | 398 | 796 | 20.87 | -21.22 | 0.0060 |

## Safety

- `orders_enabled=false`; `promotion_authority=false`.
- No dynamic candidate selection or parameter mutation occurred.
- Foundation zero trades are an engineering PASS, not alpha evidence.
