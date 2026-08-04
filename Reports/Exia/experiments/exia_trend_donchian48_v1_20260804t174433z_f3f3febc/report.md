# Exia experiment: exia_trend_donchian48_v1_20260804t174433z_f3f3febc

Candidate: `exia_trend_donchian48_v1` version `1.0.0`.
Status: `DEVELOPMENT_SCORED`.

This is an offline Freqtrade experiment. It cannot authorize paper/live.

## Runs

| Window | Cost | Trades | Fills | Mean bps | Block LCB | Drawdown |
|---|---|---:|---:|---:|---:|---:|
| development | base | 293 | 586 | -18.71 | -68.28 | 0.0118 |
| development | stress | 293 | 586 | -26.72 | -75.87 | 0.0131 |

## Safety

- `orders_enabled=false`; `promotion_authority=false`.
- No dynamic candidate selection or parameter mutation occurred.
- Foundation zero trades are an engineering PASS, not alpha evidence.
