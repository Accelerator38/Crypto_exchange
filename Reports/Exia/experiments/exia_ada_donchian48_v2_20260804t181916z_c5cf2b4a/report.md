# Exia experiment: exia_ada_donchian48_v2_20260804t181916z_c5cf2b4a

Candidate: `exia_ada_donchian48_v2` version `2.0.0`.
Status: `DEVELOPMENT_SCORED`.

This is an offline Freqtrade experiment. It cannot authorize paper/live.

## Runs

| Window | Cost | Trades | Fills | Mean bps | Block LCB | Drawdown |
|---|---|---:|---:|---:|---:|---:|
| development | base | 180 | 360 | -1.19 | -69.36 | 0.0081 |
| development | stress | 180 | 360 | -9.19 | -80.01 | 0.0093 |

## Safety

- `orders_enabled=false`; `promotion_authority=false`.
- No dynamic candidate selection or parameter mutation occurred.
- Foundation zero trades are an engineering PASS, not alpha evidence.
