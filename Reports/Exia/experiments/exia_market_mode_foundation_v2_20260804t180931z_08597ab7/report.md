# Exia experiment: exia_market_mode_foundation_v2_20260804t180931z_08597ab7

Candidate: `exia_market_mode_foundation_v2` version `2.0.0`.
Status: `ENGINEERING_PASS_NO_TRADES`.

This is an offline Freqtrade experiment. It cannot authorize paper/live.

## Runs

| Window | Cost | Trades | Fills | Mean bps | Block LCB | Drawdown |
|---|---|---:|---:|---:|---:|---:|
| development | base | 0 | 0 | n/a | n/a | 0.0000 |
| development | stress | 0 | 0 | n/a | n/a | 0.0000 |
| validation | base | 0 | 0 | n/a | n/a | 0.0000 |
| validation | stress | 0 | 0 | n/a | n/a | 0.0000 |
| oos | base | 0 | 0 | n/a | n/a | 0.0000 |
| oos | stress | 0 | 0 | n/a | n/a | 0.0000 |
| sanity | base | 0 | 0 | n/a | n/a | 0.0000 |
| sanity | stress | 0 | 0 | n/a | n/a | 0.0000 |

## Safety

- `orders_enabled=false`; `promotion_authority=false`.
- No dynamic candidate selection or parameter mutation occurred.
- Foundation zero trades are an engineering PASS, not alpha evidence.
