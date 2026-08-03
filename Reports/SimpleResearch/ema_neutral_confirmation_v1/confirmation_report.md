# ema_trend_12_48_neutral_v1 evaluation

Generated: `2026-08-03T11:51:47.697806+00:00`

## Verdict

- Status: `FAILED_SEALED_CONFIRMATION_EARLY`.
- Passed: `false`.
- Evidence class: `sealed_retrospective_holdback_not_prospective`.
- Paper/live/orders/promotion authority remain `false`.

## Data integrity

- Confirmation window: `2026-07-15T00:00:00Z` to `2026-07-29T00:00:00Z` (end exclusive).
- Calendar days: `14`; required: `30`.
- 1m-to-1h overlap rows: `5760`; OHLC mismatches: `0`.

## Aggregate metrics

| Profile | Trades | Fills | Mean net bps | LCB bps | Adjusted LCB bps | Drawdown |
|---|---:|---:|---:|---:|---:|---:|
| candidate base | 91 | 182 | -12.963 | -32.358 | -49.400 | 0.0301 |
| candidate stress | 91 | 182 | -16.963 | -36.358 | -53.400 | 0.0327 |
| unfiltered EMA stress | 70 | 140 | -35.764 | -62.538 | -86.065 | 0.0412 |

## Time stability

| Half | Trades | Stress mean bps | LCB bps |
|---|---:|---:|---:|
| first | 61 | -14.051 | -37.383 |
| second | 30 | -22.884 | -58.182 |

## Stress metrics by symbol

| Symbol | Trades | Mean net bps | LCB bps |
|---|---:|---:|---:|
| ADA/USDT | 11 | -27.184 | -98.682 |
| BNB/USDT | 11 | -22.353 | -57.572 |
| BTC/USDT | 10 | -39.892 | -99.863 |
| DOGE/USDT | 12 | -20.207 | -79.891 |
| ETH/USDT | 12 | -3.979 | -66.724 |
| LINK/USDT | 12 | -16.124 | -72.702 |
| SOL/USDT | 11 | -13.450 | -64.306 |
| XRP/USDT | 12 | 2.652 | -46.913 |

## 24-hour moving-block bootstrap

- Observed stress total return: `-1.9334%`.
- 95% total-return LCB: `-4.3181%`.
- Observations/nominal blocks: `336` / `14`.

## Gate

| Check | Passed |
|---|---|
| minimum_calendar_days | false |
| minimum_closed_trades | true |
| minimum_fills | true |
| minimum_symbols_with_trades | true |
| maximum_symbol_trade_share | true |
| positive_stress_mean_net_bps | false |
| positive_stress_lcb_95 | false |
| positive_selection_adjusted_lcb_95 | false |
| positive_stress_mean_in_both_time_halves | false |
| maximum_drawdown | true |
| positive_block_bootstrap_total_return_lcb | false |

No threshold may be changed under this candidate ID after this run.
