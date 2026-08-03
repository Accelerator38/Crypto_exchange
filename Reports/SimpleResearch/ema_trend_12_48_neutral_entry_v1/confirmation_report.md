# ema_trend_12_48_neutral_entry_v1 evaluation

Generated: `2026-08-03T11:53:30.000326+00:00`

## Verdict

- Status: `FAILED_DIAGNOSTIC_REPLAY`.
- Passed: `false`.
- Evidence class: `diagnostic_replay_not_independent_confirmation`.
- Paper/live/orders/promotion authority remain `false`.

## Data integrity

- Confirmation window: `2026-07-15T00:00:00Z` to `2026-07-29T00:00:00Z` (end exclusive).
- Calendar days: `14`; required: `30`.
- 1m-to-1h overlap rows: `5760`; OHLC mismatches: `0`.

## Aggregate metrics

| Profile | Trades | Fills | Mean net bps | LCB bps | Adjusted LCB bps | Drawdown |
|---|---:|---:|---:|---:|---:|---:|
| candidate base | 53 | 106 | -56.451 | -87.225 | -114.377 | 0.0493 |
| candidate stress | 53 | 106 | -60.451 | -91.225 | -118.377 | 0.0511 |
| unfiltered EMA stress | 70 | 140 | -35.764 | -62.538 | -86.160 | 0.0412 |

## Time stability

| Half | Trades | Stress mean bps | LCB bps |
|---|---:|---:|---:|
| first | 27 | -26.837 | -80.877 |
| second | 29 | -86.723 | -123.990 |

## Stress metrics by symbol

| Symbol | Trades | Mean net bps | LCB bps |
|---|---:|---:|---:|
| ADA/USDT | 7 | -173.735 | -241.605 |
| BNB/USDT | 6 | -36.527 | -96.852 |
| BTC/USDT | 8 | -68.088 | -109.236 |
| DOGE/USDT | 6 | 24.642 | -72.871 |
| ETH/USDT | 6 | -74.525 | -215.579 |
| LINK/USDT | 6 | -81.626 | -211.504 |
| SOL/USDT | 6 | -15.511 | -103.571 |
| XRP/USDT | 8 | -42.723 | -97.899 |

## 24-hour moving-block bootstrap

- Observed stress total return: `-3.6157%`.
- 95% total-return LCB: `-6.9686%`.
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
