# Strategy Lab V1 historical evaluation

## Safety

- Orders enabled: `false`
- Promotion authority: `false`
- Runtime actor created: `false`

## Candidates

| Candidate | Verdict | Trades | Evidence mean bps | LCB bps | Stress mean bps | Failures |
|---|---|---:|---:|---:|---:|---|
| regime_pullback_hourly_v1 | terminal_rejected_historical_oos | 1977 | -4.3569 | -18.5531 | -10.3569 | cost_stress_nonpositive_expectancy, cost_stress_nonpositive_lcb, direction_collapse:LONG, direction_collapse:SHORT, max_drawdown_exceeded, nonpositive_costed_expectancy, nonpositive_lcb, promotable_symbols_below_2, regime_collapse:bearish, regime_collapse:bullish, root_expectancy_collapse:oos, root_expectancy_collapse:sanity, root_lcb_collapse:oos, root_lcb_collapse:sanity, root_lcb_collapse:validation |
| ohlcv_compression_transition_hourly_v1 | terminal_rejected_historical_oos | 1741 | -3.9167 | -14.5651 | -9.9167 | cost_stress_nonpositive_expectancy, cost_stress_nonpositive_lcb, direction_collapse:LONG, direction_collapse:SHORT, max_drawdown_exceeded, nonpositive_costed_expectancy, nonpositive_lcb, promotable_symbols_below_2, regime_collapse:compression_transition, root_expectancy_collapse:oos, root_expectancy_collapse:sanity, root_lcb_collapse:oos, root_lcb_collapse:sanity, root_lcb_collapse:validation |

## Split metrics

### regime_pullback_hourly_v1

| Split | Signals | Fills | Closed | Mean bps | LCB bps | DD USD |
|---|---:|---:|---:|---:|---:|---:|
| development | 3379 | 1797 | 898 | -16.0290 | -33.8011 | 17.4900 |
| validation | 1716 | 905 | 452 | 16.1237 | -8.4944 | 4.9099 |
| oos | 1645 | 809 | 404 | -19.9363 | -42.0023 | 8.3678 |
| sanity | 897 | 446 | 223 | -17.6448 | -42.3998 | 5.0466 |

### ohlcv_compression_transition_hourly_v1

| Split | Signals | Fills | Closed | Mean bps | LCB bps | DD USD |
|---|---:|---:|---:|---:|---:|---:|
| development | 1775 | 1563 | 781 | -2.0798 | -14.1055 | 3.7143 |
| validation | 984 | 778 | 389 | 2.2537 | -16.0053 | 4.1206 |
| oos | 1078 | 770 | 385 | -9.8078 | -26.3720 | 4.6144 |
| sanity | 624 | 372 | 186 | -4.6274 | -24.3602 | 2.4669 |

## Data-blocked

- `funding_carry_hourly_v1`: missing `funding_settlement_rate`.
