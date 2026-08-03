# Walk-forward cointegration spread 4h development screen

## Safety

- Stage: `development_only`
- Development run budget: `1`
- Parameter sweep allowed: `false`
- Symbol allowlist: `none`
- Validation/OOS/sanity opened: `false`
- Runtime actor created: `false`
- Paper/live allowed: `false`

## Verdict

- Result: `terminal_rejected_development`
- Continuation allowed: `false`
- Failures: `cost_stress_nonpositive_expectancy`, `cost_stress_nonpositive_lcb`, `direction_collapse:LONG_SPREAD`, `direction_collapse:SHORT_SPREAD`, `max_drawdown_exceeded`, `nonpositive_lcb`

## Metrics

| Signals | Fills | Closed | Mean bps | LCB bps | Stress mean | DD USD |
|---:|---:|---:|---:|---:|---:|---:|
| 383 | 376 | 94 | 2.5739 | -27.6179 | -3.4261 | 3.0926 |

## Formation

- Refits: `99`
- Qualified refits: `78`
- Tested pairs: `2772`
- Leakage violations: `0`
- Maximum single-pair trade share: `15.96%`

## Direction

| Direction | Trades | Mean net bps | LCB bps |
|---|---:|---:|---:|
| LONG_SPREAD | 46 | 2.8732 | -35.7157 |
| SHORT_SPREAD | 48 | 2.2871 | -44.2521 |
