# Cross-sectional trend 4h development screen

## Safety

- Stage: `development_only`
- Validation/OOS/sanity opened: `false`
- Runtime actor created: `false`
- Paper/live allowed: `false`
- Orders enabled: `false`
- Promotion authority: `false`

## Verdict

- Result: `terminal_rejected_development`
- Continuation allowed: `false`
- Failures: `cost_stress_nonpositive_lcb`, `direction_collapse:LONG`, `direction_collapse:SHORT`, `does_not_beat_baseline`, `max_drawdown_exceeded`, `nonpositive_lcb`

## Metrics

| Candidate | Signals | Fills | Closed | Mean bps | LCB bps | Stress mean | DD USD |
|---|---:|---:|---:|---:|---:|---:|---:|
| cross_sectional_trend_4h_v1 | 738 | 383 | 191 | 48.2778 | -54.5255 | 42.2778 | 3.9487 |
| donchian_42_4h_atr_exit_v1 | 1821 | 254 | 127 | 111.1662 | -86.4441 | n/a | 4.8557 |

## Direction

| Direction | Trades | Mean net bps | LCB bps |
|---|---:|---:|---:|
| LONG | 96 | -28.2454 | -165.7042 |
| SHORT | 95 | 125.6066 | -26.5957 |