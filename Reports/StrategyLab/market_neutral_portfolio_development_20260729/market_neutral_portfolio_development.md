# Weekly top2/bottom2 market-neutral development screen

## Safety

- Stage: `development_only`
- Development run budget: `1`
- Parameter sweep allowed: `false`
- Validation/OOS/sanity opened: `false`
- Runtime actor created: `false`
- Paper/live allowed: `false`
- Orders enabled: `false`
- Promotion authority: `false`

## Verdict

- Result: `terminal_rejected_development`
- Continuation allowed: `false`
- Failures: `cost_stress_nonpositive_lcb`, `does_not_beat_baseline`, `max_drawdown_exceeded`, `nonpositive_lcb`

## Portfolio metrics

| Candidate | Signals | Fills | Closed portfolios | Mean bps | LCB bps | Stress mean | DD USD |
|---|---:|---:|---:|---:|---:|---:|---:|
| weekly_top2_bottom2_relative_momentum_4h_v1 | 97 | 772 | 96 | 50.8281 | -22.9922 | 44.8281 | 7.6533 |
| weekly_top2_bottom2_raw_momentum_v1 | 99 | 788 | 98 | 71.8294 | -3.9921 | n/a | 7.8464 |

Long notional: `$20.0`; short notional: `$20.0`; net: `$0.0`.

## Leg diagnostics

| Direction | Legs | Mean net bps | LCB bps |
|---|---:|---:|---:|
| LONG | 192 | 60.7476 | -105.4253 |
| SHORT | 192 | 40.9086 | -99.2339 |