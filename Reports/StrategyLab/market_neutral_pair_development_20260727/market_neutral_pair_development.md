# Market-neutral relative momentum 4h development screen

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
- Failures: `cost_stress_nonpositive_lcb`, `max_drawdown_exceeded`, `nonpositive_lcb`

## Pair metrics

| Candidate | Signals | Fills | Closed pairs | Mean bps | LCB bps | Stress mean | DD USD |
|---|---:|---:|---:|---:|---:|---:|---:|
| market_neutral_relative_momentum_4h_v1 | 684 | 386 | 96 | 13.0377 | -74.0583 | 7.0377 | 6.2940 |
| paired_raw_momentum_30d_v1 | 699 | 398 | 99 | 3.3564 | -92.4501 | n/a | 9.6724 |

Round-trip cost per leg: `12.0 bps`; total pair cost: `24.0 bps`.

## Leg diagnostics

| Direction | Legs | Mean net bps | LCB bps |
|---|---:|---:|---:|
| LONG | 96 | -32.9363 | -239.6299 |
| SHORT | 96 | 59.0117 | -164.7272 |

## Exit reasons

| Reason | Pairs | Mean net bps |
|---|---:|---:|
| max_holding | 55 | 244.0673 |
| pair_stop | 35 | -376.7528 |
| rank_inversion | 6 | 169.0442 |