# Funding Carry recent diagnostic screen

## Verdict

- Result: `terminal_rejected_recent_activation_and_cost_floor`
- Continuation allowed: `false`
- Registered historical OOS passed: `false`
- Paper/live allowed: `false`
- Orders enabled: `false`
- Promotion authority: `false`

The public Bitget funding history window does not cover the registered validation/OOS/sanity splits. This screen is diagnostic only.

## Evidence

- Funding dataset SHA: `f97b9ea582a7cd10bd5fccce482b8321c6ce97008bb9b18e52e806e7aea145b1`
- Funding manifest SHA: `a828259d3afb987008877f930f212e2bb6a4a32bf7c2a582eb3b749ecbcdb9f8`
- Allowed use: `recent_diagnostic_screen_only`
- Overlap bars: `1856`
- Candidate signals: `0`
- Filled orders: `0`
- Closed trades: `0`
- Mean net expectancy: `n/a bps`
- LCB 95: `n/a bps`
- Stress mean: `n/a bps`
- Max drawdown: `0.0000 USD`
- Cost floor: `16.0000 bps`
- Max projected carry: `7.3500 bps`

## Activation funnel

| Stage | Observations |
|---|---:|
| settlements | 1856 |
| history_ready | 1840 |
| sign_persistent | 1004 |
| last_rate_pass | 0 |
| median_rate_pass | 0 |
| projected_carry_pass | 0 |
| trend_guard_pass | 0 |
| signals | 0 |

First rejection reasons: `funding_history_short=16`, `last_below_threshold=1004`, `sign_not_persistent=836`

## Per symbol

| Symbol | Trades | Mean net bps | LCB bps |
|---|---:|---:|---:|