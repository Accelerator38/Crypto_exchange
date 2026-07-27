# Range-transition breakout audit

## Verdict

**ACTIVATION_OR_EXPECTANCY_FAILURE**

| Root | Symbol transitions | Raw events | Selected events | Mean net, bps |
|---|---:|---:|---:|---:|
| v1 | 15 | 5 | 3 | 32.72 |
| v4 | 3 | 0 | 0 | n/a |
| v5 | 6 | 2 | 2 | -29.88 |
| root003 | 9 | 3 | 3 | 18.49 |

Aggregate: 8 trades, 11.74 mean net bps, -15.82 LCB bps.

Direction metrics:

| Direction | Trades | Mean net, bps | LCB, bps |
|---|---:|---:|---:|
| LONG | 4 | 12.60 | -38.01 |
| SHORT | 4 | 10.87 | -20.45 |

Failures: closed_trades_below_10, nonpositive_lcb, inactive_root:v4, root_expectancy_collapse:v5

No runtime actor, paper order or live order is created.
