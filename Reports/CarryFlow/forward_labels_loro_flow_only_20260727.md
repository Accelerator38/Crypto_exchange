# CarryFlow forward-label entry evaluation

## Verdict

**REJECTED**

Rows: 249; selected held-out trades: 27; mean net: -0.36 bps; LCB: -30.37 bps.

| Held-out root | Test rows | Trades | Mean net, bps | LCB, bps |
|---|---:|---:|---:|---:|
| root003 | 80 | 9 | 31.30 | -24.75 |
| v1 | 68 | 4 | -29.86 | -104.82 |
| v4 | 32 | 4 | 75.25 | -10.46 |
| v5 | 69 | 10 | -47.29 | -79.79 |

Failures: nonpositive_costed_expectancy, nonpositive_lcb, root_expectancy_collapse:v1, root_expectancy_collapse:v5

The model is trained only on other roots for every row. Labels are non-overlapping per symbol and include modeled costs plus the profile stop/target.

No runtime profile, paper order or live order is created.
