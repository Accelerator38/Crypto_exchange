# CarryFlow forward-label entry evaluation

## Verdict

**REJECTED**

Rows: 209; selected held-out trades: 22; mean net: -1.42 bps; LCB: -39.66 bps.

| Held-out root | Test rows | Trades | Mean net, bps | LCB, bps |
|---|---:|---:|---:|---:|
| root003 | 80 | 7 | 11.27 | -88.98 |
| v1 | 56 | 7 | 11.86 | -38.03 |
| v4 | 20 | 3 | 58.36 | 24.17 |
| v5 | 53 | 5 | -73.64 | -115.99 |

Failures: nonpositive_costed_expectancy, nonpositive_lcb, root_expectancy_collapse:v5

The model is trained only on other roots for every row. Labels are non-overlapping per symbol and include modeled costs plus the profile stop/target.

No runtime profile, paper order or live order is created.
