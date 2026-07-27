# CarryFlow fixed-exit hypothesis

## Verdict

**REJECTED**

`hold=4` was selected on `root003` and evaluated without retuning on the remaining independent roots.

| Root | Role | Observations | Hold 4 net, bps | Hold 6 net, bps |
|---|---|---:|---:|---:|
| root003 | selection | 5 | 26.95 | -4.33 |
| v1 | evaluation | 4 | -64.59 | -46.35 |
| v4 | evaluation | 3 | 45.32 | 50.12 |
| v5 | evaluation | 4 | -19.23 | 5.54 |

## Independent evaluation

- Observations: 11
- Candidate mean net: -18.12 bps
- Baseline mean net: -1.17 bps
- Root collapses: v1, v5

## Expected-move feature screen

- Observations: 16 (minimum 20)
- Stable positive features: none

| Feature | Pooled correlation | Sign positive on every root |
|---|---:|---:|
| ranking_score | 0.13 | false |
| price_dislocation_bps | 0.16 | false |
| oi_excess_bps | 0.10 | false |

`hold=4` is not registered as a policy profile when this report is rejected. The existing six-bar profile is also not promoted by this comparison.

This report is R&D-only and cannot authorize paper or live orders.
