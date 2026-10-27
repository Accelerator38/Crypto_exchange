# Genetic_Primus V4

## Status

`DEVELOPMENT_V4_REJECTED_STRICT_GATES`.

V4 is a research-only replacement for the terminally rejected V3 experiment.
It does not authorize validation, OOS, paper, live or orders.

## What changed

- Revealed research corpus expanded from 2022-2023 to 2022-2024.
- 2024 is no longer called validation; it is disclosed V4 development data.
- New sealed validation is 2025. It remains unopened.
- Fitness uses compounded full8 portfolio daily returns, not individual trades.
- Temporal evidence is six fixed half-year folds.
- Screen compares `linear`, `compact_8` and `compact_16_8` across seeds
  `141`, `242`, `343`.
- Architecture selection uses all seeds and worst-fold robustness.
- Final research run uses independent seed `444`.
- Costs are tested at 16 and 24 bps round trip.

## V3 generalization audit

The 2024 failure was not explained by simple input drift: every feature had
PSI below 0.10. Changing only the exit did not repair the entries. The best
2024 counterfactual was fixed six-bar holding with LCB `-30.29 bps`.

## Architecture screen

No seed passed the temporal/risk contract.

| Architecture | Parameters | Passing seeds | Minimum positive folds | Minimum worst-fold mean | Median fold LCB | Maximum DD |
|---|---:|---:|---:|---:|---:|---:|
| linear | 93 | 0/3 | 2/6 | -28.48 | -12.82 | 55.57% |
| compact_8 | 275 | 0/3 | 4/6 | -38.06 | -19.96 | 63.01% |
| compact_16_8 | 659 | 0/3 | 4/6 | -22.86 | -19.51 | 66.57% |

The preregistered fallback selected `compact_16_8` for one independent final
research run because its worst seed/fold was the least negative among the two
architectures with the same minimum positive-fold count. This selection was
not a development PASS.

## Final strict freeze

| Cost | Trades | Mean daily | Median daily | Overall LCB | Positive folds | Worst-fold mean | Median fold LCB | DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 bps | 942 | +14.60 | +10.39 | -0.62 | 5/6 | -27.44 | -17.57 | 53.91% |
| 24 bps | 949 | +14.08 | +9.12 | -1.12 | 5/6 | -28.31 | -16.84 | 54.13% |

Failed gates at both costs:

- maximum drawdown;
- nonpositive overall portfolio LCB;
- nonpositive worst-fold mean;
- nonpositive median fold LCB.

The post-freeze incidence audit found another methodological failure. Both
`2024_H1` and `2024_H2` contain portfolio mark-to-market returns from positions
opened earlier, but zero new entries. Those folds cannot be treated as evidence
that the entry policy generalized into 2024. This does not change the frozen
result; it makes the rejection stronger and must become a per-fold incidence
gate in the next training design.

The frozen model has `validation_allowed=false`. The one-time stage runner
rejects any attempt to open 2025 unless the development freeze status is PASS.

## Three-day measurement foundation

The next adaptive research protocol targets a three-day deployment horizon.
Its immutable measurement contract is
`configs/genetic_primus_measurement_3d_v1.json` and does not reopen the frozen
V4 result.

- supervised forward labels are purged by seven 4h bars before the training
  cutoff, so 1/3/6-bar targets cannot consume post-cutoff prices;
- every genome in a run uses the same materialized selection bootstrap indices;
- the top selection finalists are re-evaluated on a separate materialized audit
  bank with an independent seed, and audit results are not used for selection;
- manifests pin the dataset and dataset-manifest SHA, fold SHA, measurement
  contract SHA, cutoff, purge cutoff, selection/audit bank content SHA and bank
  artifact SHA;
- the protocol remains research-only with paper, live, orders and promotion
  authority disabled.
