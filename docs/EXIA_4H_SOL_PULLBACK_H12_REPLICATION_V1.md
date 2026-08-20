# Exia 4h SOL pullback H12 validation replication v1

## Terminal decision

`VALIDATION_REPLICATION_FAILED`.

`CLOSE_4H_LONG_TREND_BRANCH`.

The selected development slice did not replicate. OOS and sanity were not opened. This result grants no paper or live authority.

## Fixed candidate

| Field | Value |
|---|---|
| Symbol | SOL/USDT |
| Event | BULL_TREND_FAST_RECLAIM |
| Direction | LONG |
| Horizon | 12 x 4h = 48 hours |
| Entry | Open after the completed event bar |
| Exit | Open after 12 bars |
| Stress round-trip cost | 16 bps |
| Minimum observations | 20 |
| Minimum label coverage | 95% |

Development evidence, its feasibility specification and symbol-metrics ledger were pinned by SHA before validation was read.

## Validation result

| Events | Labels | Coverage | Mean bps | Median bps | LCB bps | Positive rate | Median MFE | Median MAE | P95 MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 20 | 100% | 57.57 | -75.26 | -190.24 | 30% | 307.24 | 560.82 | 2122.15 |

The positive mean is produced by a small number of large winners. Fourteen of twenty stress-costed outcomes were non-positive, the median is negative and the clustered LCB is deeply negative. Adverse excursion is materially larger than favorable excursion for the typical observation.

The candidate fails positive median and LCB requirements. Positive MFE surplus does not change the decision because it is an oracle path statistic rather than an executable terminal return.

## Consequence

This was the only retained symbol/event/horizon slice after the aggregate 4h long-trend feasibility audit. Its independent validation failure closes the complete branch:

- no further EMA, breakout or pullback threshold tuning on these windows;
- no alternate ATR or trailing-stop search for these events;
- no per-symbol rescue using the already inspected development and validation periods;
- no OOS, sanity, paper or live promotion.

The next research branch must use a structurally different return source rather than another variation of directional 4h trend timing. Suitable bounded directions are market-neutral cross-sectional relative value or exchange-native carry/basis, provided their required data is available and sealed before candidate construction.

## Artifacts

- Specification: `configs/exia_4h_sol_pullback_h12_replication_v1.json`.
- Runner: `tools/run_exia_4h_sol_pullback_h12_replication_v1.py`.
- Authoritative run: `Reports/Exia/four_hour_sol_pullback_h12_replication_v1_authoritative`.
- Deterministic repeat: `Reports/Exia/four_hour_sol_pullback_h12_replication_v1_authoritative_repeat`.
- Exact repeat equality: metrics and observations.

All paper, live, order and promotion flags remain `false`.
