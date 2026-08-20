# Exia 4h sparse-event forward feasibility v1

## Decision

The aggregate experiment is closed with `NO_EVENT_HORIZON_PASSED_STAGED_GATE`.

None of the 12 alternative event/horizon combinations opened validation, OOS or sanity. The broad full8 long-trend event family is rejected. One development-only symbol slice is retained for a separate, fixed validation replication; it has no paper or live authority.

## Contract

- Event definitions are inherited by SHA from `exia_4h_sparse_event_family_v1`.
- Horizons are fixed at 1, 3, 6 and 12 bars.
- Entry is the open immediately after the completed event bar.
- Terminal price is the open after the fixed holding horizon.
- MFE and MAE use bars from entry through the bar before terminal open.
- Events whose terminal lies outside the evaluated window are excluded.
- Stress round-trip cost is 16 bps.
- Overlapping labels are retained, while bootstrap resamples complete UTC-day clusters.
- There is no stop, trailing rule or learned exit in this audit.

## Aggregate result

No alternative passed positive stress mean, median, ordinary LCB and family-wise LCB simultaneously.

The closest aggregate combination was `BULLISH_REGIME_TRANSITION + H1`:

| Observations | Mean bps | Median bps | LCB bps | Family LCB bps | Positive rate | Median MFE | Median MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 605 | 28.05 | -7.17 | 7.59 | -4.32 | 46.78% | 94.76 | 69.68 |

Its ordinary LCB is positive, but the median and 16-trial family-wise LCB are negative. This is weak development evidence, not a candidate.

The best aggregate mean was `TOP_QUARTILE_BREAKOUT20 + H12` at `+45.10 bps`, but its median was `-35.40 bps`, LCB `-18.80 bps` and family-wise LCB `-56.47 bps`.

## Information versus execution

MFE stress-surplus LCB is positive for the leading combinations, so price often moves far enough intrahorizon to cover costs. Fixed terminal returns remain unstable or negatively centered. This means the data contains path opportunity but the current event does not predict which favorable excursion will persist to a fixed terminal time.

It would be incorrect to use this observation to start another exit-parameter grid. The broad candidates have already failed both fixed exits and fixed horizons.

## Retained replication candidate

The only exploratory symbol slice with positive mean, median, ordinary LCB and reported slice-family LCB was:

| Event | Symbol | Horizon | Observations | Mean bps | Median bps | LCB bps | Slice-family LCB bps |
|---|---|---:|---:|---:|---:|---:|---:|
| BULL_TREND_FAST_RECLAIM | SOL/USDT | 12 bars | 34 | 394.22 | 239.96 | 168.68 | 13.54 |

This slice was discovered after inspecting development results and therefore cannot inherit the aggregate experiment's stage progression. It must be copied unchanged into a new single-candidate replication specification and evaluated first on the already sealed validation window. Its symbol, event, horizon, costs and thresholds must not be altered after preregistration.

All other event/horizon/symbol combinations should be retired from the current long-trend branch. If the retained SOL slice fails validation, the entire 4h long-trend branch should be closed rather than retuned.

## Artifacts

- Specification: `configs/exia_4h_forward_feasibility_v1.json`.
- Labels: `src/simple_research/forward_labels.py`.
- Runner: `tools/run_exia_4h_forward_feasibility_v1.py`.
- Authoritative run: `Reports/Exia/four_hour_forward_feasibility_v1_authoritative`.
- Deterministic repeat: `Reports/Exia/four_hour_forward_feasibility_v1_authoritative_repeat`.
- Exact repeat equality: metrics, observations, symbol metrics and regime metrics.

All paper, live, order and promotion flags remain `false`.
