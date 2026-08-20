# exia_range_band_reentry_family_v1 - development decision

Generated: `2026-08-05T07:26:01.344650+00:00`

This is a development-only research decision. It cannot authorize paper or live trading.

Status: `NO_CANDIDATE_FOR_VALIDATION`.

Taxonomy runtime status: `PASS_RESTART_STABLE`.

## Overall development stress-cost results

| Candidate | Trades | Fills | Mean net bps | Block LCB bps | Baseline diff LCB bps | Drawdown | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| exia_range_band_reentry_v1 | 481 | 962 | -18.973 | -31.584 | -32.409 | 0.008483 | FAIL |

## Development market-state breakdown

All four market states are shown for base and stress costs. These rows are diagnostic and cannot override the full-window gate.

| Candidate | Cost | Market state | Trades | Fills | Mean net bps | Block LCB bps | Baseline diff LCB bps | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| exia_range_band_reentry_v1 | base | TREND_UP | 0 | 0 | n/a | n/a | n/a | INSUFFICIENT |
| exia_range_band_reentry_v1 | base | TREND_DOWN | 0 | 0 | n/a | n/a | n/a | INSUFFICIENT |
| exia_range_band_reentry_v1 | base | RANGE | 481 | 962 | -10.972 | -23.616 | -24.108 | FAIL |
| exia_range_band_reentry_v1 | base | UNSAFE | 0 | 0 | n/a | n/a | n/a | INSUFFICIENT |
| exia_range_band_reentry_v1 | stress | TREND_UP | 0 | 0 | n/a | n/a | n/a | INSUFFICIENT |
| exia_range_band_reentry_v1 | stress | TREND_DOWN | 0 | 0 | n/a | n/a | n/a | INSUFFICIENT |
| exia_range_band_reentry_v1 | stress | RANGE | 481 | 962 | -18.973 | -32.856 | -32.108 | FAIL |
| exia_range_band_reentry_v1 | stress | UNSAFE | 0 | 0 | n/a | n/a | n/a | INSUFFICIENT |

## Exploratory positive subscopes

These rows are diagnostic only because they were observed after comparing the full family.

| Candidate | Scope | Value | Trades | Mean net bps | Block LCB bps | Baseline diff LCB bps |
|---|---|---|---:|---:|---:|---:|
| none | - | - | 0 | n/a | n/a | n/a |

## Decision

No validation, OOS, paper, or live stage is opened unless a candidate passes every preregistered development gate.

Selected for validation: `none`.

Terminally rejected on development alpha: `exia_range_band_reentry_v1`.

All order and promotion authority flags remain false.
