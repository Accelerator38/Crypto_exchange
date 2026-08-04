# exia_trend_family_v1 - development decision

Generated: `2026-08-04T18:21:28.426391+00:00`

This is a development-only research decision. It cannot authorize paper or live trading.

Status: `NO_CANDIDATE_FOR_VALIDATION`.

Taxonomy runtime status: `REJECTED_RESTART_DEPENDENT`.

## Full8 stress-cost results

| Candidate | Trades | Fills | Mean net bps | Block LCB bps | Baseline diff LCB bps | Drawdown | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| exia_trend_state_onset_v1 | 440 | 880 | -3.151 | -40.312 | -38.586 | 0.008942 | FAIL |
| exia_trend_donchian48_v1 | 293 | 586 | -26.720 | -75.866 | -77.472 | 0.013149 | FAIL |
| exia_trend_pullback_reclaim_v1 | 398 | 796 | 20.874 | -21.216 | -20.017 | 0.006007 | FAIL |

## Exploratory positive subscopes

These rows are diagnostic only because they were observed after comparing the full family.

| Candidate | Scope | Value | Trades | Mean net bps | Block LCB bps | Baseline diff LCB bps |
|---|---|---|---:|---:|---:|---:|
| exia_trend_state_onset_v1 | symbol | ADA | 48 | 142.945 | 5.552 | 11.817 |
| exia_trend_donchian48_v1 | symbol | ADA | 35 | 234.790 | 12.650 | 10.931 |

## Decision

No validation, OOS, paper, or live stage is opened unless a candidate passes every preregistered development gate.

Selected for validation: `none`.

All order and promotion authority flags remain false.
