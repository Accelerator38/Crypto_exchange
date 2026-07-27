# CarryFlow root 003: signal quality analysis

Generated: `2026-07-27T08:26:53.248960+00:00`

This report is research-only. It does not authorize paper or live orders.

## Verdict

The root exposed two independent defects: activation is too rare, and `diagnostic.edge` is not calibrated as a forward return. The fixed six-bar exit then gives back favorable excursion.

## Activation funnel

| Stage | Count | Share |
|---|---:|---:|
| Flat observations | 268 | 100.0% |
| OI passed | 28 | 10.4% |
| Raw actor candidates | 6 | 2.2% |
| Actor selected | 5 | 1.9% |
| Policy allowed | 4 | 1.5% |

## OI gate

The threshold is 150 bps over three bars. 240 observations failed it.
Median failed OI change was -40.2 bps; p95 was 112.2 bps. Only 7 failures were within 25 bps of the threshold.

Lowering the threshold is not supported: most rejected observations are materially weaker, not borderline cases.

| Symbol | Flat | OI pass | Pass rate | Median, bps | p90, bps |
|---|---:|---:|---:|---:|---:|
| BTC | 36 | 0 | 0.0% | -24.9 | 53.0 |
| ETH | 36 | 2 | 5.6% | -8.7 | 138.5 |
| SOL | 28 | 9 | 32.1% | 9.3 | 242.3 |
| BNB | 30 | 3 | 10.0% | -65.0 | 124.0 |
| XRP | 36 | 3 | 8.3% | -47.5 | 117.7 |
| DOGE | 36 | 2 | 5.6% | -70.7 | 47.5 |
| ADA | 30 | 3 | 10.0% | 5.3 | 149.6 |
| LINK | 36 | 6 | 16.7% | -34.6 | 275.2 |

## Signal calibration

Mean predicted move: 220.79 bps.
Mean realized gross move: 62.09 bps; mean net: 50.12 bps.
Realized/predicted ratio: 28.12% on only 3 trades.

`diagnostic.edge` may rank candidates, but its current linear conversion to bps is not a valid expectancy estimate.

| Edge component | Mean contribution | Share |
|---|---:|---:|
| basis | 0.901 | 36.9% |
| crowding | 0.719 | 29.4% |
| funding | 0.182 | 7.5% |
| oi_excess | 0.113 | 4.6% |
| price_divergence | 0.528 | 21.6% |

Basis and crowding dominate the score; OI is mostly a binary gate and contributes little to ranking after the threshold is passed.

| Bar | Symbol | Regime | Policy | Predicted | 6h gross | 6h net |
|---:|---|---|---|---:|---:|---:|
| 4 | BNB | range_low_vol | ALLOW_OPEN | 162.4 | 8.3 | -3.7 |
| 11 | SOL | range_low_vol | ALLOW_OPEN | 250.0 | 56.0 | 44.0 |
| 20 | ADA | range_low_vol | ALLOW_OPEN | 250.0 | 122.0 | 110.1 |
| 29 | ETH | bullish | NO_TRADE | 157.6 | 4.2 | -7.8 |
| 34 | SOL | range_low_vol | ALLOW_OPEN | 232.6 | n/a | n/a |

All five allowed trades were in `range_low_vol`; the two bearish SOL candidates were blocked by low regime confidence.

## Exit behavior

All 3 closed trades used `max_holding`. Target: 240 bps; stop: 120 bps.
Target hits by six bars: 0; stop hits: 0. All 3 trades had MFE above modeled cost.
MFE/MAE use sealed hourly OHLC and are diagnostic excursions, not proof that a trailing exit could have filled at the extreme.

| Fixed exit | Observations | Mean net, bps | Positive |
|---:|---:|---:|---:|
| 1h | 4 | -6.33 | 25.0% |
| 2h | 4 | 43.94 | 75.0% |
| 3h | 3 | 57.80 | 100.0% |
| 4h | 3 | 45.32 | 100.0% |
| 6h | 3 | 50.12 | 66.7% |
| 8h | 3 | -10.93 | 33.3% |
| 12h | 3 | 82.76 | 100.0% |

The four-hour row is an in-sample hypothesis, not a selected policy.

## Next design constraints

- Do not lower the OI threshold based on this root.
- Separate candidate ranking score from calibrated expected move.
- Calibrate expected move on independent train/validation/OOS data.
- Evaluate one profile-level cost-aware exit against the fixed six-bar exit; do not expose independent live exit knobs.
- Treat the four-bar result as a hypothesis only because it was selected after observing five trades.
