# Genetic_Primus V6

## Final status

`V6_TERMINAL_REJECTED_NO_EVENT_CAPACITY`.

V6 repaired the historical order-flow coverage problem that stopped V5. It
then tested the economic capacity of three preregistered derivatives events
before opening a simulator, baseline or neurogenetic search. None passed the
fixed stress-costed walk-forward gates, so V6 stopped before evolution.

## Immutable contract

The contract is `configs/genetic_primus_v6_contract.json`, SHA-256:

`2746fb1e3ab0119e8076e856415964f651149123e059b1a8a518f7db78d15b92`

It fixes the full8 universe, 2022-2024 research window, four-hour decisions,
next-open execution, six-bar holding period, 24 bps round-trip stress cost,
four temporal evaluation folds, event definitions and terminal kill criteria.
Threshold extension and fallback selection are forbidden.

## Data result

V6 combines the V5 Binance USD-M open-interest/account-ratio dataset with
monthly four-hour Binance USD-M kline taker flow.

| Metric | Result |
|---|---:|
| Monthly archives | 288 / 288 |
| Failed archives | 0 |
| Four-hour flow rows | 52,548 |
| Combined rows | 52,592 |
| Raw file coverage | 100.00% |
| Minimum joined flow coverage | 99.54% |
| Core-feature gate | PASS |

Dataset SHA-256:

`137bc104159dcd25f5d334299e2229d80be7c3a3e4da13519d409f7d2f4d4953`

The two March 2022 archives without a header were accepted only after strict
validation as numeric, headerless, exact 12-column Binance kline records. This
repair is recorded in the raw catalog.

## Event result

All returns use next-bar-open entry, sixth-bar-next-open exit and the fixed
24 bps cost. LCB uses a seven-day contiguous-block bootstrap with Bonferroni
alpha across the three events.

| Event | Events | Mean bps | Median bps | LCB bps | Positive folds | Worst fold mean bps | Passed |
|---|---:|---:|---:|---:|---:|---:|---|
| OI + taker + price continuation | 774 | -49.73 | -56.97 | -95.01 | 1 / 4 | -157.01 | No |
| Crowding unwind reversal | 64 | -0.37 | -28.44 | -172.82 | 3 / 4 | -102.00 | No |
| OI/price divergence reversal | 862 | -28.37 | 20.36 | -74.91 | 1 / 4 | -77.77 | No |

The divergence event has a positive overall median, but its mean, LCB, worst
fold and temporal consistency are negative. It is not a viable candidate.
The crowding event is also too sparse and statistically uncertain.

## Consequence

The source was sufficient; the preregistered signals were not. Therefore the
following V5/V6 roadmap stages remain intentionally unopened:

- bounded position simulator and cost overlays;
- logistic baseline;
- sparse neurogenetic ranker and three-seed screen;
- 2025 validation, 2026 OOS and paper campaign;
- live trading and order submission.

`tools/run_genetic_primus_v6.py --stage development` fails closed against the
terminal event manifest. Reusing the opened V6 labels to alter thresholds would
be post-selection. Any materially different event family requires a new frozen
contract and a new experiment version.

## Artifacts

- Dataset: `Retrodate/genetic_primus_v6_oi_kline_flow_v1_20220101_20241231`
- Data gate: `Reports/Exia/Genetic_Primus/v6_data_source_gate_20260824`
- Event gate: `Reports/Exia/Genetic_Primus/v6_event_feasibility_20260824`
- Stage runner: `tools/run_genetic_primus_v6.py`
