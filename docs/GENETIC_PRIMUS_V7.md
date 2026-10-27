# Genetic_Primus V7

## Final status

`V7_TERMINAL_REJECTED_NO_EVENT_CAPACITY`.

V7 tested a genuinely new source family: forced-liquidation snapshots from
Binance COIN-M perpetuals as causal cross-venue features for Bitget four-hour
execution. The data source passed its integrity gate, but none of the three
preregistered liquidation events passed the stress-costed temporal gate.

## Immutable contract

Contract: `configs/genetic_primus_v7_contract.json`.

SHA-256:

`bd2cefb2f5234a42cd364c4d338cd1e26487f07057e8a06d842b7b281b816cd8`

The contract was frozen before future-return labels were calculated. It fixes:

- source window 2023-06-25 through 2024-10-14;
- full8 COIN-M-to-Bitget symbol mapping and USD contract sizes;
- one 12-hour holding horizon, next-open entry and exit;
- 24 bps round-trip stress costs;
- five chronological evaluation folds;
- causal shifted rolling thresholds and three event definitions;
- at least 30 events in every fold and positive mean, median, LCB, worst-fold
  mean and median fold LCB;
- no extension, fallback selection, paper, live or order authority.

## Source result

USD-M liquidation archives are no longer available from the public archive.
The independently listed COIN-M daily archives were retained and verified.

| Metric | Result |
|---|---:|
| Listed archives | 3,418 |
| Downloaded and checksum-verified | 3,418 |
| Download failures | 0 |
| Four-hour feature rows | 22,944 |
| Cross-market bars | 2,868 |
| Usable cross-market coverage | 88.08% |
| Minimum symbol-day coverage | 81.38% |

The source commonly duplicates complete rows. V7 removed only exact duplicate
rows and counted every removal. A missing daily archive remains missing; it is
never interpreted as a zero-liquidation day. Zeros exist only inside a day for
which the source archive was published and checksum-verified.

Dataset SHA-256:

`656f4491ced505b74879fd3b8de897a93e799e4dd3d9477666c50c469981802f`

## Event result

| Event | Events | Mean bps | Median bps | LCB bps | Positive folds | Worst fold mean bps | Passed |
|---|---:|---:|---:|---:|---:|---:|---|
| Cross-market cascade reversal | 114 | -30.10 | -15.87 | -52.92 | 1 / 5 | -98.69 | No |
| Cross-market cascade continuation | 114 | -17.90 | -32.13 | -56.68 | 2 / 5 | -71.82 | No |
| Isolated-symbol cascade reversal | 587 | -4.84 | -2.68 | -33.86 | 2 / 5 | -59.01 | No |

The opposing cross-market hypotheses both lose after costs. This means the
12-hour post-cascade move is too noisy and too small to support either a simple
reversal or continuation rule. The isolated-symbol event has sufficient total
incidence but changes sign across time.

## Post-hoc LINK observation

The diagnostic LINK slice has 55 events, mean +113.42 bps, median +79.29 bps
and LCB +35.51 bps. It is not a passing candidate:

- LINK was selected after opening the symbol table;
- each fold contains only 8-14 events versus the fixed minimum of 30;
- WF1 mean is -0.40 bps and LCB is -113.19 bps;
- the public liquidation archive ends in October 2024, so the same source has
  no untouched later holdout.

The observation is registered only as a prospective hypothesis in
`configs/genetic_primus_v7_posthoc_hypotheses.json`. It cannot authorize model
training, paper trading or live trading.

## Consequence

The logistic/neurogenetic model screen was intentionally not implemented or
run. `tools/run_genetic_primus_v7.py --stage development` fails closed against
the terminal event manifest. Reopening V7 with altered thresholds or a LINK
filter would be post-selection.

A valid continuation requires a new prospective liquidation tape or another
independent provider covering a later period. The first test on that untouched
window must evaluate the frozen LINK hypothesis without retuning it.

## Artifacts

- Dataset: `Retrodate/genetic_primus_v7_coinm_liquidations_v1_20230625_20241014`
- Data gate: `Reports/Exia/Genetic_Primus/v7_data_source_gate_20260824`
- Event gate: `Reports/Exia/Genetic_Primus/v7_event_feasibility_20260824`
- Stage runner: `tools/run_genetic_primus_v7.py`
