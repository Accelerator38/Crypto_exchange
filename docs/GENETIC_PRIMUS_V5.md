# Genetic_Primus V5

## Final status

`V5_TERMINAL_REJECTED_DATA_SOURCE_INSUFFICIENT_COVERAGE`.

V5 was designed as a fail-closed cross-exchange derivatives experiment. Its
contract requires the data source to pass integrity and core-feature coverage
before future-return labels, simulation, scoring or evolution may be opened.

## Immutable contract

The contract is `configs/genetic_primus_v5_contract.json`, SHA-256:

`6c897d9e03fc0c1d4c8d74433ddccc81060a7409c4d13d8cb94496daf4d6cd67`

It fixes:

- full8 universe and 2022-2024 research source window;
- four-hour decisions and next-open execution;
- fixed six-bar holding and no cross-fold positions;
- three preregistered derivatives events;
- 24 bps stress cost and four walk-forward evaluation folds;
- three fixed evolution seeds and no fallback architecture selection;
- terminal kill criteria and all disabled paper/live/order flags.

## New dataset

The builder downloaded every official Binance USD-M daily metrics archive for
the fixed universe and window and verified the official checksum for each file.

| Metric | Result |
|---|---:|
| Daily archives | 8,768 / 8,768 |
| Raw five-minute rows | 2,524,126 |
| Derived four-hour rows | 52,592 |
| Raw file coverage | 100.00% |
| Four-hour bar coverage | 99.97% |
| Files with deterministic row-order repair | 199 |

The 199 files contained unique valid timestamps in a non-monotonic row order.
They were stable-sorted and recorded in `raw_catalog.parquet`; no duplicate or
invalid timestamp was accepted.

Dataset SHA-256:

`39ba79679655eddd4a29e2a93de2038dad13c7e91122cb9799316bc0b24207d5`

## Terminal gate

The contract requires at least 90% coverage for every core feature and symbol.
Open-interest and account-ratio features passed. Binance taker-ratio fields are
absent for part of early 2022, leaving `taker_imbalance_mean` coverage between
88.35% and 88.44% for all eight symbols.

The threshold and required feature list cannot be changed after observing this
result. Therefore:

- feasibility labels were not opened;
- preregistered event expectancy was not calculated;
- bounded simulator and cost overlay were not opened for V5;
- logistic baseline and neurogenetic ranker were not created;
- no seed, validation, OOS or paper stage was run;
- paper, live, orders and promotion authority remain false.

`tools/run_genetic_primus_v5.py --stage development` raises `PermissionError`
against the terminal gate report.

## Artifacts

- Contract: `configs/genetic_primus_v5_contract.json`
- Dataset builder: `tools/build_genetic_primus_v5_derivatives_dataset.py`
- Dataset: `Retrodate/genetic_primus_v5_binance_metrics_v1_20220101_20241231`
- Acceptance gate: `tools/evaluate_genetic_primus_v5_data_source.py`
- Gate report: `Reports/Exia/Genetic_Primus/v5_data_source_gate_20260824`

The dataset remains useful for a separately preregistered future experiment
whose feature contract is designed around the source's known historical
availability. It cannot be retroactively substituted into V5.
