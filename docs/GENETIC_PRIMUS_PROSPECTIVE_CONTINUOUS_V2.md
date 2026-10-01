# Genetic Primus: continuous offline evaluator V2

The V2 runner evaluates a registered, future three-day origin chain. It verifies
each per-origin immutable Parquet manifest, refits the median/IQR scaler on
purged history at each origin, and joins all decisions into one cash simulation.
An origin boundary does not itself close or reopen a position. The model SHA,
training-row hash, scaler-input hash, and committed snapshot hash accompany
every decision. The position is liquidated only at the end of the completed
prefix so that the prefix has an auditable realized PnL. When the next origin
finishes, the entire prefix is recomputed; earlier prefix PnL is provisional.

The runner is `tools/run_genetic_primus_prospective_continuous_v2.py` with
`commit`, `preflight`, and `evaluate` subcommands. All require `--contract`,
`--registry`, `--output-root`, and `--origin`. `commit` additionally requires
one or more `--snapshot` Parquet paths. The output root must be inside
`Reports/Exia/Genetic_Primus`. Only the project `.venv` Python is accepted.
All commands are local and have no trading or network capability.

Before using this runner, register a **new future `/2` contract** in a new
immutable objective registry version. The contract must use one NoTrade, one
EMA, and one registered sparse GA candidate, and add:

```json
"model_selection": {
  "window": "expanding",
  "scaler": "refit_each_origin_train_only_median_iqr_clip_8",
  "report_path": "Reports/Exia/Genetic_Primus/<selection-report>.json",
  "report_sha256": "<sha256-of-selection-report>"
},
"data_commitment": {
  "manifest_version": 2,
  "origin_snapshot_commit_rule": "after_origin_complete_before_evaluation",
  "origin_commit_deadline_hours": 72
},
"accounting": {
  "terminal_exit": "completed_prefix_final_candle_close"
}
```

These are additional or replacement fields inside the corresponding contract
sections. The runner checks the selection report's bytes and that the selection
and deployment window/scaler rules match before any manifest is committed.
The JSON report must record `selected_genome`, `window`, `scaler`, and
`selection_cutoff_utc`; the runner checks these against the contract and
requires the cutoff to precede outer start. The archived `f7a4ae11`
selection used a different 150-day window, so it cannot be declared matching
by changing contract fields. A new model selection procedure and a newly
registered outer period are required before prospective V2 use.

The September V1 results retain their original meaning and are never
converted to V2 evidence. The V2 simulation still assumes next four-hour open
execution with fixed 16/24 bps costs; it is research accounting, not a live
execution model or proof of positive return.
