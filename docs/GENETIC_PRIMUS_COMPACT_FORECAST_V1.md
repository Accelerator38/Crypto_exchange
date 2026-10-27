# Compact forecast comparison v1

This is a newly registered **gross forecast diagnostic**, not net policy training,
financial validation, promotion, or runtime integration. The explicit user request
authorizes these local offline fits only. All V8 Clean safety/authority fields remain
false; the diagnostic receipt grants no downstream launch authority. Frozen old
`gross_open_open_diagnostic_not_net_training_target` samples are not requalified.

Four fixed arms: six-feature linear (7 coefficients including bias) and
6→8 tanh→1 MLP (65 coefficients including biases), each under uniform and
half-life 90 days **with a 20% uniform anchor**. Three predeclared seeds 11/29/47
are averaged for each MLP arm, not selected on the benchmark. An ensemble stores
195 coefficients; scaler/target-normalization statistics are additional estimated
quantities, not counted as network coefficients. Linear and MLP have different
compute costs; the report logs actual fits and gradient steps.

The six features are distance to EMA48, close return18, efficiency18, return18
relative to the complete eight-symbol universe, std of 48 log returns and breadth
above EMA48. EMA starts at the first close in the frozen source history. At least
49 bars are required. Missing timestamps/symbols are rejected. Completed candles
only; zero-lag availability and instantaneous research fit readiness are assumptions,
not measured execution latency. Do not qualify this experiment for real trading.

Origins repeat every 18 four-hour bars (three days). Feature-asof is origin 00:00;
entry is the **next** open at 04:00, exit exactly 18 bars later, target availability
one more bar later. Fit train_end=origin−1 bar, label cutoff=train_end−7 bars.
All labels available strictly after that cutoff are excluded. Expanding training
is deliberate; no rolling-window search is performed.

For each origin, all arms share the same final **train-only**, unweighted feature
and target scaler (ddof=0, standardized features clipped ±8). Separate inner
scalers use only inner training rows. The last 60 days of eligible origin cohorts
provide chronological early stopping; inner training has its own seven-bar label
purge. MLP selects an epoch on unweighted inner MSE then refits from the same seed
on the whole mature prefix for exactly that many epochs. Fixed L2=0.01, learning
rate=0.001, maximum 120 full-batch Adam steps, patience=12. Biases are not penalized.
Linear solves the corresponding weighted ridge objective; both fit **every** sign.

One temporal weight per simultaneous origin:
`w = 0.2/N + 0.8 * 2**(-age_days/90) / sum(2**(-age_days/90))`; uniform uses 1/N.
Normalize both components **before** mixing; this preserves exactly 20% uniform
mass. A raw 0.2 floor does not. The first engineering attempt using that floor is
superseded and preserved (280 fits, included in the attempt budget, not valid for
the requested normalized-anchor comparison). Age is measured from the purged
cutoff using outcome availability, never current wall time. Divide each origin
mass equally across eight symbols. Report ESS of
origins, positive/negative/zero weight mass and hashes of the full weight vectors.
Origin ESS must not be multiplied by eight to imply independent temporal evidence.

The already-open August–September benchmark has 20 complete three-day origins,
160 outcomes. No hyperparameter, seed, feature or risk selection on this interval.
This is expanding walk-forward retraining: prior benchmark outcomes become train
data only after maturity and purge. It is not an untouched sealed final test.

Forecast comparison reports RMSE/MAE, sign accuracy, zero-forecast and causal
train-mean controls, per-origin errors, fixed-seed spread, and paired block-bootstrap
MSE differences. Common banks: 4,096 circular resamples, block length 2 origins,
seeds 3039 and 7079 (audit), both pinned **before** fits. These diagnostics are not
capital-growth LCBs, multiplicity-adjusted evidence or a financial promotion gate.
Trading costs, funding, turnover, exposure and policy thresholds are intentionally
not invented. No capital curve is computed for forecasts with no trading policy.

Run one short local command at a time with the workspace Python only:

1. `tools/check_genetic_primus_compact_forecast_v1.py --scope core`
2. `tools/check_genetic_primus_compact_forecast_v1.py --scope rolling`
3. `tools/compare_genetic_primus_compact_forecast_v1.py prepare --report-dir PATH`
4. Direct checks can persist to `--report-path PATH/engineering_checks.json`.
5. `... fit --report-dir PATH --origin-index N` (one origin per command).
6. `... report --report-dir PATH` after all 20 origins; no training in report.

Each step verifies the immutable preparation, input/source hashes and sample bank.
Outputs use exclusive creation and existing fit outputs are refused. No pytest,
collector, network, processes, trading runtime, keys, orders, or live supervisor.
