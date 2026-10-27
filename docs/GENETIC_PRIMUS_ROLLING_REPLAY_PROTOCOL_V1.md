# Rolling replay protocol v1

The interval is fixed: `[2026-08-01T00:00Z, 2026-10-01T00:00Z)`, including the whole
calendar day September 30. It is already revealed, not prospective. Use 20 complete
three-day origins, a next-open entry, 18-bar hold, then 20 hours of terminal cash;
do not silently create a partial final origin or use October outcomes.

The completed gross forecast fits are reused from `compact_forecast_v1_20261007_r2`.
No retraining or hyperparameter search is necessary to audit those origins. Each
forecast checkpoint binds the decision to model SHA (or a separate fixed-mean
ensemble SHA), all member SHAs, per-origin scaler, feature definition/prefix,
training samples, inner fold and weight hashes, train_end and purged cutoff.
The latest label availability is independently reconciled against the frozen sample
index. Twenty distinct per-origin scaler identities are required.

The checkpoint ledger is **not** a trade, fill, cash, or PnL ledger. Net policy and
funding/cost inputs must be registered before running `unified_measurement_v1`.
Existing `training_allowed=false` and the other eight safety fields remain false;
none of these artifacts grants runtime or live authority.

The current funding manifest is diagnostic-only and ends July 27, not September 30.
Missing funding is never assumed zero. Financial replay readiness also requires a
frozen decision policy (thresholds/abstention and exposure), shared quote capital,
allocation/cash reserve/borrow policy, normal and stress per-fill costs, input
provenance and a registered execution/availability delay. Registered sensitivity
cost assumptions may support a reference research replay but cannot be called
observed execution or qualify a live release. Verify the August 19–22 source price
spike independently before economic conclusions; retain the window in the benchmark.

The profit objective remains net log-growth under registered research DD10/20/30%,
with no extra double-charged cost penalty. LCB is a stability diagnostic. Full net
capital curves, costs, funding, exposure, turnover and position carry must be
reported once the replay really runs. Never substitute RMSE plots for these.

GA is conditional on a simple model's **verified net** signal on causal pre-August
development data: positive normal/stress absolute and paired LCBs, independent audit
bank, risk/stability gates and complete global trial registry. Forecast MSE can be
negative relative to a control while a policy still has economic value; RMSE alone
must neither authorize nor categorically reject GA. No GA or random-search runner
is called by this protocol. The current net signal is not evaluated.

The conditional GA/random design uses the same registered linear feature-mask/L2
space, origins, folds, objective and bootstrap bank. Per-origin cap: 64 unique
objective evaluations, 128 fits, 64 inner-fold evaluations, zero gradient steps.
Duplicates are logged but do not count as new evaluations; incumbent evaluations
do count. `compare_search_usage` checks **actual** unique candidate logs, counters,
completed budgets and shared identities, not just equal generation settings.
Candidate-space hash remains unset until full preregistration; no search happened.

Prospective is not sealed or started. A future seal must follow completed net replay
and stability review, freeze the final policy and trial registry, choose duration
in advance and allow only one final analysis. Start must be after sealing and all
already-consumed data; the outcomes must actually be unseen. Do not use another
August/September retune as a “new” prospective study. Zero net effect is a research
boundary, not a guaranteed non-loss target or an approved real-money risk budget.

Use workspace Python only. `tools/check_genetic_primus_rolling_replay_protocol_v1.py
check` runs only 19 named pure unittest regressions. `... package --report-dir PATH`
verifies local evidence and creates a new immutable package; it does not execute
collector, pytest, network, workers, model fits, financial replay or trading runtime.
