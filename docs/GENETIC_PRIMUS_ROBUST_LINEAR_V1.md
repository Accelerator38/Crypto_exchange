# Fixed Huber versus daily-label ridge — offline research

The direct control is frozen B from `cadence_matrix_v1_20261009`: three-day gross scheduled-close labels starting daily, decisions and causal retraining every three days, linear6 with seven parameters, uniform temporal base weights, L2=0.01, forecast strictly >20 bps. A is shown only as context because its training sample stride differs. No winner is promoted.

Only loss changes for H: mean Huber(delta=1.345) in the **same unweighted train-only target standardization** as B, plus half L2 times squared slopes. The intercept is not penalized. The scaler is fitted again at each origin and must exactly match B's causal scaler. This is not a MAD-based scaler or recency weighting experiment. Both profitable and unprofitable outcomes retain positive weights. Huber is symmetric in residuals, not a filter that keeps profitable trades.

Deterministic IRLS uses base weights 1/N multiplied by min(1, delta/absolute residual); effective weights are **not renormalized**, so regularization is not silently changed. Initialization uses B at the same origin, not a search over initializations. At most 100 IRLS iterations, parameter step <=1e-10 and objective gradient <=1e-8; nonconvergence or an increasing objective fails closed without publishing a model. Iterations are solver updates, not independent model candidates.

Training dates, purge (seven four-hour bars plus the boundary bar), features, label bank, folds, cutoff, costs, funding, decision policy, position accounting and shared block-bootstrap banks are frozen. Model SHA and causal identities are attached to every decision. Positive and negative residual tails can both be downweighted. Kish ESS is a diagnostic, not an overlap-adjusted independent sample count.

## Budget

- Development May 1–July 25, 2026: 29 new model/scaler fits.
- Report August 1–September 30: 21 new model/scaler fits.
- Total 50 Huber model fits, 50 causal scaler fits, four new normal/stress replays. Frozen A/B artifacts are reused, not rerun.
- No delta, lambda, threshold, clock, architecture or feature sweep. No phase-specific Huber refits in this contract. Cross-origin coefficient stability does not establish cross-phase stability.
- Each command performs at most seven real-data fits, foreground with one numerical thread. No pytest, collector, WS, network, background process, keys or trading runtime.

Both periods have already been revealed. This is an engineering/research comparison, not new independent validation, multiplicity-adjusted evidence, DSR/PBO proof or an investment recommendation. Fixed selection/audit banks check resampling only. Execution remains the parent's assumed OHLCV fills, not qualified historical L2 execution. Close-mark DD does not bound intrabar risk.

## Artifacts and commands

Use only `E:\Crypto_exchange\.venv\Scripts\python.exe -X utf8 tools/run_genetic_primus_robust_linear_v1.py`:

1. `prepare`: verify frozen evidence, exclusive-create registration and folds before fitting.
2. `checks`: synthetic deterministic checks, no real-data fits.
3. `train --period development|report --start-index N --count 1..7`: only missing registered indices.
4. `replay --period development|report --scenario normal|stress`: once per combination.
5. `audit`, `diagnostics`, `summarize`, `plot --period development|report`.
6. Write results report, visually inspect graphs, and run `verify` last to seal output hashes and reverify both frozen parents.

Registration pins configuration/code and planned folds/cutoffs. Source banks and shared resample hashes remain in the verified parent chain. An immutable local objective entry records this single new candidate; it does not pretend to reconstruct the complete historical trial budget. Results contain solver convergence, coefficient signs, forecast margins, both-sign weight masses, matched forecast quality, continuous capital/DD and reconciled H−B dollar attribution.

Existing files are never overwritten. After a disconnect resume at absent artifacts; do not repeat completed fit/replay commands. Registered numerical code must not be edited after preparation. All nine authority flags remain false; local research fits/replays are separately requested by the user. NoTrade remains the default deployment.
