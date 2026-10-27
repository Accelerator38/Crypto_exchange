# Label-phase sensitivity v1 — offline research

This diagnostic extends the frozen `cadence_matrix_v1_20261009` without changing it. No model selection, tuning, network, collector, keys or trading runtime is authorized by these files.

## Registered comparison

`P0` reuses A; `B` reuses the daily-label control B. `P1` and `P2` shift **only the training-label anchor** by one and two UTC calendar days. Every policy keeps the same three-day forecast, three-day decision clock, three-day retraining clock, six features, seven-parameter linear model, uniform outcome weights, L2=0.01, strict forecast >20 bps threshold, binary 10% sleeves, fees and normal/stress costs. Scalers are independently fitted on each eligible training set. Eligible labels mature before the common fit minus eight four-hour bars (one boundary bar plus seven purge bars).

The three phase banks must be a disjoint partition of the daily-label bank, including their feature and outcome contents. Their eligible subsets must partition B at every fit cutoff. This changes neither the calendar of decisions nor the calendar of model updates. Different phases necessarily have different latest eligible samples; record that difference rather than shifting the cutoff to hide it.

## Budget and provenance

Development May 1–July 25, 2026 inclusive: 29 fits per phase. Report August 1–September 30 inclusive: 21 fits per phase. Both periods were already revealed. Two new phases produce exactly 100 new fits and eight new normal/stress replays. P0/B reuse 100 parent model artifacts and eight parent replays; they are not retrained or rerun. Each command trains at most seven models, foreground and one numerical thread. Existing 4096-draw block 3/7/14 selection/audit banks are reused unchanged. No independent-validation, multiplicity-adjustment or DSR/PBO claim is made.

Registration before fits pins configuration, code, phase sample banks, fold/cutoff plans and the parent's registration/receipt. Parent receipt artifacts, original sources and numerical code are verified before preparation and again when closing the experiment. Outputs are exclusive-created and never overwritten. A disconnected run resumes at missing artifacts, not from the beginning.

## Diagnostics

- At every fit: pooled/per-asset label distributions, train sizes, last mature label age, scaler means/scales, target mean, coefficients in standardized and unclipped raw units. Raw coefficient formula applies only inside the scaler's clip.
- At every decision/asset: forecast, signed distance to threshold, disagreement with P0, causal model SHA and full-horizon outcome when it fits completely within the evaluation period. A censored final horizon has no forecast-quality label.
- Attribute actual USDT differences by decision/asset and gross mark-to-market, fee, adverse drag and funding. Mark price and funding at a right boundary belong to the preceding holding interval; fills at a left boundary belong to the following interval. Terminal closing costs belong to the final interval. Check exact aggregate and per-symbol reconciliation within 1e-7 USDT.
- Classify unequal targets as missed profit/avoided loss according to the reference interval's net outcome, extra-entry episodes, or late-exit episodes. Equal targets have a separate path/sizing/cost residual. Descriptive classification is not a causal marginal effect or a retraded counterfactual.
- Report concentration by asset, UTC boundary date and decision origin, largest close-to-close market moves, and removal of the best complete three-day log-return reporting window. These are accounting/sensitivity diagnostics, not corrected backtests or new optimization objectives. Absolute concentration uses a sum-of-absolute-contributions denominator, not a possibly near-zero net PnL.

The execution assumptions remain the frozen parent's modeled OHLCV execution, not qualified historical L2 fills. Close-mark drawdown is not an intrabar executable bound. All nine authority flags remain false; local research fits/replays were separately requested by the user.

## Bounded commands

Use only `E:\Crypto_exchange\.venv\Scripts\python.exe -X utf8` with `tools/run_genetic_primus_label_phase_v1.py`:

1. `prepare`, then `checks`.
2. `train --period development|report --phase 1|2 --start-index N --count 1..7` for missing registered indices.
3. `replay --period development|report --phase 1|2 --scenario normal|stress` once per combination.
4. `audit`, `diagnostics`, `summarize`, then `plot --period development|report`.
5. Write the results report, inspect plots and run `verify` once, last, to seal all artifacts.

Do not run pytest, collector or live/network smoke tests as part of this experiment. Do not edit registered code after `prepare`; a necessary correction requires an explicitly separate diagnostic contract, not silent replacement of evidence.

An additional read-only, zero-fit exploratory diagnostic was added after replay results: `tools/analyze_genetic_primus_label_phase_influence_v1.py analyze|plot`. Its own source pin and post-result status are stored in `influence_diagnostic_provenance.json`. It verifies the frozen ridge normal equations, decomposes stored coefficients with scaler/Gram held fixed, and reports training-label tail concentration. This is not deletion/refitting, a new candidate or a new statistical gate.
