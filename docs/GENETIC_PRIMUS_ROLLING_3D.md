# Genetic Primus rolling 3-day evaluator

## Scope

The evaluator reproduces an offline `train -> freeze -> deploy for three days ->
retrain` sequence. It imports no exchange, credentials, order or live runtime
modules. The protocol is fixed by
`configs/genetic_primus_rolling_3d_v1.json`.

- expanding training starts at `2022-01-01`;
- validation origins start at `2025-01-01`;
- deployment horizon and retrain cadence are both three days;
- seven 4h bars are purged from supervised labels before every train end;
- the scaler is fitted again using only rows available before each origin;
- positions are conservatively flattened at each deployment boundary;
- every decision, trade and portfolio row is pinned to the frozen origin model
  SHA.

The 2025 protocol contains 121 complete, non-overlapping three-day windows. The
last incomplete two days are excluded.

## Origin training

The bounded engineering trainer uses one local worker and a linear V4 policy.
Five isolated populations contain four genomes each and run for five
generations. Every origin receives a materialized common selection bootstrap
bank. The five island champions are evaluated on an independent audit bank;
audit results never change the selection-bank choice.

The model SHA binds the genome, scaler, trainer metadata, origin timestamps and
rolling protocol SHA. Each origin stores its champion, scaler, model manifest,
selection/audit banks, generation metrics and top-K audit table.

## Recency fitness

Rolling selection applies the same materialized resample bank to every genome.
Daily PnL is scored at 30, 90 and 180 day half-lives. Each weight vector is an
80% exponential decay plus a 20% uniform anchor, applied symmetrically to gains
and losses so old adverse periods cannot disappear from the objective.

For every half-life the ledger records weighted mean, weighted block-bootstrap
LCB and effective sample size (ESS). The return component is 70% of the worst
of the three recency LCBs plus 30% of the worst LCB across 12 chronological
rolling windows. Continuous penalties cover maximum drawdown, fills per day,
genome parameter count and the share of policy decisions that cause fills.
Selection and independent top-K audit use identical formulas but independent
bootstrap banks. The sealed protocol fixes every coefficient and ESS threshold.

## Model simplicity and benchmarks

The V4 architecture screen permits only a 93-parameter linear model and a
single 8-unit hidden layer with 275 parameters. The former 659-parameter
`compact_16_8` candidate is no longer accepted by the sealed contract. Rolling
training remains linear-first.

Parsimony is measured per genome at a sealed absolute-weight threshold. The
fitness penalizes architecture parameter count, active parameter count and
effective active input features. For the 8-unit model an input is active only
when it has an active path through a hidden unit to an output. Generation and
independent audit ledgers store active counts, feature sparsity and the total
parsimony penalty.

Every three-day deployment window is evaluated with identical data boundaries
and 16 bps round-trip costs against `NoTrade`, robust-scaled one-bar momentum
and its mean-reversion inverse. Starting with the second origin, the previous
origin's frozen model and scaler are evaluated as the incumbent. Benchmark
summaries and timestamp-level portfolio returns are stored separately from the
candidate decision and trade ledgers. Benchmark evidence remains offline and
cannot authorize paper/live trading or promotion.

## Island search and ensemble

Each retrain runs five isolated islands with four genomes and five generations.
Every island has its own preregistered seed and checkpoint. Exactly one genome
per island is a fresh random immigrant (25%); the other three positions are
seeded from the current supervised model, the frozen incumbent when available,
and deterministic mutations. Incumbent warm-start therefore never eliminates
fresh or supervised search paths.

Island champions are ranked using the shared selection bootstrap bank. The
top-3 and top-5 ensembles always contain champions from different island seeds
and average member logits before selecting an action. Both sizes receive a
full-model parsimony charge and are evaluated on selection and independent
audit banks. Selection chooses the deployed ensemble size; audit results are
recorded but cannot change it. The model SHA binds every ensemble member, and
the complete frozen ensemble becomes the incumbent warm-start and benchmark at
the next origin.

## Limited experiment matrix

The preregistered first stage runs only weekly retraining for five cells:
expanding/uniform, rolling 730d/uniform, rolling 730d/180d half-life, rolling
365d/90d half-life and rolling 180d/30d half-life. Uniform weighting is encoded
as a 100% uniform anchor. Rolling windows change the scaler and supervised-data
start at every origin while preserving the seven-bar label purge.

Only after all five weekly cells have completed are the two best variants
selected by the sealed ordering: audit paired LCB, positive outer-slice share,
stress-cost LCB and stable variant id. Only those two variants may enter the
1/3/7/30 day cadence stage. The weekly 7d result is reused rather than rerun.
The experiment runner accepts exactly one eligible cell per invocation and
defaults to one origin.

## Challenger promotion gate

Starting with the second origin, a challenger can replace the frozen incumbent
only when every gate passes: positive paired block-bootstrap LCB, at least 65%
positive outer slices, positive 24 bps stress-cost LCB, no drawdown degradation,
turnover and concentration limits, positive paired LCB on an independent audit
bank, action total-variation at most 0.25 on the latest outer slice and
normalized ensemble-centroid drift at most 0.75.

When any condition fails, challenger artifacts and diagnostics are retained,
but decisions continue using the incumbent model SHA and its original scaler.
The gate only controls the offline rolling challenger/incumbent ledger. All
credentials, paper/live, order and external promotion authority remain false.

## Safe command

The CLI defaults to one origin to prevent accidental full-year computation:

```powershell
E:\Crypto_exchange\.venv\Scripts\python.exe `
  tools\run_genetic_primus_rolling_3d.py `
  --output-dir Reports\Exia\Genetic_Primus\rolling_3d_<id> `
  --max-origins 1
```

The first real-data engineering smoke completed one origin with 136 pinned
decisions and 23 closed trades. It is not validation evidence and grants no
paper, live, order or promotion authority.
