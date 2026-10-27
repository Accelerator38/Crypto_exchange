# Genetic Primus: hash-bound context links v1

## Scope

This is an additive metadata layer over the frozen result catalog, not a new
financial experiment. It does not import evaluators, fit/predict with models,
decode market parquet, recompute returns, contact networks, or use trading runtime.
All execution/credentials/orders/paper/live/promotion flags are false.

Base catalog: `Reports/Exia/Genetic_Primus/result_catalog_v1_20261005_r2/catalog.json`,
byte SHA256 `cafb0375dab4e0e608004c3fe9c216302ebc30331a745a0885d2c285ddbdb08b`.
The base completion receipt and original 1784-source inventory are separately
pinned. Historical outputs and the previous catalog are never rewritten.

## Implemented joins

| Family | Linked evaluations | What is recovered/checked |
|---|---:|---|
| Reversal calibration | 42 | Final 60-day interval and 20 origin intervals; 3-day horizon/cadence, purge 7, declared 730-day window, normal 16/0-bar and stress 40/1-bar costs/delay |
| Rolling linear matrix | 15 | Config-bound 20 x 3-day training interval, 4 arm grids, purge/cutoffs, explicit 240-minute bars, 16/24-bps scenarios and named zero-cost scenario |
| September baseline economics | 10 | Result/config interval agreement, normal 16 and stress 24 bps, source manifest byte identity |

The reversal registration has a **semantic** SHA256 excluding its own hash field;
this is not its file-byte SHA256. The legacy canonical recipe is reproduced
exactly (`sort_keys`, compact separators, ASCII escaping, no NaN). Registration,
fold list, final result, all step/state payloads and models are checked. All
20 origin spans are verified against folds/config, not inferred from filenames.
The final endpoint must match the last step. Scenario costs/delay must agree in
every origin, not just one sample.

All 20 reversal model payload digests and 160 explicit per-decision model hashes
match. All 80 linear model payload digests match their parent decision batches.
Linear decisions do **not** contain individual model SHA fields: that scope is
recorded explicitly, not overstated as per-decision lineage. Reported label ends
must respect reported cutoffs; this does not re-audit raw training labels.

The fixed EMA training reference has no learned-model retrain cadence. It is
marked `not_applicable_fixed_baseline`, not silently assigned linear retraining.
Reversal bar duration is not guessed from numerical cutoff differences.

## Field provenance and unknowns

Each linked evaluation retains the original source/envelope identity, policy,
scenario and a semantic digest of the **unchanged** base metric list. Joined
fields have explicit units and evidence path/byte hash/field/rule. A rule may be
a verified formula or known-schema interpretation rather than a literal JSON
pointer; derived values are labeled as such. Conflicts with already-known base
context abort the join. Unsupported or unproven fields remain null, not zero.

Snapshot, training rows, scaler, weights, code and resample-bank hash claims are
preserved under their original field names. A matching manifest byte hash does
**not** verify its raw market sources. No raw-data/weights-bank payload hashes,
funding, real spread/fills, historical fit causality, or global DSR/PBO are newly
verified. Only explicitly recomputed semantic identities are marked verified.
SHA256 receipts are integrity checks, not signatures or an immutable global
experiment registry.

Cross-source comparison remains disabled. Metadata joins alone do not establish
equal risk/exposure, accounting, search budgets or independent/prospective
evidence. NoTrade's previous descriptive September ordering is unchanged.

## Reproducibility

Use only `E:\Crypto_exchange\.venv\Scripts\python.exe -X utf8`, one short local
command at a time. No pytest or collector is needed.

- Pure module: `src/exia/genetic_primus/result_context_links_v1.py`.
- Regressions: `tools/check_genetic_primus_result_context_links_v1.py`.
- Builder: `tools/build_genetic_primus_result_context_links_v1.py`.
  `--verify-only` checks metadata without writing outputs.
- Artifact verifier: `tools/check_genetic_primus_result_context_artifacts_v1.py`.

The builder writes only a new `result_context_links_v1_20261005` directory using
exclusive creation. Existing outputs cause a refusal. Before/after checks cover
all 1784 originals, every input sidecar, the base catalog/receipt and the frozen
future EMA contract. The completion receipt binds input bytes, output bytes,
module and builder. A partial directory without a verified receipt is incomplete.

The artifact verifier independently reconstructs metadata joins, compares exact
outputs/summary, checks every provenance reference and base metric digest, and
rechecks inputs. It never evaluates the future interval; that contract is read
for its byte hash only.

## Next scope

Consolidate the metric dictionary and objective/attempt identities. Distinguish
engineering revisions from financial trials and do not invent missing adaptive
testing history. Resolve accounting/valuation and historical provenance gaps
before enabling any cross-source ranking or new training decision. Keep the
frozen prospective interval unopened for result analysis.
