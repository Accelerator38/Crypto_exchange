# Genetic Primus: additive result catalog v1

## Purpose and boundaries

The catalog normalizes **reported metadata**, not trading performance itself. It
does not replay trades, decode market data, fit models, open network connections,
or grant execution/promotion authority. All envelope safety flags are false.

The fixed input is `development_review_v1_20261005/evidence_index_v2.json`, byte
SHA256 `5e6192ccef065c9e0f488395bfd9456d160c071d706a9985447149e172d89bed`.
Its 1784 original JSONs are read directly. Their byte sizes and SHA256 values are
checked before adaptation, immediately before publication, and after publication.
Only a new directory is written; existing outputs are never overwritten.

Latest metadata revision: `result_catalog_v1_20261005_r2`. The first snapshot
`result_catalog_v1_20261005` is preserved. Revision r2 adds explicit gate types and
adapters for closed-origin chaining and reversal origin wrappers. Neither revision
is a new research attempt or evidence of an independent profitable strategy.

## Typed contract

`src/exia/genetic_primus/result_catalog_v1.py` uses frozen standard-library
dataclasses and enums. An envelope contains:

- Original path, **byte** SHA256, byte size, and top-level reported schema.
- Artifact kind: financial result, origin step, model metadata, scaler, manifest,
  registration, research ledger, checkpoint, engineering check, search trace,
  research gate, data summary, resample-bank metadata, or unknown.
- Reported evidence class and a conservative category. A filename does not prove
  prospective or independent validation. Such claims remain unverified.
- Reported hash claims under their original names. These are not verified data
  lineage; a market-rows hash is not relabeled as a snapshot hash. For known origin
  wrappers, nested model and source hash claims retain their field prefixes.
- Per-policy/scenario evaluations with explicit units, source fields, context,
  missing-value reasons, and separate accounting semantics.
- Canonical envelope SHA256, plus non-execution flags. A digest is tamper detection,
  **not a digital signature**, trusted provenance, or calibrated trial registry.

Unknown horizon, cadence, periods, delays, and accounting remain `null`. Sidecar
joins are intentionally not performed. Metadata completeness is summarized by
field; supported metrics do not imply a fully specified measurement contract.

## Metric adapters

| Source shape | Adapter semantics |
|---|---|
| Baseline economics audit `/1` | Explicit `*_pct` returns/drawdown divided by 100; trades and fills remain distinct counts |
| Rolling linear matrix `/1` | Explicit fractional net return/drawdown, trade count; train arms are not independent validation |
| Reversal calibration result `/1` | Marked NAV net, hypothetical terminal-adjusted net, bar-close drawdown, transition count, initial-NAV fee fraction |
| Reversal rolling-step wrapper `/1` | Per-origin net, explicitly declared origin/end timestamps, cost and delay; not a full-pilot result |
| Revealed three-day summary `/1` | Fractional geometrically chained **closed-origin** returns; not continuous NAV |
| Legacy flat `mean_bps/lcb_bps` | Per-trade mean/LCB divided by 10000; fractional drawdown |
| Legacy flat `mean_daily_bps/lcb_daily_bps` | Per-calendar-day mean/LCB divided by 10000; fractional drawdown |

Unsupported financial shapes produce no inferred metrics and carry an issue.
Expected missing/invalid fields yield `null`, never zero. Observed NoTrade zeros
remain zero. Booleans, strings, NaN/infinity, invalid counts, and out-of-range
drawdowns are rejected as values. Conversions do not estimate new bootstrap LCBs,
ESS, funding, risk, slippage, or terminal liquidation costs.

Metric names distinguish trade vs day LCB, trades vs fills vs transitions,
bar-close vs sampling-unknown drawdown, and marked vs terminal-adjusted returns.
Raw daily arrays are not copied or re-evaluated.

## Comparison guard

`comparison_reasons` returns explicit blocking reasons. No generic compatibility
is inferred from identical field names or equally incomplete contexts. Currently
the only allowed comparisons are descriptive same-scenario comparisons within the
one previously audited baseline-economics JSON, pinned to byte SHA256
`4ec6c77c8d533dd4db970da5ffa7769706ccf202225e0cb4ed7e7506154daee7`.
That batch shares one reported interval/data context, with normal 16 and stress 24
bps costs; it must not be confused with the reversal pilot's stress 40 bps + delay.

Normal vs stress, changed context, unknown/incompatible metric, invalid digest,
nonfinancial artifacts, and cross-source results are blocked. Even a permitted
descriptive ordering is **not** an exposure-matched causal comparison, a paired
statistical test, or a promotion decision. Unreported execution details remain
unknown, including for the pinned batch.

No generic cross-source comparator is enabled until data identity, periods,
units, horizon/cadence, costs/funding/delays, capital/valuation rules, and evidence
scope are proven compatible. Matching metadata alone will not establish equal
search budgets, global DSR/PBO calibration, or independent evidence.

## Reproducibility and next stage

The direct checker is `tools/check_genetic_primus_result_catalog_synthetic_v1.py`.
It performs finite, deterministic regressions without pytest or collector. The
builder is `tools/build_genetic_primus_result_catalog_v1.py`. Use exclusively
`E:\Crypto_exchange\.venv\Scripts\python.exe -X utf8`.

The completion receipt binds the inventory, source-identity bank, output bytes,
adapter module, and builder. Only a present and verified completion receipt
indicates a completed build. It proves local source immutability at that time,
not ongoing immutability or the truth of historical data/execution claims.
`tools/check_genetic_primus_result_catalog_artifacts_v1.py` independently verifies
the receipt, every envelope, all 1784 original byte hashes, descriptive orderings,
and the frozen future EMA contract. It does not evaluate market data or models.

Next: validate hash-bound sidecar joins for missing measurement context and
consolidate metric/attempt definitions. Reconcile global adaptive-testing history
without inventing missing trials. Do not retrain on a rejected objective or peek
at the frozen future interval to fill catalog gaps. Continuous-NAV accounting,
frozen contracts, original result files, runtime, and credentials are untouched.
