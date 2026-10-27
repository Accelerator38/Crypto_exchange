# Superseded engineering attempt — do not use as the requested comparison

The 280 gross forecast fits and all original outputs are preserved, not overwritten.
The initial temporal weighting normalized `0.2 + 0.8*recency`. That is a raw floor,
**not** the requested normalized mixture of 20% uniform and 80% half-life-90 mass.
The earlier report's “20% uniform anchor” wording is incorrect for these receipts.

Correct contract: `0.2/N + 0.8*recency/sum(recency)`. The semantics correction is not
a change of features, architecture, seeds or hyperparameters based on performance.
The complete fixed matrix must be repeated under a new immutable preparation.
All 280 fits remain in the attempt/testing budget; no financial claim or promotion
was made. These raw-floor results cannot support the intended recency experiment.

`source_snapshots/` preserves the seven original code/config/doc files, with bytes
matching the original `preparation.json` code-input identities. Source pins now
refer to those preserved snapshots rather than the revised workspace code. Original
outputs remain covered by the unchanged `evidence_receipt.json`.
