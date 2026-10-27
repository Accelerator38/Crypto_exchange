# Draft engineering protocol and bounded full8 market-row adapter

No parquet, fit, net targets, portfolio evaluation, network, runtime or pytest.
This step implements a data/feature component, not a new experiment matrix.
Verification rows carry actual full8 symbol names but all prices are fabricated.
No claim of execution on market data or calibration. Old source/results unchanged.

The new protocol pins the old eleven-file manifest byte SHA and historical origin,
history, model family, two variants and max-two-pipeline budget. Raw-read/fit
permissions and all nine authority flags remain false. Ingestion, latency and
accounting values stay null. Engineering declarations cannot become measurements.

Legacy build_origin_examples uses an origin-anchored 18bar calendar and the old
open→last-close label. Its module imports numpy/pandas and is read only statically.
The adapter preserves the 18bar calendar, never expands opportunities to 4h,
and does not call or requalify the old label path. Candidate calendar includes
closed-prefix warmup only; label maturity and purge must gate it separately.

MarketFeatureState accepts complete eight-symbol bars with the physical source_rows
column, requires source_rows=240 and sequential timestamps from the pinned history,
and validates all symbols before committing any state. Duplicate/missing/skipped
bars reject. Counted source rows are not proof of source minute uniqueness.

Ingestion lag is explicit, with no null→0 substitution. Publication is a scenario
start+4h+lag, not observed exchange arrival. An unpublished bar returns false
without touching price values, source hash or feature state; it can be retried
after publication. Snapshots require the complete prefix at that asof and reject
retrospective/stale requests, rather than using a partially supplied history.

EMA12/48 use first-close seed and streaming adjust=False recurrence from the fixed
history. State stores only the last 13 rows and four EMA12 values per symbol;
history is not restarted at each origin. Return/efficiency use 13 closes; proposal
uses EMA gap/slope and the full history, min48. Flat path abstains with zero features.
State is history-derived feature state, NOT warm-started scaler/model parameters.

Prefix SHA is a per-bar hash chain bound to manifest/history/revision-policy identity,
not the predecessor JSON-array hash. A declared revision-policy hash is not proof
of original revisions. The adapter itself does not decode, authenticate raw files
or prove caller rows came from the bound snapshot. A future caller must verify
each original file's byte size/SHA before/after decoding and align sorted bars.

Fixtures verify formulas against the pinned old pure EMA kernel, delayed visibility,
future poison, atomic failure, retained flat abstention, complete prefix and calendar.
No model parameters/scaler statistics calculated. Compile one new pure file, direct
calendar check, then checker --run/--verify. Outputs are exclusive-create and contain
only hashes/counters. Report lies outside receipt. Actual ingestion/market execution,
delayed labels, net-price bridge and continuous ledger remain separate obligations.
