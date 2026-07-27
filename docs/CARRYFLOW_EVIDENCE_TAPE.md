# CarryFlow Bitget evidence tape

> Archived research path as of 2026-07-27. No CarryFlow experiment is an
> operational candidate. The terminal state is recorded in
> `configs/strategy_experiment_registry_v1.json`. The commands below document
> evidence provenance and cannot authorize a new campaign, paper canary or live
> order for the rejected profiles.

## Why this replaces split replay inputs

CarryFlow evidence must come from one synchronized observation. A price CSV and
a separately collected funding/OI CSV can have different windows, cadence,
symbol coverage or source code. Their combination is diagnostic only, even when
the resulting PnL is positive.

The authoritative input is now one append-only JSONL tape. Every sample contains
the full symbol set, one closed OHLCV bar, the decision price, funding, open
interest, account long/short ratios, mark/index prices and source timestamps.
Each line pins the previous line hash, collector code hash and Bitget fetcher
code hash.

Collector and policy provenance are intentionally separate. The tape records
the source revision and exact hashes used to collect market evidence. A replay
manifest records the later policy revision and runtime fingerprint being
evaluated. Historical tapes therefore remain usable after policy commits; the
report exposes both revisions and marks a cross-revision replay explicitly.

The policy manifest separately pins a runtime fingerprint over the actual
CarryFlow, adapter, policy, execution, risk, position and attribution source
files. This remains valid in a dirty worktree and fails if any byte on the
decision route changes after the manifest is sealed.

## Hard invariants

- Source is exactly `BITGET` linear swap.
- Bar interval, collection lag and derivatives age match the policy manifest.
- Every bar close is present exactly once; gaps and duplicates invalidate the
  tape.
- Bar-close timestamps satisfy the manifest cadence. Observation timestamps
  are strictly increasing and remain bounded by the declared collection lag.
- Tampering with any field breaks the SHA-256 chain.
- Incomplete derivatives context remains visible and reduces coverage. It is
  never replaced by price-only fallback evidence.
- Sealed OHLC drives protective-stop replay. An open position with missing OHLC
  invalidates the replay instead of falling back to the next hourly close.
- Split market/context input requires `--allow-legacy-split-input` and is always
  marked not promotion-eligible.

## Collector

The public read-only collector writes:

`Retrodate/bitget_carryflow_tape/<run-id>/`

It never enables orders. A new root also writes
`carryflow_warmup_seed.json`: 53 closed OHLCV bars collected and sealed before
the first evidence sample. The seed initializes EMA/RSI state but contains no
derivatives context, creates no actions and is not counted as trade evidence.
This removes 53 otherwise idle continuous hours after every clean restart.

Do not use a fixed 720-hour target. Evaluate the root sequentially and stop when
the candidate passes the event-count/statistical gates or reaches an explicit
failure boundary. Current status is available with:

```powershell
.venv\Scripts\python.exe tools\check_bitget_carryflow_tape.py
```

Once enough contiguous samples exist, authoritative replay uses:

```powershell
.venv\Scripts\python.exe tools\run_policy_replay_v1.py `
  --evidence-tape Retrodate\bitget_carryflow_tape\<run-id>\carryflow_evidence_tape.jsonl `
  --warmup-seed Retrodate\bitget_carryflow_tape\<run-id>\carryflow_warmup_seed.json `
  --profile screened_short_v1
```

`--profile` is the only algorithm setting. The profile derives direction,
regimes, OI/price lookbacks, entry conditions, exit behavior, stop, holding
horizon and feature-to-edge mapping. Its manifest `actor_config` contains only
`PROFILE_ID`; mixing it with scalar actor overrides fails closed. Exchange costs
and absolute capital limits remain explicit because they are deployment inputs,
not strategy tuning knobs.

The authoritative tape also supplies the exact symbol set and bar interval.
They are not repeated as normal replay settings. Optional `--symbols` and
`--stride-minutes` values act only as fail-closed assertions for an existing
tape; they cannot select a subset or reinterpret its cadence.

Paper canary remains forbidden until this route has at least 95% context
coverage, 20 fills, 10 closed trades, positive expectancy after costs and a
positive lower confidence bound.

## Sealed, restart-safe evidence campaign

Process uptime is not an evidence requirement; exact bar continuity is. The
campaign wrapper replaces manually managed run IDs and collector settings. Its
first invocation accepts the exact profile report, verifies that the only
failure is `nonpositive_lcb`, pins the published Git revision and runtime
fingerprint, and creates a self-contained read-only campaign lock:

```powershell
$Campaign = "Retrodate\bitget_carryflow_campaign\systemic_guard_<revision>"
.venv\Scripts\python.exe tools\run_carryflow_evidence_campaign.py `
  --campaign-dir $Campaign `
  --exact-report Reports\CarryFlow\exact_profile_evaluation_latest.json
```

Schedule the shorter form at minute `01` of every hour:

```powershell
.venv\Scripts\python.exe tools\run_carryflow_evidence_campaign.py `
  --campaign-dir $Campaign
```

There are no strategy, symbol, cadence or risk flags on this command. The lock
contains exactly one `PROFILE_ID`; full8, hourly cadence, freshness limits and
the 53-bar prospective warm-up are derived from the screened evidence contract.
The process exits after at most one public-data observation and always records
`orders_enabled=false` and `promotion_authority=false`.

The wrapper resumes the current root only for the exact next hourly bar. A
duplicate invocation is an idle no-op. A real machine/network gap automatically
starts and seals a new independent root. It never appends across the gap and
never concatenates roots. A code change, unpublished commit, modified report,
runtime fingerprint change or altered root lock fails closed and requires a new
campaign.

A completed negative root is sealed with:

```powershell
.\.venv\Scripts\python.exe tools\finalize_carryflow_negative_root.py `
  --campaign-dir Retrodate\bitget_carryflow_campaign\<campaign> `
  --root-dir Retrodate\bitget_carryflow_campaign\<campaign>\roots\<root> `
  --replay-summary Reports\PolicyReplayV1\<replay>\replay_summary.json
```

The terminal `root_verdict.json` pins the campaign/root hashes, tape SHA/head,
replay summary and manifest hashes, final metrics and safety flags. Once present,
the campaign runner returns `completed_negative` without environment
verification, public-data collection or creation of another root. The verdict
is fail-closed: mutation of the tape or verdict invalidates the campaign.

Diagnose a rejected root without changing its policy:

```powershell
.\.venv\Scripts\python.exe tools\analyze_carryflow_signal_quality.py `
  --evidence-tape Retrodate\bitget_carryflow_campaign\<campaign>\roots\<root>\carryflow_evidence_tape.jsonl `
  --replay-dir Reports\PolicyReplayV1\<replay>
```

The analyzer reconstructs the OI funnel, edge components, expected-move
calibration, per-symbol incidence and fixed-horizon/MFE/MAE diagnostics. Its
outputs are research-only and cannot select a policy from the rejected root.

CarryFlow diagnostics now distinguish candidate ranking from the feature used
by the manifest signal model. The current v1 profile still declares
`signal_feature=diagnostic.edge` for backward-compatible replay, but traces pin
the exact `feature_name`, and the actor also emits `ranking_score`,
`oi_excess_bps` and `price_dislocation_bps`. A future profile may use only a
manifest-supported diagnostic feature; an unknown or missing feature fails
closed. Adding a feature does not calibrate it and does not create promotion
evidence.

The single `hold=4` hypothesis selected from root 003 was evaluated without
retuning on the earlier independent roots:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_carryflow_exit_hypothesis.py `
  --root root003=Reports\CarryFlow\root003_signal_quality_20260727.json `
  --root v1=Reports\CarryFlow\v1_signal_quality_20260727.json `
  --root v4=Reports\CarryFlow\v4_signal_quality_20260727.json `
  --root v5=Reports\CarryFlow\v5_signal_quality_20260727.json `
  --selection-root root003 `
  --output-json Reports\CarryFlow\fixed_exit_cross_root_20260727.json `
  --output-md Reports\CarryFlow\fixed_exit_cross_root_20260727.md
```

It was rejected: the independent mean was `-18.12` net bps versus `-1.17`
net bps for the six-bar baseline, with negative candidate roots `v1` and `v5`.
Consequently no four-bar profile is registered. Do not rerun this hypothesis
as if it were untested; the next entry/exit design must be materially different
and then validated on a new prospective root.

The same report screens `ranking_score`, `price_dislocation_bps` and
`oi_excess_bps` against the subsequent six-hour gross move. There are only 16
usable observations (minimum 20), and none keeps a positive correlation sign
on every root. Therefore no alternative expected-move model or v2 profile is
registered from these data.

The broader entry redesign uses all complete, non-overlapping forward labels
instead of only historical actor entries:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_carryflow_forward_labels.py `
  --root "v1=<v1-tape>::<v1-replay-dir>" `
  --root "v4=<v4-tape>::<v4-replay-dir>" `
  --root "v5=<v5-tape>::<v5-replay-dir>" `
  --root "root003=<root003-tape>::<root003-replay-dir>" `
  --output-json Reports\CarryFlow\forward_labels_loro_hybrid_20260727.json `
  --output-md Reports\CarryFlow\forward_labels_loro_hybrid_20260727.md
```

The fixed hybrid model combines flow, price response, volume ratio, volatility,
bar structure and regime context. Every prediction is produced by a model that
was trained without that prediction's root. Labels include the profile stop,
target and per-root modeled costs. The result is rejected: 209 rows produced 22
held-out trades with `-1.42` mean net bps and `-39.66` bps LCB; root `v5`
collapsed to `-73.64` mean net bps with zero positive trades. The preceding
flow-only model was also rejected (`-0.36` mean, `-30.37` LCB, root collapses
in `v1` and `v5`).

These are terminal results for the current CarryFlow feature family. Do not add
another threshold, hold variant or linear feature combination on these roots.
No runtime profile was created. Further strategy work must define a materially
different event and reserve a new prospective root before model selection.

The materially different event audit is implemented separately:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_range_transition_breakout.py `
  --root "v1=<v1-tape>::<v1-replay-dir>" `
  --root "v4=<v4-tape>::<v4-replay-dir>" `
  --root "v5=<v5-tape>::<v5-replay-dir>" `
  --root "root003=<root003-tape>::<root003-replay-dir>" `
  --output-json Reports\CarryFlow\range_transition_breakout_20260727.json `
  --output-md Reports\CarryFlow\range_transition_breakout_20260727.md
```

Its immutable event contract requires a 12-bar range no wider than 3%, close
outside the range, volume ratio at least 1.3, three-bar OI expansion at least
1.5%, previous symbol regime `range_low_vol`, directional regime consistency,
six-bar cooldown and at most one portfolio entry per bar. The result is
promising but rejected: 8 trades, `+11.74` mean net bps and `-15.82` bps LCB.
Root `v4` had zero confirmed events and `v5` had negative expectancy. No actor
class or evidence campaign is created until a fixed event passes root incidence,
10-trade, costed expectancy and LCB gates.

`campaign_status.json` reports each root SHA/head, samples, complete samples and
weighted context coverage. Root totals are operational aggregates only; replay
continues to evaluate every root separately with endpoint censoring and
root-collapse checks.

Optional funnel diagnostics across existing roots use:

```powershell
.venv\Scripts\python.exe tools\analyze_carryflow_segments.py `
  --output Reports\CarryFlow\segment_screening_latest.json
```

This diagnostic is not a promotion stage. It may explain activation, but its
output is explicitly `promotion_authority=false`.

The single mandatory offline stage is exact profile evaluation. It requires one
warm-up seed per independent root:

```powershell
.venv\Scripts\python.exe tools\sweep_carryflow_exact_segments.py `
  --segment "<root-1-tape>::<root-1-seed>" `
  --segment "<root-2-tape>::<root-2-seed>" `
  --output Reports\CarryFlow\exact_profile_evaluation_latest.json
```

This route sets `--no-flatten-end` internally. A position still open at an
interrupted root boundary is right-censored and cannot create a synthetic
winning or losing trade. Selection additionally fails on a negative active root
even when the combined point estimate is positive, and uses the fixed `$0.20`
drawdown ceiling for the `$10` research notional. Start a prospective
paper-validation root only when the exact report contains a non-null
`selected_for_prospective_validation`. A non-null
`selected_for_evidence_extension` authorizes only additional no-order market
collection when `failures=[nonpositive_lcb]`; it cannot authorize paper or live.

Do not create a historical evidence tape before checking source capability:

```powershell
.venv\Scripts\python.exe tools\audit_bitget_historical_evidence.py `
  --output Reports\CarryFlow\bitget_historical_capability_latest.json
```

The current Bitget result is `historical_open_interest_unavailable`, so
`evidence_tape_creation_allowed=false`. Historical OHLCV, funding and ratios are
diagnostic-only because the strategy also requires timestamped historical OI.

The reduced path is therefore:

```text
independent tapes + immutable seeds
  -> exact versioned-profile evaluation with costs and LOO
  -> reject, no-order LCB evidence extension, or one prospective candidate
```

There is no scalar grid, separate candidate-policy builder or normalization
variant stage on this path.

## Sequential decision points

- At 24 samples: verify hash chain, 100% expected cadence and context coverage.
- At 48-72 post-seed samples: reject activation failures instead of waiting for
  hundreds of additional bars with zero exact actor candidates.
- Continue only while candidate incidence can realistically reach 20 fills and
  10 closed trades by the 168-hour review.
- At or before 168 hours: require positive costed expectancy and positive LCB;
  otherwise reject or redesign the candidate.
- Only a passing prospective replay proceeds to cost stress, sanity, strict
  short paper canary and strict extended paper canary.

## Production parity boundary

The Bitget policy runtime now consumes the same decision contract as the tape:

- one exact closed candle at the manifest interval;
- a decision-time ticker price;
- current complete derivatives context;
- collection/decision lag bounded by the manifest;
- current order-book spread and visible-depth slippage for the manifest maximum
  notional.

The same profile-derived absolute protective stop is used throughout the
route. Replay models an intrabar stop-market fill from sealed OHLC, including a
worse bar open after a gap and configured slippage. A Bitget policy open must
atomically include the server-side `presetStopLossPrice`; if the exchange client
cannot provide that capability, the open is rejected before order submission.
The frequent market-price check remains a fallback for reconciliation and does
not define the primary stop semantics.

At startup, the actor price state is warmed from synchronized public Bitget
closed bars. The derivatives fetcher is disabled during warmup, because copying
the current OI/funding value into historical bars would create false evidence.
Live OI history therefore becomes actionable only after genuine cadence samples
arrive.

Collector output is read-only evidence. It is never a paper canary, never a
live status artifact and never proof that an order was attempted. Existing tapes
remain valid historical inputs with their recorded collector hashes, but a new
policy replay and manifest must pin the current runtime fingerprint before any
promotion decision.
