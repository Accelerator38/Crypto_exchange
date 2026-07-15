# Pantheon vNext: strategy for removing the root causes

## Decision

The current Flash/Pantheon route is frozen as a Bitget live candidate. It remains
available for research and historical comparison, but it must not authorize new
live entries.

The replacement is a deliberately small runtime with one active policy, one
immutable manifest and one decision trace per candidate signal:

```
research agents -> evidence gates -> sealed PolicyManifest
                                      |
hash-chained market+derivatives tape -> one actor signal
                                      -> PolicyExecutorV1 -> TradeExecutor -> exchange
                                      |
                                   NoTrade
```

`PolicyExecutorV1` does not rank actors, mutate policy, promote genetics, run a
tournament or perform controlled exploration. GeneticsCore and Flash remain in
the research layer.

## Root causes in the current route

1. Research, promotion and execution are coupled. A candidate can look useful
   standalone while Flash selects nothing in the actual execution path.
2. The runtime state space is too large. `flash_allocator.py`, `main_loop.py`
   and `startup.py` contain many interacting gates, fallbacks and experimental
   flags. A passing test of one branch does not prove the complete route.
3. Matrix, canary and live do not share one immutable policy contract. Runtime
   configuration can change actor selection and eligibility after the report was
   produced.
4. A zero-fill canary is often an activation mismatch, not evidence about edge.
   Re-running it in the same blocked regime adds time but no information.
5. Evidence artifacts can be stale or mixed across candidates. The existing
   candidate policy hash identifies a JSON document, but it does not require
   validation, OOS, cost stress and sanity artifacts to belong to that policy.
6. Bypass and controlled-exploration branches turn a failed gate into another
   configuration experiment. This hides whether the strategy has executable
   positive expectancy.

## Runtime invariants

- No valid manifest means `NoTrade`.
- A `replay` manifest may omit promotion evidence, but mode matching makes it
  unusable in paper or live. It exists only to pin research code and parameters.
- Bitget live requires a `micro_live` manifest and an externally pinned SHA-256.
- Exactly one actor is active in a manifest. There is no live actor tournament.
- Actor parameters and expected-move calibration are immutable manifest fields.
- Bar interval, cadence tolerance, bar-close lag and derivatives freshness are
  immutable manifest fields. Changing stride therefore creates a different
  policy contract instead of silently changing actor behavior.
- CarryFlow replay requires point-in-time funding, open interest, account
  long/short ratios, mark and index prices. Price-only fallback is not CarryFlow
  evidence.
- Derivatives rows must precede or equal the replay bar. Future rows are
  inaccessible, stale rows are rejected, and context coverage below 95% is a
  hard evidence failure.
- Rules are exact `(symbol, regime, direction)` tuples. Wildcards are forbidden.
- Every paper manifest requires passing `validation`, `oos`, `cost_stress` and
  `sanity` evidence.
- Every micro-live manifest additionally requires passing short and extended
  paper canaries.
- Costed expectancy and its lower confidence bound must be positive. Recorded
  costs must be non-zero. Direction/regime collapse is a hard failure.
- Evidence files are content-hashed, candidate-bound, exchange-bound and fresh.
- The manifest pins a byte-level runtime fingerprint for the complete active
  actor/policy/execution route; Git revision alone is not accepted as proof of
  the code used by replay or live preflight.
- Gate metrics must match normalized evidence receipts; every receipt pins its
  source reports by path and SHA-256.
- The expected move must exceed fees, slippage and a safety buffer at runtime.
- The first failed check is the primary decision reason and is always emitted in
  the trace.
- Entry authorization is isolated in `PolicyExecutorV1`. Actor exits and
  manifest stop/holding/expiry exits use a separate close-only safety path.

## Implemented production boundary (2026-07-15)

The Bitget production wiring now enforces the contract instead of only checking
it in preflight:

- `Start_panteon.py` accepts non-virtual Bitget only with
  `PANTEON_TRADE_REGIME=policy`;
- direct `start_production()` calls require `enable_policy_runtime=True`, so a
  worker cannot bypass the launcher;
- startup loads the externally pinned `micro_live` manifest, registers exactly
  its `CarryFlowAgentV2`, forces leverage to 1 and constrains `TradeExecutor`
  risk to the manifest;
- Flash, Genetics, fallback actor selection and the production shadow
  tournament are disabled on this route;
- the transport loop may poll every five seconds, but actor evaluation happens
  only once on an exact manifest-cadence closed bar;
- the actor is warmed from an exact synchronized Bitget OHLCV window, while
  historical derivatives context is never fabricated;
- every entry cadence requires a complete public order book and uses observed
  spread plus visible-depth slippage for the declared maximum notional;
- the manifest absolute daily-loss limit latches a non-recovering manage-only
  kill switch and routes any owned position through the close-only path;
- the UTC-day loss baseline and latch survive process restarts in an atomic
  policy risk-state file; malformed state blocks startup;
- `status.json` exposes a dedicated `policy` block and always reports
  `flash.enabled=false` for this route;
- compact causal logging preserves each evaluated policy cadence, including the
  first `NoTrade` check.

The runtime fingerprint now includes the launcher, live preflight, Bitget feed
and adapter, startup, main loop, policy runtime, regime detector, actor and v2
execution stack. A manifest compiled before this boundary cannot authorize the
new code.

This implementation is not a live approval. Without fresh passing evidence,
strict paper canaries, a new sealed manifest and explicit manual confirmation,
startup remains fail-closed.

## Migration plan

### Stage 1: contract and live freeze

- Implement strict manifest parsing, evidence validation and content pinning.
- Implement deterministic entry authorization and a uniform decision trace.
- Make the Bitget launcher require a valid, pinned vNext micro-live manifest.
- Keep the existing Flash route usable only for research and virtual trading.

Exit criterion: malformed, stale, weak, mismatched or unpinned policy artifacts
cannot start Bitget live.

### Stage 2: exact replay adapter

- Adapt one actor at a time to emit `CandidateSignal`.
- Replay the exact `PolicyExecutorV1 -> TradeExecutor` route on historical bars.
- Collect Bitget public derivatives context forward and synchronize it to each
  replay bar without look-ahead. Market and derivatives data must live in one
  hash-chained evidence tape; split CSV inputs are diagnostic-only.
- Compare signal, decision, order and fill counts between replay and paper.
- Remove candidate-specific canary configuration rewrites.

Exit criterion: deterministic parity report with no unexplained activation gap.

### Stage 3: paper lifecycle

- Compile a paper manifest only from robust evidence.
- Run short paper only when the current regime matches at least one rule.
- Recompile to micro-live only after a positive extended canary.
- Archive failed candidate/version pairs instead of retuning their gates.

Exit criterion: non-smoke fills, positive costed expectancy, positive LCB and no
open owned positions after shutdown.

### Stage 4: micro-live

- Require manual confirmation of manifest hash, symbols, maximum notional,
  maximum positions, daily loss and expiry.
- Start with one manifest and one small Bitget position at a time.
- Stop on manifest expiry, reconciliation error, daily loss or evidence mismatch.

Exit criterion: reconciliation-clean micro-live session followed by a deliberate
risk review. Scaling is a later decision, not an automatic promotion.

## Explicitly removed from the live decision path

- GeneticsCore promotion/probation and all genetics bypass flags.
- Flash actor ranking, shadow-score promotion and controlled exploration.
- Dynamic fallback from an ineligible actor to another actor.
- Live-time policy mutation and terminal-deny exceptions.
- Re-running a canary when the observed regime cannot activate the manifest.

These features may continue in R&D, but their output is only a candidate evidence
artifact. They never become live authority by themselves.
