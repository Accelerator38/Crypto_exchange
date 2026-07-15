# CarryFlow Bitget evidence tape

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
- Split market/context input requires `--allow-legacy-split-input` and is always
  marked not promotion-eligible.

## Collector

The public read-only collector writes:

`Retrodate/bitget_carryflow_tape/<run-id>/`

It is aligned to hourly bar close plus 30 seconds, targets 720 samples and never
enables orders. Current status is available with:

```powershell
.venv\Scripts\python.exe tools\check_bitget_carryflow_tape.py
```

Once enough contiguous samples exist, authoritative replay uses:

```powershell
.venv\Scripts\python.exe tools\run_policy_replay_v1.py `
  --evidence-tape Retrodate\bitget_carryflow_tape\<run-id>\carryflow_evidence_tape.jsonl `
  --symbols BTC,ETH,SOL,BNB,XRP,DOGE,ADA,LINK `
  --stride-minutes 60
```

Paper canary remains forbidden until this route has at least 95% context
coverage, 20 fills, 10 closed trades, positive expectancy after costs and a
positive lower confidence bound.
