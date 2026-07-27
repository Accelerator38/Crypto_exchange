# Panteon Operator Guide

Updated: 2026-07-27

## Current verdict

Bitget real trading is **not authorized** until a current `micro_live` policy
manifest exists and all evidence gates pass. An evidence collector or replay is
not a trading process and cannot authorize an order.

There is currently no operational strategy candidate. The authoritative
research state is tracked in
`configs/strategy_experiment_registry_v1.json`; it has
`operational_candidate_id=null`, `orders_enabled=false` and
`promotion_authority=false`. Validate it with:

```powershell
.\.venv\Scripts\python.exe tools\check_strategy_experiment_registry.py
```

The old `configs/bitget_hypotheses_20260706.json` manifest is archived
research-only. Its legacy sweep and canary entry points fail closed and must not
be used to restart CarryFlow, Flash or ensemble promotion work.

The `divergence_short_systemic_guard_v1` evidence extension is complete and
rejected. Prospective root 003 finished with 74 samples, 10 fills, 5 closed
trades, negative costed expectancy and a negative 95% LCB. Its terminal
`root_verdict.json` forbids further collection in that campaign. It is not an
active policy and must not be used to create a paper or live manifest.

The complete CarryFlow flow-feature family is terminally rejected for the
current event contract. Changes limited to thresholds, fixed holding time or
another linear combination of the same flow/OHLCV features are forbidden
retries. The range-transition OI contract is also terminally rejected as
specified; only a materially different event contract that first passes
historical OOS may receive a new experiment ID.

Signal-quality analysis also rejected replacing the six-hour exit with a fixed
four-hour exit. On the independent historical roots, the four-hour candidate
averaged `-18.12` net bps versus `-1.17` net bps for the same entries at six
hours and collapsed in roots `v1` and `v5`. No four-hour policy profile was
registered.

The runtime now records the manifest's exact signal feature separately from the
actor ranking score. This is an observability and fail-closed contract change,
not a new approved model: `diagnostic.edge` remains uncalibrated, and no
alternative expected-move feature currently has sufficient validation/OOS
evidence. The cross-root screen had only 16 usable observations, and
`ranking_score`, price dislocation and OI excess each changed correlation sign
between roots. No v2 profile was registered.

Forward-label redesign did not rescue CarryFlow. A flow-only ridge and one
fixed hybrid flow/OHLCV ridge were evaluated leave-one-root-out on
non-overlapping six-hour labels. The hybrid selected 22 held-out trades but
returned `-1.42` mean net bps and `-39.66` bps LCB; `v5` collapsed to `-73.64`
bps with 0% positive trades. This closes the current CarryFlow feature family:
do not resume its campaign, lower its thresholds or add another exit variant.

A separate range-transition breakout event was audited without adding a runtime
actor. Compression + close-outside-range + volume + OI confirmation produced 8
portfolio trades with `+11.74` mean net bps, but LCB remained `-15.82` bps;
`v4` was inactive and `v5` had negative expectancy. It is research-only and
cannot start paper/live or a new evidence campaign.

The old Flash/Panteon ensemble remains available for virtual research. It is no
longer a Bitget live route.

The archived evidence-extension campaign can be audited with its sealed
command:

```powershell
.\.venv\Scripts\python.exe tools\run_carryflow_evidence_campaign.py `
  --campaign-dir Retrodate\bitget_carryflow_campaign\<campaign>
```

The terminal verdict makes this command an idempotent no-op with
`run_state=completed_negative`; it does not fetch another observation. The
campaign cannot create a paper/live manifest or submit an order. A new campaign
requires a materially changed profile that first passes offline validation,
OOS, cost stress and sanity.

## One launcher

Use only the project virtual environment and the unified launcher:

```powershell
.\.venv\Scripts\python.exe Start_panteon.py --only BITGET
```

The safe repository default is:

```text
BITGET=ON
MEXC=OFF
PANTEON_TRADE_REGIME=multi
```

That default deliberately blocks non-virtual Bitget startup. It cannot send a
real Bitget order.

## Trade regimes

| Regime | Bitget paper/R&D | Bitget real trading |
|---|---:|---:|
| `multi` | allowed | blocked |
| `singlton(<actor>)` | allowed | blocked, even with a bypass flag |
| `policy` | blocked; paper canaries use their dedicated runner | only possible live route |

`multi` runs the legacy Flash/ensemble selection. `singlton(<actor>)` is a
diagnostic single-actor wrapper. Neither is evidence-compatible with the new
Bitget live contract.

`policy` loads exactly one actor from the sealed manifest. The current supported
actor is `CarryFlowAgentV2`. Startup registers only this actor and disables
Flash, Genetics, fallback selection and the shadow tournament.

New CarryFlow manifests use one algorithm field:

```text
actor_config.PROFILE_ID=<versioned profile>
```

Actor thresholds, direction, regimes, stop and holding horizon are derived and
cross-checked against that profile. Profile plus scalar actor overrides is an
invalid configuration. Explicit exchange costs, notional and daily-loss limits
remain deployment controls.

Complete scalar actor configs remain accepted only by historical replay
diagnostics. Paper and micro-live fail closed unless `actor_config` contains
exactly one key, `PROFILE_ID`.

## Required live environment

Do not set these values until the manifest has been reviewed and manually
approved:

```text
BITGET_TRADING_MODE=live_futures
PANTEON_TRADE_REGIME=policy
BITGET_POLICY_MANIFEST_V1=Runtime/BITGET/active_policy_manifest_v1.json
BITGET_POLICY_MANIFEST_SHA256=<exact 64-character SHA-256>
```

The launcher and the direct worker both run preflight. A child process is not
spawned when the manifest is absent, stale, unpinned, tied to another Git
revision, or fails its runtime fingerprint/evidence checks.

`--dry-run` performs the same read-only regime and preflight checks but never
spawns the worker.

## Manifest gates

A Bitget `micro_live` manifest must pin one exact policy and contain passing,
fresh receipts for:

- validation;
- OOS;
- cost stress;
- sanity;
- short strict paper canary;
- extended strict paper canary.

Validation, OOS and cost stress require at least 20 fills, 10 closed trades,
positive expectancy after fees/slippage, positive LCB, non-zero costs and
drawdown within the declared limit. Direction or regime collapse is a hard
failure.

The runtime fingerprint includes the launcher, Bitget market adapter, startup,
main loop, policy runtime, bridge feed, actor, policy checks and execution stack.
Any byte change on that route invalidates a previously sealed manifest.

## Runtime behavior

The vNext route is:

```text
closed Bitget bar + current derivatives context + executable order book
  -> one CarryFlow actor
  -> exact (symbol, regime, direction) manifest rule
  -> PolicyExecutorV1
  -> TradeExecutor
  -> Bitget
```

Operational invariants:

- hourly policies evaluate once per exact closed hourly bar, not every 5-second
  transport poll;
- startup warms the actor from a synchronized public Bitget OHLCV window;
- missing candles, stale data, incomplete context or insufficient book depth
  produce `NoTrade`;
- a missing/broken real Bitget adapter terminates startup; non-virtual modes
  never fall back to `FakeExchange`;
- spread and visible-book slippage are checked against manifest limits;
- each policy open derives one absolute stop from the immutable profile and
  must attach it atomically as a Bitget server-side stop; unsupported protection
  blocks the open before order submission;
- the same stop is modeled from sealed OHLC in replay, while market-poll stop
  handling remains a live fallback alongside maximum holding time, manifest
  expiry and the global kill switch;
- reaching `max_daily_loss_usd` latches a policy-specific manage-only kill
  switch; it does not auto-recover;
- the UTC-day equity baseline and daily-loss latch are atomically persisted in
  `panteon_v2_state/bitget_policy_risk_v1.json`, so a same-day restart cannot
  clear the stop;
- no policy candidate can fall back to Flash, Genetics or another actor.

## Operator-visible proof

For the active session inspect `status.json` and
`causal_entry_decisions.jsonl` in its `Results/BITGET/...` directory.

The expected status fields are:

```text
policy.enabled=true
policy.actor=CarryFlowAgentV2
policy.manifest_sha256=<approved SHA>
flash.enabled=false
configured_actor_pool=[CarryFlowAgentV2]
```

`policy.status`, `policy.reason`, `policy.market_quality` and the policy decision
checks explain every evaluated cadence. A valid hold/no-trade cadence is kept in
the causal log even when compact logging is enabled.

## Verification

Run the focused safety tests:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  src\panteon_v2\tests\test_live_policy_runtime.py `
  src\panteon_v2\tests\test_start_panteon.py `
  src\panteon_v2\tests\test_policy_manifest_v1.py `
  src\panteon_v2\tests\test_live_preflight.py -q
```

Check preflight without starting a worker:

```powershell
.\.venv\Scripts\python.exe -c "import json,sys; from pathlib import Path; sys.path.insert(0,str(Path('src').resolve())); from panteon_v2.app.live_preflight import config_from_env, run_live_preflight; r=run_live_preflight('BITGET','live_futures',config=config_from_env(Path('.').resolve())); print(json.dumps({'passed':r.passed,'reasons':list(r.reasons)},indent=2))"
```

Real trading still requires a separate manual confirmation of the exact
manifest SHA, symbols, `max_notional_usd`, `max_open_positions`,
`max_daily_loss_usd` and expiry. Passing tests alone is not approval.
