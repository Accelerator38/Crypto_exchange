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

Three replacement research contracts were pre-registered in
`configs/strategy_candidates_v1.json`. The file is the immutable contract
snapshot, not a current authorization list. Validate it with:

```powershell
.\.venv\Scripts\python.exe tools\check_strategy_candidates.py
```

`regime_pullback_hourly_v1` and
`ohlcv_compression_transition_hourly_v1` have completed historical OOS and are
terminally rejected. Their evidence expectancy is respectively `-4.36 bps`
with `-18.55 bps` LCB and `-3.92 bps` with `-14.57 bps` LCB; both also fail
OOS/sanity, cost-stress, direction/regime-collapse and drawdown gates. Do not
retune thresholds, exits or per-symbol slices on the revealed OOS. A retry
requires a materially different event contract and a new candidate ID.
`funding_carry_hourly_v1` is also terminally rejected. A sealed public Bitget
snapshot provided 1,856 full8 settlement observations from 2026-04-28 through
2026-07-14 on the overlapping OHLCV window. After sign-persistence checks,
1,004 contexts remained, but none passed the fixed rate threshold: maximum
last funding was `3.99 bps` versus `8 bps`, and maximum projected carry was
`7.35 bps` versus the `16 bps` cost floor and `24 bps` registered gate.
Lowering the threshold would therefore enable entries that cannot cover the
registered costs.

Bitget's public funding endpoints exposed only a rolling window of about 90
days, and the official data-download page does not publish funding archives.
This recent screen cannot substitute for validation/OOS/sanity. Do not fill
older funding with zero, reconstruct it, or substitute another exchange's
rates. The unavailable older history remains a data-provenance blocker, while
the current profile is independently rejected by its activation/cost floor.
No candidate created a runtime actor or paper/live authority.

The sealed snapshot and diagnostic screen can be reproduced or verified with
one fixed command:

```powershell
.\.venv\Scripts\python.exe tools\run_funding_carry_research_v1.py
```

If the snapshot already exists, the command verifies its hashes and does not
replace it with a newer rolling API window.

The fixed `cross_sectional_trend_4h_v1` development screen is also terminally
rejected. It produced 191 closed trades and positive point expectancy
(`+48.28 bps` after costs), so activation was not the blocker. Robustness was:
the 95% LCB was `-54.53 bps`, cost-stress LCB was `-60.53 bps`, LONG averaged
`-28.25 bps`, every per-symbol LCB was negative, and drawdown exceeded the
registered limit. The fixed Donchian baseline averaged `+111.17 bps`, so the
candidate also failed the mandatory baseline comparison.

This was a development-only decision. Validation, OOS and sanity remained
sealed and were not consumed. Removing LONG, selecting winning symbols or
tuning thresholds/exits after seeing this result is forbidden under the same
family. A retry requires a materially different event and execution contract
with a new candidate ID. Reproduce the immutable profile evaluation with:

```powershell
.\.venv\Scripts\python.exe tools\run_cross_sectional_trend_development_v1.py
```

The materially different `market_neutral_relative_momentum_4h_v1` paired
contract is also terminally rejected at development. It opened equal-notional
LONG/SHORT legs atomically and produced 96 closed pairs with 386 fills. The
point estimate was positive (`+13.04 bps`) and beat the fixed raw-momentum pair
baseline (`+3.36 bps`), but the 95% LCB was `-74.06 bps`, stress LCB was
`-80.06 bps`, and drawdown was `$6.29` against the `$1` limit.

Validation/OOS/sanity stayed sealed. The max-holding exit subset was positive,
but removing the pair stop or tuning the holding horizon after seeing that
split is forbidden post-hoc selection. This profile cannot create a runtime
actor, paper canary or live manifest. Reproduce its immutable development run:

```powershell
.\.venv\Scripts\python.exe tools\run_market_neutral_pair_development_v1.py
```

Run the fixed-profile historical evaluation with no strategy flags:

```powershell
.\.venv\Scripts\python.exe tools\run_strategy_lab_v1.py
```

The evaluator verifies all registered dataset hashes, uses next-bar-open fills,
applies fees/slippage and cost stress, right-censors split-end positions and
enforces the registered validation/OOS/sanity gates. Its report is research-only
and cannot create a policy manifest. The terminal result is stored in
`Reports/StrategyLab/p2_historical_oos_20260727` and registered in the
authoritative experiment registry.

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
