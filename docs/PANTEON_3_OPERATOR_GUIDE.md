# Panteon 3 Operator Guide

Updated: 2026-06-29

This document is the current operator entrypoint for Panteon 3. It supersedes
old root launchers such as `Start_BITGET.py`, `Start_MEXC.py`,
`Start_ML.py`, `Start_ML_BITGET.py`, and `Start_panteon_v3.py`.

## Current Live Verdict

BITGET live is blocked by pre-flight as of 2026-06-29.

Latest checks:

- Matrix: `Reports/Panteon3PreLiveMatrix/bitget_focus_20260629/panteon3_pre_live_matrix_summary.json`
- Canary: `Reports/Panteon3Canary/bitget_after_context_deny/20260629_114446/canary_summary.json`
- Pre-flight reasons:
  - `matrix_failed:expectancy_usd -0.06278683036664612 <= 0`
  - `canary_failed`
  - `canary_nonpositive_expectancy`

Do not bypass this guard for real orders. The launcher will refuse
`live_futures` while these checks remain failed or stale.

## Unified Launcher

Use only:

```powershell
.\.venv\Scripts\python.exe Start_panteon.py
```

The active switches are at the top of `Start_panteon.py`:

```python
BITGET = "ON"
MEXC = "OFF"
trade_regime = "multi"
```

`BITGET = "ON"` and `MEXC = "OFF"` are the default focus. To dry-run the
launcher plan without starting workers:

```powershell
.\.venv\Scripts\python.exe Start_panteon.py --dry-run
```

To limit the run to one exchange:

```powershell
.\.venv\Scripts\python.exe Start_panteon.py --only BITGET
```

The parent process performs live pre-flight before spawning a worker. If
pre-flight fails, no child process is created and no order is sent.

## Trade Regimes

`trade_regime = "multi"` is the standard Panteon mode. Flash chooses between
eligible actors per symbol, action, regime, and recent causal evidence.

`trade_regime = "singlton(<actor>)"` is the manual single-real-actor mode. The
named actor is allowed to send real orders; the rest of the runtime remains
available for shadow/statistical collection where the production pipeline
supports it.

Examples:

```python
trade_regime = "multi"
trade_regime = "singlton(GeneticsCore)"
trade_regime = "singlton(genetic_core)"
trade_regime = "singlton(LiveVolCompress)"
trade_regime = "singlton(MomentumScalper)"
```

The spelling `singlton` is intentionally supported because that is the current
operator-facing flag. Common actor aliases are normalized, so
`genetic_core` maps to `GeneticsCore`.

## Agents, Players, Actors

An agent is the smallest signal source. Examples: `GeneticsCore`,
`MomentumScalper`, `LiveVolCompress`, `CarryFlowAgentV2`, `LiveCrashHunter`.

A player is a composed strategy/profile that may use multiple agents or a
specialized wrapper. Examples include solo wrappers such as
`Solo_MomentumScalper` and profile/ensemble players.

An actor is the label used by Flash when deciding who may trade. It can be an
agent label (`agent:LiveVolCompress`), a bare agent label (`LiveVolCompress`),
or an ensemble/solo label (`ensemble:Solo_LiveVolCompress`).

Panteon 3 must not treat an ensemble as a simple average of weak signals. The
current routing path prefers causal evidence:

- prior-bar actor/symbol/regime/action performance;
- promotion-derived best component labels;
- controlled exploration only while expectancy is not negative;
- negative-outcome context deny after failed closed trades;
- risk caps, min-notional checks, duplicate key checks, and exchange health.

## Market Regimes

The runtime assigns regimes such as `range_low_vol`, bullish, bearish, neutral,
and crash-like contexts. Regime is part of the routing key. A profitable actor
in one regime is not automatically promoted in another.

For low-volume range conditions, real execution is restricted by allowlists and
evidence. This prevents forced trading in contexts that recently produced
negative expectancy.

## Flash Routing

The Flash decision path is:

1. collect raw agent/player candidates;
2. normalize actor labels and action keys;
3. apply symbol/action/risk/exchange health checks;
4. apply causal memory and promotion-derived routing;
5. allow controlled exploration only below the configured risk cap;
6. reject actors with negative rolling expectancy for the same context;
7. emit real signals only for candidates that survive the full gate funnel.

No-trade is valid only when the gate funnel explains why candidates were
rejected. Important rejection reasons include:

- `controlled_exploration_negative_expectancy`
- `promotion_derived_negative_outcome`
- `controlled_exploration_min_notional_risk_too_high`
- `insufficient_closed_trades`
- `no_evidence`
- `foreign_position_owner`
- `pnl_below_threshold`
- `score_below_threshold`

## Execution Safety

Real positions must appear only after confirmed fills. Pending, rejected, or
unconfirmed orders must not create positions. Restart/recovery relies on:

- `OrderLedger` for sent/rejected/filled order state;
- `PositionTracker` for fill-confirmed positions;
- exchange reconciliation to correct drift;
- duplicate execution keys to prevent repeated order sends;
- pre-flight matrix/canary freshness and pass/fail gates.

## BITGET Verification Commands

Run the focused tests:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_start_panteon.py src\panteon_v2\tests\test_entrypoints.py src\panteon_v2\tests\test_live_preflight.py src\panteon_v2\tests\test_panteon3_live_canary_check.py -q
```

Run BITGET paper canary:

```powershell
.\.venv\Scripts\python.exe tools\run_panteon3_single_component_canary.py --exchange BITGET --actor LiveOIBreakout --fallback-actor LiveVolCompress --results-root Results\Panteon3BitgetCanaryAfterContextDeny --reports-dir Reports\Panteon3Canary\bitget_after_context_deny --initial-capital 2000 --exploration-risk-mult 0.25 --max-bars 180 --max-idle-polls 180 --sleep-between-polls-sec 2 --warmup-bars 1440 --lookback-minutes 1440
```

Run BITGET-focused matrix:

```powershell
.\.venv\Scripts\python.exe tools\run_panteon3_pre_live_matrix.py --years 2026 --max-bars 360 --window-skip-bars 0,72,144 --stride-minutes 60 --initial-capital 2000 --risk-capital-fraction 0.02 --results-root Results\Panteon3PreLiveMatrixBitgetFocus_20260629 --reports-dir Reports\Panteon3PreLiveMatrix\bitget_focus_20260629 --min-filled 1 --min-closed-trades 1 --min-profitable-windows 1 --min-positive-regimes 1
```

Check live pre-flight:

```powershell
.\.venv\Scripts\python.exe -c "import json; from pathlib import Path; from src.panteon_v2.app.live_preflight import config_from_env, run_live_preflight; r=run_live_preflight('BITGET','live_futures',config=config_from_env(Path('.').resolve())); print(json.dumps({'passed':r.passed,'reasons':list(r.reasons),'matrix':r.matrix_summary_path,'canary':r.canary_summary_path}, indent=2, ensure_ascii=False))"
```

Only after this returns `passed: true` should `Start_panteon.py` be used for
real BITGET `live_futures`, and only with a tiny risk cap.

## Current No-Go Fix List

Before live-size or live restart, fix and re-run evidence for:

1. positive multi-window BITGET expectancy after fees, funding, spread and
   slippage;
2. canary with nonzero signals/orders/fills and positive expectancy;
3. stable actor routing where Panteon 3 beats or justifies not using the best
   causal component;
4. explicit explanation for remaining inactive specialists such as
   `CarryFlowAgentV2`, `MomentumScalper`, and `LiveVolCompress`;
5. clean reconciliation with zero unexpected open positions.
