# Pantheon reset implementation: points 1-4

Date: 2026-08-03

Status: complete for engineering and screening scope. Live and paper promotion
remain unauthorized.

## 1. Legacy Pantheon freeze

The pre-reset worktree was preserved before any architectural changes:

- branch: `codex/panteon-reset-20260803`;
- snapshot commit: `c0fd836c77a6ce56d29cfaeefc8a8bdc382d7f0c`;
- annotated tag: `panteon-research-freeze-2026-08-03`;
- snapshot inventory:
  `configs/research_snapshots/panteon_freeze_2026-08-03.json`;
- inventory file SHA256:
  `a7a5d2f6dc0a8a87f567ed60007323cb69d65712caf1c6513830a5aea39fb921`.

`Start_panteon.py` and direct `panteon_v2.app.startup.start_production()` now
call the same fail-closed guard. Bitget external-order modes are blocked before
preflight, credential loading or exchange construction. Paper, paper-live-feed,
shadow-live-feed and demo modes remain available. There is no environment
bypass.

## 2. Isolated Freqtrade execution spike

Freqtrade `2026.7` is installed in `.venv-freqtrade`, separate from the Pantheon
runtime. The checked config has:

- `dry_run=true`;
- Bitget linear USDT futures;
- isolated margin;
- one position maximum;
- fixed `10 USDT` simulated stake;
- empty exchange credentials;
- disabled API and Telegram;
- server-side stop configuration;
- no force-entry.

Both `NoTradeStrategy` and the dry-run-only `DeterministicPulseStrategy` reached
the Freqtrade `RUNNING` state. A separate strategy-contract check emitted one
deterministic entry and one exit on synthetic candles, and rejected startup
with `dry_run=false`. Public Bitget metadata was validated:

| Field | Value |
|---|---:|
| Symbol | `BTC/USDT:USDT` |
| Active linear swap | `true` |
| Minimum notional | `5 USDT` |
| Minimum amount | `0.0001 BTC` |
| Amount precision | `0.0001` |
| Price precision | `0.1` |

Acceptance artifact:
`Reports/FreqtradeReset/acceptance_v1.json`, SHA256
`ae42fad9a7b90a13e979163ae4a526775fd781d2e080dad646f599020cf682e4`.

This validates connector/configuration plumbing only. No exchange order was
sent and no profitability claim is made.

## 3. Canonical feature and decision tapes

The independent `simple_research` package has one contract:

```text
sealed OHLCV -> bar-close features -> {-1,0,+1} target
             -> next-bar open -> shared cost model -> shared ledger
```

It does not import Pantheon, Flash, Genetics, selectors or an exchange adapter.
All strategies use base costs of `12 bps` round trip and stress costs of
`16 bps` round trip.

Generated tapes:

| Tape | Rows | Symbols | SHA256 |
|---|---:|---:|---|
| Bitget 1h | 317,952 | 8 | `ead9db0169be907f85e143d9672769fbf49044afa315c499808b399ebf05290f` |
| Bitget 1m | 506,880 | 8 | `bdf9adf0461960dce1c8378fc094fd0281cc4bf06c085ce7fd91a6b4a0c8fb2d` |
| Decision tape, 10 strategies | 317,952 | 8 | `eeb5562f6d256a3efa662673ab840e64ddbf1f31414215d48095063d9bfbcf95` |

Both feature tapes passed continuity, duplicate, finite-value and OHLCV
validation. Features contain no future labels. Signals are delayed by one bar
before execution.

## 4. Preregistered strategy batch

The registry is limited to ten trials and at most two variants per family.
Parameters and split boundaries were fixed before the batch:

- development: 2022-2024;
- validation: 2025;
- OOS: 2026-01-01 through 2026-07-14;
- multiple-testing LCB uses all ten registered trials.

The complete batch finished in `38.5 seconds`.

| Strategy | OOS trades | Mean net bps | LCB 95 bps | Max DD |
|---|---:|---:|---:|---:|
| Long-only baseline | 8 | -3645.33 | -4059.22 | 52.06% |
| EMA trend 12/48 | 956 | -10.20 | -30.80 | 50.85% |
| EMA trend 24/96 | 470 | -15.45 | -55.64 | 50.20% |
| Donchian 20 | 1,265 | -5.46 | -17.12 | 25.08% |
| Donchian 55 | 543 | -18.59 | -47.09 | 40.53% |
| Mean reversion 24 | 1,261 | -18.40 | -30.72 | 35.41% |
| Volatility compression | 355 | -16.49 | -30.42 | 9.79% |
| Regime pullback | 2,042 | -13.97 | -18.13 | 33.10% |
| Candle momentum 3 | 9,449 | -9.47 | -11.50 | 70.55% |

Flat has no trades and is not a candidate. No strategy passed validation plus
OOS plus cost stress. There are zero survivors.

Authoritative screening artifact:
`Reports/SimpleResearch/reset_v1/batch_report.json`, SHA256
`a6bb9ccd2a4f12f75510c679cfdc6998c23b1c690546f73e21b1eef13afcef5f`.

## Decision

Points 1-4 are complete, but they do not justify paper promotion or live
trading. The important result is architectural and statistical:

1. The old live path can no longer accidentally place a Bitget order.
2. A mature replacement execution framework starts correctly in dry-run.
3. A complete ten-strategy OOS batch now takes less than one minute instead of
   days of prospective collection.
4. The first bounded batch rejected every directional strategy without a new
   threshold campaign.

The next decision must follow the previously declared stop rule: do not tune
these trials against the revealed OOS. Either stop directional alpha work or
register a materially different economic hypothesis and a fresh untouched OOS
window. An ensemble cannot repair a set of components whose individual OOS
expectancy is negative.
