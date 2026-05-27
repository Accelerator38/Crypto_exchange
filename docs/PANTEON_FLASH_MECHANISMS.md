# Panteon Flash mechanisms

Live-oriented documentation for the currently trading Flash build is in
`docs/PANTEON_FLASH_LIVE_TRADING.md`. This file remains a lower-level notes
page for individual mechanisms and configuration flags.

## Selection controls

`anchor_actor_keys` marks preferred actors by `actor_type:label` or plain label.
If an anchor is close enough to the greedy winner, `anchor_min_score_advantage`
lets Flash keep the anchor instead of switching to a marginally better actor.
The current priority is explicit: greedy ranking runs first, anchor dominance
runs second, and `actor_switch_margin` can still keep the previous actor when
the challenger has not cleared the switch margin.

`portfolio_actor_keys` marks actors whose statistics should be read as a
portfolio-level record instead of a single regime record. This is useful for
actors such as broad Solo wrappers that are deliberately meant to trade across
several regimes and symbols.

`terminal_denied_signal_keys` is a hard stop list for `actor|symbol|action`
keys. If the best candidate is terminal-denied, Flash emits a NoTrade decision
and preserves the blocked actor in `original_selected_actor` for auditability.

`denied_open_regimes` and `denied_open_symbols` block new opens only. Close
signals still pass through so Flash can flatten existing exposure.

## Legacy specialist wrappers

`ActionFilterAgent` can now narrow legacy agents with open-only contextual
gates:

- `min_regime_confidence` blocks new opens when regime detection is weak.
- `funding_cost_aligned_opens` allows futures opens only when funding does not
  penalize the side (`long` avoids positive funding, `short` avoids negative
  funding).
- `min_lookback_return_pct_by_bars` and `max_lookback_return_pct_by_bars`
  require recent return context before an open can pass.

These gates are used by opt-in experimental labels such as
`CrashHunterStrict`, `VolBreakoutFundingAware`, `AfterShockRegimeOnly`,
`LiveTrendFollowBullOnly`, and `LiveMeanRevNeutralOnly`. They are registered
only through the experimental Flash actor path and remain shadow-only unless
the run explicitly allows experimental actors to trade.

`AfterShockRegimeOnly` is intentionally crash-only after the first smoke test:
the broader bearish/crash variant over-traded the early 2025 sample in shadow.

## Degradation controls

`degradation_signal_*` tracks recent realized PnL per signal key and can
temporarily deny a bad key. `degradation_actor_*` applies the same idea to an
actor or actor-regime key. `degradation_symbol_*` watches recent realized PnL
per symbol and can disable new opens on unstable symbols while still allowing
closes.

`degradation_symbol_lookback_bars` is an opt-in time-window variant of the
symbol guard. When it is greater than zero, symbol degradation is based on
closed trades inside the rolling bar lookback instead of the last
`degradation_symbol_window_closed_trades` outcomes. This is intended for tests
such as "disable symbols that lost money in roughly the last three months";
the current live candidate leaves it at `0`.

All degradation state is maintained in `app/degradation_tracking.py`. The main
loop only refreshes the state and passes the resulting deny sets into
`FlashAllocator`.

## Shadow confirmation keys

Shadow confirmation now uses two canonical lookup shapes:

- actor fallback: `label`
- symbol/action confirmation: `(label, symbol, action_name)`

Legacy partial keys such as `(label, symbol)` are intentionally ignored for
symbol-scoped confirmation. This prevents accidental reuse of stale aggregate
stats for a different action.

## Shadow symbol health

`shadow_symbol_health_enabled` adds an aggregate diagnostic key:
`("__symbol_health__", symbol, action_name)`. The value is built from
symbol/action shadow confirmations across actors and is emitted into Flash
candidate audit as:

- `shadow_symbol_health_score`
- `shadow_symbol_health_closed_trades`
- `shadow_symbol_health_pnl_per_trade_lcb_usd`
- `shadow_symbol_health_penalty`

The layer is opt-in. In live-style profiles it should currently be used for
diagnostics only: short 2026 H1 tests showed that low-sample thresholds can
remove profitable symbol-specific trades. If a penalty is enabled, it only
applies to open signals and only when the actor-specific shadow PnL LCB does
not already clear the configured symbol-health floor.

`tools/build_flash_lcb_deny_diagnostics.py` can join
`flash_signal_key_shadow_report.json` with `flash_attribution_summary.json` and
produce deny-candidate diagnostics. The generated candidates are proposals only;
each candidate must beat the current baseline in a retro run before it can be
added to a pre-live or live profile.

## Economic layer

The economic layer is opt-in and disabled by default, including in
`FLASH_PRESET_SAFE`.

`no_trade_fee_saving_score_enabled` gives NoTrade a synthetic score equal to
the expected fee saving. Fees are read from `MarketSnapshot.fees_bps_by_symbol`
or `no_trade_default_fee_bps`.

`funding_score_weight` adjusts open-candidate scores by current funding. Longs
are penalized when funding is positive, shorts are rewarded when funding is
positive, and vice versa.

`actor_risk_sizing_enabled` turns actor history into `Signal.risk_mult`.
Small or negative net edge reduces size toward `actor_risk_min_mult`; stronger
per-trade net PnL moves size toward `actor_risk_max_mult`.

`funding_risk_mult_weight` and `funding_risk_mult_cap` add a bounded sizing
tilt for funding-aligned opens.

`volatility_risk_sizing_enabled` scales `risk_mult` toward constant volatility
exposure using `MarketSnapshot.lookback_volatility_pct`.

## Technical overlay

`technical_overlay_enabled` adds RSI(14), MACD(12,26,9), and ATR(14) context
to Flash candidate audit rows. The overlay is diagnostic by default.

For open candidates:

- Long alignment requires RSI inside `technical_rsi_long_min/max` and a
  positive MACD histogram above `technical_macd_histogram_min_abs_pct`.
- Short alignment requires RSI inside `technical_rsi_short_min/max` and a
  negative MACD histogram below the negative histogram threshold.
- Aligned candidates receive `technical_score_bonus`.
- Misaligned candidates receive `technical_score_penalty`.
- `technical_hard_gate_enabled` converts misalignment into a hard NoTrade
  rejection. This remains disabled until retrodate evidence beats baseline.
- `technical_atr_risk_sizing_enabled` scales open risk toward constant ATR
  exposure with `technical_atr_target_pct / atr_14_pct`, capped by
  `technical_atr_max_mult`.

Technical indicators are computed from closed OHLC candles in Retrodate and
stored on `MarketSnapshot.technicals_by_symbol`. They are not allowed to read
future bars or execution outcomes.
