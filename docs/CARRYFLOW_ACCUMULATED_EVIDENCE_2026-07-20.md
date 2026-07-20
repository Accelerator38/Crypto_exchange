# CarryFlow accumulated evidence analysis - 2026-07-20

## Scope and evidence boundary

Five Bitget hourly roots were validated independently. They contain 166 segment
hours, 1,328 complete symbol observations and 1,288 valid adjacent OI
observations across BTC, ETH, SOL, BNB, XRP, DOGE, ADA and LINK. Context coverage
is 100%. Real gaps remain between roots; no price, OI state or position was
carried across a boundary.

The reproducible output is
`Reports/CarryFlow/segment_screening_20260720.json`. It is R&D screening only:
`segments_concatenated=false` and `promotion_authority=false`.

## Root cause of zero signals

The existing short policy requires `basis >= +4 bps`. Actual Bitget mark/index
basis was:

- minimum: `-14.79 bps`;
- median: `-5.16 bps`;
- 95th percentile: `-2.93 bps`;
- maximum: `+0.83 bps`.

The gate is therefore structurally unreachable in this sample. The full funnel
confirms it:

| Stage | Observations |
|---|---:|
| Adjacent OI observations | 1,288 |
| OI change >= 2% | 68 |
| Funding >= 0.00008 | 39 |
| Long ratio >= 0.58 | 39 |
| Existing basis >= +4 bps | 0 |
| Research basis >= -15 bps | 39 |

Collecting more data under the existing positive-basis policy is not justified.
The zero-signal result was caused by the policy definition, not by insufficient
collector uptime.

## Hypothesis screening

A fixed 16-variant grid tested OI thresholds `1.5/2.0/2.5/3.0%` and maximum
holds `6/12/18/24h`, with 12 bps round-trip costs, 1.2% stop and 2.4% target.
Selection is in-sample and requires fresh prospective validation.

The least-bad candidate is:

- short only;
- OI change >= 2%;
- funding >= 0.00008;
- long ratio >= 0.58;
- basis sanity floor `-15 bps`;
- maximum hold `6h`;
- one global position.

Segment-screening result:

| Metric | Result |
|---|---:|
| Filled orders | 20 |
| Closed trades | 10 |
| Mean net expectancy | +36.01 bps |
| Median net outcome | +61.54 bps |
| Positive trades | 70% |
| Total net outcome | +360.13 bps |
| Max drawdown | 145.25 bps |
| 95% one-sided LCB | -12.56 bps |
| Largest symbol share | 50% |
| Largest segment share | 40% |

This precondition candidate passes incidence and point-expectancy checks but
fails the LCB gate. The segment model does not include the actor's EMA/RSI
momentum confirmation or exact policy decisions, so it cannot select the next
continuous candidate by itself.

Lowering OI to 1.5% increased fills but made net expectancy negative. Longer
holds widened losses and confidence intervals. Raising OI reduced sample size
without producing a positive LCB. Those variants should not consume separate
continuous canaries.

## Exact actor and policy replay

Retrospective 53-bar OHLCV seeds were fetched for v1, v4 and v5 to evaluate every
recorded evidence bar. The seed timestamps precede evidence, but the files were
retrieved after the fact and are explicitly `prospective_for_tape=false`; they
are diagnostic only.

This replay exposed and fixed a second activation defect: policy decisions used
global cross-symbol regime agreement as the confidence of a symbol-local
regime. On v5 that rejected all five actor candidates at confidence 0.125-0.25.
Replay and live policy runtime now use the detector's local symbol confidence.

The first exact run also force-closed an open trade at the last bar of each
interrupted root. That mixed policy performance with an arbitrary file boundary.
Exact multi-root screening now uses right censoring: naturally closed trades are
counted and an open end position is reported, but no synthetic PnL is created.

After the parity and endpoint fixes, the `OI 2% / basis -15 bps / hold 6h`
exact result was:

| Root | Candidate signals | Closed trades | Net PnL | Mean net/trade |
|---|---:|---:|---:|---:|
| v1 (58h) | 4 | 3 | -$0.00565 | -$0.00188 |
| v4 (36h) | 1 | 1 | +$0.07897 | +$0.07897 |
| v5 (66h) | 3 | 3 | +$0.03456 | +$0.01152 |
| Combined diagnostic | 8 | 7 | +$0.10788 | +$0.01541 |

The positive point estimate is not sufficient. Combined one-sided 95% LCB is
`-$0.04757` per trade, v1 remains negative, there are only seven natural closes
and one v1 position is censored. A second fixed seven-policy sweep compared
normalization timing, no-normalization, 6/12h holds, OI 1.5/2%, non-range and
range-only policies. Every policy failed incidence, LCB or independent-root
expectancy.

The least-bad non-range policy produced only three closed trades: +$0.10188
total, +$0.03396 mean and `-$0.00660` LCB. Its v5 held-out trade was negative.
Changing the normalization exit did not change any trade; the suspicious
one-bar v1 loss was a real 1.2% stop, not a normalization close. The reproducible
result is `Reports/CarryFlow/exact_exit_regime_loo_20260720.json`, with
`selected_for_prospective_validation=null`.

Therefore no CarryFlow variant should start a long promotion tape yet. The next
data collection can continue as a reusable scheduled Bitget market tape, but it
must not be labeled a candidate canary until a redesigned exact policy passes
the exact profile evaluation.

## Profile simplification and OI/price divergence

The replay interface previously exposed more than ten independently editable
actor and signal settings. CarryFlow now supports a compact versioned contract:

```json
{"PROFILE_ID": "screened_short_v1"}
```

All algorithm values are derived from the source-pinned profile. Direction,
regimes, stop, holding horizon, OI and price lookbacks, entry/exit logic and the
signal mapping cannot be overridden separately. Profile/scalar mixing fails
closed. The duplicate generic futures-replay override for CarryFlow was removed.
Old complete manifests remain readable for compatibility, but new replay/live
artifacts use the compact contract. Scalar actor manifests are diagnostic replay
compatibility only; paper and micro-live reject them.

One exact evaluator now replaces the precondition-grid -> scalar-sweep -> policy
assembly chain. It evaluates complete profiles over independent roots, applies
costs, right-censors endpoints and performs root/LOO hard checks in one command.

A three-hour OI expansion plus weak-price-confirmation model was implemented and
tested as immutable profiles. Results from
`Reports/CarryFlow/exact_profile_evaluation_20260720.json` were:

| Profile | Closed | Net PnL | Mean/trade | LCB |
|---|---:|---:|---:|---:|
| `screened_short_v1` | 7 | +$0.10788 | +$0.01541 | -$0.04757 |
| `divergence_short_v1` | 14 | -$0.09220 | -$0.00659 | -$0.05479 |
| `divergence_short_strict_v1` | 5 | -$0.19821 | -$0.03964 | -$0.06666 |
| `divergence_short_non_range_v1` | 7 | -$0.32631 | -$0.04662 | -$0.09317 |

The divergence family increased activation but destroyed costed expectancy. It
is rejected before prospective collection. The evaluator again emitted
`selected_for_prospective_validation=null`; no live or paper authority was
created.

## Alternative entry without OI spike

A separate root-aware event study then tested deleveraging bounce, downside
continuation, negative-basis reversion, broad-market breadth and crowded price
overextension. Only crowded overextension had a positive preliminary mean in
all three roots, so it was implemented as one immutable profile instead of a
threshold grid:

```json
{"PROFILE_ID": "crowded_overextension_short_v1"}
```

The profile shorts a three-hour price overextension only when funding and
account-long crowding are elevated. It does not require OI expansion and ranks
simultaneous candidates by price extension rather than the legacy CarryFlow
score. The first exact replay, which could observe stops only at hourly
decisions, produced `25` candidate signals, `38` fills and `19` closed trades,
but failed economically: net PnL `-$0.14252`, mean `-$0.00750` per trade, LCB
`-$0.04587` and max drawdown `$0.62592`, with negative expectancy in v1 and v5.

This result closes the obvious hourly CarryFlow families on the current tape:
OI expansion, OI/price divergence and crowded price reversal all fail the same
unchanged cost/root/LCB gates. Continuing to tune thresholds on these 160 hours
would be data fitting, not additional evidence.

## Protective-stop execution parity

The hourly replay revealed that a configured `1.2%` stop was previously checked
only at the next decision price. Three nominal stop exits therefore realized
gross losses of approximately `1.66-2.44%`. The policy value was correct, but
the execution model did not implement its intrabar meaning.

There is now one protective-stop contract derived from the immutable profile:

- evidence snapshots retain the sealed closed-bar OHLC;
- replay checks the next bar before actor evaluation and models a stop-market
  fill at the trigger, or at the worse bar open after a gap, plus slippage;
- an open replay position without complete OHLC fails closed;
- paper/live signals carry the same absolute stop price;
- a Bitget policy open requires the exchange client to attach
  `presetStopLossPrice` atomically to the opening order; missing capability
  blocks the open before an API order call;
- market-poll stop handling remains as a fallback, not the primary live stop.

No stop CLI or second risk setting was added. `PROFILE_ID` remains the only
algorithm setting and the stop module is covered by the runtime fingerprint.

The exact five-profile evaluation after this fix produced:

| Profile | Fills | Closed | Mean net/trade | LCB | Drawdown | Failed roots |
|---|---:|---:|---:|---:|---:|---|
| `crowded_overextension_short_v1` | 40 | 20 | -$0.00355 | -$0.03627 | $0.53094 | v1, v5 |
| `divergence_short_v1` | 28 | 14 | +$0.00368 | -$0.03628 | $0.20490 | v5 |
| `divergence_short_strict_v1` | 10 | 5 | -$0.03964 | -$0.06666 | $0.19821 | v1, v4 |
| `screened_short_v1` | 14 | 7 | -$0.00071 | -$0.07112 | $0.24068 | v5 |
| `divergence_short_non_range_v1` | 14 | 7 | -$0.03962 | -$0.07770 | $0.27731 | v1, v4, v5 |

The stop correction roughly halved the crowded-overextension net loss, and
`divergence_short_v1` now has a positive combined point estimate. It still
fails the unchanged gate: negative LCB, negative v5 expectancy and drawdown
`$0.20490` above the fixed `$0.20` limit. The authoritative evaluator still
emits `selected_for_prospective_validation=null`; no paper or live authority
exists.

## Systemic selloff guard

The losing `divergence_short_v1` entries exposed two structural errors rather
than another threshold problem: SHORT was allowed in a bullish symbol regime,
and the actor could enter after a broad full8 selloff when the likely residual
risk was a synchronized bounce. The new immutable profile is:

```json
{"PROFILE_ID": "divergence_short_systemic_guard_v1"}
```

It keeps the original divergence, cost and risk parameters. It adds no manifest
or CLI setting. Its versioned model applies two causal rules:

- the manifest has no `SHORT + bullish` rule;
- the actor requires at least half of the current full8 universe to have a
  nonnegative return over the same three-hour divergence lookback. Incomplete
  cross-sectional history fails closed.

Exact root-aware replay produced:

| Profile | Fills | Closed | Mean net/trade | 95% LCB | Drawdown | Failed roots |
|---|---:|---:|---:|---:|---:|---|
| `divergence_short_v1` | 28 | 14 | +$0.00368 | -$0.03628 | $0.20490 | v5 |
| `divergence_short_systemic_guard_v1` | 22 | 11 | +$0.02460 | -$0.01896 | $0.18535 | none |

All three roots now have positive costed point expectancy and all
leave-one-root-out checks are positive. Incidence, fill count, costed
expectancy, drawdown and root-collapse gates pass. The 95% LCB remains
negative, so `screening_passed=false` and
`selected_for_prospective_validation=null` remain mandatory.

This profile was designed after inspecting v5 failures. Therefore v1/v4/v5 are
development evidence for this profile, not fresh OOS evidence, regardless of
their root separation. The result authorizes neither a paper canary nor live
orders. Adding another filter to force the development-set LCB above zero would
be outcome fitting; the next evidence must come from data not used to define
the guard.

## Historical backfill capability

The official Bitget history surface and the installed CCXT client were probed
before implementing a backfill. The result is fail-closed:

| Required source | Historical probe |
|---|---|
| market OHLCV | available |
| mark OHLCV | available |
| index OHLCV | available |
| funding rate | available |
| account long/short ratio | available |
| open interest | not supported |

Bitget's public open-interest endpoint is a current snapshot; CCXT advertises
`fetchOpenInterestHistory=false`. Current OI must not be copied into historical
rows, and zero or another exchange's OI is not an equivalent feature. The new
`audit_bitget_historical_evidence.py` tool therefore emits:

```text
primary_failure=historical_open_interest_unavailable
allowed_use=diagnostic_only
evidence_tape_creation_allowed=false
promotion_backfill_possible=false
orders_enabled=false
```

The observed capability report SHA-256 is
`d239f20470a9ab1f677b2ba6bf7d56356536a9a5b20e1ec1a19dd6f3f7b22422`.
Historical candles and ratios may support diagnostic studies, but they cannot
increase CarryFlow promotion evidence.

## Evidence extension without promotion

The exact evaluator now distinguishes a no-order evidence extension from a
promotion candidate. `evidence_extension_eligible=true` is possible only when
the complete failure set is exactly `nonpositive_lcb`; incidence, costs,
positive point expectancy, drawdown, active-root and root-collapse checks must
already pass. This is not a bypass: positive LCB remains mandatory for
prospective validation, paper and live.

The current exact report therefore contains:

```text
selected_for_evidence_extension=divergence_short_systemic_guard_v1
selected_for_prospective_validation=null
evidence_extension_orders_enabled=false
evidence_extension_is_promotion=false
```

After the runtime is published and locked, new no-order observations may be
accumulated in independent prospective roots. A machine or network gap ends a
root but does not invalidate completed roots; roots remain separate and are
aggregated by the root-aware evaluator. Missing derivative snapshots still
cannot be backfilled.

## Implemented efficiency changes

1. `CarryFlowAgentV2` now has an independent `SHORT_BASIS_FLOOR`; long-entry and
   exit semantics remain unchanged. Failed short gates are logged precisely.
2. Replay defaults for the new research hypothesis are `-15 bps` basis floor
   and `6h` maximum hold. Live authority remains false.
3. A hash-sealed 53-bar OHLCV warm-up seed initializes indicators before the
   first evidence bar. It contains no copied historical derivatives and no
   trade evidence.
4. Collector `--resume` supports one scheduled hourly process per bar. Exact
   candle continuity replaces dependence on multi-day process uptime.
5. Segment screening reuses interrupted roots for hypothesis rejection and
   ranking without pretending that they form one promotion window.
6. Exact segment replay right-censors positions at interrupted root boundaries;
   it never manufactures `replay_end_flatten` PnL for model selection.
7. One exact profile evaluator applies independent-root/leave-one-root-out
   collapse checks before any prospective collection is authorized.
8. CarryFlow audit rows distinguish stop, target, maximum-hold and normalization
   exits; production thresholds and order permissions remain unchanged.
9. New manifests expose one algorithm knob, `PROFILE_ID`; the former CarryFlow
   scalar CLI and duplicate generic replay override were removed.
10. Authoritative replay derives symbols and cadence from the immutable tape;
    optional CLI values are assertions only and cannot silently reinterpret it.
11. Replay, paper and Bitget live share one profile-derived protective stop;
    live opens fail closed unless the stop is attached server-side.
12. `divergence_short_systemic_guard_v1` folds direction-regime and full8
    breadth protection into one profile; no independent breadth setting exists.
13. Historical-source audit prevents a current OI snapshot from being reused as
    historical evidence.
14. LCB-only candidates may collect additional no-order roots without being
    mislabeled as paper-ready or promotion-ready.
15. A sealed campaign wrapper pins the exact report, published revision,
    runtime fingerprint and sole `PROFILE_ID`. It automatically rotates to a
    new independent root after a missed bar, so recovery no longer depends on
    an operator reconstructing collector flags or discarding completed roots.

## New test budget

The old path required 53 warm-up hours plus an open-ended fixed collection,
often restarted after disconnects. The new path uses zero continuous warm-up
hours and approximately one short process invocation per hour. A disconnect
ends only the current root; the next invocation seals a new root under the same
immutable no-order campaign. Interrupted roots remain useful for rejection
without being spliced into one timeline. The exact sweep now rejects the current
CarryFlow family before a new week-long canary, which is the largest test-time
saving.

Do not start another candidate-labelled CarryFlow tape from this report. A new
prospective root is justified only after an exact offline policy emits a
non-null `selected_for_prospective_validation` and then uses a seed created
before its first evidence bar. Once such a candidate exists, review at 24h,
reject activation failure at 48-72h and make the pass/fail decision by 168h
instead of continuing automatically toward 720h.

Promotion gates are unchanged: >=95% coverage, >=20 fills, >=10 closed trades,
positive costed expectancy, positive LCB, drawdown within limit,
validation+OOS+cost stress+sanity, followed by positive strict short and
extended paper canaries. Real orders remain disabled until separate manual
micro-live confirmation.
