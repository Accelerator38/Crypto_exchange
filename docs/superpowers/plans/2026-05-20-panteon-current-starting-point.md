# Panteon Allocator Current Starting Point

Date: 2026-05-20

## Selected Build

The current starting point is the production pipeline with:

- `Antonius_conservative` as the only default Antonius regime-switch player.
- `Antonius_strategy` excluded from default production candidates because it was a persistent real loser.
- Shadow-position replay enabled only through causal current-bar shadow state.
- Shadow-state raw entry gate for `Antonius_*`: raw component opens are not executable unless the same `symbol/side` has a current positive shadow position.
- Actionable fallback cannot bypass real-promotion or probation safety rejects.
- Default shadow replay max age remains `1` bar. The `age6` experiment is available for diagnostics but is not the recommended default.
- Live money execution remains disabled by default. Use paper/live-simulation only with `DRY_RUN=true`.

## Latest Validation

Unit tests:

```text
.\.venv\Scripts\python.exe -m pytest -q
704 passed, 13 subtests passed
```

## Iteration Results

| Run | Scope | PnL USD | PnL % | Max DD % | NoTrade % | Trades | Profitable leader share | Profitable trade share | Regret vs soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `U_causal_entry_export_5100` | 5100 bars | -0.67 | -0.07 | 1.68 | 98.41 | 79 | 30.0% | 51.9% | 225.01 |
| `AC_shadow_state_entry_gate_5100` | 5100 bars | -4.13 | -0.41 | 0.41 | 99.33 | 24 | 12.5% | 37.5% | 243.75 |
| `AE_shadow_state_replay_age6_5100` | 5100 bars | -13.90 | -1.39 | 1.39 | 99.12 | 31 | 0.0% | 32.3% | 253.52 |
| `Q_final_scoped_probation_2022_2026` | 2022-2026 | -3.58 | -0.36 | 1.96 | 99.76 | 90 | 21.4% | 41.1% | 342.02 |
| `AD_shadow_state_entry_gate_2022_2026` | 2022-2026 | -5.87 | -0.59 | 0.59 | 99.90 | 30 | 0.0% | 36.7% | 495.25 |

## Decision

The current build is not a financial improvement over the best baseline. It is still the correct new engineering starting point because it fixes an execution correctness bug:

- before the fix, Antonius could be selected as a shadow-state player but execute raw component-agent opens with no positive transfer-ready shadow position;
- after the fix, real Antonius entries are only replay-style or explicitly confirmed by current positive shadow state.

The `age6` experiment is rejected as a default. It proved that merely allowing older profitable shadow positions creates late-entry losses. A valid handoff needs an opportunity-quality score, not only `age <= N` and positive unrealized PnL.

## Main Blocker

Panteon still does not evaluate the current executable opportunity well enough. The selector knows that `Antonius_conservative` is a strong shadow player, but real execution does not know whether the currently visible position is still worth entering.

Observed symptoms:

- `Antonius_conservative` shadow: strong positive PnL in both short and full runs.
- `Antonius_conservative` real: small loss with strict replay, large loss with wider age replay.
- Full 2022-2026 `AD` run: `Antonius_conservative` rejected on 38,365 bars, mostly by current actionability gate.
- `NoTrade` remains too high: 99.90% in `AD`.

## Next Technical Step

Build an executable opportunity dataset and score for current shadow positions:

- position age;
- side and symbol;
- entry price and current price;
- unrealized PnL;
- distance to stop/take;
- recent move since shadow entry;
- current regime and market tags;
- component agent label;
- whether a real replay entry at this bar would have closed profitably.

Use this score to decide whether to replay a shadow position. Do not widen replay age globally until the score shows walk-forward edge.

## Test Policy Going Forward

Use staged diagnostics before full 2022-2026:

1. Run unit tests.
2. Run 5100-bar diagnostic.
3. Run one-year buckets only if 5100 improves against `U_causal_entry_export_5100`.
4. Run full 2022-2026 only if early buckets improve PnL, profitable leader share, and regret without unacceptable DD.

Do not count lower DD from suppressed trading as improvement.

## Artifacts

- `Results/neiro_genetics/RetrodateMarket_AB/AC_shadow_state_entry_gate_5100/RETRODATE_MARKET/2026-05-20_09-54-51_retrodate_market_v2`
- `Results/neiro_genetics/RetrodateMarket_AB/AD_shadow_state_entry_gate_2022_2026/RETRODATE_MARKET/2026-05-20_10-02-17_retrodate_market_v2`
- `Results/neiro_genetics/RetrodateMarket_AB/AE_shadow_state_replay_age6_5100/RETRODATE_MARKET/2026-05-20_12-23-09_retrodate_market_v2`
