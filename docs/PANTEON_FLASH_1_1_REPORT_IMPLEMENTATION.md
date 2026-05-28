# Panteon Flash 1.1 Report Implementation

Date: 2026-05-28

Source report: `C:\Users\anton\Downloads\panteon_flash_analysis.md`

## Summary

Panteon Flash 1.1 adds reproducible CLI support for the report's tunable controls and a profile runner for H1 and 5-year retests. The report-core parameter set was fully retested on the fixed 2022-2026 window and failed the promotion gate, so it is not the default live candidate.

The default `best-compatible` profile is the intended current-best-compatible control: risk `0.12`, `flash_min_score_to_trade=4.0`, `max_new_opens_per_bar=1`, `risk_max_open_positions=8`, and the selected-subset manifest. On the current code state it did not reproduce the historical `112.45%` run, so any new promotion must be judged against current-code control and the historical mismatch must be investigated before live use.

## Implemented

- Added selected-subset wrapper pass-through for:
  - `--flash-min-score-to-trade`
  - general degradation window/min/pnl/cooldown parameters
  - symbol degradation guard/lookback parameters
  - `--v3-shadow-fresh-handoff-max-age-bars`
  - Flash anchor and portfolio actor keys
- Added `tools/run_panteon_flash_1_1_retrotest.py` with three profiles:
  - `best-compatible`: default, current-best-compatible retest profile.
  - `handoff-age12`: isolated handoff freshness test, age `24 -> 12`.
  - `symbol-lookback504`: isolated symbol lookback test, lookback `0 -> 504`.
  - `signal-cooldown480`: isolated signal cooldown test, cooldown `720 -> 480`.
  - `report-core`: report degradation/handoff/symbol-lookback changes.
  - `experimental`: `report-core` plus unproven profit-expansion ideas.
- Added unit coverage for the new selected-subset CLI controls.

## Not Promoted

- `report-core` degradation relaxation was implemented and tested, but it failed the 5-year gate.
- Technical hard gate and funding-score weight are not enabled by default because the report asked for A/B validation before promotion.
- Shadow symbol health remains diagnostics-only by default; no penalty is enabled.
- Partial profit lock, lower min-score, max-new-opens `3`, volatility/actor risk sizing, and anchor actor expansion are kept under `--profile experimental`.
- Adopted-position live alerts, NoTrade decay, anchor time decay, and price-drift handoff checks require separate live/allocator changes and are not part of the 5-year profile retest.

## 5-Year Test Result

Full report-core retest:

- Command profile: `--profile report-core`
- Output: `C:\Work\Crypto_exchange\Results\PanteonFlash_1_1_20260528\full2022_2026\RETRODATE_MARKET\2026-05-28_05-35-59_retrodate_market_v2`
- Window: 2022-01-01 00:00 UTC to 2026-05-18 23:00 UTC
- Bars: `38,375`
- Panteon leader: `Panteon_Flash`
- PnL: `-25.64%`
- MaxDD: `25.64%`
- Closed trades: `1,331`

Control best full-window result:

- Output: `C:\Work\Crypto_exchange\Results\PanteonFlashRiskFractionSweep_20260526\full2022_2026_risk_12\RETRODATE_MARKET\2026-05-26_19-02-52_retrodate_market_v2`
- PnL: `112.45%`
- MaxDD: `6.24%`
- Closed trades: `658`

Conclusion: the report-core changes made Flash trade too much and destroyed the existing edge. They must not be promoted to live without narrower ablation.

## Next Ablation Gates

The report-core failure showed that broad degradation relaxation is destructive. Follow-up tests must change one parameter at a time against `best-compatible`.

Gate criteria for each ablation:

- PnL must be `>= 100%`.
- MaxDD must be `<= 7%`.
- Closed trades must be `<= 850`.

Profiles:

- `handoff-age12`: only `--v3-shadow-fresh-handoff-max-age-bars 12`.
- `symbol-lookback504`: only `--flash-degradation-symbol-lookback-bars 504`. This deliberately does not enable the symbol guard.
- `signal-cooldown480`: only `--flash-degradation-signal-cooldown-bars 480`.

## H1 Prefilter

All H1 runs used `--years 2026 --max-bars 3600`, which resolved to 2026-01-01 00:00 UTC through 2026-05-18 23:00 UTC, `3,312` hourly bars.

| Profile | PnL % | MaxDD % | Closed trades | Selected | Filled | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `best-compatible` | -0.30 | 0.39 | 25 | 66 | 50 | control only |
| `handoff-age12` | 1.24 | 0.00 | 31 | 81 | 62 | only H1-positive ablation |
| `symbol-lookback504` | -0.30 | 0.39 | 25 | 66 | 50 | no incremental effect |
| `signal-cooldown480` | -0.30 | 0.39 | 25 | 66 | 50 | no incremental effect |

Historical H1 `risk_12` from `PanteonFlashSelectedSubsetPredeploy_20260525` was `10.81%`, MaxDD `0.00%`, `14` closed trades. Current-code `best-compatible` selected and filled more trades than that historical run, despite matching the main visible config fields. This mismatch is now a separate blocker for live promotion.

## Isolated Full-Window Results

| Profile | PnL % | MaxDD % | Closed trades | Selected | Filled | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| historical `risk_12` | 112.45 | 6.24 | 658 | 1489 | 1152 | pass |
| current-code `best-compatible` | -14.82 | 15.18 | 891 | 2267 | 1782 | fail |
| `handoff-age12` | -16.41 | 16.77 | 792 | 1989 | 1584 | fail |
| `report-core` | -25.64 | 25.64 | 1331 | 3369 | 2662 | fail |

`handoff-age12` was the only H1-positive ablation, but the full 2022-2026 run rejected it. It reduced trade count versus current-code control, but worsened PnL and MaxDD. It should not be promoted.

## Verification

- `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_run_flash_selected_subset_candidate.py -q`
  - `11 passed`
- `.\.venv\Scripts\python.exe -B tools\run_panteon_flash_1_1_retrotest.py --dry-run`
  - verified default `best-compatible` args.
- `.\.venv\Scripts\python.exe -B tools\run_panteon_flash_1_1_retrotest.py --profile report-core`
  - full 5-year retest completed.
- `.\.venv\Scripts\python.exe -B tools\run_panteon_flash_1_1_retrotest.py --years 2026 --max-bars 3600 --profile ...`
  - H1 control/A/B/C prefilter completed.
- `.\.venv\Scripts\python.exe -B tools\run_panteon_flash_1_1_retrotest.py --profile best-compatible`
  - current-code full control completed.
- `.\.venv\Scripts\python.exe -B tools\run_panteon_flash_1_1_retrotest.py --profile handoff-age12`
  - full A retest completed.
- `.\.venv\Scripts\python.exe -B tools\run_panteon_flash_1_1_retrotest.py --max-bars 48`
  - default profile smoke completed.

The repeated `dashboard_latest_RETRODATE_MARKET.png` permission error is nonfatal: the run still writes `run_summary.json`, `analysis_report.md`, and dashboard files inside the timestamped run directory.

## Commands

Default best-compatible 5-year retest:

```powershell
.\.venv\Scripts\python.exe -B tools\run_panteon_flash_1_1_retrotest.py
```

Report-core failed profile:

```powershell
.\.venv\Scripts\python.exe -B tools\run_panteon_flash_1_1_retrotest.py --profile report-core
```

Experimental report ideas:

```powershell
.\.venv\Scripts\python.exe -B tools\run_panteon_flash_1_1_retrotest.py --profile experimental
```

Isolated ablations:

```powershell
.\.venv\Scripts\python.exe -B tools\run_panteon_flash_1_1_retrotest.py --profile handoff-age12
.\.venv\Scripts\python.exe -B tools\run_panteon_flash_1_1_retrotest.py --profile symbol-lookback504
.\.venv\Scripts\python.exe -B tools\run_panteon_flash_1_1_retrotest.py --profile signal-cooldown480
```
