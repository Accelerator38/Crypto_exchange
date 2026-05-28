# Panteon Flash 1.1 Report Implementation

Date: 2026-05-28

Source report: `C:\Users\anton\Downloads\panteon_flash_analysis.md`

## Summary

Panteon Flash 1.1 adds reproducible CLI support for the report's tunable controls and a profile runner for 5-year retests. The report-core parameter set was fully retested on the fixed 2022-2026 window and failed the promotion gate, so it is not the default live candidate.

The default `best-compatible` profile keeps the proven full-window profile close to the current best build: risk `0.12`, `flash_min_score_to_trade=4.0`, `max_new_opens_per_bar=1`, `risk_max_open_positions=8`, and the selected-subset manifest.

## Implemented

- Added selected-subset wrapper pass-through for:
  - `--flash-min-score-to-trade`
  - general degradation window/min/pnl/cooldown parameters
  - symbol degradation guard/lookback parameters
  - `--v3-shadow-fresh-handoff-max-age-bars`
  - Flash anchor and portfolio actor keys
- Added `tools/run_panteon_flash_1_1_retrotest.py` with three profiles:
  - `best-compatible`: default, current-best-compatible retest profile.
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

## Verification

- `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_run_flash_selected_subset_candidate.py -q`
  - `11 passed`
- `.\.venv\Scripts\python.exe -B tools\run_panteon_flash_1_1_retrotest.py --dry-run`
  - verified default `best-compatible` args.
- `.\.venv\Scripts\python.exe -B tools\run_panteon_flash_1_1_retrotest.py --profile report-core`
  - full 5-year retest completed.
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
