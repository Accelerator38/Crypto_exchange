# Panteon Flash structural retest addendum

Date: 2026-05-24.

## Scope

This addendum covers the final V8 cleanup pass after the pre-live report:

- moved Flash audit emission out of `main_loop.py` into `src/panteon_v2/app/audit_emission.py`;
- moved Flash per-symbol orchestration into `src/panteon_v2/app/decision_paths/flash.py`;
- kept `main_loop.py` compatibility wrappers for existing tests and private imports;
- added a Windows-safe guard for root-level `dashboard_latest_*` publishing after `PermissionError`;
- reran the best production candidate, `Round3 Deny8 EntryRegime`, on 2025 and on the full 2022-2026 period.

## Verification

Automated tests:

- focused new module-boundary tests first failed because the modules did not exist;
- focused module-boundary tests then passed: 3 passed;
- app + allocator integration tests passed: 234 passed, 4 subtests passed;
- full pytest passed before the dashboard-publisher fix: 966 passed.

Retest runs:

| Run | Period | Bars | PnL | Best component | Alpha | Max DD | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Early-stop control | 2025 | 8,760 | 21.11% | 18.15% | +2.96 p.p. | 4.27% | pass |
| Full structural retest | 2022-2026 | 38,375 | 105.42% | 62.34% | +43.08 p.p. | 4.19% | pass |

The full retest produced the complete run artifacts and metrics. The shell command itself hit a timeout after `run_summary.json` and `analysis_report.md` were already written, during repeated root-level latest-dashboard publication failures. That issue is now fixed in `OutputWriter`: after a `PermissionError`, latest-dashboard publishing disables itself for that writer instance while preserving the run-specific dashboard files.

## Trading Conclusion

The best launch candidate remains `Round3 Deny8 EntryRegime` / `Panteon_Flash`.

It is suitable only for a constrained pilot or live-shadow start:

- small capital;
- `max_new_opens_per_bar=1`;
- `risk_max_open_positions=8`;
- daily attribution review;
- hard stop if 2025/2026-style dominance gates fail on fresh data;
- no `symbol_guard` in the live profile, because its full 2022-2026 retest reduced PnL from 105.42% to 100.50%.

Full unattended production is still not recommended until live-shadow confirms exchange fees, slippage, order rejection behavior, and attribution parity.

## Artifacts

- Full retest run: `C:\Work\Crypto_exchange\Results\PanteonFlashRound3Deny8EntryRegime_RetestFull2022_2026_20260524\RETRODATE_MARKET\2026-05-24_09-23-27_retrodate_market_v2`
- 2025 control run: `C:\Work\Crypto_exchange\Results\PanteonFlashRound3Deny8EntryRegime_Retest2025_20260524\RETRODATE_MARKET\2026-05-24_09-10-28_retrodate_market_v2`
- Main Word report: `C:\Work\Crypto_exchange\Reports\PanteonFlashPreLive_20260524\Panteon_Flash_PreLive_Report_expanded.docx`
- Main markdown report: `C:\Work\Crypto_exchange\Reports\PanteonFlashPreLive_20260524\Panteon_Flash_PreLive_Report.md`
- Charts: `C:\Work\Crypto_exchange\Reports\PanteonFlashPreLive_20260524\charts`
