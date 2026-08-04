# Panteon research archive

This branch is a frozen research archive of the Panteon crypto-trading project.
It is not an approved live-trading release.

- Legacy Panteon execution on Bitget is hard-frozen.
- `orders_enabled=false` and `promotion_authority=false` are archive invariants.
- No trained model checkpoint is retained or approved.
- The compact `SimpleResearch` package is the supported path for reproducing
  historical strategy screens.
- The isolated Freqtrade setup is an engineering dry-run harness only.

Start with [the archive guide](docs/research_archive/README.md), then review the
[dataset catalog](docs/research_archive/DATASETS.md) and
[hypothesis register](docs/research_archive/HYPOTHESES.md).

Development on the `Exia` branch follows the
[minimal regime-system plan](docs/EXIA_REGIME_MIGRATION_PLAN.md).
The plan does not reactivate the archived Pantheon live route.

The first Exia foundation is a fail-closed Freqtrade strategy which classifies
four local 1h market states and hard-codes all entries to zero. Validate its
contracts, causality and full8 state distribution with:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_exia_foundation.py -q
.venv\Scripts\python.exe tools\audit_exia_market_modes_v1.py
```

The generated report is stored in
`Reports/Exia/market_mode_foundation_v1/distribution.md`. A passing foundation
report permits candidate design only; it does not permit paper or live orders.

Run one immutable Freqtrade experiment and rebuild the offline research catalog:

```powershell
.venv\Scripts\python.exe tools\run_exia_experiment_v1.py `
  --candidate freqtrade_pilot\candidates\exia_market_mode_foundation_v1.json
.venv\Scripts\python.exe tools\build_exia_catalog_v1.py
```

Each experiment stores its pinned manifest, normalized trade ledger and metrics
under `Reports/Exia/experiments`. The derived catalog in `Reports/Exia/catalog`
can be rebuilt from those immutable folders and is never loaded by dry/live.

The first preregistered trend family is closed with
`NO_CANDIDATE_FOR_VALIDATION`. Reproduce its development-only decision with:

```powershell
.venv\Scripts\python.exe tools\evaluate_exia_trend_family_v1.py
```

Taxonomy v1 is retained as historical evidence but is not used for new
candidates because its 4320-hour threshold was restart-dependent within the
Bitget startup limit. The restart-stable v2 foundation uses a 720-hour window,
999 startup bars and remains a no-trade strategy:

```powershell
.venv\Scripts\python.exe tools\audit_exia_market_modes_v2.py
.venv\Scripts\python.exe tools\run_exia_experiment_v1.py `
  --candidate freqtrade_pilot\candidates\exia_market_mode_foundation_v2.json
```

A passing v2 foundation authorizes one new preregistered candidate design only;
it does not authorize validation, paper, or live orders.

The first v2 candidate, `exia_ada_donchian48_v2`, was tested once on development
and terminally rejected: 180 stress-cost trades, mean `-9.19 bps`, block LCB
`-80.01 bps`. Validation, OOS and paper were not opened. Its reproducible
decision is stored in `Reports/Exia/ada_donchian48_v2/development_report.md`.

Archive verification with local datasets:

```powershell
.venv\Scripts\python.exe tools\check_research_archive.py --require-local-data
```

No command in this archive grants paper or live trading authority.
