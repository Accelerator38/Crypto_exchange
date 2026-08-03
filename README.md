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

Archive verification with local datasets:

```powershell
.venv\Scripts\python.exe tools\check_research_archive.py --require-local-data
```

No command in this archive grants paper or live trading authority.
