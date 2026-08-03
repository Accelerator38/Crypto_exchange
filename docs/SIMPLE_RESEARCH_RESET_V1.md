# SimpleResearch reset v1

`SimpleResearch` is a deliberately small replacement for strategy evaluation.
It does not import Pantheon, Flash, Genetics, selectors, promotion gates or an
exchange connector.

## Contracts

1. A feature tape contains sealed OHLCV and only bar-close features.
2. A strategy returns `-1`, `0` or `1` for every symbol/bar.
3. The simulator shifts the target by one bar and executes it at the next open.
4. Every strategy uses the same fee/slippage model and ledger schema.
5. A batch is bounded to ten strategies and three variants per family.
6. Reports are screening-only: `orders_enabled=false` and
   `promotion_authority=false`.

## Commands

```powershell
.venv\Scripts\python.exe tools\build_simple_research_tapes.py
.venv\Scripts\python.exe tools\run_simple_research_batch.py
```

Canonical outputs:

- `Retrodate/simple_research_reset_v1/bitget_full8_1h.parquet`;
- `Retrodate/simple_research_reset_v1/bitget_full8_1m.parquet`;
- `Retrodate/simple_research_reset_v1/decision_tape.parquet`;
- `Reports/SimpleResearch/reset_v1/batch_report.json`.

The first batch uses fixed development, validation and OOS boundaries. The OOS
result must not be used to retune these registered trials. Any changed threshold
is a new trial and requires a new registry version.
