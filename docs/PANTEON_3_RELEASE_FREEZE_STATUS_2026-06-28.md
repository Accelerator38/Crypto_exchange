# Panteon 3.0 release-freeze status

Дата: 2026-06-28

Статус версии: `paper-gated candidate`, не `live-ready`.

## Текущие артефакты

- Replay/walk-forward matrix:
  `Reports/Panteon3PreLiveMatrix/promotion_regime_prior_gate_cost_verified_matrix/panteon3_pre_live_matrix_summary.json`
- BITGET paper canary:
  `Reports/Panteon3Canary/paper_after_range_promotion_verify/latest_canary_summary.json`
- MEXC/BITGET prior paper canary:
  `Reports/Panteon3Canary/paper_next_both/latest_canary_summary.json`
- Последний BITGET paper run:
  `Results/Panteon3PaperCanaryAfterRangePromotion/BITGET/2026-06-27_20-32-49_v2`

## Что уже доказано

- Replay candidate дает ненулевые сделки на futures replay:
  59 fills, 29 closed trades, realized PnL 1.9949766542 USD,
  expectancy 0.0687922984 USD/trade.
- Fees/funding/slippage учитываются в walk-forward/canary parsing.
- Controlled exploration и promotion-derived routing находятся behind flags.
- После отрицательной закрытой exploration/promotion сделки actor/symbol/action блокируется повторно.
- Rejected/unconfirmed orders не должны создавать позиции; phantom-position prevention покрыт execution tests.
- Live preflight теперь блокирует live, если Panteon candidate хуже best causal component.

## Что не доказано

- Последний paper/shadow canary не дал real paper signals/orders/fills:
  `zero_signals`, `zero_orders`, `zero_fills`.
- Replay candidate не обгоняет лучший causal component:
  best component `Solo_LiveVolCompress` / `LiveVolCompress`,
  best_component_pnl_usd около 3.9583 USD против candidate 1.9950 USD.
- Текущий live status stale: старые MEXC/BITGET `status.json` показывают `running/active`,
  но python live processes отсутствуют.
- CarryFlowAgentV2 все еще не имеет полноценного futures funding/OI/crowding feed для live-quality signal.
- LiveVolCompress/MomentumScalper на текущем futures paper feed почти всегда inactive.

## Live verdict

Live restart с real orders сейчас запрещен.

Read-only preflight reasons для MEXC и BITGET:

- `matrix_not_beating_best_component`
- `canary_failed`
- `canary_zero_signals`
- `canary_zero_orders`
- `canary_zero_fills`
- `canary_nonpositive_expectancy`

## Следующие P0 шаги до live restart

1. Расширить futures replay dataset минимум до нескольких рыночных режимов и нескольких дней по MEXC/BITGET.
2. Пересобрать causal component memory на свежих futures данных, отдельно по exchange/symbol/regime/action.
3. Добиться nonzero `LiveVolCompress`/`MomentumScalper` open candidates на paper futures feed без отключения risk gates.
4. Запустить paper/shadow canary до ненулевых `signals/orders/fills`.
5. Требовать positive expectancy after fees/funding/slippage и `reconcile_ok`.
6. Только после passed preflight запускать live workers с tiny risk cap.

## Команды проверки

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_live_preflight.py src\panteon_v2\tests\test_panteon3_live_canary_check.py src\panteon_v2\tests\test_flash_allocator.py -q
```

```powershell
.\.venv\Scripts\python.exe tools\run_panteon3_live_canary_check.py --results-root Results\Panteon3PaperCanaryAfterRangePromotion --reports-dir Reports\Panteon3Canary\paper_after_range_promotion_verify --exchange BITGET --lookback-minutes 10080
```

```powershell
.\.venv\Scripts\python.exe -c "import sys; from pathlib import Path; root=Path.cwd(); sys.path.insert(0, str(root/'src')); from panteon_v2.app.live_preflight import LivePreflightConfig, run_live_preflight; matrix=root/'Reports/Panteon3PreLiveMatrix/promotion_regime_prior_gate_cost_verified_matrix/panteon3_pre_live_matrix_summary.json'; canary=root/'Reports/Panteon3Canary/paper_after_range_promotion_verify/latest_canary_summary.json'; print(run_live_preflight('BITGET', 'live_futures', config=LivePreflightConfig(project_root=root, matrix_summary_path=matrix, canary_summary_path=canary, max_age_hours=72)))"
```
