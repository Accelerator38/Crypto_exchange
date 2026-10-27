# Genetic_Primus V2

## Причина версии

V1 получил положительные общие development mean/median/LCB, но провалил 10
поддержанных разрезов. Причина была не в execution path: финальные slice-gates
не участвовали в отборе поколений.

V2 сохраняет без изменений dataset, 4h discretization, 30 признаков, сеть
`30 -> 32 -> 16 -> 3`, next-open execution и stress cost 16 bps. Единственное
экспериментальное изменение — порядок сравнения геномов.

## Лексикографический ranking

Особи сравниваются строго по уровням:

1. Выполнение structural/risk gates: fills, monthly coverage, drawdown и collapse.
2. Выполнение общих positive mean/LCB gates.
3. Минимальное число поддержанных slices с `LCB <= 0`.
4. Максимальный LCB худшего поддержанного slice.
5. Robust monthly fitness `0.20 mean + 0.30 median + 0.50 LCB`.
6. Общий LCB как последний tie-breaker.

Числовой вес не может компенсировать более высокий уровень. Tournament получает
ordinal ranks, полученные из полного tuple; поэтому высокий общий PnL не может
перекрыть дополнительный проваленный режим, direction или symbol.

## Ограничение сложности

Новых trading thresholds, position sizing, router, stop/exit модулей или
классификаторов режимов нет. V1 champion и scaler используются только как
SHA-проверенная стартовая точка. Все authority flags остаются `false`.

## Запуск

```powershell
.venv\Scripts\python.exe tools\run_genetic_primus_v2.py --smoke
```

Полный development:

```powershell
.venv\Scripts\python.exe tools\run_genetic_primus_v2.py
```

Каждое поколение обновляет `strict_gate_progress_live.png`. Validation не
открывается runner-ом даже при нуле failed slices.

## Результат development V2

Запуск `Reports/Exia/Genetic_Primus/development_v2_20260821` завершён на 50
поколениях. Выбран чемпион поколения 45.

| Метрика | V1 | V2 |
|---|---:|---:|
| Closed trades | 4272 | 3319 |
| Mean net, bps | +23.3551 | +55.4290 |
| Median net, bps | +7.6547 | +7.7002 |
| Overall LCB, bps | +8.2705 | +38.7571 |
| Worst supported slice LCB, bps | -34.3016 | +9.5006 |
| Failed supported slices | 10 | 0 |
| Max drawdown | 19.8421% | 16.5600% |

V2 полностью устранил direction/regime/symbol slice failures. Тем не менее он
заморожен как `DEVELOPMENT_V2_REJECTED_TEMPORAL_ROBUSTNESS`: median monthly
robust fitness равна `-6.1406`, только 10 из 24 period scores и 9 из 24 monthly
LCB положительны. Validation не открыта.

Это локализует следующую проблему: после прохождения slices ranking продолжал
максимизировать худший cross-sectional LCB раньше temporal fitness. Следующий
заранее фиксируемый эксперимент должен поднять `temporal_fitness > 0` на уровень
hard lexicographic gate перед оптимизацией запаса slice LCB. Данные, execution,
сеть и классификацию режимов менять одновременно нельзя.
