# Genetic Primus V8 Clean — прогресс и сравнение на 2026-08-24

## Итог

Ни одна исследованная модель сейчас не имеет оснований для promotion. Четыре V8 rolling-cell завершились и прошли проверку хэшей артефактов, а один expanding-cell восстановлен из пяти завершённых island-checkpoint. Повреждённая reboot-копия `rolling 730d + half-life 180d` сохранена только как diagnostic; этот cell успешно повторён с нуля в отдельном каталоге.

Все прогоны были локальными и offline. В конфигурациях и итоговых manifest подтверждено: `credentials_allowed=false`, `paper/live=false`, `orders_enabled=false`, `promotion_authority=false`.

## Реализованный контур V8

- измерения: purge 7 баров, общий selection bootstrap-bank, независимый audit-bank, фиксация data/fold/resample/cutoff hashes;
- rolling evaluator: train end, purge, deployment horizon 3 дня, retrain cadence, scaler fit на каждом origin, model SHA в decision ledger;
- recency fitness: half-life 30/90/180 дней, uniform anchor 20%, weighted LCB, worst-window LCB, ESS и штрафы за drawdown/turnover/complexity/action instability;
- компактные модели: linear или один hidden layer до 8 units, parsimony и feature sparsity;
- island search: 5 islands, 25% immigrants, incumbent/random seeds, 5 ограниченных generations, top-3/top-5 ensemble;
- ограниченная experiment matrix и fail-closed promotion gate с paired/audit/stress/drift проверками;
- fixed baselines исправлены: momentum/mean-reversion теперь используют сырые contract-features через identity scaler и не зависят от train window;
- добавлен явный fail-closed `--resume-partial`; по умолчанию существующий output directory по-прежнему запрещён.
- добавлены pure/offline regression checks для baseline invariance и resume guard; обе функции прошли прямой запуск без pytest/collector/evaluator.
- исправлено фактическое island-resume wiring: существующий checkpoint теперь передаётся в `run_evolution_v4` с `resume=true`; fingerprint mismatch остаётся fail-closed.

## Состояние experiment cells

| Cell | Состояние | Islands | Manifest/hash | Решение |
|---|---:|---:|---:|---|
| expanding + uniform | training завершён, post-analysis восстановлен | 5/5 | manifest нет | reject |
| rolling 730d + uniform | завершён | 5/5 | manifest есть, hashes OK | reject |
| rolling 730d + half-life 180d | свежий повтор завершён | 5/5 | manifest есть, hashes OK | reject |
| rolling 365d + half-life 90d | завершён | 5/5 | manifest есть, hashes OK | reject |
| rolling 180d + half-life 30d | завершён | 5/5 | manifest есть, hashes OK | reject |

Это engineering smoke по одному weekly origin (`2025-01-01` — `2025-01-08`), а не полная Stage 1 и не prospective/promotion evidence. Stage 2 cadence `1/3/7/30` не открывалась: один origin не позволяет обоснованно выбрать устойчивый top-2.

## Расширенный Stage 1: outer3 pilot

Начат заранее ограниченный pilot по трём последовательным weekly origins для всех пяти вариантов. Адаптивного выбора между cells не выполнялось.

| Cell | Состояние | Comparable origins | Audit paired LCB | Positive slices | Stress LCB | Promotion pass |
|---|---|---:|---:|---:|---:|---:|
| rolling 180d + HL30 | завершён, hashes OK | 2 | -14.92 bps | 0.625 | -14.64 bps | 0.0 |
| rolling 365d + HL90 | stable-env повтор завершён, hashes OK | 2 | -11.39 bps | 0.542 | -9.88 bps | 0.0 |
| rolling 730d + HL180 | завершён, hashes OK | 2 | -4.83 bps | 0.292 | -4.69 bps | 0.0 |
| rolling 730d + uniform | завершён, hashes OK | 2 | -1.55 bps | 0.458 | -1.70 bps | 0.0 |
| expanding + uniform | завершён, hashes OK | 2 | -5.08 bps | 0.250 | -5.77 bps | 0.0 |

`rolling 180d/HL30` не проходит promotion уже на двух incumbent-comparable origins: paired audit и stress LCB отрицательны. Исходный повреждённый `rolling 365d/HL90` остаётся diagnostic-only, но свежий stable-env повтор завершён: paired audit/stress LCB также отрицательны, promotion pass fraction равна нулю.

Все пять outer3 cells завершены в stable environment. Preregistered provisional ranking:

1. `rolling_730_uniform`;
2. `rolling_730_hl180`;
3. `expanding_uniform`;
4. `rolling_365_hl90`;
5. `rolling_180_hl30`.

На этом ограниченном pilot uniform weighting оказался лучше recency weighting на том же 730-дневном окне, а более короткие recency-окна были хуже. Однако у каждого cell только два incumbent-comparable origins; все audit/stress LCB отрицательны и promotion pass fraction равна нулю. Поэтому top-2 остаётся provisional, а Stage 2 не открывается.

Аудит обнаружил, что прежняя реализация `--resume-partial` разрешала каталог, но island trainer всегда передавал `resume=false`, поэтому завершённый checkpoint мог быть перезаписан. Wiring исправлен и закреплён прямой regression-проверкой. Повреждённый outer3-каталог не возобновляется. Evolution уже использует `workers=1`, поэтому multiprocessing не объясняет немой exit; причина процесса/хоста остаётся неопределённой.

## Обновление 2026-08-26: checkpoint smoke и native crash

Выполнен отдельный минимальный real checkpoint-resume smoke на синтетических данных: linear genome, population 2, generation 1, worker 1. Результат `OFFLINE_CHECKPOINT_RESUME_OK`:

- checkpoint имеет `complete=true`, `next_generation=1`;
- SHA checkpoint не изменился после resume;
- champion genome и fitness не изменились;
- неверный fingerprint был отклонён без изменения checkpoint;
- credentials/paper/live/orders/promotion отключены.

Следовательно, исправленная checkpoint-resume логика работоспособна. Немой exit тяжёлых прогонов имеет другую причину.

Read-only разбор семи Windows crash dumps за 2026-08-24 выявил одну и ту же нативную сигнатуру:

- exception `0xC0000005` — access violation;
- операция — чтение адреса `0x0`;
- fault module — `python312.dll`;
- одинаковый module offset — `0x30a36`;
- окружение: Python 3.12.4, NumPy 2.4.2, Pandas 3.0.0, PyArrow 25.0.0.

Это не Python exception и не доказательство бесконечного цикла в Genetic Primus. Без symbolized call stack нельзя определить, находится ошибка в CPython или в загруженном native extension. Heavy Stage 1 не следует продолжать в этом окружении до локализации/устранения native crash.

Локальная symbolization без сети продолжена до предела доступных артефактов:

- Windows debugger и `python312.pdb` локально отсутствуют;
- fault RVA расположен как `_PyEval_EvalFrameDefault+0x11d6` по таблице экспортов установленного DLL;
- инструкция `44 39 2b ...` обращается к `[RBX]`; во всех семи contexts `RBX=0`, `R13=0`;
- эвристический scan активной части стека от exception RSP не обнаружил адресов `.pyd`, только `python312.dll`/системные модули.

Это подтверждает единую сигнатуру null-state внутри CPython eval loop. Однако без PDB/unwind stack нельзя доказать, является ли это самостоятельной ошибкой CPython 3.12.4 или следствием более ранней порчи памяти native extension.

## Обновление 2026-08-26: отдельное Python-окружение и устранение trigger

Создано отдельное окружение `.venv_gp_v8_stable` на локальном Python 3.12.13. Сеть не использовалась. NumPy 2.3.5 и Pandas 3.0.1 взяты из bundled runtime, PyArrow 25.0.0 скопирован offline из исходного проектного venv. Запуски выполняются с `PYTHONNOUSERSITE=1`, `PYTHONFAULTHANDLER=1`, worker 1.

Checkpoint smoke и `rolling 180d/HL30` one-origin/outer3 завершились штатно. Все семь outer3 parquet frames семантически идентичны прежнему результату.

Первый запуск `rolling 365d/HL90 outer3` воспроизвёл access violation и впервые дал Python stack через faulthandler: `labels.py:43` → `DataFrame.median` → Pandas nanops → NumPy `nanmedian`. Новый Python имел эквивалентную сигнатуру `_PyEval_EvalFrameDefault+0x109d`, ту же инструкцию через `[RBX]`, `RBX=0`.

Опасная dataframe-wide медиана заменена на эквивалентную строгую медиану ровно трёх horizon-значений через pairwise `minimum/maximum`, с явным распространением NaN. Votes и support также рассчитываются из фиксированных arrays без Pandas row-reduction.

После изменения:

- `rolling 180d/HL30 outer3` завершён без crash;
- ранее падавший `rolling 365d/HL90 outer3` завершён с первого запуска;
- `rolling 730d/HL180`, `rolling 730d/uniform` и `expanding/uniform` outer3 также завершены;
- manifest/artifact SHA корректны;
- origin 0 полностью совпал со старым one-origin результатом;
- новых crash dumps не появилось.

## V8: результат одного deployment origin

Для rolling-cell ниже показан выбранный top-3/top-5 ensemble. Fitness не является доходностью и сравнивается только внутри одинакового evaluator. Исправленный fixed raw-momentum на этом origin дал `+872.03 bps`, `DD 3.15%`, 158 сделок; NoTrade — `0`; raw mean-reversion — `-1395.49 bps`.

| Вариант | Ensemble | Selection fitness | Audit fitness | Net, bps | DD | Сделки | Разница к raw momentum | Основные failures |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| rolling 730d uniform | top-3 | -73.15 | -72.11 | +717.90 | 3.68% | 30 | -154.13 | max DD; worst fold; median fold LCB |
| rolling 730d HL180 | top-3 | -67.13 | -68.37 | +628.11 | 3.36% | 35 | -243.92 | max DD; worst fold; median fold LCB |
| rolling 365d HL90 | top-5 | -102.14 | -100.68 | +52.30 | 4.06% | 42 | -819.73 | max DD; overall/worst/median LCB |
| rolling 180d HL30 | top-3 | -66.10 | -69.50 | +118.05 | 2.74% | 38 | -753.98 | worst fold; median fold LCB |
| expanding uniform (recovered) | top-5 | -130.31 | audit LCB -2.20 bps/day | -304.69 | 5.64% | 34 | -1176.72 | minimum trades; max DD; overall/worst/median LCB |

`rolling 180d/HL30` лучший по selection fitness и deployment drawdown, но намного хуже простого momentum. `rolling 730d/HL180` близок к нему по fitness и занял второе место по deployment-доходности. `rolling 730d/uniform` лучший по единичной deployment-доходности, однако уступает momentum и проваливает историческую fold-устойчивость/drawdown. Один origin не позволяет заключить, что recency weighting улучшает validation.

## Сравнение с предыдущей генетикой

Метрики прежних версий имеют другие протоколы и единицы (обычно bps/сделку против daily/weekly V8), поэтому используются только как диагностика generalization.

| Модель | Split/protocol | LCB | Max DD | Вывод |
|---|---|---:|---:|---|
| Genetic Primus V1 | development | +8.73 bps/trade | 19.84% | reject: slice gates |
| Genetic Primus V2 | development | +37.47 bps/trade | 16.56% | reject: temporal robustness |
| Genetic Primus V3 | development | +46.05 bps/trade | 16.41% | прошёл только dev |
| Genetic Primus V3 | validation | -36.94 bps/trade | 49.99% | reject: сильное переобучение |
| Genetic Primus V4 compact | strict base16 | -0.62 bps/day | 53.91% | reject; worst fold mean -27.44 bps |
| V8 expanding ensemble | recovered selection/audit | -1.48 / -2.20 bps/day LCB | 68.29% | reject |

V8 пока не даёт готового преимущества по доходности, но новый evaluator корректнее выявляет нестабильность до promotion: purge, независимый audit, recency ESS, rolling origins и explicit gates уменьшают риск повторения V3.

## Сравнение с обычными агентами

| Агент | Срез | Результат | Статус |
|---|---|---:|---|
| NoTrade | текущая deployment-неделя | 0 bps | контроль |
| raw Momentum return_1 | текущая deployment-неделя | +872.03 bps, DD 3.15% | лучший на одной неделе; не promotion evidence |
| raw MeanReversion return_1 | текущая deployment-неделя | -1395.49 bps | reject на этом срезе |
| MQ_REL_MOM24 | архивная validation | LCB -2.25 bps/trade, DD 10.88% | лучший архивный validation-control, но rejected/diagnostic |
| player_one | архивная validation | mean +19.42, LCB -29.79 bps/trade | reject |
| player_two | архивная validation | mean -6.43, LCB -21.97 bps/trade | reject |

Отдельный V8 Clean LINK holdout имеет 55 событий: frozen LINK reversal LCB `+45.27 bps/event`, а genetic gate — `+11.59 bps/event` на 20 событиях. Frozen-правило в этом отчёте полностью совпадает с обычным 4h mean-reversion, поэтому это узкая event-гипотеза, а не доказательство преимущества генетики на full-8 panel.

## Последствия перезапуска ПК

- Итоговые артефакты `rolling 730 uniform`, `rolling 365/HL90`, `rolling 180/HL30` целы: все SHA из manifest совпадают.
- Повреждённый `rolling 730/HL180` не имеет manifest и помечен `DIAGNOSTIC_INCOMPLETE.json` с `usable_as_experiment_evidence=false`, `resume_allowed=false` и SHA checkpoints.
- Свежий `rolling 730/HL180` завершён в `matrix_weekly_rolling730_hl180_smoke_20260824_fresh1`; manifest и artifact SHA корректны.
- Fixed baselines в свежем 730-дневном и завершённом 365-дневном cell полностью совпадают по policy SHA, net, drawdown, trades и fills.
- Никаких фоновых процессов после reboot не предполагается и новых процессов в рамках проверки не запускалось.

## Рекомендуемое продолжение

1. Выполнено: pure/offline regression checks закрепляют baseline invariance и fail-closed resume guard.
2. Выполнено: неполный `730/HL180` изолирован как diagnostic, а cell успешно повторён с нуля в новом каталоге.
3. Выполнено для двух вариантов: `180/HL30 outer3` и свежий `365/HL90 outer3` завершены и rejected; старый повреждённый каталог изолирован как diagnostic.
4. Выполнено: минимальный real checkpoint-resume smoke прошёл; silent exits подтверждены семью одинаковыми native crash dumps (`0xC0000005`, null read, `python312.dll+0x30a36`).
5. Частично выполнено: локальная symbolization без сети дошла до `_PyEval_EvalFrameDefault+0x11d6`; debugger/PDB отсутствуют, активный стек не указывает на конкретный `.pyd`.
6. Выполнено: создано отдельное Python 3.12.13 окружение, trigger локализован в Pandas/NumPy row median и устранён без изменения результатов.
7. Выполнено: все пять preregistered outer3 cells завершены в `.venv_gp_v8_stable`; provisional top-2 — `rolling_730_uniform` и `rolling_730_hl180`.
8. Следующий шаг: расширить weekly Stage 1 beyond outer3 до заранее выбранного достаточного числа outer slices. Не открывать cadence `1/3/7/30` по двум comparable origins.
9. Только после устойчивого Stage 1 открыть cadence для двух лучших вариантов; weekly 7d переиспользовать, не перезапускать.
10. Promotion разрешать лишь при прохождении paired audit/stress/drift gates. До этого incumbent остаётся неизменным, а runtime/live/orders — выключенными.
