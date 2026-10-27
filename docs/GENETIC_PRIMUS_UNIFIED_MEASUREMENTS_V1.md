# Genetic Primus: единые измерения v1

Модуль `src/exia/genetic_primus/unified_measurement_v1.py` объединяет один денежный счёт, непрерывные позиции,
комиссии, adverse price adjustments, подписанный funding, капитал, экспозицию и трёхдневный clock.
Это чистая offline-арифметика над явно переданными входами. Она не получает данные, не обучает модели,
не интегрируется с торговым runtime и не выдаёт полномочий для paper/live или promotion.

## Временной и экономический контракт

- Бар: 4 часа UTC. Deployment horizon и retrain cadence: 18 баров, ровно три дня.
- Вход: первый open строго после `decision_asof + execution_delay`. Выход forecast label: open через 18 баров.
- `complete_origins` использует существующий `research_execution_contract_v1.execution_window`; это общий источник clock.
- Задержка и границы не могут меняться между origins. Каждый следующий entry совпадает с предыдущим planned exit.
- `label_binding` принимает существующие diagnostic open/open samples, проверяет цены/символы/времена и пересчитывает
  cutoff как `training_asof - 7 * BAR_MS`. Старый cutoff из единственного materialization context не используется.
- `open_to_last_close` и сокращённый последний горизонт отклоняются. Admission метки не даёт разрешения на fit.
- Label — трёхдневный прогнозируемый интервал; фактическая позиция может переноситься через несколько интервалов.
  Поэтому `holding_bars` сделки не обязан равняться 18. Перенос не меняет определение labels и не вызывает refill.

Для `[2026-08-01, 2026-10-01)` получаются **20** полных origins с 1 августа по 27 сентября.
Первый entry — 1 августа 04:00 UTC, последний complete exit/terminal — 30 сентября 04:00 UTC.
После этого капитал остаётся cash до отчётной границы 1 октября 00:00 UTC. Origin 30 сентября потребовал бы
выхода 3 октября и в этот контракт не входит. Для этого cash tail не нужны дополнительные цены после terminal.

## Cash, позиции и sizing

`MeasurementSpec.initial_capital_quote` — капитал всего universe, а не каждого символа. `symbols` фиксируется
для всех моделей. Quote panel должен быть непрерывным и полным до единственного terminal.
`market_inputs_sha256` — SHA256 **canonical JSON переданного price panel**, проверяется против его содержимого.
SHA сырого файла/manifest хранится отдельно в происхождении данных следующего evaluator; подменять эти понятия нельзя.

Поддерживается long/cash без borrow/leverage. `targets` содержит все символы и неотрицательные веса с суммой ≤ 1.
При неизменном весе сохраняется quantity, даже если model/scaler SHA изменились или цена создала drift веса.
При изменении положительного веса позиция полностью закрывается и открывается заново. Это явная политика sizing,
а не скрытый ребаланс на каждом mark; частичный delta-rebalance является отдельным будущим вариантом контракта.

На ребалансе сначала закрываются изменённые позиции. Новый желаемый allocation = weight × NAV после sells.
При нехватке cash новые allocations пропорционально уменьшаются; scale фиксируется в decision ledger.
Уже перенесённые позиции не продаются ради нового target. Buy allocation включает buy fee:

`quantity = allocated_quote / (reference_open * (1 + buy_adverse_bps/10000) * (1 + buy_fee_bps/10000))`.

Существует одно зарегистрированное финальное закрытие всех позиций. На остальных retrains принудительного
закрытия нет. Повторный вход, ресайз, exit и terminal каждый платят свои per-fill costs.

## Funding и последовательность событий

Funding input содержит `complete=true`, точный интервал покрытия, universe, полный упорядоченный список
settlements и SHA evidence. Пустой список означает подтверждённое отсутствие settlements; неизвестные данные
не заменяются нулём. Проверка declared completeness не заменяет внешнюю проверку источника funding.

На одинаковом timestamp порядок: **funding → sells → buys → mark**. Для long положительный signed rate списывает
`quantity × settlement_mark × rate`; отрицательный зачисляет. Позиция владеет settlement при
`entry_time < settlement_time <= exit_time`. Дубликаты и неправильный порядок отклоняются.

Funding списывается из cash во время settlement. Если cash reserve не покрывает дебет, replay завершается ошибкой;
неявное финансирование, перенос неоплаченного funding на exit и отрицательный cash не разрешаются.
Резерв должен задаваться торговой политикой; этот модуль не выбирает его за модель.

## Метрики и идентичность

На каждом 4h mark: `NAV = cash + Σ quantity × reference_open`. Независимая проверка:

`NAV = initial + realized_gross + unrealized_gross - fees - adverse_drag - signed_funding`.

После terminal сумма `closed_trade.net_pnl_quote` также должна совпадать с `final_capital - initial_capital`.
Начальный капитал входит в максимум equity, поэтому первая комиссия создаёт DD, а не исчезает из измерения.
DD считается на 4h reference opens; внутрибара экстремумы этим измерением не покрываются.

`net_return` — последовательное отношение общего NAV. Интервал `(previous_mark, current_mark]` относится к
calendar bucket `current_mark - 1ms`, включая terminal ровно на границе дня. `daily_measurements` компаундирует
такие интервалы, включая cash-дни. Совокупная дневная доходность сходится с конечным капиталом.
Turnover хранится как quote notional fills / initial total capital; fills/day использует весь calendar interval.
Средняя экспозиция — равновзвешенное по длительности 4h интервалов отношение exposure/NAV после предыдущего mark.

Каждое решение фиксирует model SHA, scaler SHA, causal feature-prefix SHA и training identity SHA.
SHA всех decisions, costs, funding inputs и всего replay входа сохраняются отдельно. Output SHA включает
ledger, equity, decisions, closed trades и метрики. Адаптеры проверяют output SHA перед использованием.

`compare_policies` получает одну панель цен, funding, initial capital, universe и reporting interval для всех
агентов. Decision clocks всех конкурентов должны совпадать. Normal/stress задаются явно, без defaults costs.

## Подключение к scoring

`as_primus_simulation` переводит новый ledger в существующий формат fitness/bootstrap. Equity нормируется на
общий initial capital; `net_return` остаётся долей, DD получает отрицательный знак, ожидаемый старым scorer.
Market state по entry и opportunity diagnostics должны поступать из причинных proposal logs. Без такого
источника state=`unknown`; искусственные режимы и opportunities адаптер не создаёт.

В `fitness_v4.py` ranking теперь создаётся **один раз после всех веток**:

1. Наличие/отсутствие всех failed gates.
2. Число уникальных нарушенных gates.
3. Итоговый penalized scalar fitness.
4. Меньшие DD, fills/day и active parameter count при равенстве предыдущих полей.

Evidence и concentration gates больше не теряются в paired/constrained branch. Регуляризация и другие
штрафы внутри fitness влияют на search, а не только на поздний tie-break. Неопределённые/NaN score отклоняются.
Это исправление ранжирования; формула нового profit/risk fitness относится к следующему этапу разработки.

Старые frozen результаты, replay/L2-модули и open/last-close evaluator сохраняют архивную семантику.
Их результаты не переводятся в новый контракт простым переименованием метрик. Новые модельные сравнения
должны использовать `compare_policies` и при необходимости `as_primus_simulation`; сами fitters ещё не подключены.

## Проверки и воспроизводимость

Использовать только `E:\Crypto_exchange\.venv\Scripts\python.exe`. Checker работает без pytest,
discovery, collector, ожиданий по clock или фоновых процессов. Тесты — small synthetic fixtures;
scoring regression использует 12 дневных значений и один общий банк из четырёх resamples.

```powershell
& 'E:\Crypto_exchange\.venv\Scripts\python.exe' -X utf8 tools/check_genetic_primus_unified_measurement_v1.py --scope core
& 'E:\Crypto_exchange\.venv\Scripts\python.exe' -X utf8 tools/check_genetic_primus_unified_measurement_v1.py --scope ranking
```

Для полной проверки и reviewable инженерного отчёта можно передать `--scope all --report-dir` с **новым**
подкаталогом `Reports/Exia/Genetic_Primus`. Существующие артефакты не перезаписываются. Отчёт содержит
точную fixture, normal/stress ledger, дневные результаты, PNG/SVG капитала/DD/exposure и evidence receipt.
Его графики обязательно помечены как синтетические. Обучение и market evaluations остаются равны нулю.

Следующий шаг — причинный evaluator: построить общий dataset manifest, каждому origin передать собственный
допустимый training prefix, зарегистрировать costs и обеспечить проверенное полное funding coverage.
Только затем подключать реальные прогнозы моделей и запускать август–сентябрьский market replay.
