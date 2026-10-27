# Genetic Primus: net-growth fitness v1

Цель поиска — максимизировать чистый рост общего капитала при явном ограничении просадки.
Scorer принимает проверенный непрерывный ledger из `unified_measurement_v1`, а не gross returns или сумму
процентных доходностей отдельных символов. Издержки уже входят в NAV: per-fill fees, adverse drag и signed funding.

## Search objective

`raw_fitness = 10000 × ln(final_shared_NAV / initial_shared_capital) / calendar_days`.

`fitness = raw_fitness − active_parameters / 100 × registered_parameter_penalty − active_features × registered_feature_penalty`.

При одинаковых периоде и initial capital без parsimony штрафов максимум fitness совпадает с максимумом
конечного капитала. Calendar days включает все cash-дни. Losses уменьшают log wealth, а не отбрасываются.
Число сделок не является положительной целью и не создаёт требование торговать ради fitness.

В текущем machine contract parsimony penalties явно равны нулю: новый этап меняет profit/risk семантику,
не добавляя неподтверждённых коэффициентов регуляризации. Ненулевые penalties требуют зарегистрированного
коэффициента и фактических counts архитектуры; пропущенные counts вызывают ошибку. Они входят в основной
scalar score, а не только в tie-break. Коэффициенты и drawdown budget включены в objective SHA.

Комиссии/turnover повторно не штрафуются. Optional fill limit — операционный hard gate; это отдельный
лимит частоты исполнения, а не ещё одно вычитание уже учтённых trading costs. В текущем contract limit=null.

## Исследовательский риск

Зафиксирована матрица **10/20/30% max DD**. Это варианты исследований, не согласованный риск реальных денег.
Точное равенство лимиту допустимо. DD включает initial capital peak и costs первого входа; считается по
4h reference opens непрерывного ledger. Для оценки внутрибара риска необходим отдельный execution/risk этап.

Когда предоставлены normal и stress ledger, gate требует `max(normal_DD, stress_DD) <= budget`.
Оба должны иметь одинаковые prices, universe, capital, funding, decisions и clock. Без stress search
показывает `stress_evaluated=false`; статистический аудит такой пары выполнить нельзя.

Ranking: прохождение search risk gates → меньше нарушенных risk gates → penalized net growth → меньшие
DD/turnover/complexity при точном равенстве предыдущих компонентов. Даже недопустимый кандидат сохраняет
численный score для анализа, но проигрывает допустимому. NoTrade имеет raw fitness 0 и DD 0; у него нет
положительного статистического эффекта, но он остаётся baseline/fallback для убыточных моделей.

Все три risk variants оцениваются на уже рассчитанных ledger. Они не требуют трёх новых обучений, трёх
resample banks или трёх прогонов сигналов. Нельзя выбирать budget после просмотра августа–сентября и
представлять его как заранее выбранный. Отчёт показывает всю зарегистрированную матрицу.

## LCB и статистические проверки

LCB **не входит в fitness, ranking, search_failed_gates или временной минимум цели**. Нет лексикографического
приоритета worst-fold LCB или weighted LCB над чистым капиталом. Recency weights для обучения модели
определяются в следующем этапе; новая equity-цель пока использует весь reporting interval.

`growth_diagnostics` считает descriptive daily/fold means и, при явно переданном общем bank, LCB.
По умолчанию новый search scorer не исполняет bootstrap для каждого genome. Его `lcb_daily_bps=null`,
`stability_bootstrap_evaluated=false`, `validation_status=not_evaluated`. Fold summaries не являются outer slices.

`check_profit_stability` предназначен для final candidates. Требования:

- normal и stress outcomes;
- общий фиксированный selection seed bank и отдельный независимый audit seed bank;
- одинаковые protocol, samples, block length, cutoff и fold definitions;
- на обоих banks normal/stress LCB средней дневной net log доходности строго > 0;
- paired LCB дневной разницы log доходности challenger−incumbent строго > 0; без incumbent явно используется NoTrade;
- при paired incumbent нужны обе cost scenarios и одинаковые prices/capital/funding/costs/clock;
- все cash-дни остаются в sample; дневной bootstrap не смешивает полные и неполные calendar days;
- фактическое содержимое resample banks проверяется по SHA; повторное использование selection seed в audit запрещено.

Статистика: `mean(log(1 + daily_net_return)) × 10000`. Paired difference суммируется в log отношение
конечных капиталов при одинаковом периоде. Для совместимости legacy `mean_daily_bps/lcb_daily_bps` остаются
арифметическими метриками и подписаны отдельно от `mean_net_log_bps/net_log_lcb_daily_bps`.
На passed/failed LCB результатах search fitness не меняется. Положительный итог с отрицательным LCB
сохраняет прибыльный search score, но не получает подтверждения устойчивости.
Каждый audit сохраняет alpha и SHA сравниваемых normal/stress/incumbent ledger, а также metadata обоих banks.

Это статистическая проверка, не полный promotion gate: нужны отдельные temporal outer slices, проверка
multiple-testing, независимый новый период, concentration/action drift и execution evidence.
`promotion_authority=false`, `temporal_gate_passed=false` в search остаются явными.

## Интеграция и конфигурация

Основной модуль: `src/exia/genetic_primus/profit_fitness_v1.py`.
Machine contract: `configs/genetic_primus_profit_fitness_v1.json`.

`objectives_from_contract(contract)` возвращает три `ProfitObjective`. Core API:

```python
objectives = objectives_from_contract(contract)
scores = [score_net_growth(normal, objective=o, stress_result=stress,
                           complexity_metrics=actual_architecture_counts)
          for o in objectives]
```

Существующий `score_simulation_v4` использует новый objective при явном ключе `config['search_objective']`
с `asdict(ProfitObjective(...))`. Нужен `as_primus_simulation(normal, stress_result=stress)`:
он передаёт проверенный source ledger, а scorer сверяет нормализованный portfolio с этим ledger.
Legacy simulation без source ledger или изменённый portfolio отклоняются новым objective.

Для нового scorer `compute_stability_diagnostics=false` по умолчанию. Включение этого параметра требует
общего bootstrap bank и добавляет диагностический LCB, не меняя score/ranking. Generation log записывает
objective SHA, budget, raw net growth и validation status; пропущенный LCB остаётся null, а не фиктивным -1e9.

Существующие legacy objectives доступны для воспроизведения архивов. Новая модельная программа должна
выбрать новый contract явно и зарегистрировать objective/model/data/fold/bank/cutoff identities до
рыночного эксперимента. Fitters и causal per-origin prefixes подключаются следующим этапом; эта работа их не запускает.

## Проверки

Прямой checker использует только synthetic fixtures и unittest, без pytest/discovery/collector:

```powershell
& 'E:\Crypto_exchange\.venv\Scripts\python.exe' -X utf8 tools/check_genetic_primus_profit_fitness_v1.py --scope core
& 'E:\Crypto_exchange\.venv\Scripts\python.exe' -X utf8 tools/check_genetic_primus_profit_fitness_v1.py --scope integration
```

Полный отчёт: `--scope all --report-dir <новый подкаталог Reports/Exia/Genetic_Primus>`.
Существующие артефакты не перезаписываются. Выход: ledger, точная fixture, матрица 10/20/30%, статистический
аудит, сохранённые resample indices, графики капитала/DD/exposure и risk/net-growth, PNG/SVG, evidence receipt.
Все графики помечены как синтетические. Обучение, рынок, сеть, paper/live и promotion не выполняются.
