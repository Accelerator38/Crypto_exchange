# Genetic_Primus V1

## Назначение

`Genetic_Primus` — изолированный исследовательский нейрогенетический агент для
Bitget USDT futures. Он не зарегистрирован в live runtime, не загружает старые
GeneticCore checkpoints и не имеет права включать paper/live ордера.

## Неизменяемый контракт V1

| Поле | Значение |
|---|---|
| Таймфрейм | 4 часа |
| Данные | immutable Bitget full8 2022-01-01..2026-07-14 |
| Решение | после закрытия 4h бара |
| Исполнение | следующий open |
| Действия | `SHORT`, `FLAT`, `LONG` |
| Сеть | `30 -> 32 -> 16 -> 3`, `tanh` |
| Position sizing | вне генома |
| Parallel evaluation | 16 worker processes; deterministic ordered reduction |
| Costs | base 8 bps, stress 16 bps round trip |
| Fitness period | календарный месяц |
| Fitness | `0.20 mean + 0.30 median + 0.50 LCB95` |

Фьючерсная позиция представлена целевым состоянием. Переходы и комиссии
обрабатывает один simulator contract; сеть не выбирает размер позиции, плечо,
stop loss или отдельную exit-политику.

## Рыночный контекст

Сеть получает непрерывные trend, volatility, breadth, dispersion и transition
features. Для анализа они причинно сводятся к пяти состояниям:

- `trend_up`;
- `trend_down`;
- `quiet_range`;
- `volatile_range`;
- `transition`.

Отдельных геномов и router по режимам нет. Режимы используются для диагностики
collapse и условной expectancy, а не для выбора другого execution path.

## Временные окна

| Окно | Период | Назначение |
|---|---|---|
| development | 2022-2023 | обучение и эволюция |
| validation | 2024 | выбор после заморозки development |
| OOS | 2025 | однократная проверка кандидата |
| sanity | 2026-01-01..2026-07-14 | историческая устойчивость |

Эти окна уже раскрывались в прежних исследованиях и не дают live authority.
После исторического успеха обязателен новый sealed prospective holdout.

## Evidence bundle

Каждый запуск сохраняет:

- `manifest.json` с dataset/config/code SHA и safety flags;
- `generation_metrics.parquet` и `experiment_tables.xlsx`;
- `generation_champions/` с лучшим геномом каждого поколения;
- `evolution_checkpoint.npz` с популяцией и RNG state для точного resume;
- `evolution_progress.png` и `market_state_progress_live.png`, обновляемые после поколения;
- `evolution_overview.png`;
- `market_state_progress.png`;
- `champion_equity_drawdown.png`;
- period, slice и trade ledgers;
- читаемый `report.md` и машинный `summary.json`.

В XLSX каждое логическое поле записывается в отдельную ячейку; листы не
содержат объединённых строк с разделителями.

## Запуск engineering smoke

```powershell
.venv\Scripts\python.exe tools\run_genetic_primus_v1.py --smoke
```

Smoke проверяет data/features/labels/evolution/reporting plumbing на первых
180 днях development. Он не открывает validation/OOS и ничего не говорит о
прибыльности.

После прерывания тот же запуск продолжается без повторного отбора:

```powershell
.venv\Scripts\python.exe tools\run_genetic_primus_v1.py `
  --smoke `
  --output-dir Reports\Exia\Genetic_Primus\<run> `
  --resume
```

## Полный development run

```powershell
.venv\Scripts\python.exe tools\run_genetic_primus_v1.py
```

Полный запуск также остаётся development-only. Validation должна запускаться
отдельной командой только после фиксации champion SHA и scaler SHA.

## Строгая заморозка development

После завершения эволюции чемпион повторно оценивается с тем же stress-cost и
bootstrap seed, зависящим только от SHA генома:

```powershell
.venv\Scripts\python.exe tools\evaluate_genetic_primus_v1.py `
  --run-dir Reports\Exia\Genetic_Primus\<development-run>
```

Команда проверяет SHA config, training code, dataset, champion и scaler. Она
создаёт `freeze_manifest.json`, строгие period/slice таблицы, XLSX с отдельными
ячейками и heatmap mean/median/LCB. Validation разрешается только если нет ни
одного общего, risk/collapse или поддержанного slice LCB gate failure. Сам
evaluator не открывает validation и не имеет paper/live authority.

## Результат development V1

Запуск `Reports/Exia/Genetic_Primus/development_v1_20260821` завершён и
заморожен со статусом `DEVELOPMENT_REJECTED_SLICE_GATES`.

| Метрика | Результат |
|---|---:|
| Закрытые сделки | 4272 |
| Mean net | +23.3551 bps |
| Median net | +7.6547 bps |
| LCB 95% net | +8.2705 bps |
| Max drawdown | 19.8421% |
| Поддержанные месяцы | 24/24 |

Несмотря на положительный общий профиль, LCB неположительна в `trend_up`,
`volatile_range`, `transition`, направлении `SHORT` и шести символах. Прошли
только режимы `trend_down`/`quiet_range` и символы `LINK/USDT`/`SOL/USDT`.
Validation не открыта.

Корневая архитектурная проблема V1: slice-gates были post-development
проверкой, но не входили в ranking особей при эволюции. Следующая версия должна
лексикографически минимизировать число проваленных поддержанных slices и
максимизировать худший slice LCB до оптимизации общего monthly fitness. Снижать
gate или выбирать удачные slices V1 задним числом запрещено.
