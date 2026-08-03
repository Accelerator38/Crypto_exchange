# Исследовательский архив Panteon

Дата заморозки: 2026-08-03

Ветка: `codex/panteon-research-archive-20260803`

Тег исходного freeze: `panteon-research-freeze-2026-08-03`

Финальный архивный тег: `panteon-research-archive-2026-08-03`

## Статус

Panteon больше не является live-контуром. Проект сохранён как история
исследований, набор отрицательных результатов, data tooling и библиотека
инженерных защит.

Архивные инварианты:

- legacy Bitget live-route блокируется в исходном коде;
- `orders_enabled=false`;
- `promotion_authority=false`;
- активных торговых кандидатов нет;
- валидированных model checkpoints нет;
- Flash, Genetics и ensemble не разрешены как runtime-путь;
- Freqtrade используется только как изолированный dry-run harness.

## Что является основным

| Область | Основной источник |
|---|---|
| Архитектурное решение | `docs/PANTEON_REAL_TRADING_RESET_2026-08-03.md` |
| Реализация freeze/reset | `docs/PANTEON_RESET_IMPLEMENTATION_2026-08-03.md` |
| Каталог данных | `docs/research_archive/DATASETS.md` |
| Реестр гипотез | `docs/research_archive/HYPOTHESES.md` |
| Карта Git-истории | `docs/research_archive/GIT_HISTORY.md` |
| Машинный manifest архива | `configs/research_archive/archive_manifest_v1.json` |
| Машинный реестр гипотез | `configs/research_archive/hypotheses_v1.json` |
| Единый исследовательский pipeline | `src/simple_research` |
| Dry-run execution harness | `freqtrade_reset` |

Остальные документы в `docs/`, legacy runtime в `src/panteon_runtime` и сложный
контур в `src/panteon_v2` сохранены для истории и извлечения отдельных
компонентов. Они не являются инструкцией по запуску торговли.

## Проверка архива

Проверка Git-части, freeze и доступных локальных данных:

```powershell
.venv\Scripts\python.exe tools\check_research_archive.py --require-local-data
```

На другом компьютере без больших локальных datasets:

```powershell
python tools\check_research_archive.py
```

Второй вариант проверяет код, safety contract, registry и присутствующие
datasets, но не требует локальные файлы из `Retrodate`.

Воспроизведение SimpleResearch tapes и batch:

```powershell
.venv\Scripts\python.exe tools\build_simple_research_tapes.py
.venv\Scripts\python.exe tools\run_simple_research_batch.py
```

Результат batch является screening evidence. Он не создаёт actor, policy,
paper-допуск или live-допуск.

## Git

На момент архивации remote содержал только `main`. Локально были только `main`
и рабочая reset-ветка, которая переименована в архивную. Поэтому удалять
ветки без потери истории не требовалось.

`main` оставлен без изменений как опубликованный исторический базис. Полный
freeze/reset и структурированные отрицательные результаты публикуются в
отдельной архивной ветке и фиксируются annotated tag.

## Возобновление исследований

Возобновлять Panteon runtime не следует. Новая гипотеза должна получить новый
ID и registry version, использовать `SimpleResearch` или другой компактный
независимый simulator и открывать новый holdout только после preregistration.
Terminal-rejected гипотезы нельзя подбирать повторно на уже раскрытом OOS.
