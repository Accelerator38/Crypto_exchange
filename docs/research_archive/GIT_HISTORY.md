# Карта Git-истории Panteon

## Состояние refs перед публикацией архива

| Ref | Commit | Назначение |
|---|---|---|
| `origin/main` | `91d1052` | Последний опубликованный strategy/data lab до freeze |
| `main` | `91d1052` | Локальная tracking-ветка без divergence |
| `codex/panteon-research-archive-20260803` | архивный HEAD | Freeze, SimpleResearch, dry-run harness и документация |
| `panteon-research-freeze-2026-08-03` | `c0fd836` | Snapshot состояния до reset |

Remote до публикации содержал только `main`. Stale remote-tracking refs и
merged topic branches отсутствовали. Поэтому удаление веток не выполнялось:
оно не освободило бы данные, но ухудшило бы доступность истории. Рабочая
`codex/panteon-reset-20260803` была переименована в архивную, а не продублирована.

## Основные этапы

| Commit | Этап | Результат |
|---|---|---|
| `776624c` | Singleton и capability guard | Начало ограничения полного ансамбля |
| `7df69ff` | Panteon 3 live gate | Формализация pre-live blockers |
| `def55d8` | GeneticCore WFA | Guarded genetics pipeline без доказанного edge |
| `4f1c512` | Genetics readiness gates | Validation/OOS и collapse стали обязательными |
| `65891bb` | Bitget policy-only runtime | Переход от полного ансамбля к фиксированной policy |
| `640c66a` | CarryFlow evidence parity | Sealed tape и execution parity вместо старых canary artifacts |
| `b65a4f8` | CarryFlow terminal rejection | Отрицательный эксперимент закрыт |
| `91d1052` | Strategy development lab | Новые directional/neutral hypotheses и data layer |
| `c0fd836` | Research freeze | Dirty research state сохранён отдельным commit/tag |
| `9f87000` | Reset implementation | Live freeze, SimpleResearch и Freqtrade dry-run harness |

Финальный archive commit идёт после `9f87000` и добавляет очистку, dataset
manifest, hypothesis registry и archive checker.

## Что показала история

1. Большая часть изменений улучшала безопасность, fidelity и диагностику, но не
   создавала торговое преимущество.
2. Retro, shadow, paper и live долго имели разные пути и настройки. Исправление
   parity порождало следующий слой gates и artifacts.
3. Ensemble был последовательно сужен до singleton/policy-only, что подтвердило:
   сложность selector не была корнем отсутствия edge.
4. Genetics, CarryFlow и девять strategy families не прошли costed evidence.
5. Gates сработали правильно, не допустив отрицательные стратегии к деньгам.
   Ошибкой было ожидание, что новый gate или bypass улучшит expectancy.
6. Последний reset отделил исследование стратегии от execution plumbing и
   позволил проверить десять preregistered trials примерно за 39 секунд.

## Политика сохранения

- История не squash/rebase и не переписывается.
- `main` не получает архивный код автоматически.
- Архив публикуется отдельной branch и annotated tag.
- Старые runtime-модули остаются как source history, но закрыты freeze guard.
- Новые исследования не продолжаются в этой ветке: для них создаётся новая
  ветка от минимального research/execution baseline.
