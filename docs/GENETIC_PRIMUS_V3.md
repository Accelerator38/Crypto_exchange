# Genetic_Primus V3

## Цель

V2 устранил все regime/direction/symbol LCB failures, но median monthly robust
fitness осталась отрицательной. V3 меняет только ranking; dataset, 4h features,
network, costs и execution остаются идентичными.

## Temporal-first ranking

Фиксированный порядок сравнения:

1. Structural/risk gates.
2. Overall positive mean/LCB gates.
3. Temporal gate: одновременно `median monthly fitness > 0` и `overall median > 0`.
4. Число поддержанных slices с `LCB <= 0`.
5. LCB худшего поддержанного slice.
6. Median monthly fitness и overall LCB как tie-breakers.

Пока temporal gate не пройден, оптимизируется `min(monthly fitness, overall
median)`. После прохождения любой прошедший temporal-кандидат строго сильнее
любого непрошедшего, а приоритет возвращается к cross-sectional slices.

## Search budget

До запуска фиксируются `64` особи и `100` поколений. Лимит увеличен относительно
V2, потому что V2 впервые прошёл slice gates на поколении 28 и нашёл более
сбалансированного чемпиона на поколении 45. После запуска лимит не продлевается
до успеха. Validation остаётся нераскрытой и компенсирует повышенный риск
development-overfit от большего search budget.

## Запуск

```powershell
.venv\Scripts\python.exe tools\run_genetic_primus_v3.py --smoke
```

Полный фиксированный development:

```powershell
.venv\Scripts\python.exe tools\run_genetic_primus_v3.py
```

Runner не имеет права открывать validation, OOS, paper или live.

## Результат 100 поколений

Полный запуск `development_v3_100g_20260821` выбрал поколение 92. Кандидат
прошёл строгую development freeze:

| Метрика | V2, 50 gen | V3, 100 gen |
|---|---:|---:|
| Closed trades | 3319 | 3480 |
| Mean net, bps | +55.4290 | +62.8603 |
| Median net, bps | +7.7002 | +39.0656 |
| Overall LCB, bps | +38.7571 | +47.0452 |
| Temporal margin | -6.1406 | +16.0522 |
| Worst slice LCB, bps | +9.5006 | +17.6730 |
| Failed slices | 0 | 0 |
| Max drawdown | 16.5600% | 16.4140% |

Лучшие улучшения появились на поколениях 68, 73 и 92. Поэтому именно для V3
лимит 100 оказался полезнее 50. Увеличение выше 100 не обосновано.

## One-time validation

Frozen champion SHA `b37f4e455a1a5e3b4b8858dbabcd00f5bc78acdb134cfea18ec375fbe0335076`
однократно проверен на validation 2024 и отклонён:

| Метрика | Validation |
|---|---:|
| Closed trades | 1887 |
| Mean net, bps | -13.4577 |
| Median net, bps | +3.3888 |
| Overall LCB, bps | -38.3076 |
| Temporal fitness | -47.7941 |
| Worst slice LCB, bps | -147.9619 |
| Failed slices | 15/15 |
| Max drawdown | 49.9860% |

Статус: `VALIDATION_V3_REJECTED`. OOS закрыт. Нельзя проверять на раскрытой
validation другие поколения, менять ranking или продолжать evolution: это
превратит validation в development. Для следующего независимого исследования
нужна другая постановка и новый sealed validation период, а не больший budget.

## Terminal status

V3 окончательно закрыт со статусом
`TERMINAL_REJECTED_VALIDATION_GENERALIZATION`. Его champion, поколения и
validation 2024 нельзя переобучать, перебирать или использовать для открытия
OOS. Раскрытый 2024 может входить только в новый research-контракт V4; новым
однократным validation для V4 становится ранее не открытый 2025 год.
