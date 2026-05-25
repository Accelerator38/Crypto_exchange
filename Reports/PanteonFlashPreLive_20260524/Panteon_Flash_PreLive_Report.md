# Panteon Flash pre-live отчет

Дата: 2026-05-24.

## Резюме

Лучший кандидат для ограниченного pilot/live-shadow: `Round3 Deny8 EntryRegime`.
Полный 2022-2026 PnL: **105.42%**, max DD: **4.19%**, alpha к лучшему компоненту: **43.08 п.п.**
Лучший компонент: `Antonius_conservative` (62.34%).

Запуск реальных торгов допустим только как ограниченный пилот: малый капитал, `max_new_opens_per_bar=1`, ежедневный attribution-контроль и kill-switch. Полный production без ограничений не рекомендован.

## Как читать отчет: что здесь является самим Пантеоном

В этом отчете самим реально торгующим Пантеоном является агрегат `Round3 Deny8 EntryRegime` / `Panteon_Flash`. Это не отдельный агент и не отдельный игрок, а контур выбора и исполнения: `FlashAllocator` выбирает лучшего актора по каждому символу, затем execution-слой пропускает выбранные сигналы через лимиты позиций и отправляет их в executor.

Именно этот контур дал **105.42%** PnL: 1067 выбранных сигналов -> 836 executable -> 835 filled -> 496 closed trades, realized PnL **1054.19 USD**.

| Строка в отчете | Что это | Это сам Пантеон? |
| --- | --- | --- |
| `Round3 Deny8 EntryRegime` / `Panteon_Flash` | боевой Flash-контур выбора и исполнения | да, это агрегат реально исполненных сделок Пантеона |
| `Antonius_conservative` | лучший отдельный компонент-бенчмарк | нет, это игрок-кандидат; его сделки внутри Flash учитываются как вклад выбранного актора |
| `Solo_MomentumScalper`, `LiveOIBreakout`, другие actor rows | агенты или ensemble-игроки, выбранные на отдельных символах | нет как самостоятельная система; да как вклад в сделки, которые исполнил `Panteon_Flash` |
| Standalone vs Flash-selected | сравнение полного самостоятельного поведения актора с subset, выбранным Пантеоном | standalone - нет; Flash-selected - часть реальных сделок Пантеона |
| `NoTrade` | явное решение не торговать по символу | часть decision layer, но не сделка |

## Как работают агенты, игроки и Пантеон

Агент - это источник торговой логики с интерфейсом `act(market) -> dict[symbol, Action]`. Он предлагает действие по монете, но сам не решает, можно ли это действие реально исполнять.

Игрок - это ансамблевый актор (`actor_type="ensemble"`). `EnsemblePlayer` вызывает несколько агентов, агрегирует их голоса через voting policy, назначает основного contributor и возвращает сигналы. В Flash игрок конкурирует с одиночными агентами как обычный кандидат, а не выбирается глобальным лидером на весь рынок.

Пантеон в режиме Flash - это decision/execution слой. На каждом баре он собирает real-executable агентов и ensemble-игроков, строит кандидатов, оценивает их по каждому символу, выбирает одного победителя или `NoTrade`, затем передает только прошедшие сигналы в execution.

Схема решения:

```text
MarketSnapshot -> agents + ensemble players -> FlashAllocator
    -> one FlashDecision per symbol -> risk/position filters
    -> TradeExecutor -> EventLog + attribution
```

Фильтры перед исполнением: карантин, наличие статистики, минимум закрытых сделок, PnL-порог, shadow confirmation, degradation/deny keys, score gate, лимит новых открытий на бар и лимит открытых позиций.

## Профиль реально торгующего контура

| Параметр | Значение | Смысл |
| --- | --- | --- |
| Leader | Panteon_Flash | имя, под которым сделки проходят как Пантеон |
| Run profile | Round3 Deny8 EntryRegime | выбранная pre-live конфигурация |
| Flash enabled | да | используется per-symbol Flash selection |
| Min score | 4.00 | минимальная оценка кандидата |
| Min closed trades | 3 | минимальная история для допуска |
| Shadow confirmation | да | проверка через shadow-статистику |
| Symbol shadow confirmation | да | подтверждение качества по символу |
| Max signals per actor | 3 | защита от концентрации в одном акторе |
| Max new opens per bar | 1 | боевой лимит новых открытий |
| Risk max open positions | 8 | общий лимит открытых позиций |
| Degradation guard | да | временное отключение деградировавших signal keys |
| Actor degradation guard | да, scope=actor_regime | контроль деградации актора по режиму |
| Deny keys | 55 | ручные запреты плохих actor/symbol/action |
| Terminal deny keys | 1 | жесткие запреты, ведущие к NoTrade |

## Работа Пантеона на полном прогоне

Flash обработал 745734 решений по символам. Из них 744667 закончились `NoTrade`, поэтому активная торговля была редкой и отфильтрованной: no-trade share **99.86%**.

| Показатель | Значение |
| --- | --- |
| Selected signals | 1067 |
| Executable selected | 836 |
| Filtered before execution | 231 |
| Filled signals | 835 |
| Blocked signals | 1 |
| Closed trades | 496 |
| Winning / losing trades | 212 / 284 |
| Win rate | 42.74% |
| Realized PnL | 1054.19 USD |

## Вклад агентов и игроков внутри Пантеона

Следующая таблица не показывает отдельные самостоятельные торговые системы. Она показывает, какие акторы были выбраны самим `Panteon_Flash` и какой вклад дали в его реальные/ретро-реальные сделки.

| Actor | Type | Selected | Executable | Closed | PnL USD |
| --- | --- | --- | --- | --- | --- |
| ensemble:Antonius_conservative | ensemble | 185 | 117 | 102 | 762.41 |
| ensemble:Solo_MomentumScalper | ensemble | 521 | 450 | 260 | 276.56 |
| agent:LiveOIBreakout | agent | 120 | 68 | 41 | 112.87 |
| ensemble:Solo_LiveCrashHunter | ensemble | 34 | 23 | 19 | 67.56 |
| ensemble:Solo_FundingArb | ensemble | 20 | 12 | 10 | 32.28 |
| agent:LiveMeanRev | agent | 11 | 10 | 1 | 11.31 |
| agent:LiveRegimePullback | agent | 46 | 43 | 14 | 3.41 |
| agent:BullRotationAgent | agent | 7 | 7 | 0 | 0.00 |

## Финальный тест symbol guard

Новый symbol guard был проверен на полном 2022-2026 прогоне: PnL **100.50%** против **105.42%** у Round3.
Итог: symbol guard ухудшил результат на 4.92 п.п. и не включается в live-профиль. Код оставлен отключенным по умолчанию как инструмент будущих экспериментов.
Деградированные символы появлялись на 8087 барах; чаще всего: [('LTC/USDT', 1440), ('LINK/USDT', 1440), ('APT/USDT', 1440), ('BNB/USDT', 1440), ('NEAR/USDT', 1440)].

## Standalone vs Flash

Для Solo_MomentumScalper и LiveOIBreakout standalone-суммарно дал 68.20%, а Flash-selected subset дал 38.94%.
Это значит, что Пантеон прибыльнее полного набора компонентов за счет других игроков/фильтров, но selection subset еще требует улучшения.

| Actor | Standalone | Flash-selected | Selected | Executable | Closed | Selection alpha |
| --- | --- | --- | --- | --- | --- | --- |
| Solo_MomentumScalper | 58.92% | 27.66% | 521 | 450 | 260 | -31.26 п.п. |
| LiveOIBreakout | 9.28% | 11.29% | 120 | 68 | 41 | 2.00 п.п. |

## Визуализации

![01_full_pnl_vs_component](C:/Work/Crypto_exchange/Reports/PanteonFlashPreLive_20260524/charts/01_full_pnl_vs_component.png)

![02_full_quality_gates](C:/Work/Crypto_exchange/Reports/PanteonFlashPreLive_20260524/charts/02_full_quality_gates.png)

![03_entry_regime_pnl_usd](C:/Work/Crypto_exchange/Reports/PanteonFlashPreLive_20260524/charts/03_entry_regime_pnl_usd.png)

![04_symbol_pnl_usd](C:/Work/Crypto_exchange/Reports/PanteonFlashPreLive_20260524/charts/04_symbol_pnl_usd.png)

![05_actor_contribution_usd](C:/Work/Crypto_exchange/Reports/PanteonFlashPreLive_20260524/charts/05_actor_contribution_usd.png)

![06_iteration_alpha](C:/Work/Crypto_exchange/Reports/PanteonFlashPreLive_20260524/charts/06_iteration_alpha.png)

![07_signal_funnel](C:/Work/Crypto_exchange/Reports/PanteonFlashPreLive_20260524/charts/07_signal_funnel.png)
