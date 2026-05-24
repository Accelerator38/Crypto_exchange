# Panteon Flash pre-live отчет

Дата: 2026-05-24.

## Резюме

Лучший кандидат для ограниченного pilot/live-shadow: `Round3 Deny8 EntryRegime`.
Полный 2022-2026 PnL: **105.42%**, max DD: **4.19%**, alpha к лучшему компоненту: **43.08 п.п.**
Лучший компонент: `Antonius_conservative` (62.34%).

Запуск реальных торгов допустим только как ограниченный пилот: малый капитал, `max_new_opens_per_bar=1`, ежедневный attribution-контроль и kill-switch. Полный production без ограничений не рекомендован.

## Финальный тест symbol guard

Новый symbol guard был проверен на полном 2022-2026 прогоне: PnL **100.50%** против **105.42%** у Round3.
Итог: symbol guard ухудшил результат на 4.92 п.п. и не включается в live-профиль. Код оставлен отключенным по умолчанию как инструмент будущих экспериментов.
Деградированные символы появлялись на 8087 барах; чаще всего: [('LTC/USDT', 1440), ('LINK/USDT', 1440), ('APT/USDT', 1440), ('BNB/USDT', 1440), ('NEAR/USDT', 1440)].

## Standalone vs Flash

Для Solo_MomentumScalper и LiveOIBreakout standalone-суммарно дал 68.20%, а Flash-selected subset дал 38.94%.
Это значит, что Пантеон прибыльнее полного набора компонентов за счет других игроков/фильтров, но selection subset еще требует улучшения.

## Визуализации

![01_full_pnl_vs_component](C:/Work/Crypto_exchange/Reports/PanteonFlashPreLive_20260524/charts/01_full_pnl_vs_component.png)

![02_full_quality_gates](C:/Work/Crypto_exchange/Reports/PanteonFlashPreLive_20260524/charts/02_full_quality_gates.png)

![03_entry_regime_pnl_usd](C:/Work/Crypto_exchange/Reports/PanteonFlashPreLive_20260524/charts/03_entry_regime_pnl_usd.png)

![04_symbol_pnl_usd](C:/Work/Crypto_exchange/Reports/PanteonFlashPreLive_20260524/charts/04_symbol_pnl_usd.png)

![05_actor_contribution_usd](C:/Work/Crypto_exchange/Reports/PanteonFlashPreLive_20260524/charts/05_actor_contribution_usd.png)

![06_iteration_alpha](C:/Work/Crypto_exchange/Reports/PanteonFlashPreLive_20260524/charts/06_iteration_alpha.png)

![07_signal_funnel](C:/Work/Crypto_exchange/Reports/PanteonFlashPreLive_20260524/charts/07_signal_funnel.png)
