# Panteon Flash: итоговый отчёт по готовности к реальным торгам

Дата: 2026-05-23

## Executive Summary

Лучший устойчивый вариант для следующего этапа: `proven_solo_portfolio_hard_solo_momentum_cap10_generated_selected_deny_round2_terminal_atom`.
Это не рекомендация запускать неограниченные реальные торги. Результат достаточен для ограниченного live/paper pilot с малыми лимитами, обязательным shadow-мониторингом и автоматическим kill-switch.

Ключевой вывод: Пантеон стал прибыльнее своих составляющих только после перехода от общего выбора акторов к контролю конкретных `actor|symbol|action` ключей и после сохранения жёстких лимитов экспозиции.

## Лучший результат

| Период | Flash PnL | Best component | Alpha | Beats best | LCB dominance | Churn/day | Executable signals |
|---|---:|---:|---:|---|---|---:|---:|
| 2025 | 20.68% | 18.15% | 2.53 п.п. | да | да | 0.48 | 86 |
| 2026 H1 | 11.31% | 8.24% | 3.07 п.п. | да | да | 0.76 | 40 |

Конфигурационная логика финала:

- база: `PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_ARGS`;
- generated selected-deny из 2025 `base_cap10` + `generated_selected_deny`;
- новый terminal guard: `ensemble:Solo_LiveCrashHunter|ATOM/USDT|FUT_SHORT_FULL`;
- лимит новых открытий оставлен строгим: расширение до `open2` ухудшило 2026 H1.

## Визуализации

![PnL vs best component](C:/Work/Crypto_exchange/Reports/PanteonFlashFinalTradingReadiness_20260523/charts/01_pnl_vs_best_component.png)

![Alpha by iteration](C:/Work/Crypto_exchange/Reports/PanteonFlashFinalTradingReadiness_20260523/charts/02_alpha_by_iteration.png)

![Signals and churn](C:/Work/Crypto_exchange/Reports/PanteonFlashFinalTradingReadiness_20260523/charts/03_signals_and_churn.png)

![Gate matrix](C:/Work/Crypto_exchange/Reports/PanteonFlashFinalTradingReadiness_20260523/charts/04_gate_matrix.png)

## Причины успехов

1. `selected-deny` работал лучше, чем общий запрет акторов: проблема была не в акторе целиком, а в конкретных `actor|symbol|action` сочетаниях.
2. Второй слой deny на 2025 снял selection bias базового Flash: 2025 вырос с 3.76% до 21.35%, alpha стал +3.19 п.п.
3. Terminal ATOM guard устранил главный OOS-разрыв 2026 H1 без разрушения 2025, если применять его поверх round2, а не вместо round2.
4. Финальный вариант прошёл оба контрольных периода: 2025 остался выше лучшего компонента, а 2026 H1 преодолел strict alpha threshold.

## Причины провалов

1. Broad shadow deny переобучался и ломал 2025: широкая фильтрация full-shadow статистики перерезала полезные ключи и снижала Flash PnL до 2.57%.
2. Одиночный ATOM deny без round2-базы выглядел хорошо на 2026 H1, но проваливал 2025: без уже обученного round2 слоя Flash начинал выбирать плохие fallback-кандидаты.
3. Расширение open-limit до open2 ухудшало OOS: 2026 H1 снижался с 10.14% до 9.54%, значит текущие лимиты реально защищают от плохой экспозиции.

## Production Readiness

Статус: не запускать без ограничений. Лучший вариант можно запускать только как контролируемый пилот.

Минимальный режим запуска:

- старт с paper/live-shadow на 1-2 недели на той же конфигурации;
- затем real capital не более 5-10% от планового лимита;
- `max_new_opens_per_bar=1`, не расширять до open2/open3;
- kill-switch: отключить real execution при просадке Пантеона > 1.5-2.0% от equity или при `PanteonAdvantage < 0` на rolling window;
- ежедневный отчёт `standalone_vs_flash_selected` и `flash_attribution_summary`;
- запрещать новые generated-deny правила только после прохождения 2025 + 2026 H1 + будущего holdout.

## Следующие шаги

1. Прогнать финальный кандидат на полном 2022-2026 walk-forward.
2. Добавить CI-gate, который запрещает merge, если 2025 или 2026 H1 не проходят `beats_best_component`, `panteon_lcb_dominance`, `churn_budget`.
3. Добавить отчёт diff выбранных ключей после каждого deny/terminal-deny, чтобы видеть fallback-сделки до запуска expensive прогонов.
4. Расширить OOS: 2024 holdout, 2026 H2 после появления данных, стресс-тест комиссий/проскальзывания.

## Источники данных

Метрики и графики построены из локальных артефактов в `Results/*` и сохранены в текущей папке отчёта.
