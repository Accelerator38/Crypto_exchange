# Реестр гипотез Panteon

Машинный источник: `configs/research_archive/hypotheses_v1.json`.

## Итог

Активных кандидатов: **0**. Ни одна стратегия или модель не имеет paper/live
authority. Положительный средний результат без положительного LCB не считается
edge.

Статусы:

- `ARCHIVED_ARCHITECTURE`: код сохраняется для истории, runtime запрещён;
- `TERMINAL_REJECTED`: независимая или costed проверка провалена;
- `DATA_BLOCKED`: валидной выборки для заявленного теста нет;
- `CALIBRATION_ONLY`: данные пригодны только для execution calibration;
- `ENGINEERING_ONLY`: проверено подключение, но не прибыльность.

## Архитектура и модели

| ID | Статус | Вывод | Что можно переиспользовать |
|---|---|---|---|
| `ARCH-PANTHEON-ENSEMBLE-V1` | ARCHIVED | Flash/ensemble не доказали robust costed edge, а атрибуция стала слишком сложной | adapters, guards, evidence formats, diagnostics |
| `ARCH-GENETICS-V2` | ARCHIVED | Ни один checkpoint не прошёл validation/OOS/cost contract | WFA contracts, collapse checks, fitness research |
| `EXEC-FREQTRADE-PULSE-V1` | ENGINEERING_ONLY | Dry-run wiring работает, `dry_run=false` блокируется | будущий минимальный execution harness |
| `DATA-MICROSTRUCTURE-V3` | CALIBRATION_ONLY | 48 часов достаточно для spread/depth calibration, но не для alpha | единая модель costs/execution |

Все непромоутированные checkpoints удалены. Genetics не находится в live
whitelist и не имеет bypass.

## CarryFlow

| ID | Основной результат | Решение |
|---|---|---|
| `CF-BASE-ROOT003` | 5 сделок, net -4.33 bps, все exits по max_holding | Terminal reject |
| `CF-FIXED-EXIT-4H` | Independent mean -18.12 bps против baseline -1.17 bps | Terminal reject |
| `CF-LORO-FLOW` | 27 held-out trades, mean -0.36, LCB -30.37 bps | Terminal reject |
| `CF-LORO-HYBRID` | 22 trades, mean -1.42, LCB -39.66 bps | Terminal reject |
| `CF-RANGE-TRANSITION` | 8 trades, mean 11.74, LCB -15.82 bps, root collapse | Terminal reject |

Полезный вывод CarryFlow: ranking score нельзя интерпретировать как expected
move; понижение OI threshold не поддержано данными; exit tuning на пяти сделках
не переносится между roots.

## StrategyLab

| ID | Результат | Решение |
|---|---|---|
| `LAB-REGIME-PULLBACK` | OOS mean -19.94 bps, direction/regime collapse | Terminal reject |
| `LAB-COMPRESSION-TRANSITION` | OOS mean -9.81, LCB -26.37 bps | Terminal reject |
| `LAB-FUNDING-CARRY` | max projected carry 7.35 bps при cost floor 16 bps | Data-blocked/current definition rejected |
| `LAB-CROSS-SECTIONAL-TREND` | Positive dev mean, но отрицательный LCB и excessive DD | Terminal reject |
| `LAB-MARKET-NEUTRAL-PAIR` | 96 pairs, mean 13.04, LCB -74.06 bps | Terminal reject |
| `LAB-MARKET-NEUTRAL-PORTFOLIO` | 96 portfolios, mean 50.83, LCB -22.99 bps | Terminal reject |
| `LAB-COINTEGRATION-SPREAD` | mean 2.57, LCB -27.62, stress -3.43 bps | Terminal reject |

Positive development means у cross-sectional/neutral подходов являются
полезной идеей для совершенно нового исследования на новых данных, но не
основанием продолжать подбор текущих параметров.

## Uniform SimpleResearch batch

В одном simulator и cost contract были preregistered: flat, long-only, два EMA
trend, два Donchian, mean reversion, volatility compression, regime pullback и
candle momentum. Результат: **0 из 10 прошли**.

Наименее отрицательный OOS mean был у `donchian_20_v1` (-5.46 bps), но его LCB
(-17.12 bps) и drawdown (25.08%) также провалили gate. `vol_compression` имел
умеренный drawdown 9.79%, но mean -16.49 и LCB -30.42 bps. Эти варианты не
являются кандидатами для paper.

## Протокол новой гипотезы

1. Новый economic claim, а не ещё один threshold старой гипотезы.
2. Новый ID и registry version до просмотра holdout.
3. Не более трёх variants на family.
4. Один next-bar execution contract и один cost model.
5. Сначала development, затем validation; OOS открывается один раз.
6. Требуются positive costed mean, positive LCB, stress pass и DD в лимите.
7. Только после этого возможен отдельный strict paper этап.

Возвращение к полному ансамблю или Genetics допустимо только после того, как
минимальная одиночная стратегия независимо докажет edge. Ансамбль не должен
использоваться для создания edge из отрицательных компонентов.
