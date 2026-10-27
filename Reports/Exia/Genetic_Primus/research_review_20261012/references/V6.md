# Genetic Primus V6 — итог sealed outer60

Дата завершения: 2026-08-27.

## Итог

V6 objective `overlay_3d_v6_relative_opportunity_sealed` окончательно отклонён. Полный зарегистрированный interval `[2026-01-01, 2026-07-01)` просмотрен один раз: 60 origins с cadence 3 дня. Ни один challenger не прошёл predeployment gates, поэтому active strategy всё время оставалась NoTrade и не совершила сделок.

Повторять прогон, менять thresholds/seeds/features по результатам этого интервала или использовать его как новый sealed evidence запрещено.

## Целостность эксперимента

- 60/60 origins завершены;
- 8160 decision rows;
- cadence: 3 дня;
- deployment horizon: 72 часа;
- purge: 7 баров × 4 часа = 28 часов;
- все decisions содержат model SHA, совпадающий с origin ledger;
- все 10 root artifact hashes совпали;
- config SHA совпал с immutable registry;
- objective registry: 21 зарегистрированная гипотеза;
- credentials, paper/live, orders и promotion authority выключены.

Поле `OFFLINE_ROLLING_3D_ENGINEERING` в manifest является общим legacy status runner. Sealed-статус этого запуска задают opening receipt, зарегистрированный config SHA, фиксированный интервал и completion receipt.

## Доходность на sealed interval

| Политика | Compound, bps | Среднее на origin, bps | Положительных origins | Trades | Max drawdown |
|---|---:|---:|---:|---:|---:|
| GeneticPrimusActive | 0.00 | 0.00 | 0/60 | 0 | 0.00% |
| NoTrade | 0.00 | 0.00 | 0/60 | 0 | 0.00% |
| V6 diagnostic challenger | −416.10 | −7.04 | 4/60 | 38 | 5.26% |
| V6 challenger, stress cost | −452.46 | −7.67 | 4/60 | 38 | 5.53% |
| RelativeMomentumRank_return_12 | −2733.41 | −52.72 | 13/60 | 988 | 29.67% |
| Momentum_return_1 | −3613.60 | −64.74 | 24/60 | 4413 | 49.45% |
| MeanReversion_return_1 | −7590.82 | −225.86 | 18/60 | 4413 | 80.88% |

V6 abstained полностью в 42/60 origins. В оставшихся 18 origins было 4 положительных и 14 отрицательных результата.

## Paired outer diagnostics

Официальный promotion gate не использует diagnostic challengers и поэтому при нуле eligible origins выдаёт fail-closed sentinel. Для анализа сохранённых результатов та же зарегистрированная block-bootstrap процедура была применена отдельно:

- против NoTrade: mean delta −7.04 bps, selection LCB −12.48 bps, independent audit LCB −12.77 bps, positive fraction 6.7%;
- stress против NoTrade: mean delta −7.67 bps, audit LCB −13.51 bps;
- против RelativeMomentumRank: mean delta +45.68 bps, audit LCB +32.45 bps, positive fraction 73.3%;
- против Momentum_return_1: mean delta +57.70 bps, но audit LCB −70.27 bps.

Вывод: genetic overlay существенно фильтрует плохой relative-momentum proposal, но после фильтрации всё равно проигрывает NoTrade. Это провал базовой opportunity-гипотезы, а не только promotion threshold.

## Predeployment gates

Все 60 challengers получили `NO_ELIGIBLE_CHALLENGER`.

- median selection LCB: −4.92 bps; максимум +0.22; положительный только 1/60;
- median independent audit LCB: −4.67 bps; положительных 0/60;
- temporal validation LCB failed: 59/60;
- temporal audit LCB failed: 60/60;
- feature stability failed: 60/60;
- validation-eligible finalists: 0/60;
- action diversity failed: 29/60.

У audit champions paired LCB против NoTrade неположителен в 59/60 origins. Feature stability count равен нулю во всех origins, хотя выбранные модели имели от 0 до 4 активных features. Functional action drift превысил лимит 0.20 в 43/60 origins; functional logit drift превысил 0.15 в 4/60.

## DSR/PBO

Hash-locked comparison set содержит V6 challenger и три frozen reference policies: NoTrade, RelativeMomentumRank и active incumbent. Использовано ровно 60 origin-level returns, CSCV 10 slices, 252 комбинации.

- observed Sharpe: −0.2463;
- expected maximum Sharpe с учётом 21 trial: 0.5684;
- deflated Sharpe probability: 0.0 при требовании ≥0.95 — gate failed;
- PBO: 0.0397 при лимите ≤0.5 — gate passed.

Низкий PBO не спасает стратегию: он не означает прибыльность, а лишь не показывает типичный rank reversal внутри данного comparison set. Отрицательный Sharpe и нулевой DSR являются достаточным основанием для отказа.

## Сравнение с V5

V5 на своём sealed interval 2025 года потерял −1383.93 bps, совершил 520 diagnostic trades и имел drawdown 15.56%. V6 потерял −416.10 bps, совершил 38 trades и имел drawdown 5.26%. V6 заметно лучше ограничивает активность и риск благодаря abstention и sparse overlay.

Это не прямое head-to-head сравнение: V5 и V6 оценены на разных календарных интервалах и с разными proposal hypotheses. Корректный общий вывод только один: обе версии проиграли NoTrade и обе не имеют права на paper/live promotion.

## Решение и дальнейшее направление

V6 не переводить в paper, shadow или live. Не калибровать её по интервалу 2026-01-01—2026-07-01.

Сохранить удачную архитектурную часть — linear sparse binary overlay и сильное abstention — но отказаться от `RelativeMomentumRank_return_12` как самостоятельного opportunity generator. Следующий objective должен быть материально новой, заранее зарегистрированной гипотезой. Разрабатывать его можно только на уже раскрытых данных; для следующего честного sealed evaluation потребуется новый, ещё не просмотренный интервал после 2026-07-01 и достаточное число 3-дневных origins.
