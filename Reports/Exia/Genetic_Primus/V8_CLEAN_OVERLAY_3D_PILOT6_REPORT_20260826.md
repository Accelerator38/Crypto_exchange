# Genetic Primus V8 Clean — overlay 3-day pilot6

## Статус

Pilot завершён и отклонён по заранее зафиксированному decision rule. Live/paper promotion не выполнялся.

- preregistration SHA256: `077d65493a1a4ddd5011265571ed055b4d5c2b25b890618ebfe6d24de88a6208`;
- source manifest SHA256: `45cf7870f7a0dd5ca4191f5ecfd90c5ecdbd2fec0b18f18f3900b9e0e1104ad9`;
- validation: `2025-01-01T00:00:00Z` — `2025-01-19T00:00:00Z` exclusive;
- 6 последовательных неперекрывающихся origins по 72 часа;
- 5 incumbent-comparable origins.

## Promotion evidence

| Metric | Result | Required |
|---|---:|---:|
| Minimum comparable audit paired LCB | 0.0 bps | > 0 |
| Minimum comparable stress-cost LCB | 0.0 bps | > 0 |
| Mean positive outer-slice fraction | 0% | >= 65% |
| Promotion pass fraction | 0% | > 0 |

Решение: `REJECT`.

Origin 0 имел selection paired-min LCB `-15.621 bps/day`, audit paired-min LCB `-15.739 bps/day` и allowed fraction 16.8%. На origins 1–5 selected challenger полностью abstained: paired LCB к NoTrade был ровно 0, поэтому строгий gate `>0` не выполнен.

## Deployment comparison

| Agent | Total net, bps | Positive origins | Trades | Worst DD |
|---|---:|---:|---:|---:|
| GeneticPrimusActive | 0.000 | 0/6 | 0 | 0.00% |
| GeneticPrimusChallenger | -81.657 | 0/6 | 12 | 1.18% |
| NoTrade | 0.000 | 0/6 | 0 | 0.00% |
| Momentum return_1 | +117.047 | 3/6 | 424 | 7.42% |
| MeanReversion return_1 | -1874.407 | 0/6 | 424 | 7.03% |

Raw momentum был положителен суммарно, но выиграл только 3 из 6 origins и имел высокий turnover. Overlay не смог сохранить его прибыльные slices: origin 0 потерял `-81.657 bps`, после чего search выбрал полный NoTrade на остальных origins.

## Safety и provenance

- root artifact hashes: verified;
- hypothesis-budget payload hashes: verified для всех шести origins;
- registered budget: 100 hypotheses/origin, превышений нет;
- selection bootstrap banks: 6 уникальных;
- audit bootstrap banks: 6 уникальных;
- selection/audit collisions: 0;
- active model SHA: один deterministic NoTrade;
- challenger model SHA: 6 уникальных;
- challenger scaler SHA: 6 уникальных;
- promotions: 0/6;
- credentials/paper/live/orders/promotion authority: false.

## Диагноз и следующий дизайн

Fail-closed механизм работает правильно, но текущий search objective имеет устойчивую точку `NoTrade`: нулевой paired LCB лучше отрицательных исследуемых genomes, поэтому после появления flat incumbent population теряет торговое разнообразие.

Не следует ослаблять promotion gate или менять coverage threshold по этому уже открытому pilot. Следующая версия должна быть новой preregistered гипотезой и не использовать этот период как final evidence.

Основная рекомендуемая правка: заменить directional supervised warm start на overlay-specific target:

- `ALLOW`, если исторический momentum signal после costs был полезен;
- `ABSTAIN`, если сигнал был убыточен или слаб;
- direction по-прежнему задаёт только momentum.

Дополнительно следует сохранять Pareto/coverage diversity во время evolution, например отдельные coverage niches, чтобы все islands не коллапсировали в нулевую торговлю. Promotion gate при этом остаётся неизменным и требует положительного paired audit/stress evidence.
