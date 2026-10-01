# Этап 5 — вход в независимую prospective-проверку

Статус на 2026-10-01: **NOT REGISTERABLE**. Ни одно новое outer-окно не зарегистрировано и не оценивалось. Это намеренная остановка по заранее заданному правилу, а не отрицательный результат на будущем рынке.

## Зафиксированный вход

- Selection result: `Reports/Exia/Genetic_Primus/selection_development_v1_20261001/result.json`, SHA-256 `b159550c28ade51ae9b272d6a803a7a85e1d6683588162cc9fc54ce966e35ec7`.
- Selection config: `configs/genetic_primus_selection_development_v1.json`, SHA-256 `771ce17d96cec35e2933b2cc9ecb4603efbb670fc9581e443221baae18bd6092`.
- Предъявлено 24 из 24 зарегистрированных кандидатов. `selected_genome=null`; `validation_result=null`; `final_policy=NoTrade`; cutoff `2026-08-30T00:00:00Z`.
- Причины отклонения даже лучшего по train genome: `NO_VALIDATED_CHALLENGER`, `GENOME_NOT_SELECTED`, `VALIDATION_MISSING_OR_MISMATCHED`.

## Исправление допуска

Offline V2 runner теперь требует не только совпадения SHA, genome, scaler, окна и cutoff, но также положительного train score, соблюдения бюджета поиска, полного прохождения заранее определённых validation checks, точного итогового статуса выбранной модели и выключенных прав сети/исполнения/автопромоушена. Отчёт с `selected_genome`, но проваленным validation gate, больше нельзя зарегистрировать как prospective challenger. Проверка не даёт торговых полномочий.

## Условия возобновления этапа

1. Отдельная новая гипотеза и её ограниченный development-бюджет регистрируются **до** следующей оценки; ни один уже раскрытый августовский/сентябрьский origin не переименовывается в prospective.
2. Только если новый отбор имеет положительный train score и проходит неизменённый inner-validation gate, его точные байты, genome, scaler, cutoff и V2 future contract фиксируются до поступления первого outer-бара.
3. В контракте заранее указываются 3-дневные origins, порядок реобучения/рефита scaler на каждом origin, единый будущий интервал и момент **одного** финального анализа. До него не меняются пороги, модель или число попыток; при неуспехе — NoTrade.
4. Отдельная проверка исполнения и допуск к деньгам относятся к более позднему этапу. Этот этап не запускал сеть, collector, supervisor или торговый runtime.

Локальная проверка: 4 deterministic unit tests, без рыночных данных и pytest. Достоверность экономической эффективности не изменилась: будущих наблюдений нет.
