# Genetic Primus V5 — sealed outer60

Дата завершения: 2026-08-27. Статус: `TERMINAL_REJECTED_ON_SEALED_OUTER60`.

## Контракт и период

- Frozen contract SHA: `41a90b03f7c166d07331dc51a45972173f9a0c03cbb2b1888c9ab62a80932d70`.
- Objective registry SHA: `766aed82b8c26b76ebe9a565a53fe565bcceda4a32f7360b76ff77d847de1273`.
- Период: `2025-02-01T00:00:00Z` — `2025-07-31T00:00:00Z`.
- 60 последовательных 3-дневных origins, retrain каждые 3 дня.
- 8 160 записанных deployment decisions.
- Сеть, collector, paper/live и orders не использовались.

До первого завершённого origin были исправлены два engineering-дефекта нового кода: потерянный return в `_dynamic_features` и неверное имя ledger-поля `timestamp` вместо `entry_timestamp`. Fitness, thresholds, seeds и временные границы после раскрытия outcomes не изменялись.

## Promotion outcome

- `NO_ELIGIBLE_CHALLENGER`: 60/60 origins.
- Deployable challengers: 0/60.
- Predeployment gates passed: 0/60.
- Active deployment trades: 0; активным оставался `NoTrade`.
- Итоговый promotion gate: `false`.

Основные predeployment blockers:

- temporal audit LCB: 60/60;
- no eligible validation finalist: 60/60;
- temporal validation LCB: 58/60;
- feature stability: 43/60;
- action diversity: 2/60.

Median selection paired LCB: `-20.369 bps`; лучший `+9.786 bps`. Median audit paired LCB: `-29.492 bps`; лучший audit LCB всё равно отрицателен: `-3.018 bps`.

## Диагностический challenger

Хотя challenger не допускался к deployment, runner сохранил отдельный diagnostic outcome:

- compound: `-1383.929 bps` (`-13.84%`);
- mean per origin: `-24.244 bps`;
- positive origins: 17/60 (`28.3%`);
- trades: 520;
- aggregate max drawdown: `15.56%`;
- paired LCB против NoTrade: `-39.479 / -39.750 bps` на selection/audit seeds;
- stress compound: `-1821.016 bps`;
- stress aggregate drawdown: `19.61%`.

Абсолютный drawdown budget 10% был бы нарушен даже при прохождении ранних gates.

## Сравнение с обычными агентами

На тех же 60 origins:

| Агент | Compound, bps | Positive origins | Trades | Aggregate DD |
|---|---:|---:|---:|---:|
| NoTrade | 0.000 | 0/60 | 0 | 0.00% |
| V5 diagnostic challenger | -1383.929 | 17/60 | 520 | 15.56% |
| Momentum return_1 | -4303.739 | 20/60 | 4364 | 56.22% |
| Mean-reversion return_1 | -7428.407 | 20/60 | 4364 | 75.08% |

V5 заметно уменьшил потери и turnover относительно momentum, но не создал положительную доходность. Самая сильная стратегия на sealed interval — `NoTrade`.

## Сравнение с прежними genetic pilots

На ранее раскрытых шести январских origins compound составлял:

- V1: `-81.657 bps`;
- V3: `+206.626 bps`;
- V4: `+371.102 bps`;
- Momentum: `+91.629 bps`.

Положительные V3/V4 pilot6 не перенеслись на длинный новый период. Это подтверждает selection/temporal overfitting малой январской выборки и оправдывает запрет возвращаться к ней для настройки.

## DSR/PBO

На 60 одинаковых origins для заранее зарегистрированных V5, NoTrade и Momentum:

- V5 Sharpe: `-0.229257`;
- Momentum Sharpe: `-0.169449`;
- NoTrade Sharpe: `0`;
- Deflated Sharpe Probability при global trial count 20: `0.000429` против gate `0.95`;
- CSCV PBO: `0.150794` на 252 combinations против gate `0.50`.

PBO ниже лимита означает лишь отсутствие сильного признака перестановочного выбора среди этих трёх рядов. Он не превращает отрицательную стратегию в пригодную: DSR, paired LCB, drawdown и absolute return провалены.

## Дополнительные наблюдения

- Stability selection прошла только в 17/60 retrain cycles.
- Functional action drift превысил 0.20 в 59/60 cycles; median `0.474`, max `0.680`.
- Functional logit drift превысил 0.15 в 19/60 cycles; median `0.112`, max `0.254`.
- Audit allowed fraction имела median `11.4%`, то есть модель не была тривиальным постоянным abstention.
- Audit opportunities: median 2360 на origin; недостаток рыночных возможностей не объясняет провал.
- Recency ESS на 60-дневном audit окне равен примерно 59.3, поэтому контрактный minimum ESS 120 на этом уровне недостижим. Это измерительный дефект контракта, но не причина отрицательного paired LCB: лучший audit LCB всё равно `-3.018 bps`.

## Решение

Не ослаблять gates и не донастраивать V5 на раскрытом периоде. Текущий linear momentum overlay закрыть как rejected objective. Shadow, paper и canary не запускать.

Следующая версия должна быть новым зарегистрированным objective, а не V5 с изменёнными порогами. Основной вопрос — замена слабого momentum baseline или переход к opportunity detector, который сначала доказывает положительную conditional expectancy против NoTrade. Для новой проверки нужен новый, ещё не раскрытый interval минимум на 60 origins.

## Artifact hashes

- Run manifest: `99af77bed89b3cafbc1f986862e4b9afe071890ced7509f0c48e74a401d80cc9`.
- Benchmarks: `9a3185a61d306f4754babdcc78737f7a008ab7df866be5cb45c450893f741b9a`.
- Global DSR/PBO evidence: `58495ef62a59e3f0df9acc661d5f987cabcd52ee673a47425575fa94fb367b49`.
