# Compact post-close research v1

Локальный исследовательский прогон от 08.10.2026, отдельно от замороженных результатов.
Разрешение пользователя относится к offline-обучению и сравнению с явными допущениями.
Все девять authority flags false: они запрещают downstream runtime, сеть и торговлю;
не отменяют прямо заказанные диагностические fits/replays. Никаких API-ключей.

## Исправления и измерения

- Label — ровно три дня между scheduled закрытиями 4h свечей. Прежний benchmark
  использовал следующий open спустя четыре часа; его результаты не считать paired comparison.
- Решение не требует известного будущего outcome. 21 origin с 1 августа по 30 сентября,
  20 полных трёхдневных labels; последний deployment ограничен одним днём до 1 октября UTC.
- Только labels с available <= origin - 8 bars: train_end=origin-1bar, purge=7bars.
- Полный причинный EMA-prefix, отдельный scaler на каждом origin. Inner-validation 60 дней
  также имеет отдельный purge. Seeds MLP 11/29/47 усредняются, лучший seed не выбирается.
- Линейная модель 7 параметров, MLP 65 на seed / 195 в ансамбле; uniform и half-life90
  с 20% uniform anchor, все исходы получают вес. L2=.01, Adam=.001, максимум120 epochs,
  early stopping только на причинном inner-validation, затем полный refit на mature samples.

## Исполнение — допущение, не квалифицированный IOC

Цена — последняя закрытая 4h свеча; предполагается доступность при закрытии, compute100ms,
fill350ms после boundary и неизменность этой reference-цены за350ms. Применяется adverse
price2bps (stress8bps) и комиссия6bps на каждый fill. Стакан, ticks, min-notional,
historical account fee, impact, liquidation и consumer-availability не подтверждены.

Капитал10000USDT, fully collateralized long/cash research, не futures-margin модель.
Восемь binary sleeves по10%, без плеча/заимствования; неизменный target сохраняет units.
Ledger непрерывен, включает fees, adverse drag, signed funding и terminal liquidation.
Funding использует reported historical rates и 4h-close вместо settlement mark.
Включены отдельно найденные terminal reports 1 октября, до terminal fill. Исходные
records `consumable_as_funding_cashflow=false` не изменяются: в новом прогоне они
используются исключительно как зарегистрированные модельные rates, не observed cashflows.
DD измерен по 4h-close и при моделируемых операциях, не intrabar executable equity.
Бюджеты10/20/30% — исследовательские post-gates, не согласованный live-риск и не stop-loss.

## Ограничение подбора и статистики

До любых fits фиксируются config, data/source/code pins, cutoff/folds rule, common circular
three-calendar-day block bootstrap banks4096draws; seeds3039(selection)/7079(audit).
Пороги20/50/100bps выбираются для каждого arm только на20 июня–25 июля, по stress net growth
при DD<=10%. Август–сентябрь уже раскрыт: это engineering research, не независимый outer.
LCB — диагностическая проверка устойчивости, а не единственная цель ranking.
GA/random search пропускаются без положительного stress-LCB простой модели на обеих banks.
Даже положительный gate не разрешает расширять search: нужен отдельный равнобюджетный контракт.
Immutable registry entry нового objective не заменяет отсутствующий глобальный registry;
DSR/PBO и multiplicity-adjusted inference не заявляются. Audit seed bank не независимые данные.
Сохранены model/scaler/feature-prefix/training/fold SHA каждого решения.

## Последовательные команды

Только `E:\Crypto_exchange\.venv\Scripts\python.exe -X utf8` и один foreground шаг:
`tools/run_genetic_primus_compact_postclose_research_v1.py prepare`, затем `checks`,
`cal-origin --index N` отдельно для0..11, `calibrate`, `origin --index N` отдельно для0..20,
`replay --policy POLICY --scenario normal|stress` отдельно для каждой пары,
`summarize`, `plot`, `verify`. Результаты exclusive-create, повтор не перезаписывает файлы.
Без pytest, collector, ожиданий wall-clock, фоновых процессов или сети.
