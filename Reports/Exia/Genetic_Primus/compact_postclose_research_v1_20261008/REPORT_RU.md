# Обучение и сравнение compact post-close — 8 октября 2026

Лучший результат данного раскрытого ретропериода — **linear6 uniform**:
капитал10000→11784,04USDT, +17,8404%; при stress-costs10000→11724,00, +17,2400%.
Однако stress-DD12,4649% превышает исследовательский бюджет10%, audit-LCB отрицательный,
и положительный incremental эффект против EMA/прежней модели статистически не подтверждён.
Это НЕ допуск к деньгам. Default остаётся NoTrade.

## Что реально выполнено

- 12 pre-August origins:20 июня–23 июля, ledger до26 июляUTC. Только здесь выбраны
  пороги входа из20/50/100bps. Calibrator — stress net capital growth при sampled DD<=10%.
- 21 validation/research origin:1 августа–30 сентября2026; terminal liquidation1 октябряUTC
  плюс модельные350ms. 61 календарный день,20 полных3-дневных deployment и последний1-дневный.
  Прогнозная ошибка оценивается лишь на20 полных labels /160asset-outcomes.
- Причинное полное переобучение каждые3дня, отдельные outer/inner scaler на каждом origin,
  только available labels<=origin−8*4h: train_end=origin−1bar плюс purge7bars.
- 462 fits:168доавгустовских и294за август–сентябрь. Внутри:66линейных,
  198MLP early-stopping fits и198MLP full refits. 66scaler fits. Seeds11/29/47 усреднены.
- 19 прямых бухгалтерских/clock/bank проверок +963 причинные регрессии =982 checks.
  Проверены все33 origins, hashes training/folds/prefix, causal scaler means и прогнозы.
  Изменение будущих цен не изменяет признаки прошлого. Pytest/collector не запускались.
- 12 успешно сохранённых pre-August threshold replays,12 основных scenario replays,
  8 дополнительных frozen-model controls. Из-за ошибки JSON-export первые12 calibration
  replays пришлось детерминированно повторить: итого44 executions,32 успешных distinct
  scenario configurations. Это не44независимых гипотезы. Ни один fit не повторялся.
- Сеть, API-ключи, торговый runtime, реальные ордера, paper/live и фоновые процессы не использованы.
  Старые данные, модели, source-файлы и receipts не изменены.

## Зарегистрированные допущения

Цена reference — последняя закрытая4h свеча, предположительно доступная в момент boundary;
signal=+100ms, fill=+350ms, цена за эти350ms предполагается неизменной. Это модель, не реальный
IOC. Прежний forecast/replay использовал next4h open и другой label-clock.
Новый label: scheduled close→close ровно3дня. Никакого требования будущего deployment-label
для текущего решения. Последний неполный deployment закрывается отдельно от полного label.

Капитал10000USDT; восемь binary long/cash sleeves10%, без плеча/заимствования. Если target
не изменился, units сохраняются и веса дрейфуют. Комиссия6bps/fill, adverse drag2bps normal
и8bps stress. Quantity дробная: historical tick/lot/min-notional не квалифицированы.
Funding — signed reported rates ×4h-close valuation, включая8terminal reports1октября перед
продажами. Это НЕ observed settlement cashflows/marks. Их frozen status=false не изменён.
DD — по4h-close и моделируемым операциям; intrabar drawdown может быть хуже. Это полностью
обеспеченный long/cash исследовательский ledger, не Bitget futures margin/liquidation model.

## Рейтинг новых моделей по stress net growth

1. **Linear uniform**,7параметров, threshold20bps. Normal+17,8404%, stress+17,2400%;
   DD normal12,2693% /stress12,4649%;84fills. Normal fees57,04USDT,
   modeled funding−62,73USDT. Средняя exposure45,22%, turnover9,507 initial capitals.
   Stress август+16,5691%, сентябрь+0,5756%.
2. **EMA48 long-only**, необучаемый причинный comparator на тех же3-дневных origins.
   Normal+11,7463%, stress+11,3229%; DD11,2328% /11,3619%;62fills.
   Средняя exposure50,58%, turnover6,785. Stress август+10,2155%, сентябрь+1,0048%.
3. **Linear HL90 +20%uniform anchor**,7параметров, threshold100bps.
   Normal+1,2574%, stress+1,1833%; DD1,0983% /1,1216%;12fills.
   Средняя exposure4,17%. В августе не торговал; stress-сентябрь+1,1833%.
   Малая DD объясняется в том числе сильной abstention, а не доказанным преимуществом recency.
4. **MLP HL90**,6→8→1,65параметров/seed,195 в фиксированном ensemble; threshold20bps.
   Normal+0,8769%, stress+0,5859%; DD4,2360% /4,2921%;48fills.
5. **NoTrade**:10000USDT,0% net,0% DD,0fills во всех сценариях.
6. **MLP uniform**,65/seed,195вensemble; threshold50bps.
   Normal−1,3456%, stress−1,6050%; DD4,5248% /4,5754%;44fills.

При строгом исследовательском DD<=10% лидер доходности — linear HL90, а не linear uniform.
При DD<=20%/30% — linear uniform. Это post-ranking risk budgets, не действующие stop-loss
и не согласованный риск реального счёта. Нельзя пересматривать этот выбор как live-рекомендацию.

## Устойчивость и признаки переобучения

Общие circular block-bootstrap banks:4096draws,3календарных дня/block; seeds3039и7079.
Основная и audit bank используют один открытый61-дневный sample: независим только RNG,
не данные. LCB5% mean daily log return, в bps/day; не обещанная доходность и не гарантия.

- Linear uniform: audit stress LCB **−13,181bps/day**; paired LCB против EMA **−5,744**.
  Положительны12/21 outer slices,57,14%, включая последний partial slice, ниже65%gate.
  Лучший3-дневный участок даёт111,3% полного log-growth. Если описательно вычесть этот
  участок из log returns, итог **−1,787%**. Это concentration diagnostic, не новое обучение
  без этого периода и не полноценный контрфактический retraining.
- EMA: LCB−15,522bps/day;10/21positive slices; без лучшего3-дневного log-window **−1,619%**.
- Linear HL90: LCB−1,563;3/21positive slices; без лучшего window+0,139%, но слишком мало
  активных возможностей и малый средний эффект. Нельзя требовать активности во всех cash slices.
- MLP HL90: LCB−6,854;9/21positive slices; без лучшего window−0,676%.
- MLP uniform: LCB−13,853;5/21positive slices; без лучшего window−5,114%.

RMSE gross3-day forecast, bps: linear uniform768,92; linear HL90775,15;
MLP uniform778,48; MLP HL90780,17; нулевой прогноз774,56. Только linear uniform слегка
лучше нулевого поMSE. Это усреднение всех asset outcomes, не net-profit skill/financial gate.
HL90 ESS≈124origins против553–573uniform. Вес имеют прибыльные и убыточные исходы.
Максимальный functional drift (одинаковый текущий feature probe) у linear uniform7,82bps,
но adjacent action changes50,63%: небольшой drift модели не означает стабильные решения.
Loss обучения остаётся regularized gross-forecast MSE; net growth используется в threshold
calibration/ranking. MSE и прибыль не эквивалентны — важное ограничение текущей схемы.

## Сопоставление с замороженными результатами

Исходный отчёт7октября, next4h-open reference и20benchmark origins:
linear uniform+13,0705% /stress+12,5058%, DDnormal9,3083%;
EMA+8,5750% /+8,1633%; linear HL90+7,5929% /+7,3680%;
MLP uniform−3,0925% /−3,4182%; MLP HL90−1,2382% /−1,5446%.
Эти numbers — справочные, НЕ paired доказательство улучшения: изменились label-clock,
исполнение, thresholds и последний origin/terminal handling.

Дополнительный descriptive контроль использовал те же новые prices, costs, funding,
капитал и выбранные до августа thresholds, но замороженные старые модели:

- Старый linear uniform: normal+12,8051%, stress+12,2431%, DDstress10,6890%.
  Новый имеет **+4,9969pp** stress net difference, но paired audit LCB **−3,7091bps/day**.
- Старый linear HL90: stress+1,1226%; новый+1,1833%, difference+0,0607pp,
  paired LCB−1,7454bps/day.
- Старый MLP uniform при matched50bps: stress+0,7393%; новый−1,6050%,
  difference−2,3443pp; paired LCB−14,0106bps/day. Новый MLP uniform хуже этого контроля.
- Старый MLP HL90: stress+0,5224%; новый+0,5859%, difference+0,0635pp,
  paired LCB−1,2762bps/day.

У старого банка20origins; на30сентября модель27сентября применена к актуальным признакам
без дополнительного fit. Этот контроль не изолирует исключительно label-change: у нового
варианта есть последний дополнительный retrain. Регистрация control сделана после просмотра
новых основных результатов; это transparent diagnostic, не независимый challenger gate.
Результаты V7/V8 ранних objectives не ранжируются вместе с этими compact моделями без общего
капитала, clocks, costs и ledger; повторных старых collector/GA обучений здесь не было.

## Почему не запускался GA и что улучшать дальше

В pre-August stress-калибровке выбранный linear uniform дал−0,2001%, audit LCB−26,321bps/day;
HL90 выбранный100bps дал NoTrade0%. MLP uniform+1,2842%, LCB−8,264; MLP HL90+1,0646%,
LCB−9,818. Подтверждённого сигнала простой модели нет. Поэтому GA/RSfits=0.
Расширять search после слабого gate означало бы увеличивать testing budget без подтверждения edge.

Уже выполнены улучшения: корректный3-day label-clock, решение без будущего outcome,
последнее partial deployment, индивидуальные pre-August thresholds, общий ledger с terminal
funding и общие resample banks. Не подбирали новые параметры на раскрытом августе–сентябре.

Следующий зарегистрированный маленький эксперимент, до просмотра нового outer:

1. Оставить linear7 как основной исследовательский кандидат и прежние модели как immutable
   controls. MLP пока не раздувать: текущий ансамбль не даёт доказанного выигрыша.
2. Проверить pre-August net-calibration для отдельного cost-aware abstention layer:
   expected gross effect минус roundtrip fee/drag/funding allowance; зарегистрировать один
   hysteresis/cooldown вариант и контролировать opportunities отдельно от executed trades.
   Не выбирать threshold по августовскому скачку и не навязывать minimum trade count.
3. Задать единый заранее определённый exposure/risk budget для comparative overlay/EMA;
   отделить gross signal contribution от turnover, holding duration и ставок на общий market move.
4. Небольшой equal-budget GA/RS эксперимент только после positive pre-August simple-signal
   gate; ограниченное число objectives, полный append-only global registry, без заявлений
   DSR/PBO до восстановления реального числа независимых попыток.
5. Новый sealed prospective после фиксации алгоритма. Длительность рассчитывать по заранее
   согласованному minimal effect: условие «неубыточный итог» соответствует нулевой границе,
   но не задаёт power/длительность без минимального положительного эффекта и variance estimate.
6. Observed execution qualification — отдельный этап: full8 L2/consumer availability,
   actual fees/rules/funding marks, fault injection. Без этого никакого real-money promotion.

## Артефакты и исправление экспорта

`registration.json`, `objective_registry_entry.json`: data/code/cutoff/banks contract перед fits.
`cal_origin_*.json`, `report_origin_*.json`: все модели, scaler и hashes решений.
`calibration.json`, `comparison.json`, `legacy_comparison.json`: результаты и gates.
`replay_*.json`, `legacy_control_*.json`: полный ledger, funding, quantities, equity и атрибуция.
`capital_and_drawdown.png`: накопленный net капитал и DD; пунктир — stress capital.
`direct_checks.json`, `causal_regressions.json`, `receipt.json`: проверки/финальные hashes.

Единственный engineering failure — JSON-export numpy.bool_ после первой calibration.
Пинованные исходники/модели не редактировались. Отдельный serialization-only adapter
конвертирует numpy scalars в Python scalars без округления/изменения ranking;
`serialization_adapter_registration.json` создан до benchmark fits.
Глобальный registry всех прошлых adaptive попыток неполон; DSR/PBO и multiplicity-adjusted
значимость не заявлены. Audit bank не исправляет data reuse. Все9authority flags false.
