# Exia: полный 4h-эксперимент агентов по режимам рынка

Дата фиксации: 2026-08-20
Статус: завершенный диагностический эксперимент, без права paper/live promotion.

## 1. Цель и контракт

Эксперимент проверяет максимально доступное число компонентов, совместимых с единым causal-контрактом OHLCV full8 и исполнением на следующем открытии 4h-бара. Режим рынка используется только как метка результата после входа. Внешний `NoTrade` по bullish, bearish или neutral отключен.

- Компонентов: **143**.
- Текущих компонентов: **43**.
- Вариантов текущих торговых компонентов: **90**.
- Новых агентов: **10**.
- Таймфрейм: **4 часа**.
- Сигнал: на закрытии завершенного бара.
- Исполнение: на открытии следующего бара.
- Издержки: **8 bps** base и **16 bps** stress round trip.
- Сделок в итоговом ledger: **327 192**.
- Покрытие режимной меткой: **100%**.
- `orders_enabled=false`, `promotion_authority=false`.

Для каждого компонента зафиксированы **среднее и медиана** costed PnL:

1. по всем режимам вместе;
2. отдельно в bullish;
3. отдельно в bearish;
4. отдельно в neutral;
5. отдельно по development, validation, OOS и sanity;
6. отдельно на untouched prospective holdout.

## 2. Данные и окна

| Набор | Окно | Период |
|---|---|---|
| full8 4h | development | 2022-01-01 - 2023-12-31 |
| full8 4h | validation | 2024-01-01 - 2024-12-31 |
| full8 4h | OOS | 2025-01-01 - 2025-12-31 |
| full8 4h | sanity | 2026-01-01 - 2026-07-14 |
| untouched 4h | prospective holdout | 2026-07-15 - 2026-08-17 |

Историческое распределение баров: bullish **29.82%**, bearish **32.65%**, neutral **37.53%**. На holdout преобладал neutral: bullish **16.67%**, bearish **19.61%**, neutral **63.73%**.

## 3. Состав эксперимента

В тест вошли 22 доступных runtime-агента, 11 игроков/контролей Пантеона, 10 простых исследовательских компонентов и варианты торговых компонентов. Для 22 runtime-агентов и 8 действующих простых стратегий добавлены три варианта временных параметров:

- `FAST067`: окна уменьшены до 0.67 от базовых;
- `SLOW150`: окна увеличены до 1.5 от базовых;
- `SLOW200`: окна увеличены до 2.0 от базовых.

Добавлены десять новых causal-агентов, не дублирующих существующие реализации:

1. `NEW_KeltnerBreakout`;
2. `NEW_DualThrust`;
3. `NEW_RSIFailureSwing`;
4. `NEW_VolumePriceDivergence`;
5. `NEW_EfficiencyRatioTrend`;
6. `NEW_VolatilityRegimeSwitch`;
7. `NEW_RollingVWAPReversion`;
8. `NEW_CandleStructureBreakout`;
9. `NEW_CrossSectionalResidualMomentum`;
10. `NEW_CrossSectionalShortReversal`.

Настоящий L2-агент не включен: для общего окна 2022-2026 отсутствуют непрерывные point-in-time снимки стакана. Подмена L2 OHLCV-прокси исказила бы смысл эксперимента.

## 4. Метод ранжирования

Основная величина - доходность закрытой сделки после stress-издержек 16 bps. Порядок сравнения:

1. статус надежности;
2. число поддержанных временных окон;
3. familywise LCB с поправкой на 143 компонента и 4 режима;
4. обычный bootstrap LCB;
5. медиана;
6. среднее;
7. просадка.

Статусы:

- `ROBUST_POSITIVE`: mean > 0, median > 0, LCB > 0 и familywise LCB > 0;
- `POSITIVE_UNCERTAIN`: mean > 0, median > 0 и обычный LCB > 0, но familywise LCB не подтвержден;
- `MIXED`: положительно только среднее или только медиана;
- `NEGATIVE`: и среднее, и медиана неположительны;
- `LOW_INCIDENCE`: меньше 20 сделок;
- `NO_ACTIVATION`: нет сделок.

Положительное среднее при отрицательной медиане означает, что результат держится на небольшом числе крупных выигрышей, а типичная сделка убыточна. Такой профиль не считается устойчивым.

## 5. Итог исторического теста

Ни в одном режиме нет `ROBUST_POSITIVE` компонента. Единственные `POSITIVE_UNCERTAIN` срезы найдены в bearish, но они не прошли familywise LCB и prospective holdout.

### Общий результат

| Место | Компонент | Статус | Сделки | Mean, bps | Median, bps | LCB, bps |
|---:|---|---|---:|---:|---:|---:|
| 1 | PanteonNextResearch | MIXED | 1080 | 24.88 | -126.70 | -6.16 |
| 2 | LiveRegimePullback__FAST067 | MIXED | 3414 | 3.99 | -109.19 | -11.07 |
| 3 | PlayerFundingOHLCVFallback | MIXED | 3844 | 0.55 | -269.84 | -16.26 |
| 4 | ResearchValidatorAgent | MIXED | 2784 | 21.44 | -128.87 | -4.06 |
| 5 | LiveRegimePullback | MIXED | 2844 | 1.17 | -114.63 | -17.13 |

Вывод: общий положительный mean у лидеров не поддерживается медианой и LCB. Единую универсальную стратегию или полный ансамбль формировать нельзя.

### Bullish

| Место | Компонент | Статус | Сделки | Mean, bps | Median, bps | LCB, bps |
|---:|---|---|---:|---:|---:|---:|
| 1 | MomentumScalper__SLOW200 | MIXED | 468 | 77.12 | -284.58 | 27.15 |
| 2 | NEW_CrossSectionalResidualMomentum | MIXED | 2941 | 10.32 | -33.66 | -2.79 |
| 3 | ResearchValidatorAgent | MIXED | 1005 | 60.71 | -74.09 | 22.72 |
| 4 | PanteonResilient | MIXED | 1108 | 22.75 | -58.82 | 0.14 |
| 5 | LiveOIBreakoutVolumeFallback | MIXED | 1614 | 43.08 | -52.47 | 11.37 |

У всех лидеров отрицательная медиана. Положительный обычный LCB отдельных компонентов недостаточен: familywise LCB отрицателен, а типичная сделка остается убыточной.

### Bearish

| Место | Компонент | Статус | Сделки | Mean, bps | Median, bps | LCB, bps | Familywise LCB, bps |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | LiveCrashHunter__SLOW150 | POSITIVE_UNCERTAIN | 430 | 52.84 | 23.67 | 10.11 | -27.91 |
| 2 | LiveCrashHunter__SLOW200 | POSITIVE_UNCERTAIN | 429 | 66.75 | 23.03 | 15.51 | -34.88 |
| 3 | PanteonResilient | MIXED | 1117 | 15.75 | -60.56 | -7.67 | -25.48 |
| 4 | SR_CandleMomentum3__SLOW150 | MIXED | 4556 | 4.55 | -47.74 | -12.26 | -27.86 |
| 5 | CandlePatternAgent__FAST067 | MIXED | 968 | 25.62 | -13.91 | -1.40 | -29.37 |

`LiveCrashHunter__SLOW150` - лучший исследовательский след, но не готовый кандидат. Его mean по окнам: development **-2.45**, validation **73.62**, OOS **109.77**, sanity **83.34** bps. `SLOW200` имеет sanity mean **-23.81** bps. Это нестабильность, а не подтвержденный edge.

### Neutral

| Место | Компонент | Статус | Сделки | Mean, bps | Median, bps | LCB, bps |
|---:|---|---|---:|---:|---:|---:|
| 1 | NEW_VolatilityRegimeSwitch | MIXED | 3944 | -1.08 | 0.31 | -9.79 |
| 2 | NEW_VolumePriceDivergence | MIXED | 1812 | -10.83 | 60.22 | -31.58 |
| 3 | SR_RegimePullback12_48__SLOW200 | MIXED | 937 | 1.49 | 76.86 | -24.66 |
| 4 | MeanRevConfirmedAgent__FAST067 | MIXED | 570 | 16.63 | 54.47 | -17.98 |
| 5 | LiveTrendFollow | MIXED | 882 | 2.90 | -203.10 | -31.89 |

У neutral нет положительного LCB. Лучшие срезы расходятся по знаку mean и median либо ломаются между окнами. Например, `MeanRevConfirmedAgent__FAST067` имеет validation mean **-33.70** bps при положительных остальных окнах.

## 6. Untouched prospective holdout

Holdout не подтвердил исторических лидеров:

- `PanteonNextResearch`, все режимы: mean **-33.79**, median **-103.28**, 25 сделок;
- `MomentumScalper__SLOW200`, все режимы: mean **-35.94**, median **-182.39**, 20 сделок;
- `LiveCrashHunter__SLOW150`, bearish: 5 сделок, mean **-223.78**, median **-208.15**;
- `LiveCrashHunter__SLOW200`, bearish: 1 сделка, mean **-31.97**.

На holdout появились новые диагностические следы:

| Режим | Компонент | Сделки | Mean, bps | Median, bps | LCB, bps | Оценка |
|---|---|---:|---:|---:|---:|---|
| all | NEW_CrossSectionalResidualMomentum | 189 | 1.99 | -20.10 | -18.37 | хвостовой mean, неустойчиво |
| bullish | NEW_CrossSectionalResidualMomentum | 27 | 13.43 | -5.69 | -50.24 | мало данных, median < 0 |
| bearish | NEW_CrossSectionalResidualMomentum | 40 | 50.10 | -26.27 | -25.98 | мало данных, median < 0 |
| neutral | LiveVolCompress | 35 | 43.56 | 17.40 | -3.01 | ближайший к LCB > 0, но исторический neutral отрицателен |

`LiveVolCompress` нельзя продвигать: его исторический neutral mean **-9.59**, median **-78.32**, LCB **-40.36** bps. Holdout-след полезен только как гипотеза о смене рынка или неправильной границе neutral.

## 7. Что дали новые агенты и варианты

- Лучший новый bullish-агент: `NEW_CrossSectionalResidualMomentum`, но median и LCB отрицательны.
- Лучший новый neutral-агент: `NEW_VolatilityRegimeSwitch`, mean отрицателен.
- `NEW_KeltnerBreakout` показывает высокий bullish mean **84.02** bps и LCB **9.37** bps, но median **-148.32** bps; результат сформирован редкими крупными движениями.
- Медленные варианты полезны в bearish для `LiveCrashHunter`, но эффект не перенесся на holdout.
- Быстрый вариант `MeanRevConfirmedAgent__FAST067` улучшил neutral mean и median, но не LCB и не validation.
- Простое размножение параметров не устранило главную проблему: распределение сделок асимметрично, а устойчивость между окнами отсутствует.

## 8. Ограничения

1. Отчет не тестирует настоящий L2, funding, basis и синхронные cross-exchange данные из-за отсутствия общего point-in-time датасета.
2. Часть legacy-строк Пантеона работала через текущий runtime fallback: optional-модуль `crypto_exchange` с генетикой недоступен. Эти строки не доказывают качество полноценного генетического ансамбля.
3. Режим присваивается по состоянию рынка на момент входа. Сделка может завершиться уже в другом режиме.
4. Один и тот же набор использован для discovery 143 компонентов; поэтому обычный LCB недостаточен и применяется familywise-поправка.
5. Результаты являются исследовательскими и не дают права на paper/live.

## 9. Решение и следующий узкий эксперимент

Полный ансамбль возвращать рано. Он смешает отрицательные и хвостовые стратегии и скроет источник результата. Следующий шаг должен сократить пространство, а не породить еще сотни вариантов:

1. Оставить три независимые гипотезы: bearish crash continuation, neutral compression breakout и cross-sectional residual momentum.
2. Для каждой заменить примитивный выход на единый контракт: signal invalidation + ATR stop + ATR trailing exit, а `max_holding` оставить аварийным ограничением.
3. Не оптимизировать вход и выход одновременно: сначала три заранее заданных exit-профиля на фиксированном входе.
4. Повторить nested walk-forward только для 3 x 3 конфигураций, сохранив mean, median, LCB и drawdown по каждому режиму и окну.
5. Требовать положительные mean и median во всех поддержанных validation/OOS/sanity окнах, положительный familywise LCB на истории и положительный обычный LCB на untouched holdout.
6. Только после этого формировать режимную policy из отдельных специалистов и запускать strict paper canary.

Полные отдельные строки всех 143 компонентов находятся в `Reports/Exia/four_hour_regime_discovery_v1/agent_summary.tsv`; подробные метрики по режимам и окнам - в `agent_regime_metrics.tsv` и `agent_regime_split_metrics.tsv`.
