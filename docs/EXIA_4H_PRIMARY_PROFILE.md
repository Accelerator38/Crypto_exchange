# Exia: основной 4h-профиль

## Решение

Основной исследовательский таймфрейм Exia зафиксирован на четырех часах:

- profile: `exia_4h_v1`;
- источник решения: только завершенная 4h-свеча;
- исполнение ретротеста: open следующей 4h-свечи;
- единица всех периодов, cooldown и holding: завершенная 4h-свеча;
- Bitget USDT futures;
- round-trip costs: 8 bps base и 16 bps stress.

Единый машинный контракт находится в
`configs/exia_primary_timeframe_v1.json`. Он является источником timeframe,
costs, research gate и safety-флагов. Генератор audit-spec не дублирует эти
настройки вручную.

## Что изменено

`src/panteon_runtime/timeframe_profiles.py` содержит один sealed-профиль для
всех 43 сравнимых компонентов. Профиль рекурсивно применяется к standalone
агентам, игрокам и вложенным агентам Pantheon. Периоды EMA, breakout,
volatility, cooldown, holding и rotation выражены непосредственно в 4h-барах.

Bitget runtime создает shadow и real-player компоненты через профильную
фабрику. Профиль hard-fail на любом входном timeframe, отличном от 4h. Парсер
настроек поддерживает `timeframe = 4h` как `poll_interval=14400`, `bar=1`,
`tf_kline=4h`.

Ретроисполнитель также исправлен:

- session/season context берет время исторического бара, а не время запуска;
- символы и равные голоса сортируются канонически;
- два независимых запуска дают одинаковые trade ledger и metrics.

## Данные и методика

Основной panel: Bitget full8 с 2022-01-01 по 2026-07-14. Свежий holdout:
2026-07-15..2026-08-17, 391680 исходных 1m-строк и 1632 агрегированных
symbol-bars, то есть 204 полных общерыночных 4h-наблюдения.

Профиль сравнивался с прежним `native_bar` на 4h. Holdout был подключен после
выбора параметров и не использовался для их подбора. Проверка включает
development, validation, OOS, sanity, общий UTC-day bootstrap LCB и
family-wise LCB.

## Результаты

| Метрика | Старый native 4h | Sealed 4h | Fresh holdout |
|---|---:|---:|---:|
| Достаточных конфигураций | 33 | 36 | 26 |
| Закрытых сделок | 54445 | 88970 | 1573 |
| Pooled stress net, bps/trade | 1.87 | -0.78 | -47.07 |
| Медиана mean по компонентам | 10.45 | -0.49 | -52.88 |
| Компонентов с mean > 0 | 19 | 17 | 3 |
| Компонентов с LCB > 0 | 3 | 1 | 0 |
| LCB > 0 во всех обязательных окнах | 0 | 0 | 0 |

Адаптация улучшила mean у 10 из 43 компонентов, LCB у 21 и устранила
недостаточную активацию у 5. При этом старые редкие высокодоходные выбросы
исчезли после перевода интервалов в реальные 4h-бары. Это улучшение
достоверности и диагностируемости, а не подтверждение прибыльности.

Лучший full-history результат `SR_EMATrend12_48`: 1973 сделки, stress mean
`+67.04 bps`, обычный LCB `+2.80 bps`. Он не проходит обязательные окна:
family-wise LCB `-43.04 bps`, а свежий holdout имеет mean около `-101.79 bps`
и отрицательный LCB. Ни один агент, игрок или ансамбль не прошел полный gate.

## Вердикт и использование

Вердикт: `NO_FIXED_PROFILE_STABLE_COMPONENT`.

4h остается основным исследовательским контрактом и общей базой для следующих
итераций. Текущий профиль не имеет promotion authority и не разрешает paper
или live. Следующая работа должна создавать узкие preregistered варианты на
этом же 4h-контракте, а не менять timeframe или снижать gate после просмотра
holdout.

Авторитетные артефакты:

- `Reports/Exia/four_hour_profile_v1_deterministic_c/report.md`;
- `Reports/Exia/four_hour_profile_v1_deterministic_c/metrics.parquet`;
- `Reports/Exia/four_hour_profile_v1_deterministic_c/trades.parquet`;
- `Reports/Exia/four_hour_profile_v1_deterministic_c/comparison/comparison.md`;
- `Reports/Exia/four_hour_profile_v1_deterministic_c/comparison/component_comparison.parquet`.

Повтор `four_hour_profile_v1_deterministic_d` совпал с авторитетным прогоном:
90543 ledger-строки, 301 metric-строка, ноль различающихся ячеек. Число ledger
строк включает все панели; 88970 и 1573 закрытые сделки относятся к main и
holdout соответственно.

## Воспроизведение

```powershell
.venv\Scripts\python.exe tools\prepare_exia_4h_profile_spec_v1.py
.venv\Scripts\python.exe tools\run_exia_timebase_audit_sharded_v1.py `
  --spec configs\exia_4h_profile_audit_v1.json `
  --output Reports\Exia\four_hour_profile_v1_recheck `
  --workers 8
.venv\Scripts\python.exe tools\compare_exia_4h_profile_v1.py `
  --baseline-dir Reports\Exia\timebase_audit_v4_full_minute `
  --adapted-dir Reports\Exia\four_hour_profile_v1_recheck `
  --output-dir Reports\Exia\four_hour_profile_v1_recheck\comparison
```

Эти команды выполняют только retrospective research. Они не отправляют
ордера и не меняют promotion status.
