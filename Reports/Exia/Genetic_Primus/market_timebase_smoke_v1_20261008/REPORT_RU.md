# Trades / derivative ticker: bounded offline timebase smoke

Дата запуска: 2026-10-08. Исходные данные: BTCUSDT, 2024-12-01.

## Результат

- 42/42 явных unittest проверок; deterministic повтор равен.
- Прочитано по 64 trade и 64 ticker строк.
- По архивам прочитано 4096 / 4096
  сжатых и 8192 / 8192
  распакованных байт. Полный архив/hash/CRC НЕ проверялись.
- Опубликовано 44 уникальных trade events и 63 ticker events.
  Последние недоказанно завершённые пакеты: 20 / 1 events исключены.
- 12 соседних trade pairs с одинаковым receive timestamp:
  никакого вымышленного within-batch порядка; duplicate IDs: 0.
- Переиспользованы 17 замороженных L2 states, span 1.569347 s.
  Старые source/artifact hashes проверены: 10 / 10.
- Сопоставлений trade с предшествующим свежим BBO: 5;
  вне наблюдаемого spread: 0. Это не доказательство fill или ошибки данных.
- Старый pure mark parser согласен в 63 случаях.

## Часы и funding

Все timestamps сохранены в UTC микросекундах. As-of: provider receive строго
раньше cutoff; equal-clock boundary ambiguous, оба возраста <=500000 us.
Статусы joins: `{"ticker": {"fresh": 6, "missing": 11}, "trades": {"fresh": 1, "fresh_batch": 4, "missing": 5, "stale": 7}}`.
Это некалиброванный engineering freshness budget, не latency ордеров.

Reported funding rate: `['0.000195']`;
reported funding timestamps, us: `[1733040000000000]`;
отношение к времени receipt: `{'future': 63}`.
Пустой predicted rate в 64 raw строках сохранён как missing/null.
Ни один ticker не превращён в settlement или cashflow. Будущее/прошлое/rate=0
не доказывает final rate, фактическое списание или settlement mark.

## Ограничения и следующий шаг

Это только короткое пересечение двух префиксов с ранее проверенными 15 levels/side
L2. Не подтверждены venue sequence, полнота глубины, доступность стратегии,
реальное исполнение, account/historical fees/rules и funding settlements.
Архив book ticker января 2025 не присоединялся к декабрю 2024.
График — engineering цены/часы, не накопленный капитал или доходность.

0 fits, 0 economic labels, 0 market strategy replays, 0 network requests.
Все 9 authority flags false; financial_replay_ready=false.
Положительный PnL предыдущих ретротестов здесь не перепроверен и не квалифицирован.
BTC 2024 не заменяет отсутствующий verified full8 execution bundle августа–сентября 2026.

Далее: детерминированный microsecond-to-postclose clock adapter с полной
provenance и явным fail-closed на недоказанные available/funding fields.
Новый financial replay/обучение/сеть не разрешены этим отчётом.
