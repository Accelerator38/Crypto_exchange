# Genetic Primus V5 — фиксация измерительного контракта

Дата: 2026-08-26. Статус: `CONTRACT_FROZEN / SEALED_INTERVAL_UNOPENED`.

## Что исправлено

- Drawdown gate всегда применяет абсолютный бюджет 10%. Сравнение с incumbent включается только при наличии хотя бы одного fill у incumbent. Переход из `NoTrade` больше не сравнивается с нулевой просадкой как с торгующей стратегией.
- Functional drift разделён на action disagreement (`<=0.20`) и bounded-logit distance (`<=0.15`). Старый parameter-space порог `0.75` в V5 не используется.
- Разведены minimum opportunities (180), minimum trades (12), minimum trade days (12) и recency ESS (120 дней). Abstention не обязан повышать число сделок для прохождения opportunity/ESS evidence.
- Создан append-only hash-chain registry из 20 objectives. DSR должен быть не ниже 0.95, PBO — не выше 0.50. Отсутствующее или не связанное с SHA evidence блокирует promotion.
- Архитектура сохранена: linear binary overlay, не более 6 активных features. Добавлены stability selection (`frequency>=0.60`, `sign agreement>=0.80`) и контроль знака относительно торгующего incumbent.
- Concentration считается как allocation lift относительно доступных opportunities; excess lift выше 2.0 штрафуется в fitness.
- Hysteresis/cooldown зарегистрированы до открытия interval: entry logit `0.15`, exit logit `-0.05`, cooldown 2 бара.

## Новый sealed interval

- Период: `2025-02-01T00:00:00Z` — `2026-01-01T00:00:00Z`.
- Январь 2025 исключён из tuning и evidence.
- Контракт: `41a90b03f7c166d07331dc51a45972173f9a0c03cbb2b1888c9ab62a80932d70`.
- Registry V2: `766aed82b8c26b76ebe9a565a53fe565bcceda4a32f7360b76ff77d847de1273`.
- Доходности нового interval не читались, обучение и outer evaluation не запускались.

## Проверки

- `py_compile`: 8 изменённых Python-файлов — успешно.
- Прямые V5 regression checks без pytest: 6/6.
- Обратная совместимость V4 прямыми regression checks: 5/5.
- Сеть, collector, supervisor, background processes, paper/live и реальные ордера не запускались.

## Последовательность допуска

1. Сформировать полный зарегистрированный outcome matrix и независимый DSR/PBO evidence-файл.
2. Один раз открыть sealed interval и выполнить offline rolling evaluation без изменения контракта.
3. При прохождении всех outer gates провести только offline fault injection.
4. Shadow и paper требуют отдельного явного разрешения и минимум 30 дней paper evidence.
5. Canary остаётся отключённым; если позднее будет отдельно разрешён, только 1x, одна позиция, без автоматического увеличения капитала и с kill switch.

Текущий V5 не имеет promotion authority и не готов к реальным деньгам: глобальный DSR/PBO evidence ещё не материализован, sealed interval не открыт, shadow/paper/fault evidence отсутствуют.
