# Post-close microsecond clock adapter v1

Price-free инженерный адаптер. Не financial labels, не execution adapter,
не обучение/ретротест/проверка прибыли и не торговый runtime.
Все 9 authority flags false; real availability/coverage/fills не подтверждаются.
Нет сети, collector, pytest, sleep, scheduler или фонового процесса.

## Наследование и точность

Зарегистрированный `postclose_label_plan_v1` остаётся неизменным. Новый контракт
пинит его file SHA и canonical body SHA. Timing values умножаются на 1000
в integer arithmetic: compute=100000 us, order latency=250000 us,
reference wait=5000000 us, TTL=2000000 us, freshness=500000 us.
Трёхдневные deployment/retrain, 4h bars, purge=7, offset=1 сохранены.
Это перенос clock semantics, а не утверждение созревших labels или калибровка latency.

План выхода привязан к следующему scheduled origin, не entry fill + 3 дня.
Новый feature availability может иметь микросекундную часть. Между микросекундами
и старым ms форматам нет floor/ceil/round. Lossless bridge принимает только
synthetic resolved cases, где **все** clocks и half-open bounds делятся на 1000.
Иначе он отклоняет вход, не меняя временную семантику ради совместимости.
Bridge сохраняет unqualified flags; он не подключён к runtime.

## Неизвестная доступность остаётся неизвестной

У recorded BTC book fragments сохраняются exchange/received us, source prefix
SHA, исходный snapshot SHA, message-row SHA и source row ordinal.
Нормализованные clock observations не содержат prices/volumes.
`available_us=null`: provider receipt, timestamp загрузки файла и exchange time
не подменяют strategy/consumer availability. Самодекларируемая запись actual
available time без нового независимого evidence contract отвергается.

Recorded window всегда BLOCKED_UNKNOWN_STRATEGY_AVAILABILITY: без reference
и endpoint. Trade или reported funding ticker не может стать book endpoint.
Нет funding settlement, cashflow, final rate или funding mark из ticker-поля.
Для будущих реальных availability receipts нужна отдельная квалификация
происхождения данных, venue order/sequence и consumer clock; этот API её не выдаёт.

## Синтетический resolver

Работает только с явно указанным `synthetic_fixture`, максимум 256 observations.
Предварительно валидирует весь bounded window: типы, UTC us, hashes, causal
exchange <= received <= available, упорядоченный available clock, уникальные
snapshot IDs/hashes. Вход только из clock metadata; prices/OHLCV fallback запрещён.

Первый свежий post-signal reference, затем distinct post-arrival endpoint.
Endpoint snapshot не может совпадать с reference ни по ID, ни по evidence hash.
Задержанная receipt снимка, созданного до arrival, не может имитировать post-arrival.
Freshness включает реальную available - exchange задержку synthetic fixture.
Reference deadline и TTL — exclusive.

**Уточнение относительно старого resolver:** отсутствие последовательности
на equality signal/arrival boundary или несколько первых снимков с одинаковым
available clock дают AMBIGUOUS, не произвольный выбор по порядку списка.
Даже stale peer того же available batch не позволяет объявить порядок доказанным.
Старый resolver не изменяется; compatibility подтверждается только на
aligned/unambiguous synthetic cases. Clock resolution никогда не означает fill.

## Проверки и evidence

Первым после правки — py_compile одного pure module; затем одна pure function.
Далее только явно названный unittest-класс, вывод перехватывается <=1 KB.
One-shot checker исключает existing output root и пинит source/artifact hashes.
Проверяются неизменность предыдущих postclose plan, L2 и timebase evidence.
Исходные gzip не читаются повторно: используются сохранённые L2 events.

График — синтетические clock boundary cases, не график NAV/доходности.
Фрагменты BTC 2024 не являются full8 execution evidence августа–сентября 2026.

## Практический следующий шаг

Инженерный блок exact clocks завершён; не добавлять новые прибыльные оценки
на старых next-open labels. Для новой финансовой проверки нужен квалифицированный
historical full8 bundle целевого периода: L2/sequence/coverage, strategy availability,
marks, settlements (включая terminal settlement), historical fee/rule provenance.
Реальные receipts должны пройти отдельный независимый input gate.
Пока данных нет — не заменять их synthetic fixtures, OHLCV-open или provider receive.
Этот этап не разрешает сеть, обучение, paper/live или реальные ордера.
