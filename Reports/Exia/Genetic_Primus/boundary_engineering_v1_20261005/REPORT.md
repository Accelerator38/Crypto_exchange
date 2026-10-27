# Forward boundary contract: результаты этапа

5 октября 2026 года.

## Что завершено

Добавлены новый инженерный контракт и pure функции, разделяющие переобучение и liquidation. В зарегистрированной инженерной схеме retrain/deployment занимают 3 дня; capital непрерывен, при неизменном target позиция переносится. Обязательное закрытие — только final candle close всего интервала. Это не переопределение старого `origin_final_candle_close` контракта и не новый финансовый эксперимент.

Дополнительное статическое подтверждение: `force_flat_timestamps` в рассмотренной текущей версии accounting закрывает позицию по open. Передавать туда границы origin как реализацию previous-close liquidation было бы неверно.

## Полученные доказательства

- 134 детерминированные регрессии прошли; новые Python-файлы компилируются.
- Опубликованы 3 synthetic origins, все 9 переходов short/flat/long, 4 artificial single-long cash identities.
- Проверены carry без retrain-induced trade, различие SHA модели решения и SHA модели входа позиции, late-model rejection, purge declaration, partial-origin rejection, nonfinite arithmetic rejection и fail-closed flags.
- Новый контракт, три рассмотренных accounting/helper source hashes и инженерный код связаны receipt; input identity проверена до и после публикации.
- Independent artifact checker подтвердил hashes, параметры и identities.
- 1784 исторических JSON-файла и прежние контракты не изменены; прежние runner не модифицированы.

Артефакты: `fixtures.json`, `summary.json`, `completion_receipt.json`. Fixture timeline искусственная, не рыночный период. Markdown-документация не входит в hashed output bank.

## Что не доказано

Соответствие реального evaluator новому контракту ещё не проверено. Полный граф фактически загруженных зависимостей, environment/distributions, training latency, data/model payloads и realistic exchange execution не подтверждены. Receipt является инженерным, не launch receipt торгового или исследовательского прогона.

Новых model fits и финансовых оценок — **0**. Доходность моделей не изменилась; отсутствие убытков не доказано. Credentials, orders, paper, live, execution, promotion, training, financial evaluation и network остаются false. Исторические terminal ambiguities и accounting unknowns остаются в прежнем audit.

## Дальнейшее действие

Проверить отдельный adapter на injected stub: параметры interval/cost/force-flat/details, final boundary и ledger identities. Затем закрепить dependency bank и launch prerequisites. Старые или prospective расчёты не перезапускать автоматически. Новый финансовый запуск и изменение liquidation policy требуют отдельного зарегистрированного решения; этот этап такого разрешения не даёт.

Техническое описание: `docs/GENETIC_PRIMUS_BOUNDARY_ENGINEERING_V1.md`.
