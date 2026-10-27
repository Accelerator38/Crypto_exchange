# Post-close clock: точные микросекунды и fail-closed availability

Дата запуска: 2026-10-08. Engineering only, не financial replay.

## Итоги

- 45/45 bounded deterministic unittest проверок прошли.
- Новый clock contract точно наследует pinned ms parent, без изменения старого кода/результатов.
- 17 сохранённых BTCUSDT L2 observations (2024-12-01) адаптированы без потери us precision.
- Strategy available time отсутствует в 17 observations:
  available_us=null, статус BLOCKED_UNKNOWN_STRATEGY_AVAILABILITY, reference/endpoint отсутствуют.
- Provider timestamps с неделимой на 1000 частью: exchange=0, receive=17.
  Ни один timestamp не округлён и receive не объявлен strategy availability.
- 12 synthetic clock cases; один aligned case полностью согласен с legacy resolver.
  11 остальных bridge attempts корректно отклонены по precision/ambiguity/invalid input.
- Native requests сохраняют injected 17 us feature delay и scheduled exit через 3 дня,
  а не entry fill + 3 дня. Этот injected delay не является измерением реальной модели.

## Сценарии clocks

| Синтетический случай | Native статус | Legacy bridge |
| --- | --- | --- |
| aligned_resolved | CLOCK_RESOLVED_NOT_FILL_QUALIFIED | accepted: exact agreement |
| fractional_resolved | CLOCK_RESOLVED_NOT_FILL_QUALIFIED | rejected |
| signal_equality | AMBIGUOUS_REFERENCE_BOUNDARY | rejected |
| reference_batch | AMBIGUOUS_REFERENCE_BATCH | rejected |
| arrival_minus1 | EXPIRED_NO_POST_ARRIVAL | rejected |
| arrival_equal | AMBIGUOUS_POST_ARRIVAL_BOUNDARY | rejected |
| arrival_plus1 | CLOCK_RESOLVED_NOT_FILL_QUALIFIED | rejected |
| expiry_minus1 | CLOCK_RESOLVED_NOT_FILL_QUALIFIED | rejected |
| expiry_equal | EXPIRED_NO_POST_ARRIVAL | rejected |
| expiry_plus1 | EXPIRED_NO_POST_ARRIVAL | rejected |
| endpoint_batch | AMBIGUOUS_POST_ARRIVAL_BATCH | rejected |
| aliased_snapshot | INPUT_REJECTED | rejected |

TTL exclusive: endpoint за 1 us до expiry допустим, ровно на expiry/после — нет.
Ровно на signal/arrival без порядка событий — ambiguous; несколько снимков
одного available batch также ambiguous. Reference и endpoint обязаны быть
distinct по ID и evidence hash. Aliased snapshot не может стать новым событием.
Задержанная receipt pre-arrival snapshot не превращается в post-arrival.

Это **консервативное уточнение** старой ms семантики: старый resolver допускал
equality и выбирал первый элемент списка. Он не изменён. Compatibility statement
относится только к aligned/unambiguous synthetic case, не ко всем старым endpoint outcomes.
Lossless bridge принимает только resolved synthetic metadata и сохраняет unqualified flags.

## Происхождение и пределы

Проверены source/artifact hashes трёх прошлых evidence roots:
`{"l2_format_smoke_v1_20261008": {"artifacts_verified": 10, "sources_verified": 10}, "market_timebase_smoke_v1_20261008": {"artifacts_verified": 12, "sources_verified": 11}, "postclose_label_preflight_v1_20261008": {"artifacts_verified": 7, "sources_verified": 12}}`.
Исходные gzip не открывались повторно; book observations взяты из pinned reconstruction.json.
Clock observations содержат hashes и timestamps, без цен/объёмов. Reported ticker
funding не может стать endpoint, settlement, final rate или cashflow.
Provider receive, время скачивания и exchange time не доказывают consumer availability.
Actual availability input API намеренно не реализован до независимого evidence contract.

Все 9 authority flags false. 0 fits, 0 labels, 0 market strategy replays,
0 network requests. Real availability, coverage, fills и financial replay не квалифицированы.
График — synthetic clock boundaries, не капитал или доходность.

## Следующее действие

Exact-clock инженерный блок завершён. Следующий содержательный этап — проверка
наличия/происхождения execution данных целевого августа–сентября 2026 для full8:
depth/sequence/coverage, strategy availability, marks/funding settlements,
historical rules и account fees. BTC 2024 не заменяет этот bundle.
Не запускать новый training/replay на старых next-open labels и не подменять
недоказанные availability/costs synthetic данными. Сеть требует отдельного разрешения.
