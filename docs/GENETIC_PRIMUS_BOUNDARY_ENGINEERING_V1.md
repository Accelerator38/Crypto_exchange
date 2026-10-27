# Genetic Primus: forward boundary engineering contract v1

Дата: 2026-10-05. Контракт: `configs/genetic_primus_boundary_engineering_v1.json`.

## Назначение и границы

Это отдельная инженерная спецификация для искусственных fixtures, не новый финансовый objective и не изменение старых/prospective контрактов. Старые runner и бухгалтерский simulator не изменены и не исполнялись. Исторические dependency gaps и terminal mismatch этим этапом не исправлены задним числом.

Основание: предыдущий accounting audit показал неоднозначность между origin-final-close в контракте и whole-interval callsites. Дополнительное статическое чтение показывает: текущий `force_flat_timestamps` вызывает выход на **open** указанной свечи. Это не механизм закрытия на предыдущем close; эти цены нельзя считать эквивалентными при гэпе.

## Явные правила нового инженерного контракта

- UTC epoch milliseconds, свеча 4 часа. Интервал `[start, end)`, только полные origins.
- Deployment horizon и retrain cadence — 3 дня. Они не являются liquidation cadence.
- Train end — origin open, исключён из обучения. Feature open должен быть строго раньше `origin − 7 bars`, label end — не позже origin. Это декларация cutoff, не выполненный аудит labels или обучение scaler.
- Initial equity — 1; равные начальные cash sleeves, без межсимвольной ребалансировки. Cash и capital непрерывны между origins, не сбрасываются и не нормируются повторно.
- Переобучение само по себе не создаёт сделку. При неизменном target позиция переносится без изменения quantity и без новых entry/exit fees только из-за retrain.
- При изменении target: выход, затем вход при ненулевом новом target на следующем open. Short/flat/long направления проверены на уровне семантики переходов, не PnL коротких позиций.
- В конце всего интервала — liquidation на final candle close с fees. `force_flat_timestamps=[]` передаётся явно. `record_details=true` также фиксируется явно.
- В каждом решении нужен SHA текущей модели. SHA модели входа позиции сохраняется до выхода и может отличаться от SHA модели решения после retrain.
- Предыдущий close, retrain, decision и open-fill имеют определённый порядок даже при одинаковом числовом timestamp. Модель должна быть доступна до fill; фактическая latency обучения не проверена и не даёт execution authority.

Origin-close liquidation не поддерживается этой спецификацией. Если понадобится такой эксперимент, потребуется отдельный versioned контракт и проверенный close-price adapter, а не незаметное добавление origin timestamps в старый open-reset параметр.

Normal 16 и stress 24 round-trip bps — лишь явные reference synthetic costs для данной инженерной спецификации, не утверждение достаточности реального stress test. Funding/borrow, exchange fills и внутрисвечные риски не подтверждены.

## Реализация

`src/exia/genetic_primus/boundary_contract_v1.py` — stdlib-only функции:

- Strict contract validation: незарегистрированная смена политики, дополнительных полей или safety flags отклоняется; `false` не подменяется числом `0`.
- Boundary plan: train/cutoff/origin boundaries, explicit accounting parameters, final terminal boundary. Неполный origin, неправильный grid и слишком большой fixture отклоняются.
- Boundary transition: nine-direction state table и две различные model identities; не команды ордеров и не fill simulator.
- Single-long artificial cash identity: проверка gross − same-trajectory fees = net на заданных числах. Не market evaluator; не доказательство short/funding/exchange accounting.

Ни одна функция не импортирует существующие models, accounting simulator, collector или runtime. Размер synthetic plan ограничен 100 origins; опубликованный fixture содержит 3 origins, 9 переходов и 4 числовые cash identities.

## Provenance и receipt

Builder связывает предыдущий accounting receipt, его исходные byte hashes, новый контракт и инженерный код. Три исходных accounting/helper файла прочитаны как bytes/AST с проверкой зарегистрированного SHA, без импорта. Статически закреплён open-price reset callsite. До и после публикации перепроверяются 1784 исходных JSON-файла и predecessor inputs.

`completion_receipt.json` имеет тип **engineering_only_not_financial_launch_receipt**. Фиксирует byte/semantic hashes, наблюдаемые Python path/version и future launch requirements. Это не полный fingerprint runtime/environment и не доказательство исторического запуска.

Для будущего финансового запуска остаются обязательными: полный transitive dependency bank, проверка фактически загруженных зависимостей, версии Python/distributions, hashes data/folds/resample/cutoff/models/scalers, все call parameters, before/after identity и проверенный evaluator adapter. Ни одна из этих полноценных launch gates не считается открытой этим fixture.

## Проверки и ограничения запуска

Выполнено: py_compile по одному новому файлу, минимальная pure function проверка, preflight, **134 детерминированные регрессии**, публикация и отдельный artifact checker. Все прошли. Первоначальный preflight выявил преждевременное сравнение predecessor input bank; порядок исправлен до публикации, никакие старые результаты не менялись.

Команды запускаются по одной, только `E:\Crypto_exchange\.venv\Scripts\python.exe -X utf8`:

- `tools/check_genetic_primus_boundary_contract_v1.py`.
- `tools/build_genetic_primus_boundary_contract_v1.py --verify-only`.
- Builder без флага — exclusive publication в новый каталог, без перезаписи существующего.
- `tools/check_genetic_primus_boundary_contract_artifacts_v1.py`.

Никаких pytest, market replay, collector, сетевых проверок, background workers или model fits. Все flags credentials/orders/paper/live/execution/promotion/training/financial evaluation/network остаются false.

## Следующая ограниченная задача

Создать отдельный boundary adapter с injected stub, проверить передачу всех параметров и согласованность decision/entry identities без запуска simulator. Собрать статический dependency bank будущего adapter и не называть его полным loaded-runtime bank. Только после устранения этих engineering gaps можно отдельно рассматривать разрешение нового финансового расчёта по зарегистрированному контракту; это не разрешение изменять sealed interval или искать новые objectives на прежних данных.
