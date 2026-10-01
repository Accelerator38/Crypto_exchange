# Этапы 5–6: входные условия и решение

Дата проверки: 2026-10-01. Решение: **NoTrade; stage 5 blocked; stage 6 blocked**. Никакой prospective outer-результат не создавался, paper/shadow/live не запускались, торговых полномочий нет.

## Этап 5 — prospective всей адаптивной процедуры

В ограниченном genetic search на 20 train-origins проверены 24 заранее указанных sparse genomes. Ни один не получил положительную худшую block-bootstrap LCB; `selected_genome=null`, `validation_result=null`, `final_policy=NoTrade` (результат SHA-256 `b159550c28ade51ae9b272d6a803a7a85e1d6683588162cc9fc54ce966e35ec7`). Prospective admission теперь проверяет и train, и полный validation gate, а не только наличие совпадающего genome.

Для проверки новой *экономической* основы, а не увеличения genetic budget, отдельно зарегистрирована ровно одна фиксированная cross-sectional 3-day momentum гипотеза (коммит контракта `9d0373a`). На тех же **раскрытых development** 20 origins она дала gross −1,567%, net −1,943% / −2,130% при 16/24 bps, худшую LCB −0,1698% в день. Результат SHA-256 `88a3b53a9c41cfe27b182d2f8ed782271496626b9f259241845181b057082a0b`. По предзаданному правилу inner validation не открыта, гипотеза закрыта. Это вторая линия исследования, **не** независимое подтверждение общей гипотезы.

Следовательно, фиксировать будущий outer-интервал для challenger сейчас некорректно: нечего предварительно выбрать. Август/сентябрь не могут быть переименованы в новый holdout. Обновление модели каждые 3 дня может стать частью будущего запечатанного контракта только после успешного внутреннего выбора и до первого будущего бара. Число origins и единственный момент финального анализа нужно определить по заранее заданному минимальному эффекту и мощности, а не продлевать до первого pass.

## Этап 6 — execution / qualification

Локальная offline-проверка существующего release-контракта `configs/genetic_primus_release_candidate_v1.json` вернула `INVALID`: pinned SHA `src/exia/genetic_primus/measurement_lab_v2.py` (`c09835cd...`) не равен текущему (`ff9ec7f1...`). Этот контракт и старый execution-код существуют в рабочем дереве как untracked пользовательские файлы; они **не** включены в данную ветку и не были изменены. Простая замена SHA в старом контракте задним числом не является квалификацией новой модели.

Даже без SHA-ошибки contract оставляет null все 8 обязательных evidence: instrument rules, account fees, book/mark/funding dataset, latency/slippage calibration, independent statistical audit, global objectives audit, shadow/fault/reconciliation audit и risk-owner approval. Не определены 4 количественных risk-поля: capital, max order notional, max loss, max drawdown. Все safety-флаги остаются false. Положительного исследовательского и prospective результата также нет, поэтому запуск shadow/paper или canary не был бы обоснован.

## Порядок разблокировки без подмены доказательств

1. Сформулировать *другой* механизм edge с внешней экономической причиной и заранее ограниченным числом проверок; учитывать обе новые/старые раскрытые попытки в глобальном testing ledger. На текущих признаках и с простым увеличением search budget genetic search не возобновлять.
2. Если новый development train **и** inner validation пройдут абсолютный cost-aware gate, заморозить точный код/данные/model SHA/cutoff и весь 3-дневный retrain-протокол до будущего интервала. Запустить один заранее назначенный prospective анализ; при провале — NoTrade.
3. Только после положительного prospective результата создать **новый**, а не переписанный старый, release-контракт с согласованными source pins, реальными venue data и числовыми лимитами владельца риска. Далее отдельно проверить recorded execution, fault injection, reconciliation и shadow/paper. Canary без leverage — лишь после отдельного решения владельца риска; текущая работа его не разрешает.

Проведены только короткие локальные deterministic/unit проверки и offline исторический расчёт. Сеть биржи, collector, supervisor, paper/live runtime, ключи и ордера не затронуты.
