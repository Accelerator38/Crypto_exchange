# Genetic Primus prospective 3-day protocol v1

Зафиксирован 30 августа 2026 до начала outer-интервала. Это контракт измерения, а не разрешение на paper/live торговлю.

## Гипотеза и сравнение

Единственный challenger — ранее выбранный sparse linear overlay `f7a4ae11…` с неизменяемым геномом `[0, 0, 1, 1, -1]`. Он сравнивается одновременно с `NoTrade` и простым `EMA_TrendConsensus`. Основной contrast — paired daily return overlay минус EMA; NoTrade остаётся абсолютным risk reference.

Outer-интервал: `[2026-09-01 00:00, 2026-10-01 00:00) UTC`. Он состоит из десяти неперекрывающихся трёхдневных origins. Для каждого origin scaler заново оценивается только на доступной истории; последние семь четырёхчасовых баров исключаются как purge, равный label horizon. Genome, candidates, costs и gates во время outer не меняются.

## Почему обучение пока не запущено

На момент регистрации будущих баров ещё нет. Их hash невозможно честно зафиксировать заранее. До первого расчёта обязателен immutable manifest полного append-only snapshot с SHA-256; смотреть outcomes до фиксации manifest запрещено. Отсутствующие либо пересмотренные строки инвалидируют соответствующий origin. Никаких сетевых загрузок этот контракт не выполняет.

## Измерение

- accounting v2 и одинаковые normal/stress round-trip costs 16/24 bps;
- 30 paired daily observations, qualified HAC small-sample-t diagnostic;
- положительный normal и stress LCB;
- положительный результат минимум в 70% трёхдневных origins;
- отдельные gates для opportunities/ESS и числа сделок;
- абсолютная просадка не выше 10%; сравнение просадки с incumbent применяется только если incumbent действительно совершил минимум пять сделок;
- concentration не выше 40%, turnover не выше 1.25 от EMA baseline;
- model SHA, data SHA, train rows SHA и scaler SHA записываются для каждого решения.

Даже прохождение всех gates означает только допуск к отдельному длинному audit, а не promotion и не live. Провал любого gate сохраняет `NoTrade`.

## Multiple testing

В outer допускается ровно один заранее замороженный overlay. Genetic search, замена challenger, настройка порогов и выбор по outer запрещены. Любое изменение требует нового objective и нового будущего интервала. Registry v5 ссылается на immutable registry v4 и содержит SHA этого контракта.

## Локальная проверка

`prospective_contract_v1.py` детерминированно проверяет временные границы, десять origins, purge, фиксированный candidate set, отсутствие outer search и все fail-closed разрешения. Проверка не читает рыночные данные, не запускает collector и не использует сеть.

`prospective_manifest_v1.py` принимает только локальные parquet-файлы внутри workspace. Он проверяет SHA-256 каждого файла, OHLCV schema, конечность и знак значений, отсутствие дубликатов и полный 4-часовой grid по восьми символам. Manifest создаётся эксклюзивно и привязывается к SHA контракта. Изменение хотя бы одного байта snapshot блокирует preflight.

`prospective_evaluator_v1.py` оценивает ровно один завершённый origin. Он обрезает panel по `origin_end`, заново строит causal features, применяет purge, обучает scaler только на прошлом и исполняет три фиксированных кандидата при normal/stress costs. В ledger каждой строки входят model, snapshot, train rows и scaler SHA. Runner сначала создаёт immutable started-receipt; повторная попытка того же origin запрещена.

CLI `tools/run_genetic_primus_prospective_3d_v1.py` имеет отдельные команды `manifest`, `preflight` и `evaluate`. Все артефакты ограничены каталогом `Reports/Exia/Genetic_Primus/prospective_3d_v1_202609`. Runner проверяет project `.venv`, registry v5, его ancestry и все safety flags. На текущем этапе CLI с рыночными данными не запускался: первый origin ещё не завершён.
