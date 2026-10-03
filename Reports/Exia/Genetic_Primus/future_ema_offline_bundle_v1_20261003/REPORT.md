# Future EMA: offline восстановление пакетов

Дата: 2026-10-03. Реализация и evidence: локальный commit `2c727e6`.

## Итог

Подготовлен локальный комплект шести wheels и выполнено успешное offline-восстановление package target. Исходная `.venv`, Python, контракт и evaluator не заменялись. Новых model fits, genomes, PnL, сети, collector или trading runtime нет.

Проверенные версии: numpy 2.4.2, pandas 3.0.0, pyarrow 25.0.0, python-dateutil 2.9.0.post0, six 1.17.0, tzdata 2025.3. Полный комплект wheels — **50 580 597 bytes**. **3915 payload-файлов** прошли CRC/RECORD SHA256 и побайтное hash-сравнение с исходными установленными файлами. RECORD CSV сам себя не хеширует; это явно исключено из payload-сравнения.

## Поиск и проверка

В wheel cache обнаружились только устаревшие numpy cp38. В Downloads подходящих wheels/installer не найдено. HTTP-v2 cache проверен последовательно: 1882 файла, порции до 160. Найдены три совпадающих кандидата: pyarrow, six, python-dateutil. Legacy cache проверен до нахождения всего набора: первые 1280 из 1779 файлов; найдены pandas, tzdata и numpy. Произвольная десериализация cache не выполнялась. Неподходящие и повреждённые кандидаты не используются.

Из cache файлов извлечён только ZIP payload, без HTTP envelope. Затем каждый архив проверен целиком и скопирован эксклюзивно. Первый build six остановился из-за ошибки переноса source path в конфигурацию; путь исправлен до создания wheel или установки. Это инженерный отказ, не повтор статистического анализа.

Синтетические архивные регрессии: **6 tests, OK**, **0,027 s**. Проверены границы ZIP в legacy envelope, запрет overwrite, RECORD mismatch, опасные пути, несовпадение установленного payload, metadata/compatibility filters и scan budgets.

## Фактическая offline-установка

В новый `isolated_target` пакеты установлены только разрешённым workspace Python и pip с `--isolated --no-index --only-binary=:all: --no-deps --no-compile --require-hashes --no-cache-dir`; exit code 0. Установочный процесс завершён, фоновых процессов не оставлено.

Smoke с тем же Python и `-I -S -B`: **OFFLINE_PACKAGE_TARGET_VERIFIED**. Все шесть imports пришли из нового target; исходные site-packages не были видны. NumPy dot, pandas EWM и Arrow in-memory roundtrip проверены на трёх синтетических строках. Никаких рыночных данных или модельного обучения.

После установки исходный environment checker снова дал **ENVIRONMENT_MATCH**. Эта повторная проверка обоснована установкой в отдельный target; старые модельные/collector тесты не повторялись.

Bundle manifest SHA256: `fc74409c9cadc180dc18e6c7a0fb47befba4b06ff60529aaf6292b6be2c69cef`. Он связывает package evidence, wheel SHA, LF-normalized requirements SHA и restore smoke SHA. Финальная проверка всех этих связей прошла.

## Сохранение и границы

`wheels/` и `isolated_target/` сохранены локально в этом каталоге; бинарные артефакты не добавлены в Git. В Git сохранены code, manifest, package metadata, requirements и smoke evidence. Клон репозитория без отдельно сохранённых wheels не является автономным комплектом восстановления. Инструкция: `docs/GENETIC_PRIMUS_FUTURE_EMA_OFFLINE_BUNDLE_V1.md`.

Проверено восстановление **пакетов на этом ПК с общим Python 3.12.4**, а не независимого интерпретатора/ОС/другой машины. Python DLL/runtime не архивировались; независимая PyPI authenticity и power-loss durability не доказаны. Исходный immutable environment registration не изменён; новые доказательства записаны отдельно.

Следующие существенные release доказательства — независимое сохранение полного комплекта и безопасная проверка восстановления на отдельной машине, затем execution/statistical evidence. Они не заменяются повторением уже успешных тестов. GA, promotion и real trading остаются запрещены. Единственный registered future analysis полного интервала остаётся не ранее **2027-04-03 00:00 UTC**.
