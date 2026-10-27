# Диагностический v1: schema guard, не завершённый аудит данных

7 октября 2026. Prepare успешно выполнил 26 pure checks без parquet read.
File1 совпал по size/SHA, но exact-schema guard отверг physical `source_rows`,
который отсутствует в normalized manifest.columns. Это ошибка строгости auditor,
а не доказательство повреждения market данных. Keys/price values не декодировались.

Completion receipt и per-file verified outputs отсутствуют. Этот root не является
успешной input qualification. Original helper/tool source texts и hashes сохранены
в diagnostic_source_snapshot.json до исправления; preparation не перезаписан.

Проверка продолжена в новом output root offline_input_verification_v2_20261007
с явным разделением physical/normalized schema и единственным разрешённым
integer nonfeature extra source_rows. Все старые законченные artifacts сохранены.
Model fits, network, collector, trades и prospective/sealed reads = 0.
