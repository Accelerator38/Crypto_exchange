# Каталог datasets исследовательского архива

Большие datasets локальны и намеренно не помещаются в Git. Git хранит их
manifest SHA, dataset SHA, код построения и компактные результаты. Полный
машинный каталог находится в
`configs/research_archive/archive_manifest_v1.json`.

## Сохранённые данные

| Dataset | Покрытие и объём | Назначение | Статус |
|---|---:|---|---|
| `bitget_futures_history_v3_2022_20260714` | full8, 1h, 317 952 строк, 31.27 MiB | Длинный development/validation/OOS | Канонический source |
| `bitget_1m_discovery_v1_20260615_20260728` | full8, 1m, 506 880 строк, 48.08 MiB | Intraday cadence и быстрый screening | Канонический source |
| `bitget_funding_history_v1_20260727` | full8, 2 160 settlements, 0.06 MiB | Только recent funding diagnostics | Ограниченное покрытие |
| `bitget_data_v3` | 3 sessions, около 48 часов, 30.74 MiB | Spread/depth/execution calibration | Calibration-only |
| `simple_research_reset_v1` | 2 feature tapes и decision tape, 103.69 MiB | Быстрый повтор batch | Производный, воспроизводимый |
| `evidence/bitget_carryflow` | 4 независимых разорванных roots, 0.57 MiB | Отрицательное CarryFlow evidence | Отслеживается Git |

## Integrity identifiers

| Dataset | SHA256 |
|---|---|
| full8 1h source dataset | `f1479f4e3b2d32a51dd588b8c9c241a8ef71d85da33a27d8dfd5b3a18314481e` |
| full8 1m source dataset | `280e4b4c96402342ccdaa46bade4ad23814a38cb0ab995538951f7608ee52796` |
| recent funding dataset | `f97b9ea582a7cd10bd5fccce482b8321c6ce97008bb9b18e52e806e7aea145b1` |
| SimpleResearch 1h tape | `ead9db0169be907f85e143d9672769fbf49044afa315c499808b399ebf05290f` |
| SimpleResearch 1m tape | `bdf9adf0461960dce1c8378fc094fd0281cc4bf06c085ce7fd91a6b4a0c8fb2d` |
| SimpleResearch decision tape | `eeb5562f6d256a3efa662673ab840e64ddbf1f31414215d48095063d9bfbcf95` |

Три microstructure SQLite имеют отдельные hashes в archive manifest. Они не
склеиваются в единую непрерывную торговую сессию.

## Что удалено

Очистка освободила около 13.15 GiB:

| Класс | До очистки | Причина удаления |
|---|---:|---|
| `bitget_data_v1` | 10.04 GiB | Raw event segments; многократный промежуточный слой |
| `bitget_data_v2` | 1.41 GiB | Производный dataset store; не нужен текущему batch |
| `logs` | 1.03 GiB | Runtime output без promotion authority |
| `Results` | 0.58 GiB | Старые runs и непромоутированные model checkpoints |
| `panteon_v2_state` | 0.05 GiB | Stale runtime state |
| старые CarryFlow campaigns/tapes | около 0.003 GiB | Разорванные и уже сведённые в tracked reports roots |
| старые MEXC/Binance datasets | около 0.027 GiB | Вне Bitget research scope |
| generated reports/ledgers/caches | около 0.06 GiB | Воспроизводимые промежуточные артефакты |

Также удалены generated Freqtrade SQLite/logs, Python/test caches и все
checkpoint extensions вне виртуальных окружений. Единственный tracked
`agent_meta.py.truncated_backup` удалён из Git.

Не удалялись `.venv`, `.venv-freqtrade`, локальные secrets/settings и исходные
отчёты пользователя.

## Правила дальнейших данных

1. Один source dataset получает immutable integrity manifest.
2. Производные tapes должны ссылаться на source SHA и быть воспроизводимыми.
3. Runtime logs и ledgers не считаются архивным evidence без отдельного summary.
4. Разорванные prospective roots никогда не склеиваются.
5. Raw microstructure хранится только до построения проверенного compact layer.
6. Git хранит небольшие manifests/reports, но не большие локальные datasets.
