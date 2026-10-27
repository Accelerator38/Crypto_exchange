# Genetic Primus V6 — readiness-отчёт контракта

Дата фиксации: 2026-08-27.

## Статус

V6 подготовлена как новый, существенно отличающийся objective. Обучение не запускалось, sealed interval не открывался, результатов доходности V6 ещё нет. Live-сеть, collector, runtime, ключи и ордера не использовались.

## Зачем нужен V6

V5 на запечатанном 60-origin интервале была окончательно отклонена: 0 допущенных решений, paired audit LCB против NoTrade отрицательный. Поэтому повторный поиск с тем же time-series momentum objective не оправдан.

V6 проверяет другую гипотезу: фиксированный cross-sectional relative-momentum proposal по `return_12`, а генетическая модель имеет только бинарное право разрешить или отклонить возможность. Направление сделки генетика менять не может.

## Зафиксированный контракт

- deployment horizon и retrain cadence: 3 дня;
- purge: 7 баров;
- training window: rolling 730 дней;
- sealed outer interval: `[2026-01-01, 2026-07-01)`, ровно 60 origins;
- состояние интервала: `UNOPENED`;
- proposal: `RelativeMomentumRank_return_12`;
- opportunity threshold: абсолютный feature threshold `|rank| > 0.5`;
- linear binary overlay, не более 4 активных features;
- minimum opportunities: 180;
- minimum trades и trade days: 0, чтобы abstention не принуждал модель торговать;
- recency half-life: 90 дней, uniform anchor: 20%;
- ESS gate: `min(120, 75% фактически наблюдаемых дней)`;
- paired references: NoTrade, relative-momentum baseline, incumbent;
- promotion authority, credentials, orders, paper/live: выключены.

SHA-256 V6 config: `aea9dbb1b3acd926afc1b0509742f870f4fbdf844036e63f25d3375a91b07d93`.

Immutable objective registry v3 содержит 21 objective и связан с предыдущим registry v2. SHA-256 registry v3: `cedcb11b736951be18a618a3d6473b6345e50c54ce2c5e938f4935aa1c2a9a92`.

## Исправления измерительной семантики

1. Порог opportunity теперь имеет явные единицы: `return_bps` либо `absolute_feature`. Это исключает применение bps-порога к нормированному rank-feature.
2. ESS gate зависит от длины реально доступного окна. Для 60-дневного audit threshold равен 45, для длинного окна ограничен 120.
3. Возможности и фактические сделки разделены. Минимум наблюдений остаётся, но принудительного minimum trade count нет.
4. Новый objective зарегистрирован до открытия результатов; интервал, seeds и contract hash фиксированы.

## Выполненные локальные проверки

- компиляция нового V6 contract-test: пройдена;
- 6 чистых V6 contract checks напрямую, без pytest: пройдены;
- 7 прежних V5 contract regressions напрямую: пройдены;
- загрузка V6 protocol и проверка 60 origins: пройдены;
- проверка append-only registry v3 и его hash: пройдена.

Эти проверки подтверждают контракт и вычислительную семантику, но не являются evidence качества стратегии.

## Следующий безопасный этап

Сначала выполнить один engineering smoke на уже раскрытом периоде 2025 года. Его результат нельзя использовать для выбора параметров или promotion. Если ledger, hashes и gates корректны, можно отдельным решением открыть V6 sealed interval и выполнить один заранее зарегистрированный 60-origin прогон без изменения objective по ходу оценки.
