# Genetic Primus V8 Clean — 3-day abstention overlay

Дата: 2026-08-26. Статус: offline engineering only.

## Реализованный contract

Новый sealed protocol: `configs/genetic_primus_overlay_3d_v1.json`.

- timeframe: 4h;
- deployment horizon: 3 дня;
- retrain cadence: 3 дня;
- origins: непрерывные, неперекрывающиеся, по 72 часа;
- label purge: 7 баров / 28 часов;
- architecture: linear;
- 5 islands × 4 genomes × 5 generations;
- random immigrants: 25%;
- top-3/top-5 ensemble audit;
- workers: 1;
- все live/paper/order/credential/promotion-authority flags: false.

Старый weekly experiment contract и его frozen outer6 artifacts не изменены.

## Genetic momentum overlay

Направление задаёт raw `Momentum_return_1`. Momentum opportunity создаётся только при абсолютном 4h move более 16 bps. Genome не может открыть противоположное направление:

- если genome target совпал с momentum — сигнал разрешается;
- если target не совпал — `FLAT`;
- если momentum находится внутри deadband — `FLAT`.

Simulator сохраняет counters:

- `overlay_opportunities`;
- `overlay_allowed`;
- `overlay_abstained`;
- `overlay_allowed_fraction`.

Fitness допускает не более 35% momentum opportunities. Превышение добавляет penalty и gate failure `overlay_action_coverage`.

## Paired incremental fitness

Для каждого genome дневная доходность парно сравнивается с:

- `NoTrade`;
- raw `Momentum_return_1`;
- текущим incumbent, когда он существует.

Используется общий materialized block-bootstrap bank. Для каждой reference рассчитывается LCB разницы `candidate − reference`. Fitness получает penalty за отрицательный LCB, а ranking сначала учитывает число paired failures и худший paired LCB.

Independent audit bank применяет ту же paired-метрику только к пяти island champions.

## Bootstrap promotion gate

Origin 0 больше не получает автоматический promotion только из-за отсутствия incumbent.

Первый challenger активируется только если:

- selection paired-min LCB > 0;
- independent-audit paired-min LCB > 0;
- overlay allowed fraction ≤ 35%.

Если gate не пройден, active policy становится детерминированным flat genome (`NoTrade`). Challenger и его SHA сохраняются отдельно для диагностики. На следующих origins flat genome является обычным incumbent.

## Adaptive multiple-testing control

На origin зарегистрирован верхний бюджет:

`5 islands × 4 genomes × 5 generations = 100 hypotheses`.

Дополнительно:

- fitness уменьшается на Gaussian-maximum selection-bias estimate `SE × sqrt(2 log 100)`;
- audit открывается только для пяти island champions;
- `hypothesis_budget.json` сохраняет параметры, статус `within_registered_budget` и canonical payload SHA;
- budget ledger входит в trainer metadata и model SHA.

## Финальный one-origin smoke

Artifact: `Reports/Exia/Genetic_Primus/overlay_3d_origin1_smoke_20260826_r2`.

- deployment interval: 72 часа;
- completed origins: 1;
- active trades: 0;
- active net: 0 bps;
- challenger trades: 12;
- challenger net: -81.657 bps;
- raw momentum: +471.699 bps;
- raw mean-reversion: -708.984 bps;
- challenger selection paired-min LCB: -15.621 bps/day;
- challenger audit paired-min LCB: -15.739 bps/day;
- overlay allowed fraction: 16.8%;
- coverage penalty: 0;
- adaptive selection-bias penalty: 7.473 bps;
- promotion approved: false;
- active trainer: `deterministic_no_trade_bootstrap/1`;
- hypothesis budget SHA verified;
- orders enabled: false.

Smoke подтверждает корректную fail-closed реакцию: отрицательный challenger не становится active, несмотря на отсутствие genetic incumbent.

## Следующий допустимый эксперимент

Следующий прогон должен быть заранее зарегистрирован как отдельный 3-day outer pilot. Минимальная матрица:

1. raw momentum;
2. current direct `rolling_730_uniform` diagnostic baseline;
3. new momentum-abstention overlay;
4. NoTrade.

Сначала рекомендуется 6 non-overlapping 3-day origins. Расширение допускается только без изменения contract/hash и без подбора thresholds по результатам pilot.
