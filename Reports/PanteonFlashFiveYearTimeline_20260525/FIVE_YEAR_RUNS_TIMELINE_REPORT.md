# Panteon Flash: timeline полных 5-летних прогонов

- Сформировано: 2026-05-25 23:10
- Фильтр включения: `executed_years` содержит 2022-2026, `bars_processed >= 38000`, период 2022-01-01 -> 2026-05-18.
- Включено полных 5-летних прогонов: 31.
- Исключено неполных/годовых/H1 прогонов: 481.

## Главная динамика

- Первый полный Flash-прогон: 2026-05-19 14:32:38 — -0.62% PnL, MaxDD 1.41%.
- Лучший полный прогон: 2026-05-25 11:09:53 — 106.31% PnL, MaxDD 4.12%, `PanteonFlashSelectedSubsetPredeploy_20260525/candidate_no_weak_cut_full2022_2026`.
- Последний полный прогон: 2026-05-25 16:14:18 — 106.31% PnL, MaxDD 4.12%, `PanteonFlashSelectedSubsetPredeploy_20260525/candidate_no_weak_cut_full2022_2026_final`.
- Прирост от первого к последнему: 106.93 п.п.
- Медиана по всем полным версиям: 8.82% PnL, 5.25% MaxDD.
- Доля версий, где Flash обошёл лучший standalone-компонент: 32.3%.

## Графики

![PnL timeline](Reports/PanteonFlashFiveYearTimeline_20260525/five_year_pnl_timeline.png)

![Risk timeline](Reports/PanteonFlashFiveYearTimeline_20260525/five_year_maxdd_timeline.png)

![Alpha timeline](Reports/PanteonFlashFiveYearTimeline_20260525/five_year_alpha_timeline.png)

## Топ-10 полных 5-летних прогонов

| # | Дата версии | PnL % | MaxDD % | Alpha п.п. | Closed | Best component | Версия |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | 2026-05-25 14:13:29 | 106.31 | 4.12 | 43.97 | 495 | Antonius_conservative | `PanteonFlashSelectedSubsetPredeploy_20260525/candidate_no_weak_cut_full2022_2026_compact_v2` |
| 2 | 2026-05-25 12:11:35 | 106.31 | 4.12 | 43.97 | 495 | Antonius_conservative | `PanteonFlashSelectedSubsetPredeploy_20260525/candidate_no_weak_cut_full2022_2026_compact` |
| 3 | 2026-05-25 11:09:53 | 106.31 | 4.12 | 43.97 | 495 | Antonius_conservative | `PanteonFlashSelectedSubsetPredeploy_20260525/candidate_no_weak_cut_full2022_2026` |
| 4 | 2026-05-25 16:14:18 | 106.31 | 4.12 | 43.97 | 495 | Antonius_conservative | `PanteonFlashSelectedSubsetPredeploy_20260525/candidate_no_weak_cut_full2022_2026_final` |
| 5 | 2026-05-25 07:12:52 | 105.64 | 4.19 | 43.30 | 495 | Antonius_conservative | `PanteonFlashTargetedLcbDenyRegimePullbackTrxFull2022_2026_20260525` |
| 6 | 2026-05-23 17:49:26 | 105.42 | 4.19 | 43.08 | 496 | Antonius_conservative | `PanteonFlashCooldown720ActorRegime72Round3Deny8_Full2022_2026_20260523` |
| 7 | 2026-05-23 21:15:29 | 105.42 | 4.19 | 43.08 | 496 | Antonius_conservative | `PanteonFlashRound3Deny8EntryRegime_Full2022_2026_20260524` |
| 8 | 2026-05-24 09:23:27 | 105.42 | 4.19 | 43.08 | 496 | Antonius_conservative | `PanteonFlashRound3Deny8EntryRegime_RetestFull2022_2026_20260524` |
| 9 | 2026-05-24 00:20:33 | 100.50 | 4.29 | 38.16 | 494 | Antonius_conservative | `PanteonFlashPreLiveSymbolGuard_Full2022_2026_20260524` |
| 10 | 2026-05-23 14:26:41 | 70.68 | 5.25 | 8.34 | 320 | Antonius_conservative | `PanteonFlashCooldown720ActorRegime72_Full2022_2026_20260523` |

## Legacy scaled comparison

- Added long legacy runs: 39.
- Scaling rule: `scaled = ((1 + raw_pct / 100) ** (38375 / bars_processed) - 1) * 100`.
- Best scaled legacy run: 2026-05-17 14:14:04 — raw 10.42%, scaled 11.46%, `RetrodateMarket`.
- Latest scaled legacy run: 2026-05-19 07:52:06 — raw -1.91%, scaled -2.09%, `neiro_genetics/RetrodateMarket`.

![Legacy scaled timeline](Reports/PanteonFlashFiveYearTimeline_20260525/legacy_scaled_to_full_window_timeline.png)

| # | Version date | Raw PnL % | Scaled PnL % | Bars | MaxDD raw % | Version |
|---:|---|---:|---:|---:|---:|---|
| 1 | 2026-05-17 16:17:35 | 10.42 | 11.46 | 35063 | 5.02 | `RetrodateMarket` |
| 2 | 2026-05-17 16:32:17 | 10.42 | 11.46 | 35063 | 5.02 | `RetrodateMarket` |
| 3 | 2026-05-17 17:33:13 | 10.42 | 11.46 | 35063 | 5.02 | `RetrodateMarket` |
| 4 | 2026-05-17 15:36:43 | 10.42 | 11.46 | 35063 | 5.02 | `RetrodateMarket` |
| 5 | 2026-05-17 14:14:04 | 10.42 | 11.46 | 35063 | nan | `RetrodateMarket` |
| 6 | 2026-05-17 21:10:18 | 0.00 | 0.00 | 35063 | 0.00 | `RetrodateMarket` |
| 7 | 2026-05-18 23:16:18 | -0.31 | -0.34 | 35063 | 1.90 | `neiro_genetics/RetrodateMarket` |
| 8 | 2026-05-18 14:26:44 | -0.49 | -0.54 | 35063 | 7.99 | `RetrodateMarket` |

## Интерпретация

- Ранние версии до stable Flash давали отрицательный или около нуля результат: selection ещё не защищал от плохих клеток и переизбыточной активности.
- Основной скачок появился после shadow-confirmation/cap/attribution-итераций: Flash перестал выбирать большую часть шумовых сигналов.
- Поздние deny-итерации давали рост, но broad-deny был хрупким: часть улучшений переобучалась и не переносилась между 2025/2026.
- Лучшая текущая линия — targeted LCB/terminal-deny + selected-subset positive boosts: она даёт максимум PnL при снижении MaxDD относительно предыдущего лучшего полного прогона.
