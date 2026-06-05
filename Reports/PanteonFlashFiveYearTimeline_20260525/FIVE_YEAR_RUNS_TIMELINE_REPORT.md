# Panteon Flash: timeline полных 5-летних прогонов

- Сформировано: 2026-06-01 08:26
- Фильтр включения: `executed_years` содержит 2022-2026, `bars_processed >= 38000`, период 2022-01-01 -> 2026-05-18.
- Включено полных 5-летних прогонов: 13.
- Исключено неполных/годовых/H1 прогонов: 22.

## Главная динамика

- Первый полный Flash-прогон: 2026-05-31 09:26:32 — -23.82% PnL, MaxDD 26.17%.
- Лучший полный прогон: 2026-05-31 15:14:44 — 40.71% PnL, MaxDD 4.42%, `PanteonLegend_experiments_dynamic_memory_20260531_soft_symbol_gate_core_full/soft_regime_top3_w144_m20_cap45_cash15_only_core_symbol_gate_full2022_2026`.
- Последний полный прогон: 2026-05-31 18:37:48 — -10.12% PnL, MaxDD 26.31%, `PanteonLegend_latest_build_20260531_symbol_gate_dyn_genetics_full/soft_regime_top3_w144_m20_cap45_cash15_only_core_symbol_gate_dyn_full2022_2026`.
- Прирост от первого к последнему: 13.70 п.п.
- Медиана по всем полным версиям: -2.20% PnL, 23.01% MaxDD.
- Доля версий, где Flash обошёл лучший standalone-компонент: 0.0%.

## Графики

![PnL timeline](Reports/PanteonFlashFiveYearTimeline_20260525/five_year_pnl_timeline.png)

![Risk timeline](Reports/PanteonFlashFiveYearTimeline_20260525/five_year_maxdd_timeline.png)

![Alpha timeline](Reports/PanteonFlashFiveYearTimeline_20260525/five_year_alpha_timeline.png)

## Топ-10 полных 5-летних прогонов

| # | Дата версии | PnL % | MaxDD % | Alpha п.п. | Closed | Best component | Версия |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | 2026-05-31 15:14:44 | 40.71 | 4.42 | -13.08 | 0 | Legend_BearDefense | `PanteonLegend_experiments_dynamic_memory_20260531_soft_symbol_gate_core_full/soft_regime_top3_w144_m20_cap45_cash15_only_core_symbol_gate_full2022_2026` |
| 2 | 2026-05-31 13:55:44 | 22.96 | 10.55 | -30.84 | 0 | Legend_BearDefense | `PanteonLegend_experiments_dynamic_memory_20260531_soft_only_core_full/soft_regime_top3_w144_m20_cap45_cash15_only_core_full2022_2026` |
| 3 | 2026-05-31 14:27:57 | 18.96 | 13.47 | -34.83 | 0 | Legend_BearDefense | `PanteonLegend_experiments_dynamic_memory_20260531_soft_dyn_core_full/soft_regime_top3_w144_m20_cap45_cash15_only_core_dyn_full2022_2026` |
| 4 | 2026-05-31 14:46:58 | 14.25 | 43.40 | -39.55 | 0 | Legend_BearDefense | `PanteonLegend_experiments_dynamic_memory_20260531_soft_fast_core_full/soft_regime_top1_w48_m8_cap100_cash0_only_core_full2022_2026` |
| 5 | 2026-05-31 13:36:54 | 11.61 | 17.74 | -42.19 | 0 | Legend_BearDefense | `PanteonLegend_experiments_dynamic_memory_20260531_soft_only_positive_hard_full/soft_regime_top3_w144_m20_cap45_cash15_only_positive_full2022_2026` |
| 6 | 2026-05-31 14:46:59 | 4.91 | 23.01 | -48.89 | 0 | Legend_BearDefense | `PanteonLegend_experiments_dynamic_memory_20260531_soft_fast_core_full/soft_regime_top2_w72_m10_cap55_cash10_only_core_full2022_2026` |
| 7 | 2026-05-31 12:32:01 | -2.20 | 17.42 | -56.00 | 0 | Legend_BearDefense | `PanteonLegend_experiments_dynamic_memory_20260531_soft_only_full/soft_regime_top3_w144_m20_cap45_cash15_only_full2022_2026` |
| 8 | 2026-05-31 12:58:18 | -2.20 | 17.42 | -56.00 | 0 | Legend_BearDefense | `PanteonLegend_experiments_dynamic_memory_20260531_soft_only_positive_full/soft_regime_top3_w144_m20_cap45_cash15_only_positive_full2022_2026` |
| 9 | 2026-05-31 18:37:48 | -10.12 | 26.31 | -71.00 | 0 | GeneticsBullish | `PanteonLegend_latest_build_20260531_symbol_gate_dyn_genetics_full/soft_regime_top3_w144_m20_cap45_cash15_only_core_symbol_gate_dyn_full2022_2026` |
| 10 | 2026-05-31 09:26:32 | -23.82 | 26.17 | -77.62 | 0 | Legend_BearDefense | `PanteonLegend_latest_dynamic_memory_20260531_full2022_2026` |

## Legacy scaled comparison

- No long legacy runs were found that can be scaled to the full 2022-2026 window.

## Интерпретация

- Ранние версии до stable Flash давали отрицательный или около нуля результат: selection ещё не защищал от плохих клеток и переизбыточной активности.
- Основной скачок появился после shadow-confirmation/cap/attribution-итераций: Flash перестал выбирать большую часть шумовых сигналов.
- Поздние deny-итерации давали рост, но broad-deny был хрупким: часть улучшений переобучалась и не переносилась между 2025/2026.
- Лучшая текущая линия — targeted LCB/terminal-deny + selected-subset positive boosts: она даёт максимум PnL при снижении MaxDD относительно предыдущего лучшего полного прогона.
