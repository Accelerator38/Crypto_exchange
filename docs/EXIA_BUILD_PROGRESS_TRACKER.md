# Exia: tracker of key build milestones

## Purpose

The tracker records only architecture milestones that changed the tested hypothesis or the evaluation method. It does not add a point for every repeated run. This keeps the chart useful for comparing research progress instead of run noise.

The plotted value is the mean stress-costed result in basis points per closed trade or fixed-horizon observation. A positive mean is not a promotion decision: LCB, median, family-wise correction, coverage and the required evaluation windows remain mandatory gates.

## Regime normalization

| Tracker series | Source regimes |
|---|---|
| Trend Up | `TREND_UP`, `bullish` |
| Trend Down | `TREND_DOWN`, `bearish` |
| Range / Neutral | `RANGE`, `neutral` |

An absent regime result is stored as an empty cell, never as zero. Therefore the chart does not invent performance for a regime that a build did not trade.

## Selected milestones

| Milestone | Main change | Decision |
|---|---|---|
| M01 | First costed 1h trend baseline with separate up/down slices | Fail: negative mean and LCB |
| M02 | Dedicated 1h range reentry candidate | Fail: negative range expectancy |
| M03 | Entries only on range-to-trend transitions | Fail: both trend regimes negative |
| M04 | One 4h full8 market quorum; neutral became NoTrade | Fail: positive mean, negative LCB |
| M05 | Gap-aware ATR and bounded tail exits | Fail: median remained negative |
| M06 | Sparse one-shot events | Fail: negative expectancy and LCB |
| M07 | Direct fixed-horizon feasibility labels | Fail: family-wise LCB and median negative |
| M08 | Independent SOL validation without retuning | Validation fail; close the 4h long-trend branch |

M08 is the current best engineering milestone because it produced a decisive independent validation result and stopped further tuning on a failed branch. Its positive arithmetic mean is driven by a few large winners and is not evidence of a live-ready strategy: median and LCB are strongly negative.

## Reproduction

Build the source snapshot:

```powershell
.venv\Scripts\python.exe tools\build_exia_progress_milestones_v1.py
```

Source definitions are pinned in `configs/exia_progress_milestones_v1.json`. The generated JSON and TSV snapshot, workbook and PNG are stored in `Reports/Exia/build_progress_tracker_v1`. The TSV uses tab-separated columns with one field per spreadsheet cell.

The tracker has no paper, live, order or promotion authority.
