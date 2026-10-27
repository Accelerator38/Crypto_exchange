# Genetic_Primus V3 generalization audit

Status: diagnostic only; V3 remains terminally rejected.

## Feature drift

- High-PSI features: none.
- High/moderate/low counts: 0/0/25.

## Entry/exit counterfactual

| Year | Exit | Trades | Mean | Median | LCB | Hold bars |
|---:|---|---:|---:|---:|---:|---:|
| 2022 | fixed_h1 | 2104 | -2.74 | -5.70 | -17.20 | 1.00 |
| 2022 | fixed_h3 | 2104 | 29.46 | 7.36 | 7.11 | 3.00 |
| 2022 | fixed_h6 | 2104 | 45.19 | 38.62 | 16.00 | 6.00 |
| 2022 | neural_dynamic | 2104 | 81.38 | 49.68 | 56.98 | 3.18 |
| 2022 | regime_degradation_h12 | 2104 | 14.98 | 2.41 | -1.17 | 2.49 |
| 2023 | fixed_h1 | 1376 | 3.52 | 1.28 | -6.41 | 1.00 |
| 2023 | fixed_h3 | 1376 | 14.77 | 11.29 | -2.48 | 3.00 |
| 2023 | fixed_h6 | 1376 | 25.69 | 9.99 | 4.05 | 6.00 |
| 2023 | neural_dynamic | 1376 | 32.97 | 25.39 | 14.57 | 2.87 |
| 2023 | regime_degradation_h12 | 1376 | 8.39 | 7.39 | -6.78 | 2.30 |
| 2024 | fixed_h1 | 1883 | -20.15 | -14.54 | -32.10 | 1.00 |
| 2024 | fixed_h3 | 1883 | -16.38 | -8.09 | -35.72 | 3.00 |
| 2024 | fixed_h6 | 1883 | -4.32 | 16.90 | -30.29 | 6.00 |
| 2024 | neural_dynamic | 1883 | -12.40 | 4.78 | -37.71 | 2.87 |
| 2024 | regime_degradation_h12 | 1883 | -25.24 | -9.11 | -41.53 | 2.46 |

Best 2024 counterfactual by LCB: `fixed_h6`, LCB -30.29 bps.
A nonpositive best LCB means that changing only the exit cannot repair V3 entries.

No OOS, paper or live stage was opened.
