# Подробные метрики logistic 10m

13 дней: 1–13 октября 2026 UTC. Описательные показатели раскрытого периода; точность не является торговой доходностью.

## Условное raw-направление

Только фактически ненулевые движения; даже если исходный argmax neutral, направление выбрано по p(up)/(p(up)+p(down)). N указывается отдельно. Интервал accuracy: pointwise 95%, первый frozen seed bank; второй и остальные метрики доступны в JSON. Это не интервал с поправкой на все просмотры.

| Актив | Метод | N | Accuracy, % | 95% interval, % | AUC | Brier | Log-loss |
|---|---|---:|---:|---:|---:|---:|---:|
| ADA/USDT | original_C0p1 | 1760 | 52,90 | 50,94–54,64 | 0,52422 | 0,250791 | 0,694949 |
| ADA/USDT | original_C0p01 | 1760 | 52,95 | 50,95–54,76 | 0,52361 | 0,250773 | 0,694902 |
| ADA/USDT | half_C0p1 | 1760 | 52,16 | 49,89–54,32 | 0,52270 | 0,250761 | 0,694857 |
| ADA/USDT | half_C0p01 | 1760 | 52,27 | 50,14–54,35 | 0,52210 | 0,250751 | 0,694830 |
| ADA/USDT | train_prior | 1760 | 50,17 | 48,25–52,09 | 0,50062 | 0,250006 | 0,693160 |
| ADA/USDT | random_coin | 1760 | 50,97 | 48,89–53,05 | 0,50000 | 0,250000 | 0,693147 |
| BNB/USDT | original_C0p1 | 1858 | 49,57 | 47,42–52,02 | 0,48950 | 0,250890 | 0,694956 |
| BNB/USDT | original_C0p01 | 1858 | 49,46 | 47,34–52,02 | 0,48959 | 0,250861 | 0,694895 |
| BNB/USDT | half_C0p1 | 1858 | 49,52 | 47,71–51,80 | 0,49320 | 0,250615 | 0,694396 |
| BNB/USDT | half_C0p01 | 1858 | 49,35 | 47,48–51,65 | 0,49340 | 0,250600 | 0,694366 |
| BNB/USDT | train_prior | 1858 | 51,35 | 49,65–52,97 | 0,51621 | 0,249934 | 0,693016 |
| BNB/USDT | random_coin | 1858 | 49,52 | 46,11–52,63 | 0,50000 | 0,250000 | 0,693147 |
| BTC/USDT | original_C0p1 | 1871 | 52,32 | 49,68–54,12 | 0,51939 | 0,249731 | 0,692618 |
| BTC/USDT | original_C0p01 | 1871 | 52,32 | 49,84–54,06 | 0,51977 | 0,249709 | 0,692573 |
| BTC/USDT | half_C0p1 | 1871 | 51,47 | 50,00–52,97 | 0,52335 | 0,249584 | 0,692318 |
| BTC/USDT | half_C0p01 | 1871 | 51,31 | 49,41–53,07 | 0,52357 | 0,249573 | 0,692297 |
| BTC/USDT | train_prior | 1871 | 50,24 | 48,18–52,57 | 0,50609 | 0,249997 | 0,693140 |
| BTC/USDT | random_coin | 1871 | 50,19 | 47,38–52,38 | 0,50000 | 0,250000 | 0,693147 |
| DOGE/USDT | original_C0p1 | 1811 | 52,79 | 50,80–54,83 | 0,53591 | 0,249768 | 0,692819 |
| DOGE/USDT | original_C0p01 | 1811 | 52,84 | 50,91–54,83 | 0,53658 | 0,249739 | 0,692754 |
| DOGE/USDT | half_C0p1 | 1811 | 51,68 | 50,00–53,58 | 0,53713 | 0,249643 | 0,692534 |
| DOGE/USDT | half_C0p01 | 1811 | 51,91 | 50,25–53,60 | 0,53729 | 0,249628 | 0,692499 |
| DOGE/USDT | train_prior | 1811 | 50,03 | 48,69–51,54 | 0,49304 | 0,250018 | 0,693184 |
| DOGE/USDT | random_coin | 1811 | 49,03 | 45,81–51,65 | 0,50000 | 0,250000 | 0,693147 |
| ETH/USDT | original_C0p1 | 1871 | 52,27 | 51,42–53,26 | 0,52755 | 0,249857 | 0,692890 |
| ETH/USDT | original_C0p01 | 1871 | 52,59 | 51,74–53,45 | 0,52868 | 0,249831 | 0,692836 |
| ETH/USDT | half_C0p1 | 1871 | 52,11 | 50,16–53,74 | 0,53422 | 0,249669 | 0,692505 |
| ETH/USDT | half_C0p01 | 1871 | 52,32 | 50,32–54,01 | 0,53465 | 0,249657 | 0,692479 |
| ETH/USDT | train_prior | 1871 | 50,88 | 49,63–52,27 | 0,51505 | 0,249926 | 0,692999 |
| ETH/USDT | random_coin | 1871 | 49,33 | 47,41–51,12 | 0,50000 | 0,250000 | 0,693147 |
| LINK/USDT | original_C0p1 | 1839 | 52,09 | 50,49–54,00 | 0,52828 | 0,249992 | 0,693161 |
| LINK/USDT | original_C0p01 | 1839 | 51,77 | 50,43–53,44 | 0,52846 | 0,249968 | 0,693110 |
| LINK/USDT | half_C0p1 | 1839 | 51,88 | 49,73–54,15 | 0,53081 | 0,249654 | 0,692475 |
| LINK/USDT | half_C0p01 | 1839 | 51,88 | 49,62–54,25 | 0,53051 | 0,249646 | 0,692457 |
| LINK/USDT | train_prior | 1839 | 48,99 | 47,88–50,00 | 0,49295 | 0,250173 | 0,693493 |
| LINK/USDT | random_coin | 1839 | 50,08 | 47,56–52,39 | 0,50000 | 0,250000 | 0,693147 |
| SOL/USDT | original_C0p1 | 1870 | 53,48 | 51,20–56,13 | 0,54139 | 0,248998 | 0,691126 |
| SOL/USDT | original_C0p01 | 1870 | 53,10 | 50,40–56,02 | 0,54200 | 0,248981 | 0,691092 |
| SOL/USDT | half_C0p1 | 1870 | 52,83 | 50,05–55,59 | 0,54133 | 0,248996 | 0,691124 |
| SOL/USDT | half_C0p01 | 1870 | 52,89 | 50,08–55,70 | 0,54116 | 0,248990 | 0,691112 |
| SOL/USDT | train_prior | 1870 | 48,07 | 46,71–49,60 | 0,48272 | 0,250072 | 0,693291 |
| SOL/USDT | random_coin | 1870 | 48,50 | 46,76–50,08 | 0,50000 | 0,250000 | 0,693147 |
| XRP/USDT | original_C0p1 | 1825 | 51,34 | 49,01–53,71 | 0,51933 | 0,249869 | 0,692869 |
| XRP/USDT | original_C0p01 | 1825 | 51,73 | 49,70–53,80 | 0,52032 | 0,249841 | 0,692814 |
| XRP/USDT | half_C0p1 | 1825 | 51,84 | 49,92–53,71 | 0,52546 | 0,249617 | 0,692363 |
| XRP/USDT | half_C0p01 | 1825 | 51,29 | 49,37–53,16 | 0,52576 | 0,249604 | 0,692337 |
| XRP/USDT | train_prior | 1825 | 49,32 | 47,63–50,96 | 0,51080 | 0,250011 | 0,693170 |
| XRP/USDT | random_coin | 1825 | 50,85 | 49,18–52,76 | 0,50000 | 0,250000 | 0,693147 |

## Трёхклассовое предсказание на каждом timestamp

Противоположные ошибки = true up & predicted down или true down & predicted up. Два знаменателя: все timestamp и только фактически выданные up/down прогнозы. Neutral-промахи не включаются в противоположные ошибки. Покрытие — доля всех timestamp с направленным argmax. N=1872 на каждую строку.

### Общая truth: ±0,064%

| Актив | Метод | Accuracy, % | Macro-F1 | Neutral truth, % | Вызовы up/down, % | Opposite / all, % | Opposite / calls, % |
|---|---|---:|---:|---:|---:|---:|---:|
| ADA/USDT | original_C0p1 | 40,81 | 0,3830 | 18,16 | 82,96 | 33,81 | 40,76 |
| ADA/USDT | original_C0p01 | 40,65 | 0,3812 | 18,16 | 83,01 | 33,92 | 40,86 |
| ADA/USDT | half_C0p1 | 42,63 | 0,3124 | 18,16 | 100,00 | 39,21 | 39,21 |
| ADA/USDT | half_C0p01 | 42,74 | 0,3132 | 18,16 | 100,00 | 39,10 | 39,10 |
| ADA/USDT | majority_original | 40,44 | 0,1920 | 18,16 | 100,00 | 41,40 | 41,40 |
| ADA/USDT | majority_half | 41,29 | 0,2334 | 18,16 | 100,00 | 40,54 | 40,54 |
| ADA/USDT | random_uniform_3 | 35,68 | 0,3427 | 18,16 | 68,48 | 26,18 | 38,22 |
| ADA/USDT | momentum_original | 36,86 | 0,3374 | 18,16 | 81,84 | 34,08 | 41,64 |
| ADA/USDT | momentum_half | 39,05 | 0,3175 | 18,16 | 94,07 | 39,00 | 41,45 |
| BNB/USDT | original_C0p1 | 50,16 | 0,3042 | 49,52 | 10,47 | 3,53 | 33,67 |
| BNB/USDT | original_C0p01 | 50,27 | 0,3069 | 49,52 | 10,63 | 3,58 | 33,67 |
| BNB/USDT | half_C0p1 | 24,89 | 0,2245 | 49,52 | 98,93 | 25,59 | 25,86 |
| BNB/USDT | half_C0p01 | 24,63 | 0,2209 | 49,52 | 99,31 | 25,91 | 26,09 |
| BNB/USDT | majority_original | 49,52 | 0,2208 | 49,52 | 0,00 | 0,00 | н/д |
| BNB/USDT | majority_half | 25,75 | 0,2281 | 49,52 | 100,00 | 24,73 | 24,73 |
| BNB/USDT | random_uniform_3 | 34,19 | 0,3317 | 49,52 | 65,97 | 16,45 | 24,94 |
| BNB/USDT | momentum_original | 40,44 | 0,3612 | 49,52 | 50,53 | 14,00 | 27,70 |
| BNB/USDT | momentum_half | 33,76 | 0,3331 | 49,52 | 71,96 | 18,80 | 26,13 |
| BTC/USDT | original_C0p1 | 55,72 | 0,3439 | 54,54 | 11,00 | 3,10 | 28,16 |
| BTC/USDT | original_C0p01 | 55,82 | 0,3473 | 54,54 | 11,22 | 3,10 | 27,62 |
| BTC/USDT | half_C0p1 | 32,80 | 0,3276 | 54,54 | 89,48 | 21,15 | 23,64 |
| BTC/USDT | half_C0p01 | 31,57 | 0,3143 | 54,54 | 91,13 | 21,15 | 23,21 |
| BTC/USDT | majority_original | 54,54 | 0,2353 | 54,54 | 0,00 | 0,00 | н/д |
| BTC/USDT | majority_half | 22,70 | 0,1947 | 54,54 | 100,00 | 22,76 | 22,76 |
| BTC/USDT | random_uniform_3 | 31,84 | 0,3059 | 54,54 | 67,25 | 14,64 | 21,76 |
| BTC/USDT | momentum_original | 45,94 | 0,3799 | 54,54 | 45,46 | 13,68 | 30,08 |
| BTC/USDT | momentum_half | 37,66 | 0,3530 | 54,54 | 66,83 | 18,64 | 27,90 |
| DOGE/USDT | original_C0p1 | 39,00 | 0,3770 | 32,96 | 50,53 | 17,15 | 33,93 |
| DOGE/USDT | original_C0p01 | 39,05 | 0,3777 | 32,96 | 50,64 | 17,20 | 33,97 |
| DOGE/USDT | half_C0p1 | 35,68 | 0,2894 | 32,96 | 99,63 | 31,73 | 31,85 |
| DOGE/USDT | half_C0p01 | 35,63 | 0,2876 | 32,96 | 99,73 | 31,68 | 31,76 |
| DOGE/USDT | majority_original | 32,85 | 0,1649 | 32,96 | 100,00 | 34,19 | 34,19 |
| DOGE/USDT | majority_half | 33,28 | 0,2438 | 32,96 | 100,00 | 33,76 | 33,76 |
| DOGE/USDT | random_uniform_3 | 31,94 | 0,3192 | 32,96 | 67,74 | 22,60 | 33,36 |
| DOGE/USDT | momentum_original | 32,64 | 0,3265 | 32,96 | 67,09 | 25,00 | 37,26 |
| DOGE/USDT | momentum_half | 31,30 | 0,3021 | 32,96 | 84,13 | 30,66 | 36,44 |
| ETH/USDT | original_C0p1 | 48,82 | 0,3707 | 46,42 | 23,18 | 7,10 | 30,65 |
| ETH/USDT | original_C0p01 | 48,99 | 0,3728 | 46,42 | 23,24 | 7,10 | 30,57 |
| ETH/USDT | half_C0p1 | 30,29 | 0,2846 | 46,42 | 95,78 | 25,91 | 27,05 |
| ETH/USDT | half_C0p01 | 29,70 | 0,2751 | 46,42 | 96,69 | 25,80 | 26,69 |
| ETH/USDT | majority_original | 46,42 | 0,2114 | 46,42 | 0,00 | 0,00 | н/д |
| ETH/USDT | majority_half | 27,40 | 0,2325 | 46,42 | 100,00 | 26,18 | 26,18 |
| ETH/USDT | random_uniform_3 | 32,48 | 0,3193 | 46,42 | 66,35 | 17,36 | 26,17 |
| ETH/USDT | momentum_original | 39,16 | 0,3556 | 46,42 | 53,58 | 18,00 | 33,60 |
| ETH/USDT | momentum_half | 32,64 | 0,3241 | 46,42 | 74,15 | 22,44 | 30,26 |
| LINK/USDT | original_C0p1 | 37,98 | 0,3771 | 25,69 | 60,15 | 22,97 | 38,19 |
| LINK/USDT | original_C0p01 | 38,03 | 0,3780 | 25,69 | 60,36 | 23,13 | 38,32 |
| LINK/USDT | half_C0p1 | 38,51 | 0,2945 | 25,69 | 100,00 | 35,79 | 35,79 |
| LINK/USDT | half_C0p01 | 38,35 | 0,2932 | 25,69 | 100,00 | 35,95 | 35,95 |
| LINK/USDT | majority_original | 36,70 | 0,1790 | 25,69 | 100,00 | 37,61 | 37,61 |
| LINK/USDT | majority_half | 36,70 | 0,1790 | 25,69 | 100,00 | 37,61 | 37,61 |
| LINK/USDT | random_uniform_3 | 32,26 | 0,3191 | 25,69 | 67,36 | 24,68 | 36,64 |
| LINK/USDT | momentum_original | 33,49 | 0,3299 | 25,69 | 74,25 | 29,81 | 40,14 |
| LINK/USDT | momentum_half | 35,58 | 0,3332 | 25,69 | 86,75 | 34,24 | 39,47 |
| SOL/USDT | original_C0p1 | 42,09 | 0,3901 | 34,62 | 43,48 | 16,35 | 37,59 |
| SOL/USDT | original_C0p01 | 41,99 | 0,3895 | 34,62 | 43,54 | 16,29 | 37,42 |
| SOL/USDT | half_C0p1 | 34,78 | 0,2801 | 34,62 | 100,00 | 30,61 | 30,61 |
| SOL/USDT | half_C0p01 | 34,72 | 0,2798 | 34,62 | 100,00 | 30,66 | 30,66 |
| SOL/USDT | majority_original | 31,89 | 0,1612 | 34,62 | 100,00 | 33,49 | 33,49 |
| SOL/USDT | majority_half | 32,48 | 0,2500 | 34,62 | 100,00 | 32,91 | 32,91 |
| SOL/USDT | random_uniform_3 | 32,59 | 0,3255 | 34,62 | 66,03 | 22,97 | 34,79 |
| SOL/USDT | momentum_original | 32,91 | 0,3283 | 34,62 | 65,38 | 22,33 | 34,15 |
| SOL/USDT | momentum_half | 33,07 | 0,3242 | 34,62 | 81,30 | 27,51 | 33,84 |
| XRP/USDT | original_C0p1 | 41,35 | 0,3776 | 36,54 | 41,40 | 14,16 | 34,19 |
| XRP/USDT | original_C0p01 | 41,45 | 0,3793 | 36,54 | 41,40 | 13,94 | 33,68 |
| XRP/USDT | half_C0p1 | 33,44 | 0,2731 | 36,54 | 99,89 | 30,13 | 30,16 |
| XRP/USDT | half_C0p01 | 32,85 | 0,2672 | 36,54 | 100,00 | 30,61 | 30,61 |
| XRP/USDT | majority_original | 31,41 | 0,2352 | 36,54 | 100,00 | 32,05 | 32,05 |
| XRP/USDT | majority_half | 31,57 | 0,1600 | 36,54 | 100,00 | 31,89 | 31,89 |
| XRP/USDT | random_uniform_3 | 32,96 | 0,3292 | 36,54 | 67,25 | 22,12 | 32,88 |
| XRP/USDT | momentum_original | 35,42 | 0,3516 | 36,54 | 63,46 | 21,10 | 33,25 |
| XRP/USDT | momentum_half | 33,01 | 0,3262 | 36,54 | 80,02 | 26,07 | 32,58 |
### Общая truth: ±0,032%

| Актив | Метод | Accuracy, % | Macro-F1 | Neutral truth, % | Вызовы up/down, % | Opposite / all, % | Opposite / calls, % |
|---|---|---:|---:|---:|---:|---:|---:|
| ADA/USDT | original_C0p1 | 42,31 | 0,3454 | 5,98 | 82,96 | 37,23 | 44,88 |
| ADA/USDT | original_C0p01 | 42,20 | 0,3447 | 5,98 | 83,01 | 37,39 | 45,05 |
| ADA/USDT | half_C0p1 | 49,04 | 0,3365 | 5,98 | 100,00 | 44,98 | 44,98 |
| ADA/USDT | half_C0p01 | 49,15 | 0,3374 | 5,98 | 100,00 | 44,87 | 44,87 |
| ADA/USDT | majority_original | 47,22 | 0,2138 | 5,98 | 100,00 | 46,79 | 46,79 |
| ADA/USDT | majority_half | 47,17 | 0,2543 | 5,98 | 100,00 | 46,85 | 46,85 |
| ADA/USDT | random_uniform_3 | 36,43 | 0,3188 | 5,98 | 68,48 | 30,24 | 44,15 |
| ADA/USDT | momentum_original | 38,89 | 0,3173 | 5,98 | 81,84 | 39,21 | 47,91 |
| ADA/USDT | momentum_half | 43,80 | 0,3314 | 5,98 | 94,07 | 45,14 | 47,98 |
| BNB/USDT | original_C0p1 | 31,14 | 0,2231 | 28,04 | 10,47 | 4,59 | 43,88 |
| BNB/USDT | original_C0p01 | 31,30 | 0,2259 | 28,04 | 10,63 | 4,59 | 43,22 |
| BNB/USDT | half_C0p1 | 35,95 | 0,2861 | 28,04 | 98,93 | 35,68 | 36,07 |
| BNB/USDT | half_C0p01 | 35,68 | 0,2811 | 28,04 | 99,31 | 36,00 | 36,26 |
| BNB/USDT | majority_original | 28,04 | 0,1460 | 28,04 | 0,00 | 0,00 | н/д |
| BNB/USDT | majority_half | 37,07 | 0,2872 | 28,04 | 100,00 | 34,88 | 34,88 |
| BNB/USDT | random_uniform_3 | 34,40 | 0,3429 | 28,04 | 65,97 | 23,61 | 35,79 |
| BNB/USDT | momentum_original | 33,33 | 0,3289 | 28,04 | 50,53 | 18,96 | 37,53 |
| BNB/USDT | momentum_half | 34,78 | 0,3453 | 28,04 | 71,96 | 26,66 | 37,05 |
| BTC/USDT | original_C0p1 | 36,81 | 0,2614 | 33,17 | 11,00 | 4,17 | 37,86 |
| BTC/USDT | original_C0p01 | 37,02 | 0,2650 | 33,17 | 11,22 | 4,17 | 37,14 |
| BTC/USDT | half_C0p1 | 39,58 | 0,3808 | 33,17 | 89,48 | 29,54 | 33,01 |
| BTC/USDT | half_C0p01 | 39,26 | 0,3731 | 33,17 | 91,13 | 29,91 | 32,83 |
| BTC/USDT | majority_original | 33,17 | 0,1661 | 33,17 | 0,00 | 0,00 | н/д |
| BTC/USDT | majority_half | 32,59 | 0,2414 | 33,17 | 100,00 | 34,24 | 34,24 |
| BTC/USDT | random_uniform_3 | 32,64 | 0,3263 | 33,17 | 67,25 | 21,96 | 32,64 |
| BTC/USDT | momentum_original | 37,34 | 0,3509 | 33,17 | 45,46 | 18,22 | 40,07 |
| BTC/USDT | momentum_half | 35,42 | 0,3542 | 33,17 | 66,83 | 26,12 | 39,09 |
| DOGE/USDT | original_C0p1 | 32,26 | 0,3202 | 15,92 | 50,53 | 20,83 | 41,23 |
| DOGE/USDT | original_C0p01 | 32,32 | 0,3207 | 15,92 | 50,64 | 20,89 | 41,24 |
| DOGE/USDT | half_C0p1 | 43,38 | 0,3179 | 15,92 | 99,63 | 40,54 | 40,70 |
| DOGE/USDT | half_C0p01 | 43,48 | 0,3168 | 15,92 | 99,73 | 40,44 | 40,55 |
| DOGE/USDT | majority_original | 41,61 | 0,1959 | 15,92 | 100,00 | 42,47 | 42,47 |
| DOGE/USDT | majority_half | 41,24 | 0,2752 | 15,92 | 100,00 | 42,84 | 42,84 |
| DOGE/USDT | random_uniform_3 | 32,85 | 0,3130 | 15,92 | 67,74 | 28,69 | 42,35 |
| DOGE/USDT | momentum_original | 31,78 | 0,3069 | 15,92 | 67,09 | 30,40 | 45,30 |
| DOGE/USDT | momentum_half | 35,63 | 0,3145 | 15,92 | 84,13 | 37,50 | 44,57 |
| ETH/USDT | original_C0p1 | 33,07 | 0,2916 | 25,85 | 23,18 | 8,17 | 35,25 |
| ETH/USDT | original_C0p01 | 33,23 | 0,2934 | 25,85 | 23,24 | 8,17 | 35,17 |
| ETH/USDT | half_C0p1 | 39,85 | 0,3454 | 25,85 | 95,78 | 34,67 | 36,20 |
| ETH/USDT | half_C0p01 | 39,90 | 0,3402 | 25,85 | 96,69 | 34,78 | 35,97 |
| ETH/USDT | majority_original | 25,85 | 0,1370 | 25,85 | 0,00 | 0,00 | н/д |
| ETH/USDT | majority_half | 37,29 | 0,2761 | 25,85 | 100,00 | 36,86 | 36,86 |
| ETH/USDT | random_uniform_3 | 32,21 | 0,3195 | 25,85 | 66,35 | 24,41 | 36,80 |
| ETH/USDT | momentum_original | 31,62 | 0,3138 | 25,85 | 53,58 | 23,02 | 42,97 |
| ETH/USDT | momentum_half | 33,07 | 0,3277 | 25,85 | 74,15 | 30,82 | 41,57 |
| LINK/USDT | original_C0p1 | 36,27 | 0,3528 | 13,19 | 60,15 | 25,96 | 43,16 |
| LINK/USDT | original_C0p01 | 36,38 | 0,3541 | 13,19 | 60,36 | 26,07 | 43,19 |
| LINK/USDT | half_C0p1 | 45,25 | 0,3228 | 13,19 | 100,00 | 41,56 | 41,56 |
| LINK/USDT | half_C0p01 | 45,09 | 0,3217 | 13,19 | 100,00 | 41,72 | 41,72 |
| LINK/USDT | majority_original | 42,74 | 0,1996 | 13,19 | 100,00 | 44,07 | 44,07 |
| LINK/USDT | majority_half | 42,74 | 0,1996 | 13,19 | 100,00 | 44,07 | 44,07 |
| LINK/USDT | random_uniform_3 | 33,44 | 0,3128 | 13,19 | 67,36 | 28,74 | 42,66 |
| LINK/USDT | momentum_original | 35,04 | 0,3278 | 13,19 | 74,25 | 34,56 | 46,55 |
| LINK/USDT | momentum_half | 39,16 | 0,3457 | 13,19 | 86,75 | 39,53 | 45,57 |
| SOL/USDT | original_C0p1 | 33,71 | 0,3320 | 18,70 | 43,48 | 18,64 | 42,87 |
| SOL/USDT | original_C0p01 | 33,65 | 0,3316 | 18,70 | 43,54 | 18,64 | 42,82 |
| SOL/USDT | half_C0p1 | 43,22 | 0,3173 | 18,70 | 100,00 | 38,09 | 38,09 |
| SOL/USDT | half_C0p01 | 43,16 | 0,3170 | 18,70 | 100,00 | 38,14 | 38,14 |
| SOL/USDT | majority_original | 40,71 | 0,1929 | 18,70 | 100,00 | 40,60 | 40,60 |
| SOL/USDT | majority_half | 40,28 | 0,2856 | 18,70 | 100,00 | 41,03 | 41,03 |
| SOL/USDT | random_uniform_3 | 31,73 | 0,3105 | 18,70 | 66,03 | 28,53 | 43,20 |
| SOL/USDT | momentum_original | 32,21 | 0,3164 | 18,70 | 65,38 | 28,37 | 43,38 |
| SOL/USDT | momentum_half | 35,74 | 0,3321 | 18,70 | 81,30 | 34,99 | 43,04 |
| XRP/USDT | original_C0p1 | 32,91 | 0,3231 | 19,93 | 41,40 | 17,09 | 41,29 |
| XRP/USDT | original_C0p01 | 33,12 | 0,3255 | 19,93 | 41,40 | 16,88 | 40,77 |
| XRP/USDT | half_C0p1 | 42,31 | 0,3143 | 19,93 | 99,89 | 37,77 | 37,81 |
| XRP/USDT | half_C0p01 | 41,67 | 0,3081 | 19,93 | 100,00 | 38,41 | 38,41 |
| XRP/USDT | majority_original | 39,42 | 0,2691 | 19,93 | 100,00 | 40,65 | 40,65 |
| XRP/USDT | majority_half | 39,37 | 0,1883 | 19,93 | 100,00 | 40,71 | 40,71 |
| XRP/USDT | random_uniform_3 | 34,72 | 0,3408 | 19,93 | 67,25 | 27,35 | 40,67 |
| XRP/USDT | momentum_original | 33,12 | 0,3273 | 19,93 | 63,46 | 26,01 | 40,99 |
| XRP/USDT | momentum_half | 35,95 | 0,3360 | 19,93 | 80,02 | 32,91 | 41,12 |
### Общая truth: строгий raw-zero

| Актив | Метод | Accuracy, % | Macro-F1 | Neutral truth, % | Вызовы up/down, % | Opposite / all, % | Opposite / calls, % |
|---|---|---:|---:|---:|---:|---:|---:|
| ADA/USDT | original_C0p1 | 42,31 | 0,3454 | 5,98 | 82,96 | 37,23 | 44,88 |
| ADA/USDT | original_C0p01 | 42,20 | 0,3447 | 5,98 | 83,01 | 37,39 | 45,05 |
| ADA/USDT | half_C0p1 | 49,04 | 0,3365 | 5,98 | 100,00 | 44,98 | 44,98 |
| ADA/USDT | half_C0p01 | 49,15 | 0,3374 | 5,98 | 100,00 | 44,87 | 44,87 |
| ADA/USDT | majority_original | 47,22 | 0,2138 | 5,98 | 100,00 | 46,79 | 46,79 |
| ADA/USDT | majority_half | 47,17 | 0,2543 | 5,98 | 100,00 | 46,85 | 46,85 |
| ADA/USDT | random_uniform_3 | 36,43 | 0,3188 | 5,98 | 68,48 | 30,24 | 44,15 |
| ADA/USDT | momentum_original | 38,89 | 0,3173 | 5,98 | 81,84 | 39,21 | 47,91 |
| ADA/USDT | momentum_half | 43,80 | 0,3314 | 5,98 | 94,07 | 45,14 | 47,98 |
| BNB/USDT | original_C0p1 | 5,61 | 0,0647 | 0,75 | 10,47 | 5,40 | 51,53 |
| BNB/USDT | original_C0p01 | 5,82 | 0,0674 | 0,75 | 10,63 | 5,34 | 50,25 |
| BNB/USDT | half_C0p1 | 48,72 | 0,3278 | 0,75 | 98,93 | 49,47 | 50,00 |
| BNB/USDT | half_C0p01 | 48,72 | 0,3271 | 0,75 | 99,31 | 49,84 | 50,19 |
| BNB/USDT | majority_original | 0,75 | 0,0049 | 0,75 | 0,00 | 0,00 | н/д |
| BNB/USDT | majority_half | 50,96 | 0,3404 | 0,75 | 100,00 | 48,29 | 48,29 |
| BNB/USDT | random_uniform_3 | 33,23 | 0,2708 | 0,75 | 65,97 | 32,53 | 49,31 |
| BNB/USDT | momentum_original | 25,59 | 0,2299 | 0,75 | 50,53 | 25,16 | 49,79 |
| BNB/USDT | momentum_half | 35,68 | 0,2820 | 0,75 | 71,96 | 36,06 | 50,11 |
| BTC/USDT | original_C0p1 | 6,41 | 0,0762 | 0,05 | 11,00 | 4,65 | 42,23 |
| BTC/USDT | original_C0p01 | 6,62 | 0,0785 | 0,05 | 11,22 | 4,65 | 41,43 |
| BTC/USDT | half_C0p1 | 46,58 | 0,3309 | 0,05 | 89,48 | 42,95 | 48,00 |
| BTC/USDT | half_C0p01 | 47,28 | 0,3335 | 0,05 | 91,13 | 43,91 | 48,18 |
| BTC/USDT | majority_original | 0,05 | 0,0004 | 0,05 | 0,00 | 0,00 | н/д |
| BTC/USDT | majority_half | 49,25 | 0,3015 | 0,05 | 100,00 | 50,69 | 50,69 |
| BTC/USDT | random_uniform_3 | 33,81 | 0,2696 | 0,05 | 67,25 | 33,39 | 49,64 |
| BTC/USDT | momentum_original | 21,69 | 0,1990 | 0,05 | 45,46 | 23,82 | 52,41 |
| BTC/USDT | momentum_half | 31,52 | 0,2526 | 0,05 | 66,83 | 35,36 | 52,92 |
| DOGE/USDT | original_C0p1 | 27,88 | 0,2569 | 3,26 | 50,53 | 22,92 | 45,35 |
| DOGE/USDT | original_C0p01 | 27,94 | 0,2573 | 3,26 | 50,64 | 22,97 | 45,36 |
| DOGE/USDT | half_C0p1 | 49,95 | 0,3481 | 3,26 | 99,63 | 46,53 | 46,70 |
| DOGE/USDT | half_C0p01 | 50,16 | 0,3497 | 3,26 | 99,73 | 46,42 | 46,55 |
| DOGE/USDT | majority_original | 48,08 | 0,2165 | 3,26 | 100,00 | 48,66 | 48,66 |
| DOGE/USDT | majority_half | 47,81 | 0,2983 | 3,26 | 100,00 | 48,93 | 48,93 |
| DOGE/USDT | random_uniform_3 | 33,55 | 0,2822 | 3,26 | 67,74 | 32,85 | 48,50 |
| DOGE/USDT | momentum_original | 31,89 | 0,2722 | 3,26 | 67,09 | 34,40 | 51,27 |
| DOGE/USDT | momentum_half | 39,53 | 0,3076 | 3,26 | 84,13 | 42,52 | 50,54 |
| ETH/USDT | original_C0p1 | 13,03 | 0,1390 | 0,05 | 23,18 | 10,20 | 44,01 |
| ETH/USDT | original_C0p01 | 13,03 | 0,1389 | 0,05 | 23,24 | 10,26 | 44,14 |
| ETH/USDT | half_C0p1 | 50,16 | 0,3416 | 0,05 | 95,78 | 45,57 | 47,57 |
| ETH/USDT | half_C0p01 | 50,75 | 0,3440 | 0,05 | 96,69 | 45,89 | 47,46 |
| ETH/USDT | majority_original | 0,05 | 0,0004 | 0,05 | 0,00 | 0,00 | н/д |
| ETH/USDT | majority_half | 50,91 | 0,3264 | 0,05 | 100,00 | 49,04 | 49,04 |
| ETH/USDT | random_uniform_3 | 32,75 | 0,2632 | 0,05 | 66,35 | 33,65 | 50,72 |
| ETH/USDT | momentum_original | 24,09 | 0,2095 | 0,05 | 53,58 | 29,54 | 55,13 |
| ETH/USDT | momentum_half | 33,92 | 0,2598 | 0,05 | 74,15 | 40,17 | 54,18 |
| LINK/USDT | original_C0p1 | 31,84 | 0,2720 | 1,76 | 60,15 | 28,26 | 46,98 |
| LINK/USDT | original_C0p01 | 31,94 | 0,2729 | 1,76 | 60,36 | 28,37 | 46,99 |
| LINK/USDT | half_C0p1 | 50,96 | 0,3428 | 1,76 | 100,00 | 47,28 | 47,28 |
| LINK/USDT | half_C0p01 | 50,96 | 0,3428 | 1,76 | 100,00 | 47,28 | 47,28 |
| LINK/USDT | majority_original | 48,13 | 0,2166 | 1,76 | 100,00 | 50,11 | 50,11 |
| LINK/USDT | majority_half | 48,13 | 0,2166 | 1,76 | 100,00 | 50,11 | 50,11 |
| LINK/USDT | random_uniform_3 | 33,39 | 0,2742 | 1,76 | 67,36 | 33,17 | 49,25 |
| LINK/USDT | momentum_original | 34,99 | 0,2802 | 1,76 | 74,25 | 38,68 | 52,09 |
| LINK/USDT | momentum_half | 41,40 | 0,3123 | 1,76 | 86,75 | 44,34 | 51,11 |
| SOL/USDT | original_C0p1 | 23,08 | 0,2112 | 0,11 | 43,48 | 20,41 | 46,93 |
| SOL/USDT | original_C0p01 | 23,02 | 0,2108 | 0,11 | 43,54 | 20,51 | 47,12 |
| SOL/USDT | half_C0p1 | 52,78 | 0,3513 | 0,11 | 100,00 | 47,12 | 47,12 |
| SOL/USDT | half_C0p01 | 52,83 | 0,3517 | 0,11 | 100,00 | 47,06 | 47,06 |
| SOL/USDT | majority_original | 50,96 | 0,2251 | 0,11 | 100,00 | 48,93 | 48,93 |
| SOL/USDT | majority_half | 48,88 | 0,3143 | 0,11 | 100,00 | 51,01 | 51,01 |
| SOL/USDT | random_uniform_3 | 31,41 | 0,2524 | 0,11 | 66,03 | 34,51 | 52,27 |
| SOL/USDT | momentum_original | 31,30 | 0,2531 | 0,11 | 65,38 | 34,08 | 52,12 |
| SOL/USDT | momentum_half | 39,26 | 0,2890 | 0,11 | 81,30 | 41,93 | 51,58 |
| XRP/USDT | original_C0p1 | 22,86 | 0,2196 | 2,51 | 41,40 | 19,76 | 47,74 |
| XRP/USDT | original_C0p01 | 23,08 | 0,2217 | 2,51 | 41,40 | 19,55 | 47,23 |
| XRP/USDT | half_C0p1 | 50,53 | 0,3408 | 2,51 | 99,89 | 46,85 | 46,90 |
| XRP/USDT | half_C0p01 | 50,00 | 0,3371 | 2,51 | 100,00 | 47,49 | 47,49 |
| XRP/USDT | majority_original | 48,08 | 0,2995 | 2,51 | 100,00 | 49,41 | 49,41 |
| XRP/USDT | majority_half | 48,08 | 0,2165 | 2,51 | 100,00 | 49,41 | 49,41 |
| XRP/USDT | random_uniform_3 | 33,97 | 0,2859 | 2,51 | 67,25 | 32,80 | 48,77 |
| XRP/USDT | momentum_original | 31,52 | 0,2696 | 2,51 | 63,46 | 31,36 | 49,41 |
| XRP/USDT | momentum_half | 39,10 | 0,3079 | 2,51 | 80,02 | 39,69 | 49,60 |

## Парные proper-loss сравнения

Положительная добавка лучше. Обе LCB одновременные по 64 сравнениям, но не по всей адаптивной истории исследования.

| Актив | Challenger | Контроль | Loss | Добавка | LCB bank 1 | LCB bank 2 |
|---|---|---|---|---:|---:|---:|
| ADA/USDT | original_C0p01 | original_C0p1 | brier | 0,0000178 | -0,0000059 | -0,0000053 |
| ADA/USDT | original_C0p01 | original_C0p1 | log_loss | 0,0000471 | -0,0000109 | -0,0000086 |
| ADA/USDT | original_C0p01 | train_prior | brier | -0,0007664 | -0,0024592 | -0,0023802 |
| ADA/USDT | original_C0p01 | train_prior | log_loss | -0,0017415 | -0,0053278 | -0,0051530 |
| ADA/USDT | half_C0p01 | half_C0p1 | brier | 0,0000097 | -0,0000070 | -0,0000070 |
| ADA/USDT | half_C0p01 | half_C0p1 | log_loss | 0,0000276 | -0,0000123 | -0,0000115 |
| ADA/USDT | half_C0p01 | train_prior | brier | -0,0007447 | -0,0023738 | -0,0022975 |
| ADA/USDT | half_C0p01 | train_prior | log_loss | -0,0016694 | -0,0051210 | -0,0049520 |
| BNB/USDT | original_C0p01 | original_C0p1 | brier | 0,0000293 | -0,0000080 | -0,0000063 |
| BNB/USDT | original_C0p01 | original_C0p1 | log_loss | 0,0000606 | -0,0000166 | -0,0000129 |
| BNB/USDT | original_C0p01 | train_prior | brier | -0,0009268 | -0,0023374 | -0,0022729 |
| BNB/USDT | original_C0p01 | train_prior | log_loss | -0,0018795 | -0,0047322 | -0,0045998 |
| BNB/USDT | half_C0p01 | half_C0p1 | brier | 0,0000148 | -0,0000062 | -0,0000052 |
| BNB/USDT | half_C0p01 | half_C0p1 | log_loss | 0,0000308 | -0,0000124 | -0,0000103 |
| BNB/USDT | half_C0p01 | train_prior | brier | -0,0006658 | -0,0018510 | -0,0017940 |
| BNB/USDT | half_C0p01 | train_prior | log_loss | -0,0013497 | -0,0037421 | -0,0036259 |
| BTC/USDT | original_C0p01 | original_C0p1 | brier | 0,0000221 | -0,0000056 | -0,0000047 |
| BTC/USDT | original_C0p01 | original_C0p1 | log_loss | 0,0000451 | -0,0000112 | -0,0000092 |
| BTC/USDT | original_C0p01 | train_prior | brier | 0,0002874 | -0,0001792 | -0,0001525 |
| BTC/USDT | original_C0p01 | train_prior | log_loss | 0,0005678 | -0,0003610 | -0,0003078 |
| BTC/USDT | half_C0p01 | half_C0p1 | brier | 0,0000104 | -0,0000076 | -0,0000070 |
| BTC/USDT | half_C0p01 | half_C0p1 | log_loss | 0,0000211 | -0,0000149 | -0,0000138 |
| BTC/USDT | half_C0p01 | train_prior | brier | 0,0004233 | -0,0003083 | -0,0003001 |
| BTC/USDT | half_C0p01 | train_prior | log_loss | 0,0008430 | -0,0006169 | -0,0006009 |
| DOGE/USDT | original_C0p01 | original_C0p1 | brier | 0,0000291 | 0,0000063 | 0,0000081 |
| DOGE/USDT | original_C0p01 | original_C0p1 | log_loss | 0,0000652 | 0,0000110 | 0,0000155 |
| DOGE/USDT | original_C0p01 | train_prior | brier | 0,0002789 | -0,0011287 | -0,0010618 |
| DOGE/USDT | original_C0p01 | train_prior | log_loss | 0,0004295 | -0,0025323 | -0,0023858 |
| DOGE/USDT | half_C0p01 | half_C0p1 | brier | 0,0000149 | -0,0000017 | -0,0000005 |
| DOGE/USDT | half_C0p01 | half_C0p1 | log_loss | 0,0000341 | -0,0000037 | -0,0000008 |
| DOGE/USDT | half_C0p01 | train_prior | brier | 0,0003901 | -0,0008836 | -0,0008301 |
| DOGE/USDT | half_C0p01 | train_prior | log_loss | 0,0006842 | -0,0019634 | -0,0018484 |
| ETH/USDT | original_C0p01 | original_C0p1 | brier | 0,0000259 | -0,0000022 | -0,0000008 |
| ETH/USDT | original_C0p01 | original_C0p1 | log_loss | 0,0000540 | -0,0000065 | -0,0000032 |
| ETH/USDT | original_C0p01 | train_prior | brier | 0,0000944 | -0,0011674 | -0,0011102 |
| ETH/USDT | original_C0p01 | train_prior | log_loss | 0,0001626 | -0,0023999 | -0,0022842 |
| ETH/USDT | half_C0p01 | half_C0p1 | brier | 0,0000120 | -0,0000072 | -0,0000062 |
| ETH/USDT | half_C0p01 | half_C0p1 | log_loss | 0,0000253 | -0,0000153 | -0,0000131 |
| ETH/USDT | half_C0p01 | train_prior | brier | 0,0002689 | -0,0006661 | -0,0006065 |
| ETH/USDT | half_C0p01 | train_prior | log_loss | 0,0005196 | -0,0013794 | -0,0012596 |
| LINK/USDT | original_C0p01 | original_C0p1 | brier | 0,0000240 | -0,0000189 | -0,0000153 |
| LINK/USDT | original_C0p01 | original_C0p1 | log_loss | 0,0000511 | -0,0000418 | -0,0000338 |
| LINK/USDT | original_C0p01 | train_prior | brier | 0,0002053 | -0,0010692 | -0,0009967 |
| LINK/USDT | original_C0p01 | train_prior | log_loss | 0,0003824 | -0,0022458 | -0,0020942 |
| LINK/USDT | half_C0p01 | half_C0p1 | brier | 0,0000083 | -0,0000173 | -0,0000152 |
| LINK/USDT | half_C0p01 | half_C0p1 | log_loss | 0,0000184 | -0,0000372 | -0,0000325 |
| LINK/USDT | half_C0p01 | train_prior | brier | 0,0005268 | -0,0005474 | -0,0004887 |
| LINK/USDT | half_C0p01 | train_prior | log_loss | 0,0010362 | -0,0011632 | -0,0010420 |
| SOL/USDT | original_C0p01 | original_C0p1 | brier | 0,0000170 | -0,0000220 | -0,0000194 |
| SOL/USDT | original_C0p01 | original_C0p1 | log_loss | 0,0000343 | -0,0000433 | -0,0000382 |
| SOL/USDT | original_C0p01 | train_prior | brier | 0,0010903 | -0,0002049 | -0,0001595 |
| SOL/USDT | original_C0p01 | train_prior | log_loss | 0,0021985 | -0,0003788 | -0,0002898 |
| SOL/USDT | half_C0p01 | half_C0p1 | brier | 0,0000056 | -0,0000171 | -0,0000158 |
| SOL/USDT | half_C0p01 | half_C0p1 | log_loss | 0,0000113 | -0,0000335 | -0,0000310 |
| SOL/USDT | half_C0p01 | train_prior | brier | 0,0010813 | 0,0001264 | 0,0001577 |
| SOL/USDT | half_C0p01 | train_prior | log_loss | 0,0021782 | 0,0002757 | 0,0003383 |
| XRP/USDT | original_C0p01 | original_C0p1 | brier | 0,0000275 | -0,0000062 | -0,0000045 |
| XRP/USDT | original_C0p01 | original_C0p1 | log_loss | 0,0000554 | -0,0000147 | -0,0000111 |
| XRP/USDT | original_C0p01 | train_prior | brier | 0,0001698 | -0,0013556 | -0,0012890 |
| XRP/USDT | original_C0p01 | train_prior | log_loss | 0,0003561 | -0,0027231 | -0,0025904 |
| XRP/USDT | half_C0p01 | half_C0p1 | brier | 0,0000134 | -0,0000081 | -0,0000072 |
| XRP/USDT | half_C0p01 | half_C0p1 | log_loss | 0,0000267 | -0,0000174 | -0,0000155 |
| XRP/USDT | half_C0p01 | train_prior | brier | 0,0004076 | -0,0009020 | -0,0008531 |
| XRP/USDT | half_C0p01 | train_prior | log_loss | 0,0008330 | -0,0018026 | -0,0017055 |

## Изменение нормы коэффициентов (без intercept)

| Origin | Модель | L2 до | L2 после | Итераций |
|---:|---|---:|---:|---:|
| 0 | original_C0p01 | 0,471003 | 0,460584 | 24 |
| 0 | half_C0p01 | 0,527345 | 0,503951 | 31 |
| 1 | original_C0p01 | 0,521238 | 0,508606 | 26 |
| 1 | half_C0p01 | 0,564666 | 0,540657 | 33 |
| 2 | original_C0p01 | 0,506791 | 0,494920 | 26 |
| 2 | half_C0p01 | 0,535884 | 0,514715 | 33 |
| 3 | original_C0p01 | 0,525299 | 0,511312 | 26 |
| 3 | half_C0p01 | 0,548924 | 0,524236 | 36 |
| 4 | original_C0p01 | 0,527715 | 0,515204 | 25 |
| 4 | half_C0p01 | 0,553687 | 0,532312 | 33 |
