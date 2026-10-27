# Общая модель против отдельных моделей валют: прогноз на 10 минут

Отчёт от 12 октября 2026. Только направление close → следующий close. Прибыль и исполнение не рассчитывались.

## 1. Результат и область применимости

Выполнены 336 новых fit: 2 метода × 8 валют × 21 origin. Повторно использованы 42 прежних pooled fit, без нового обучения общих моделей. Отдельное обучение дало положительную описательную добавку в 3 из 16 пар; в обоих календарных месяцах — в 2 из 16. После приблизительной семейной поправки на 16 сравнений нижняя граница добавки положительна на обоих bootstrap banks у 0 пар.

Максимальная accuracy ненулевых среди отдельных моделей: 52,48% — RandomForest ETH/USDT; accuracy по всем интервалам у неё 52,36%. 80% не достигнуто. Это описательный максимум, не независимая оценка выбранного победителя.

Август–сентябрь уже раскрыты. Это адаптивное исследовательское продолжение предыдущего эксперимента, не новый outer/prospective. Audit seed bank меняет bootstrap-пересэмплирование, но не добавляет новых данных. Поправка на текущие 16 пар не компенсирует всю историю исследовательских попыток.

## 2. Данные, контракт и вычислительный путь

[Контракт](E:/Crypto_exchange/configs/genetic_primus_direction_specialization_v1.json) · [Регистрация и hashes](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/direction_specialization_v1_20261012/registration.json) · [Предыдущие данные и протокол](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/direction_multiscale_v1_20261012/registration.json) · [Реестр objectives v13](E:/Crypto_exchange/configs/genetic_primus_objective_registry_v13.json).

Используется неизменённый 10-минутный банк из 15 локальных 1m OHLCV Parquet: ADA, BNB, BTC, DOGE, ETH, LINK, SOL, XRP против USDT. Оценка 2026-08-01 00:00 UTC — 2026-10-01 00:00 UTC, конец исключён. 8 784 интервала на валюту; август 4 464, сентябрь 4 320. Новых скачиваний и сетевых запросов нет.

Сохранены 12 OHLCV-признаков, горизонт 10 минут, порог 0,5, окно train 30 дней, retrain каждые 3 дня, purge 7 native bars (70 минут), прежние train/test time indices и зрелые labels. На каждом origin scaler обучается только на ненулевых train labels; у pooled scaler общий, у asset — отдельный. Нулевая train-доходность исключается одинаково, но все flat тестового периода остаются в основном знаменателе.

Logistic12: 13 параметров, C=0,1, max_iter=120, tol=1e-6. RandomForest: 24 дерева, max_depth=4, min_samples_leaf=128, max_features=0,75, n_jobs=1. Seed=20261012. Параметры точно совпадают с прежними общими моделями; поиска и подбора seed нет.

Одинаковые даты не означают одинаковое число строк: pooled получает данные всех восьми валют, asset — одной. Восемь asset fit также дороже одного pooled fit. Фиксированный абсолютный min_samples_leaf=128 задаёт более сильную относительную регуляризацию на меньшей выборке. Эксперимент измеряет весь эффект такого разделения, включая отдельный scaler и размер выборки; он не изолирует чистую специализацию от этих составляющих.

### Объём обучения по валютам

| Валюта | Минимум nonflat train строк | Максимум |
|---|---:|---:|
| ADA/USDT | 3549 | 3737 |
| BNB/USDT | 3963 | 3988 |
| BTC/USDT | 3978 | 3998 |
| DOGE/USDT | 3801 | 3917 |
| ETH/USDT | 3985 | 3995 |
| LINK/USDT | 3864 | 3950 |
| SOL/USDT | 3982 | 3992 |
| XRP/USDT | 3860 | 3949 |

## 3. Определение метрик

Верно: прогноз вверх при следующем close > текущего, либо вниз при следующем close < текущего. Flat не угадан ни одним бинарным ответом. Accuracy all = correct / all; accuracy nonflat = correct / (up + down). Balanced accuracy = (recall up + recall down) / 2. Coverage=100%, abstention отсутствует. Random — прежняя фиксированная монета 50/50, без выбора удачного seed.

Парная добавка = accuracy asset − accuracy pooled на одних и тех же датах одной валюты. Bootstrap: 5 000 общих круговых календарных 3-day block samples из 61 дня, плюс отдельный audit bank. Для семейного one-sided LCB используется point − q95(max centered standardized bootstrap errors) × bootstrap SE, общая семья 16 пар. Это приблизительный block-bootstrap контроль текущей семьи, не точная конечновыборочная гарантия.

## 4. Все результаты сравнения

### Accuracy по всем интервалам — основной показатель

| Валюта | Метод | Общая, % | Отдельная, % | Добавка, п.п. |
|---|---|---:|---:|---:|
| ADA/USDT | Logistic12 | 47,68 | 47,21 | -0,47 |
| BNB/USDT | Logistic12 | 50,84 | 50,31 | -0,54 |
| BTC/USDT | Logistic12 | 51,63 | 50,96 | -0,67 |
| DOGE/USDT | Logistic12 | 49,94 | 49,60 | -0,34 |
| ETH/USDT | Logistic12 | 51,51 | 50,83 | -0,68 |
| LINK/USDT | Logistic12 | 50,10 | 50,00 | -0,10 |
| SOL/USDT | Logistic12 | 51,23 | 50,17 | -1,06 |
| XRP/USDT | Logistic12 | 49,85 | 49,13 | -0,72 |
| ADA/USDT | RandomForest | 47,28 | 47,02 | -0,26 |
| BNB/USDT | RandomForest | 50,72 | 50,79 | 0,07 |
| BTC/USDT | RandomForest | 51,14 | 51,05 | -0,09 |
| DOGE/USDT | RandomForest | 49,56 | 49,92 | 0,36 |
| ETH/USDT | RandomForest | 51,80 | 52,36 | 0,56 |
| LINK/USDT | RandomForest | 50,25 | 49,58 | -0,67 |
| SOL/USDT | RandomForest | 50,47 | 50,00 | -0,47 |
| XRP/USDT | RandomForest | 49,57 | 49,07 | -0,50 |

### Accuracy среди ненулевых движений — дополнительный показатель

| Валюта | Метод | Общая, % | Отдельная, % | Добавка, п.п. |
|---|---|---:|---:|---:|
| ADA/USDT | Logistic12 | 51,63 | 51,13 | -0,51 |
| BNB/USDT | Logistic12 | 51,13 | 50,60 | -0,54 |
| BTC/USDT | Logistic12 | 51,79 | 51,12 | -0,67 |
| DOGE/USDT | Logistic12 | 51,59 | 51,24 | -0,35 |
| ETH/USDT | Logistic12 | 51,64 | 50,95 | -0,68 |
| LINK/USDT | Logistic12 | 51,10 | 50,99 | -0,10 |
| SOL/USDT | Logistic12 | 51,40 | 50,34 | -1,06 |
| XRP/USDT | Logistic12 | 50,92 | 50,19 | -0,73 |
| ADA/USDT | RandomForest | 51,20 | 50,92 | -0,28 |
| BNB/USDT | RandomForest | 51,01 | 51,08 | 0,07 |
| BTC/USDT | RandomForest | 51,30 | 51,21 | -0,09 |
| DOGE/USDT | RandomForest | 51,19 | 51,57 | 0,38 |
| ETH/USDT | RandomForest | 51,92 | 52,48 | 0,56 |
| LINK/USDT | RandomForest | 51,25 | 50,56 | -0,69 |
| SOL/USDT | RandomForest | 50,63 | 50,17 | -0,47 |
| XRP/USDT | RandomForest | 50,63 | 50,12 | -0,51 |

### Парная устойчивость по основному показателю

| Валюта | Метод | CI95 добавки, п.п. | Family LCB primary, п.п. | Family LCB audit, п.п. | Исправлено / испорчено прогнозов |
|---|---|---:|---:|---:|---:|
| ADA/USDT | Logistic12 | [-1,33; 0,42] | -1,69 | -1,69 | 928 / 969 |
| BNB/USDT | Logistic12 | [-1,46; 0,39] | -1,82 | -1,79 | 1006 / 1053 |
| BTC/USDT | Logistic12 | [-1,59; 0,20] | -1,94 | -1,93 | 992 / 1051 |
| DOGE/USDT | Logistic12 | [-1,33; 0,69] | -1,76 | -1,74 | 707 / 737 |
| ETH/USDT | Logistic12 | [-1,47; 0,01] | -1,72 | -1,71 | 697 / 757 |
| LINK/USDT | Logistic12 | [-0,89; 0,67] | -1,20 | -1,17 | 974 / 983 |
| SOL/USDT | Logistic12 | [-2,02; -0,08] | -2,41 | -2,39 | 992 / 1085 |
| XRP/USDT | Logistic12 | [-1,76; 0,28] | -2,16 | -2,13 | 1075 / 1138 |
| ADA/USDT | RandomForest | [-1,00; 0,48] | -1,28 | -1,25 | 1164 / 1187 |
| BNB/USDT | RandomForest | [-1,37; 1,50] | -1,95 | -1,86 | 1411 / 1405 |
| BTC/USDT | RandomForest | [-1,16; 1,10] | -1,65 | -1,59 | 1296 / 1304 |
| DOGE/USDT | RandomForest | [-0,34; 1,08] | -0,64 | -0,63 | 1155 / 1123 |
| ETH/USDT | RandomForest | [-0,36; 1,49] | -0,72 | -0,73 | 1190 / 1141 |
| LINK/USDT | RandomForest | [-1,78; 0,51] | -2,27 | -2,21 | 1249 / 1308 |
| SOL/USDT | RandomForest | [-1,50; 0,57] | -1,90 | -1,84 | 1309 / 1350 |
| XRP/USDT | RandomForest | [-1,67; 0,66] | -2,10 | -2,09 | 1259 / 1303 |

### Добавка по календарным месяцам

| Валюта | Метод | Август, п.п. | Сентябрь, п.п. |
|---|---|---:|---:|
| ADA/USDT | Logistic12 | -0,49 | -0,44 |
| BNB/USDT | Logistic12 | -0,25 | -0,83 |
| BTC/USDT | Logistic12 | -0,85 | -0,49 |
| DOGE/USDT | Logistic12 | -1,05 | 0,39 |
| ETH/USDT | Logistic12 | -0,13 | -1,25 |
| LINK/USDT | Logistic12 | -0,36 | 0,16 |
| SOL/USDT | Logistic12 | -1,52 | -0,58 |
| XRP/USDT | Logistic12 | -0,85 | -0,58 |
| ADA/USDT | RandomForest | -0,45 | -0,07 |
| BNB/USDT | RandomForest | 0,22 | -0,09 |
| BTC/USDT | RandomForest | 0,38 | -0,58 |
| DOGE/USDT | RandomForest | 0,38 | 0,35 |
| ETH/USDT | RandomForest | 0,43 | 0,69 |
| LINK/USDT | RandomForest | -0,52 | -0,83 |
| SOL/USDT | RandomForest | -0,76 | -0,16 |
| XRP/USDT | RandomForest | -1,28 | 0,30 |

### Простые контроли: accuracy по всем интервалам, %

| Валюта | Random 50/50 | Train-majority валюты | Против последнего знака |
|---|---:|---:|---:|
| ADA/USDT | 46,89 | 46,62 | 47,21 |
| BNB/USDT | 50,52 | 49,46 | 50,17 |
| BTC/USDT | 50,42 | 50,10 | 50,82 |
| DOGE/USDT | 48,55 | 48,06 | 49,72 |
| ETH/USDT | 49,10 | 50,03 | 51,16 |
| LINK/USDT | 49,69 | 48,84 | 50,01 |
| SOL/USDT | 49,84 | 49,70 | 50,75 |
| XRP/USDT | 48,78 | 49,25 | 49,17 |

## 5. Графики

Сопоставление на единой шкале: рост сложности сам по себе не означает рост точности.

![Accuracy по валютам](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/direction_specialization_v1_20261012/accuracy_by_asset.png)

Парная разница показывает размер эффекта и обычный CI; семейные LCB приведены выше.

![Парная добавка](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/direction_specialization_v1_20261012/paired_accuracy_delta.png)

Разные знаки между месяцами означают отсутствие стабильной положительной описательной добавки на обоих блоках.

![Добавка по месяцам](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/direction_specialization_v1_20261012/monthly_accuracy_delta.png)

## 6. Проверки, ограничения и дальнейший шаг

Компиляция и 6 прямых synthetic checks выполнены до обучения; pytest/collector не запускались. Для всех 336 новых моделей проверены SHA модели и прогнозов, train-only scaler, train hashes, зрелость labels/purge, восстановление прогнозов из joblib и совпадение ценовых labels. Logistic probabilities дополнительно вычислены прямо из coefficients/intercept. Для pooled восстановление допускает численную погрешность ≤1e-14 из-за размера BLAS batch; бинарные решения должны совпасть точно. Проверены 281 088 new+pooled probability cells.

Метрики всех 168 строк проверены вторым расчётом через знак изменения цены и матрицу 3 фактических × 2 прогнозных классов. Сохранённые pooled full-period accuracy точно совпали с предыдущим расчётом. Warnings обучения: 0. Таблица TSV проверяется повторным чтением: 168 строк, 27 колонок, UTF-8 BOM, decimal comma, отдельные поля.

Если семейные LCB не положительны, нет основания объявлять отдельное обучение улучшением. Даже положительный текущий LCB — лишь исследовательская зацепка: следующая проверка должна удерживать выборку и даты одинаковыми и проверять эффект ограничения числа train строк, а не сразу расширять архитектуры. После этого — одинаковые wall-clock окна признаков. При появлении устойчивого сигнала его потребуется проверить на действительно новых данных.

80–90% не обещается. Этот опыт не доказывает ни универсальную бесполезность векторов, ни возможность добиться такого уровня на OHLCV. Данные о сделках, стакане и конкретных игроках сюда не включались. Никакие торговые разрешения не изменены: credentials/orders/paper/live/runtime/network/promotion=false.

[Полная статистика для Excel](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/direction_specialization_v1_20261012/statistics_excel_ru.tsv) · [Машинная статистика и bootstrap](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/direction_specialization_v1_20261012/comparison.json) · [Проверка таблицы](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/direction_specialization_v1_20261012/table_verification.json).
