# Диагностический v1 — не финальная версия этого этапа

7 октября 2026. В этой версии вычислены искусственные completed labels;
95 direct checks и 21 artifact check прошли. Однако дополнительный контроль
выявил дефект: training payload SHA зависит от общего snapshot, включающего
поздние deploy OHLCV. Это нарушение причинности **training identity**, хотя
numerical labels/features правильно отсечены по времени.

V1 outputs/receipt и связанные исходники сохранены неизменными. Прошедшие
checks не означают, что v1 можно применять для causal training admission.
Финальное исправление и negative control — в v2:

[Отчёт v2](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/causal_training_labels_v2_20261007/REPORT.md).

Реальных данных, fitting, network/runtime/trading не было. Все authority flags
false, model/market/real-money readiness false. Это диагностический результат,
не новая оценка доходности и не допуск frozen experiment.
