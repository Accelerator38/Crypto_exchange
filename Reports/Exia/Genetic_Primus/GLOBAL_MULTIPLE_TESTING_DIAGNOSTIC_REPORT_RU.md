# Genetic Primus — глобальный DSR/PBO diagnostic

Дата: 2026-08-27. Статус: `DIAGNOSTIC_ONLY / NOT_PROMOTION_EVIDENCE`.

## Материализованные результаты

Найдены три hash-связанных overlay-результата на одном и том же наборе из шести 3-дневных origins января 2025 года:

- `overlay_3d_v1`: Sharpe `-0.408248`;
- `overlay_3d_v3`: Sharpe `1.014057`;
- `overlay_3d_v4`: Sharpe `0.336089`.

V2 не включён: сопоставимый артефакт, содержащий SHA V2-контракта, не найден. Подстановка похожего запуска запрещена.

## Результат

- Deflated Sharpe Probability V4 при глобальном зарегистрированном trial count 20: `0.002041`.
- CSCV PBO для трёх доступных overlay objectives: `0.0`.
- Материализовано objectives: `3/20`.
- Observations: `6/60` минимально требуемых.
- Отсутствуют 17 исторических outcome-рядов; они учитываются в DSR trial count, но не требуются в PBO, если не входят в заранее зарегистрированный сравнимый subset.

PBO `0.0` не является положительным доказательством: при 3 стратегиях и 6 observations оценка слишком дискретна и статистически недостаточна. DSR показывает, что V4 не сохраняет убедительность после учёта глобальной истории попыток.

## Fail-closed состояние

Не пройдены gates:

- `minimum_observations`;
- `sealed_candidate`;
- `sealed_outer_role`.

Builder запрещает диагностическому результату занять канонический путь `global_multiple_testing_v2.json`. Этот файл отсутствует, следовательно V5 promotion остаётся заблокированным.

## Hashes

- Outcome catalog: `8bf3233218c6abd775352a3d577cb203711709f452d85cbab1b60aecde661717`.
- Diagnostic evidence: `48cb4b81eb42494c7fbaa60fc85209b22faf055eb106d01a093448d87af3de79`.
- Objective registry: `766aed82b8c26b76ebe9a565a53fe565bcceda4a32f7360b76ff77d847de1273`.

## Следующий допустимый шаг

Не пытаться восстанавливать отсутствующие старые ряды приблизительно. Для V5 нужно один раз открыть новый sealed interval и сохранить полный ряд минимум из 60 outer origins. DSR считается с глобальным trial count; PBO сравнивает V5 только с заранее зарегистрированным набором baselines/challengers на тех же origins. До этого shadow, paper и canary остаются запрещены.
