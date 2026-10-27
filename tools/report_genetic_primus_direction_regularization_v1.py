"""Independent frozen-forecast audit and Russian report; no fitting or network."""
import json
import sys
from pathlib import Path

import numpy as np

import run_genetic_primus_direction_regularization_v1 as run

ROOT = run.ROOT
OUT = run.OUT
SELF = Path(__file__)
ASSETS = ("ADA", "AVAX", "BTC", "DOGE", "ETH", "LINK", "SOL", "XRP")


def close(a, b):
    assert np.allclose(a, b, rtol=1e-11, atol=1e-12, equal_nan=False), "numeric audit mismatch"


def auc(p, y):
    """Mann-Whitney AUC with average ranks for ties, independent of scorer."""
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = (ends - counts + 1 + ends) / 2
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[inv][y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def class_metrics(cm):
    true = cm.sum(axis=1)
    pred = cm.sum(axis=0)
    tp = cm.diagonal()
    recall = np.divide(tp, true, out=np.zeros(3), where=true > 0)
    precision = np.divide(tp, pred, out=np.zeros(3), where=pred > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=precision + recall > 0)
    n = cm.sum()
    den = np.sqrt((n * n - pred @ pred) * (n * n - true @ true))
    mcc = float((tp.sum() * n - pred @ true) / den) if den else None
    return tp.sum() / n, float(f1.mean()), recall, precision, mcc


def checks():
    close(auc(np.array([.1, .5, .5, .9]), np.array([0, 0, 1, 1])), .875)
    close(auc(np.array([.5, .5]), np.array([0, 1])), .5)
    a, f, rec, pre, mcc = class_metrics(np.diag([2, 3, 4]))
    close([a, f, mcc], [1, 1, 1])
    close(rec, np.ones(3))
    close(pre, np.ones(3))
    assert class_metrics(np.array([[0, 2, 0], [0, 3, 0], [0, 4, 0]]))[-1] is None
    return ["hand_AUC_with_ties", "constant_score_AUC", "perfect_three_class_confusion", "constant_classifier_MCC_undefined"]


def load_forecasts():
    """Read each origin separately and verify reused baseline arrays exactly."""
    chunks = []
    for i in range(5):
        folder = OUT / ("origin_%02d" % i)
        m = run.base.read(folder / "ledger.json")
        assert run.base.digest(folder / "ledger.npz") == m["ledger_sha256"]
        with np.load(folder / "ledger.npz", allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        with np.load(run.PARENT / ("origin_%02d/tf_10m.npz" % i), allow_pickle=False) as old:
            for k in ("times", "return_pct", "past_return_pct", "uniform", "train_priors"):
                assert np.array_equal(a[k], old[k]), k
            assert np.array_equal(a["probability"][0], old["probability"][0])
            assert np.array_equal(a["probability"][2], old["probability"][2])
        n = len(a["times"])
        a["train_priors"] = np.repeat(a["train_priors"][:, None], n, axis=1)
        a["raw_train_priors"] = np.repeat(a["raw_train_priors"][None], n, axis=0)
        chunks.append(a)
    f = {}
    for k in ("times", "return_pct", "past_return_pct", "uniform", "probability", "train_priors", "raw_train_priors"):
        f[k] = np.concatenate([a[k] for a in chunks], axis=1 if k in ("probability", "train_priors") else 0)
    return f


def audit():
    assert not (OUT / "statistical_audit.json").exists(), "audit is immutable"
    run.registered()
    import score_genetic_primus_direction_regularization_v2 as scoring
    scoring.registered()
    pins = {}
    for file in ("registration.json", "scoring_registration_v2.json"):
        reg = run.base.read(OUT / file)
        for k, v in reg["source_pins"].items():
            assert k not in pins or pins[k] == v
            pins[k] = v
    for path, sha in pins.items():
        assert run.base.digest(ROOT / path) == sha, path
    model_audit = run.base.read(OUT / "model_audit.json")
    assert model_audit["models"] == 10 and model_audit["warnings"] == 0
    assert model_audit["exact_probability_replay"] and model_audit["identical_scalers_and_folds"]
    f = load_forecasts()
    pc = run.base.config()
    results = run.base.read(OUT / "results.json")
    times = f["times"]
    assert np.array_equal(times, np.arange(pc["test_start_ms"], pc["test_end_exclusive_ms"], 600000))
    days = ((times - pc["test_start_ms"]) // 86400000).astype(int)
    assert np.array_equal(np.bincount(days), np.full(13, 144))
    with np.load(OUT / "daily_metrics.npz", allow_pickle=False) as z:
        stored = {k: z[k] for k in z.files}
    banks = []
    for seed in pc["bootstrap"]["seeds"]:
        with np.load(run.PARENT / ("bootstrap_bank_%s.npz" % seed), allow_pickle=False) as z:
            indices = z["indices"]
            assert indices.shape == (1000, 13)
            banks.append(np.eye(13, dtype=np.int64)[indices].sum(axis=1))
    raw = f["return_pct"]
    p3 = f["probability"]
    denom = p3[..., 0] + p3[..., 2]
    conditional = np.divide(p3[..., 2], denom, out=np.full(denom.shape, .5), where=denom > 0)
    prior3 = f["raw_train_priors"]
    prior = prior3[..., 2] / (prior3[..., 0] + prior3[..., 2])
    p = np.concatenate([conditional, prior[None], np.full((1, *raw.shape), .5)], axis=0)
    direct = np.zeros((8, 6, 13, 4))
    di = {(x["symbol"], x["method"]): x for x in results["direction"]}
    for s, symbol in enumerate(pc["symbols"]):
        valid = raw[:, s] != 0
        y = (raw[valid, s] > 0).astype(int)
        assert 0 < y.sum() < len(y)
        for k, name in enumerate(run.DIRECTION_NAMES):
            pp = p[k, valid, s]
            yy = (f["uniform"][valid, s] > .5) if k == 5 else pp > .5
            loss_b = (pp - y) ** 2
            q = np.clip(pp, 1e-15, 1 - 1e-15)
            loss_l = -(y * np.log(q) + (1 - y) * np.log(1 - q))
            hit = yy == y
            row = di[symbol, name]
            assert row["n"] == len(y)
            close([row[v] for v in ("accuracy", "brier", "log_loss", "auc")],
                  [hit.mean(), loss_b.mean(), loss_l.mean(), auc(pp, y)])
            for day in range(13):
                mask = days[valid] == day
                direct[s, k, day] = [mask.sum(), loss_b[mask].sum(), loss_l[mask].sum(), hit[mask].sum()]
            close(direct[s, k], stored["direction"][s, k])
            for metric, col in (("accuracy", 3), ("brier", 1), ("log_loss", 2)):
                for bank, w in enumerate(banks):
                    values = (w @ direct[s, k, :, col]) / (w @ direct[s, k, :, 0])
                    close(np.quantile(values, [.025, .975]), row["ci95_banks"][metric][bank])
    pred = [np.argmax(prob, axis=-1) - 1 for prob in p3]
    pred.extend(np.argmax(prob, axis=-1) - 1 for prob in f["train_priors"])
    pred.append(np.floor(3 * f["uniform"]).astype(int) - 1)
    for band in (.064, .032):
        past = f["past_return_pct"]
        pred.append(np.where(past > band, 1, np.where(past < -band, -1, 0)))
    ci = {(x["symbol"], x["method"], x["evaluation_band_factor"]): x for x in results["classification"]}
    for t, band in enumerate((.064, .032, 0.)):
        truth = np.where(raw > band, 1, np.where(raw < -band, -1, 0))
        factor = (1., .5, 0.)[t]
        for s, symbol in enumerate(pc["symbols"]):
            for k, name in enumerate(run.CLASS_NAMES):
                cm_days = []
                for day in range(13):
                    loc = days == day
                    cm_days.append(np.bincount((truth[loc, s] + 1) * 3 + pred[k][loc, s] + 1, minlength=9).reshape(3, 3))
                cm_days = np.array(cm_days)
                assert np.array_equal(cm_days, stored["classification"][t, s, k])
                cm = cm_days.sum(axis=0)
                row = ci[symbol, name, factor]
                a, macro_f, rec, pre, mcc = class_metrics(cm)
                close([row[v] for v in ("accuracy", "macro_f1")], [a, macro_f])
                if mcc is None:
                    assert row["mcc"] is None
                else:
                    close(row["mcc"], mcc)
                for label in range(3):
                    if cm[label].sum():
                        close(row["recall_down_neutral_up"][label], rec[label])
                    else:
                        assert row["recall_down_neutral_up"][label] is None
                    if cm[:, label].sum():
                        close(row["precision_down_neutral_up"][label], pre[label])
                    else:
                        assert row["precision_down_neutral_up"][label] is None
                assert row["n"] == 1872
                if k < 4:
                    probs = p3[k, :, s]
                    yidx = truth[:, s] + 1
                    ll = -np.log(np.clip(probs[np.arange(1872), yidx], 1e-15, 1)).mean()
                    bb = ((probs - np.eye(3)[yidx]) ** 2).sum(axis=1).mean()
                    close([row["log_loss"], row["multiclass_brier"]], [ll, bb])
    comps = results["comparisons"]
    obs = []
    boot = [[], []]
    for row in comps:
        s = pc["symbols"].index(row["symbol"])
        k = run.DIRECTION_NAMES.index(row["candidate"])
        j = run.DIRECTION_NAMES.index(row["control"])
        col = {"brier": 1, "log_loss": 2}[row["metric"]]
        delta = direct[s, j, :, col] - direct[s, k, :, col]
        nn = direct[s, k, :, 0]
        value = float(delta.sum() / nn.sum())
        close(value, row["improvement"])
        obs.append(value)
        for b, w in enumerate(banks):
            boot[b].append((w @ delta) / (w @ nn))
    obs = np.array(obs)
    for b in range(2):
        array = np.array(boot[b]).T
        close(array, stored["bank%s" % b])
        se = array.std(axis=0, ddof=1)
        active = se > 1e-12
        critical = max(0., float(np.quantile(((obs[active] - array[:, active]) / se[active]).max(axis=1), .95)))
        lower = obs - critical * se
        close(critical, results["max_t_criticals"][b])
        close(lower, [row["simultaneous_lcb_bank%s" % b] for row in comps])
    for gate in results["gates"]:
        matching = [row for row in comps if row["symbol"] == gate["symbol"] and row["candidate"] == gate["candidate"]]
        assert len(matching) == 4
        assert gate["all_losses_controls_banks_positive"] == all(row["simultaneous_lcb_bank0"] > 0 and row["simultaneous_lcb_bank1"] > 0 for row in matching)
    summary = run.base.read(OUT / "summary.json")
    assert summary["research_gates_passed"] == sum(g["all_losses_controls_banks_positive"] for g in results["gates"])
    assert all(summary[k] is False for k in run.base.reuse.fixed.SAFETY)
    result = dict(pure_checks=checks(), source_pins_verified=len(pins), direction_cells=48,
                  classification_cells=216, paired_comparisons=64, bootstrap_banks=2,
                  bootstrap_draws_per_bank=1000, pointwise_intervals=288, full_days=13,
                  daily_timestamps=144, exact_parent_forecast_reuse=True, independent_auc=True,
                  independent_confusion_metrics=True, independent_losses_and_daily_sums=True,
                  independent_draws_and_family_correction=True, new_fits=0,
                  code_pins={run.base.rel(SELF): run.base.digest(SELF)})
    run.base.write(OUT / "statistical_audit.json", result)
    return result


def fmt(value, digits=2):
    return ("%.*f" % (digits, value)).replace(".", ",")


def report():
    assert not (OUT / "REPORT_RU.md").exists(), "report is immutable"
    audit_data = run.base.read(OUT / "statistical_audit.json")
    assert audit_data["code_pins"][run.base.rel(SELF)] == run.base.digest(SELF)
    d = run.base.read(OUT / "results.json")
    pc = run.base.config()
    symbols = pc["symbols"]
    names = run.DIRECTION_NAMES
    di = {(x["symbol"], x["method"]): x for x in d["direction"]}
    ci = {(x["symbol"], x["method"], x["evaluation_band_factor"]): x for x in d["classification"]}
    stats = {m: {key: float(np.median([di[s, m][key] for s in symbols])) for key in ("accuracy", "auc", "brier", "log_loss")} for m in names}
    norm_rows = []
    for origin in range(5):
        for name in ("original_C0p01", "half_C0p01"):
            meta = run.base.read(OUT / ("origin_%02d/%s.json" % (origin, name)))
            norm_rows.append(dict(origin=origin, method=name, before=meta["baseline_coefficient_l2_norm"], after=meta["coefficient_l2_norm"], fitted_iterations=meta["fitted_iterations"]))
    counts = {}
    for candidate, old in ((names[1], names[0]), (names[3], names[2])):
        counts[candidate] = {}
        for control in (old, "train_prior"):
            counts[candidate][control] = {}
            for metric in ("brier", "log_loss"):
                rr = [r for r in d["comparisons"] if r["candidate"] == candidate and r["control"] == control and r["metric"] == metric]
                counts[candidate][control][metric] = dict(point_positive=sum(r["improvement"] > 0 for r in rr), both_banks_positive=sum(r["simultaneous_lcb_bank0"] > 0 and r["simultaneous_lcb_bank1"] > 0 for r in rr), median_improvement=float(np.median([r["improvement"] for r in rr])))
    run.base.write(OUT / "report_statistics.json", dict(median_across_assets=stats, pair_counts=counts, norms=norm_rows, new_fits=0))
    make_figures(d, symbols)
    text = ["# Logistic 10m: единственная проверка усиления регуляризации", "", "Дата отчёта: 18 октября 2026 года. Статус: раскрытое адаптивное историческое исследование, не новая валидация и не допуск к торгам.", "",
            "## Итог", "",
            "Уменьшение C с 0,1 до 0,01 немного снизило вероятностные ошибки, но не дало подтверждённого улучшения: 0 из 16 задач прошли зарегистрированный совместный gate. Медианная точность условного направления остаётся около 52%, а не 80–90%. Нового сигнала, пригодного для торговли, этот эксперимент не доказывает.", "",
            "## Что изменили и что сохранили", "",
            "Изменена только сила L2-регуляризации: меньшее C означает больший штраф. Архитектура прежняя: pooled multiclass logistic, 12 причинных OHLCV-признаков, 39 номинальных параметров, включая intercept (3 × (12 + 1)); softmax имеет избыточность. Две обучающие разметки: нейтральная полоса ±0,064% и ±0,032%. Веса uniform для всех исходов, включая убытки и neutral. Новые микроструктурные признаки не добавлялись.", "",
            "Период 1–13 октября 2026 включительно по UTC: 13 календарных дней, 1872 десятиминутных timestamp на актив, 8 активов. Origins: 1/4/7/10/13 октября, только причинное обучение, собственный train-only scaler на origin, те же train/cal индексы и purge 7 баров. Окно 30 дней включает 7 дней calibration; это не 30 полных дней estimator-training. Десять базовых моделей C=0,1 не переобучались: прогнозы и контроли переиспользованы побитно.", "",
            "Выполнено ровно 10 новых estimator fits: 5 origins × 2 полосы. Предупреждений 0. Прогноз на каждом допустимом timestamp; здесь нет отбора по уверенности. Цена сравнивается от close до следующего 10m close; используемые timestamps — close-time. Ни сеть, ни торговый evaluator не запускались.", "",
            "## Что именно измеряется", "",
            "Первичная диагностическая задача: только при фактически ненулевом raw-return p(up | down или up) = p(up) / [p(down) + p(up)]. При нулевой сумме используется 0,5. Исключение фактически неизменившихся цен нужно только для этой условной метрики, оно не исполнимо как live-фильтр. Условный знак принудительно считается также там, где исходный трёхклассовый argmax был neutral. Поэтому точность в следующей таблице нельзя выдавать за точность реальных направленных вызовов или full-coverage трёхклассовую accuracy.", "",
            "Для y ∈ {0,1}: Brier = mean[(p−y)²]; log-loss = −mean[y ln p + (1−y) ln(1−p)] с clipping 10⁻¹⁵. Меньше лучше. AUC — ранговое качество без выбора порога. Random coin имеет фиксированный случайный бинарный вызов для accuracy, но честную вероятность 0,5 для proper losses и AUC. Prior — причинная train-only частота raw-направлений на каждом origin/активе, а не большинство итогового теста.", "",
            "### Медианы по восьми активам (не pooled-результат)", "",
            "| Метод | Accuracy, % | AUC | Brier ↓ | Log-loss ↓ |", "|---|---:|---:|---:|---:|"]
    for m in names:
        s = stats[m]
        text.append(f"| {m} | {fmt(100*s['accuracy'])} | {fmt(s['auc'], 5)} | {fmt(s['brier'], 6)} | {fmt(s['log_loss'], 6)} |")
    text += ["", "![Условная точность направления](conditional_accuracy.png)", "", "Подпись: наблюдаемая условная точность; интервалы приведены в подробных таблицах, а не на этом описательном графике. Это не трёхклассовая accuracy.", "",
             "### Поактивные изменения proper loss", "", "Δ = ошибка контроля − ошибка нового C=0,01; положительное значение лучше. Число точечных улучшений не заменяет проверку неопределённости.", "",
             "| Новая модель | Контроль | Brier: лучше / 8 | Log-loss: лучше / 8 | LCB > 0 в обоих банках, Brier / Log-loss |", "|---|---|---:|---:|---:|"]
    for new, controls in counts.items():
        for control, metrics in controls.items():
            bb, ll = metrics["brier"], metrics["log_loss"]
            text.append(f"| {new} | {control} | {bb['point_positive']} | {ll['point_positive']} | {bb['both_banks_positive']} / {ll['both_banks_positive']} |")
    text += ["", "![Изменение Brier с совместными нижними границами](brier_improvements.png)", "",
             "Подпись: точка — наблюдаемая добавка; отрезок идёт до худшей из двух одновременных 95% нижних границ. Семейство фиксировано до fits: 64 сравнения (2 challenger × 2 контроля × 2 loss × 8 активов). Все активы в каждом draw используют общие дни, сохраняя межактивную зависимость.", "",
             "## Полная трёхклассовая задача — отдельно", "",
             "Все 1872 timestamp на актив включены, фактически неизменившиеся цены не удалены. Truth имеет состояния down / neutral / up, границы neutral включены. Оба набора моделей cross-score на ±0,064%, ±0,032% и строго нулевой полосе. Разные ширины neutral нельзя сравнивать по accuracy как одинаковую задачу. Подробнее: DETAILS_RU.md.", "",
             "![Accuracy и противоположные вызовы при одинаковой разметке](three_class_accuracy_errors.png)", "",
             "Подпись: на каждом столбце одна и та же truth ±0,032%, все timestamp; справа считаются только down→up и up→down, а не neutral-ошибки. Чем меньше направленных вызовов, тем механически меньше может быть эта доля: покрытие приводится рядом в подробных таблицах.", "",
             "## Статистическая и инженерная проверка", "",
             "Повторно использованы два заранее замороженных calendar block-bootstrap банка: seeds 2026101426 / 2026101427, 1000 повторов, блок 3 дня, 13 дней. В обеих версиях банк сэмплирует одинаковые дни для всех моделей. Одновременная max-t нижняя граница рассчитана по семейству 64. Gate требует положительной LCB обеих proper losses против соответствующего C=0,1 и train-prior в обоих банках, отдельно для каждого актива и полосы. Прошло 0/16. Это локальный исследовательский gate, не глобальная поправка за всю историю разработки.", "",
             "Отдельный audit пересчитал из frozen-ledgers 48 условных метрик, 216 трёхклассовых ячеек, 64 парных сравнения, все draws обоих банков, 288 pointwise интервалов и max-t. AUC дополнительно проверен ранговой формулой с ties; confusion-derived метрики — независимо. Все 13 дней и 144 timestamp в каждом дне сохранены. SHA источников, folds, scaler, models, ledgers, конфигов и bootstrap банков зафиксированы транзитивно в registration и delivery manifest. Exact-predict/scaler/fold audit выполнен для всех 10 новых моделей.", "",
             "Первый scorer после успешного обучения перезаписывал дневную агрегацию предыдущего origin. Контроль bootstrap поймал нулевой знаменатель и остановил расчёт до записи результатов. Попытка раскрыта в score_failure_disclosure.json, исходный frozen-код не переписан; исправление зарегистрировано scoring_v2 (registry v29). Скорректированный scorer агрегирует объединённые прогнозы один раз, проверяет полный знаменатель и каждый день. Повторных fits из-за исправления не было.", "",
             "## Ограничения и следующий шаг", "",
             "13 дней дают лишь 4 полных трёхдневных блока и остаток: uncertainty/bootstrap при таком количестве блоков сами нестабильны. Исследователь неоднократно видел этот период; второй seed bank не делает его независимым. Измеряется направление цены OHLCV, не прибыль, исполнимость, комиссии или slippage. Сохранённые joblib — исследовательские артефакты, не production-модели.", "",
             "Не расширять сейчас перебор C, порогов и архитектур на этом же октябре. Слабое изменение калибровки не решает нехватку сигнала. Следующее содержательное действие — согласовать и зафиксировать действительно будущий период с полным freeze до его начала; старый DRAFT не является sealed/scheduled. До этого допустим отдельный инженерный аудит причинности и качества признаков без настройки по результату. Если нужен новый источник сигнала (например, order-flow), это отдельный контракт и данные; имеющийся OHLCV его не содержит. Автоскачивание, новое обучение и новый период этой задачей не запускались.", "",
             "Все credentials/orders/paper/live/runtime/network/promotion разрешения остаются false. Реальные деньги и торговый runtime не затронуты.", "",
             "## Проверяемые артефакты", "",
             "- registration.json, scoring_registration_v2.json — предрегистрация и поправка scorer.", "- model_audit.json, statistical_audit.json — точное воспроизведение и независимый пересчёт.", "- results.json, daily_metrics.npz — все ячейки, дневные агрегаты и draws.", "- DETAILS_RU.md, report_statistics.json — поактивные таблицы и числовые данные отчёта.", "- delivery_manifest.json — финальная фиксация файлов; появился после визуальной проверки рисунков.", ""]
    (OUT / "REPORT_RU.md").write_text("\n".join(text), encoding="utf-8")
    details = ["# Подробные метрики logistic 10m", "", "13 дней: 1–13 октября 2026 UTC. Описательные показатели раскрытого периода; точность не является торговой доходностью.", "", "## Условное raw-направление", "", "Только фактически ненулевые движения; даже если исходный argmax neutral, направление выбрано по p(up)/(p(up)+p(down)). N указывается отдельно. Интервал accuracy: pointwise 95%, первый frozen seed bank; второй и остальные метрики доступны в JSON. Это не интервал с поправкой на все просмотры.", "", "| Актив | Метод | N | Accuracy, % | 95% interval, % | AUC | Brier | Log-loss |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for symbol in symbols:
        for m in names:
            row = di[symbol, m]
            lo, hi = row["ci95_banks"]["accuracy"][0]
            details.append(f"| {symbol} | {m} | {row['n']} | {fmt(100*row['accuracy'])} | {fmt(100*lo)}–{fmt(100*hi)} | {fmt(row['auc'], 5)} | {fmt(row['brier'], 6)} | {fmt(row['log_loss'], 6)} |")
    details += ["", "## Трёхклассовое предсказание на каждом timestamp", "", "Противоположные ошибки = true up & predicted down или true down & predicted up. Два знаменателя: все timestamp и только фактически выданные up/down прогнозы. Neutral-промахи не включаются в противоположные ошибки. Покрытие — доля всех timestamp с направленным argmax. N=1872 на каждую строку.", ""]
    for factor, label in ((1., "±0,064%"), (.5, "±0,032%"), (0., "строгий raw-zero")):
        details += [f"### Общая truth: {label}", "", "| Актив | Метод | Accuracy, % | Macro-F1 | Neutral truth, % | Вызовы up/down, % | Opposite / all, % | Opposite / calls, % |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for symbol in symbols:
            for m in run.CLASS_NAMES:
                row = ci[symbol, m, factor]
                cm = np.array(row["confusion_down_neutral_up"])
                opposite = int(cm[0, 2] + cm[2, 0])
                calls = int(cm[:, [0, 2]].sum())
                ratio = fmt(100 * opposite / calls) if calls else "н/д"
                details.append(f"| {symbol} | {m} | {fmt(100*row['accuracy'])} | {fmt(row['macro_f1'], 4)} | {fmt(100*row['actual_neutral_fraction'])} | {fmt(100*calls/1872)} | {fmt(100*opposite/1872)} | {ratio} |")
    details += ["", "## Парные proper-loss сравнения", "", "Положительная добавка лучше. Обе LCB одновременные по 64 сравнениям, но не по всей адаптивной истории исследования.", "", "| Актив | Challenger | Контроль | Loss | Добавка | LCB bank 1 | LCB bank 2 |", "|---|---|---|---|---:|---:|---:|"]
    for row in d["comparisons"]:
        details.append(f"| {row['symbol']} | {row['candidate']} | {row['control']} | {row['metric']} | {fmt(row['improvement'], 7)} | {fmt(row['simultaneous_lcb_bank0'], 7)} | {fmt(row['simultaneous_lcb_bank1'], 7)} |")
    details += ["", "## Изменение нормы коэффициентов (без intercept)", "", "| Origin | Модель | L2 до | L2 после | Итераций |", "|---:|---|---:|---:|---:|"]
    for row in norm_rows:
        details.append(f"| {row['origin']} | {row['method']} | {fmt(row['before'], 6)} | {fmt(row['after'], 6)} | {row['fitted_iterations']} |")
    (OUT / "DETAILS_RU.md").write_text("\n".join(details) + "\n", encoding="utf-8")
    return dict(report="REPORT_RU.md", detail="DETAILS_RU.md", figures=3, new_fits=0)


def make_figures(d, symbols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    labels = [s.split("/")[0] for s in symbols]
    x = np.arange(8)
    di = {(r["symbol"], r["method"]): r for r in d["direction"]}
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for ax, (old, new, title) in zip(axes, [("original_C0p1", "original_C0p01", "Обучение с neutral ±0,064%"), ("half_C0p1", "half_C0p01", "Обучение с neutral ±0,032%")]):
        for m, label, color in [(old, "C=0,1", "#3266a8"), (new, "C=0,01", "#dd7429"), ("train_prior", "Train-only prior", "#6b7280")]:
            ax.plot(x, [100 * di[s, m]["accuracy"] for s in symbols], "o-", color=color, label=label)
        ax.axhline(50, color="black", linestyle=":", linewidth=.8)
        ax.set_title(title)
        ax.set_ylabel("Условная raw-accuracy, %")
        ax.grid(alpha=.2)
        ax.legend(ncol=3, fontsize=8)
    axes[-1].set_xticks(x, labels)
    fig.suptitle("10m: усиление регуляризации почти не меняет точность\n1–13 октября 2026; фактически ненулевые движения, не full-coverage 3-class")
    fig.savefig(OUT / "conditional_accuracy.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for ax, (candidate, matched) in zip(axes, [("original_C0p01", "original_C0p1"), ("half_C0p01", "half_C0p1")]):
        for offset, control, color, label in [(-.12, matched, "#3266a8", "Против C=0,1"), (.12, "train_prior", "#dd7429", "Против train prior")]:
            rows = [next(r for r in d["comparisons"] if r["symbol"] == s and r["candidate"] == candidate and r["control"] == control and r["metric"] == "brier") for s in symbols]
            observed = np.array([r["improvement"] for r in rows]) * 10000
            lower = np.array([min(r["simultaneous_lcb_bank0"], r["simultaneous_lcb_bank1"]) for r in rows]) * 10000
            ax.vlines(x + offset, lower, observed, color=color, linewidth=1.4)
            ax.scatter(x + offset, lower, marker="_", color=color)
            ax.scatter(x + offset, observed, s=25, color=color, label=label)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_title(candidate)
        ax.set_ylabel("Δ Brier × 10⁴ (больше лучше)")
        ax.grid(alpha=.2)
        ax.legend(ncol=2)
    axes[-1].set_xticks(x, labels)
    fig.suptitle("Добавка C=0,01 и худшая одновременная 95% LCB двух банков\n64 сравнения; только раскрытый 13-дневный период")
    fig.savefig(OUT / "brier_improvements.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for off, m, color in [(-.24, "original_C0p1", "#3266a8"), (-.08, "original_C0p01", "#dd7429"), (.08, "half_C0p1", "#37805b"), (.24, "half_C0p01", "#9059aa")]:
        rows = [next(r for r in d["classification"] if r["symbol"] == s and r["method"] == m and r["evaluation_band_factor"] == .5) for s in symbols]
        axes[0].bar(x + off, [100 * r["accuracy"] for r in rows], width=.16, color=color, label=m)
        opposite = [100 * (np.array(r["confusion_down_neutral_up"])[0, 2] + np.array(r["confusion_down_neutral_up"])[2, 0]) / 1872 for r in rows]
        axes[1].bar(x + off, opposite, width=.16, color=color)
    axes[0].set_ylabel("Full-coverage accuracy, %")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set_ylabel("Opposite / все timestamp, %")
    axes[1].set_xticks(x, labels)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Одинаковая трёхклассовая truth ±0,032%, 1872 timestamp на актив\nВерх: accuracy; низ: только противоположные направления, не все ошибки")
    fig.savefig(OUT / "three_class_accuracy_errors.png", dpi=150)
    plt.close(fig)


def seal():
    assert not (OUT / "delivery_manifest.json").exists(), "manifest is immutable"
    audit_data = run.base.read(OUT / "statistical_audit.json")
    assert audit_data["code_pins"][run.base.rel(SELF)] == run.base.digest(SELF)
    sources = {}
    for file in ("registration.json", "scoring_registration_v2.json"):
        reg = run.base.read(OUT / file)
        sources.update(reg["source_pins"])
        sources.update(reg["code_pins"])
    sources.update({run.base.rel(SELF): run.base.digest(SELF)})
    for file in ("configs/genetic_primus_objective_registry_v28.json", "configs/genetic_primus_objective_registry_v29.json"):
        sources[file] = run.base.digest(ROOT / file)
    artifacts = {run.base.rel(p): run.base.digest(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    for file, sha in {**sources, **artifacts}.items():
        assert run.base.digest(ROOT / file) == sha, file
    run.base.write(OUT / "delivery_manifest.json", dict(source_pins=sources, artifact_pins=artifacts,
                   new_historical_fits=10, scoring_extra_fits=0, independent_validation=False,
                   **{k: False for k in run.base.reuse.fixed.SAFETY}))
    return dict(source_pins=len(sources), artifact_pins=len(artifacts), new_fits=0)


def verify():
    data = run.base.read(OUT / "delivery_manifest.json")
    count = 0
    for collection in ("source_pins", "artifact_pins"):
        for file, sha in data[collection].items():
            assert run.base.digest(ROOT / file) == sha, file
            count += 1
    assert all(data[k] is False for k in run.base.reuse.fixed.SAFETY)
    return dict(verified_pins=count, new_historical_fits=data["new_historical_fits"], new_fits=0)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "checks":
        value = dict(pure_checks=checks(), new_fits=0)
    else:
        value = {"audit": audit, "report": report, "seal": seal, "verify": verify}[stage]()
    print(json.dumps(value, ensure_ascii=False))
