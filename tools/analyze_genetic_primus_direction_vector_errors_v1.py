"""Saved forecast diagnostics only: no estimator fitting, runtime or network."""
from __future__ import annotations
import run_genetic_primus_direction_ternary_hypotheses_v1 as run
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = run.ROOT
OUT = ROOT / "Reports/Exia/Genetic_Primus/direction_vector_errors_v1_20261014"
CONFIG = ROOT / "configs/genetic_primus_direction_vector_errors_v1.json"
REGISTRY = ROOT / "configs/genetic_primus_objective_registry_v24.json"

def pins(): return {run.rel(p): run.parent.digest(p) for p in (CONFIG, Path(__file__))}
def config():
    c = run.read(CONFIG)
    assert all(c[k] is False for k in run.fixed.SAFETY)
    assert c["band_factors"] == [1., .5, .25, 0.]
    run.config()  # Enforce the exact workspace Python and original safety flags.
    return c

def registered():
    c = config(); reg = run.read(OUT / "registration.json")
    assert reg["code_pins"] == pins()
    assert reg["parent_manifest_sha256"] == run.parent.digest(run.OUT / "delivery_manifest.json")
    return c, reg

def ratio(a, b): return None if b == 0 else float(a/b)

def error_counts(cm):
    cm = np.asarray(cm, np.int64)
    n = int(cm.sum()); calls = int(cm[:, [0, 2]].sum())
    same = int(cm[0, 0]+cm[2, 2]); opposite = int(cm[0, 2]+cm[2, 0]); flat = int(cm[1, 0]+cm[1, 2])
    assert calls == same+opposite+flat
    return dict(n=n, directional_calls=calls, same_direction_count=same, opposite_direction_count=opposite,
        neutral_outcome_on_directional_call_count=flat,
        missed_move_on_neutral_prediction_count=int(cm[0, 1]+cm[2, 1]),
        exact_class_correct_count=int(np.trace(cm)), actual_down_neutral_up=cm.sum(axis=1).tolist(),
        predicted_down_neutral_up=cm.sum(axis=0).tolist(), confusion_down_neutral_up=cm.tolist(),
        directional_call_fraction=ratio(calls, n), same_direction_per_directional_call=ratio(same, calls),
        opposite_per_directional_call=ratio(opposite, calls), neutral_per_directional_call=ratio(flat, calls),
        opposite_per_all_intervals=ratio(opposite, n),
        opposite_given_called_and_actual_move=ratio(opposite, opposite+same),
        opposite_when_predict_up=ratio(int(cm[0, 2]), int(cm[:, 2].sum())),
        opposite_when_predict_down=ratio(int(cm[2, 0]), int(cm[:, 0].sum())),
        actual_neutral_fraction=ratio(int(cm[1].sum()), n), accuracy=ratio(int(np.trace(cm)), n))

def direct_checks():
    truth = np.array([-1, -1, 0, 1, 1, 0, -1, 1]); pred = np.array([1, -1, 1, -1, 1, 0, 0, 0])
    a = error_counts(run.tri.confusion(truth, pred))
    assert a["n"] == 8 and a["directional_calls"] == 5
    assert a["opposite_direction_count"] == 2 and a["neutral_outcome_on_directional_call_count"] == 1
    assert a["opposite_per_directional_call"] == .4 and a["opposite_per_all_intervals"] == .25
    assert a["opposite_when_predict_up"] == 1/3 and a["opposite_when_predict_down"] == .5
    assert a["opposite_given_called_and_actual_move"] == .5
    assert a["accuracy"] == 3/8 and a["missed_move_on_neutral_prediction_count"] == 2
    zero = error_counts(run.tri.confusion(truth, np.zeros(8, int)))
    assert zero["opposite_per_directional_call"] is None and zero["opposite_per_all_intervals"] == 0
    returns = np.array([-.02, -.01, 0., .01, .02])
    assert np.array_equal(run.tri.labels(returns, .02), np.zeros(5, int))
    assert np.array_equal(run.tri.labels(returns, .005), [-1, -1, 0, 1, 1])
    assert np.array_equal(run.tri.labels(returns, 0), np.sign(returns))
    return ["hand_opposites_and_all_denominators", "separate_neutral_errors_and_omissions", "zero_directional_calls_is_undefined_not_zero", "inclusive_band_and_raw_sign_edges"]

def init():
    c = config(); pure = direct_checks(); OUT.mkdir(parents=True, exist_ok=True)
    manifest = run.read(run.OUT / "delivery_manifest.json")
    for path, sha in manifest["artifact_pins"].items(): assert run.parent.digest(ROOT / path) == sha, path
    parent_reg = run.read(run.OUT / "registration.json")
    run.parent.write(OUT / "registration.json", dict(contract=c, code_pins=pins(), versions=run.versions(),
        parent_manifest_sha256=run.parent.digest(run.OUT / "delivery_manifest.json"),
        parent_input_pins=parent_reg["data_pins"], pure_checks=pure, estimator_fits=0,
        independent_validation=False, **{k: False for k in run.fixed.SAFETY}))
    previous_path = ROOT / c["previous_registry"]; previous = run.read(previous_path); previous.pop("sha256", None)
    previous.update(registry_id=REGISTRY.stem, previous_registry_path=c["previous_registry"],
        previous_registry_sha256=run.parent.digest(previous_path))
    previous["objectives"].append(dict(objective_id=c["objective_id"], family="frozen_prediction_opposite_error_band_sensitivity",
        registration_path=run.rel(OUT / "registration.json"), registration_sha256=run.parent.digest(OUT / "registration.json"),
        historical_fit_budget=0, posthoc_diagnostic=True, no_threshold_selection=True,
        independent_validation=False, promotion_authority=False))
    run.parent.write(REGISTRY, previous)
    print(json.dumps(dict(pure_checks=len(pure), frozen_artifact_pins_verified=len(manifest["artifact_pins"]), fits=0, registry="v24")))

def confidence(daily, weights):
    counts = np.einsum("bd,dij->bij", weights, daily)
    calls = counts[:, :, [0, 2]].sum(axis=(1, 2)); opposite = counts[:, 0, 2]+counts[:, 2, 0]
    values = np.divide(opposite, calls, out=np.full(len(calls), np.nan), where=calls != 0)
    finite = values[np.isfinite(values)]
    return dict(ci95=None if not len(finite) else np.quantile(finite, [.025, .975]).tolist(), valid_replicates=len(finite))

def score(minutes):
    c, reg = registered(); pc = run.config(); names = run.methods(pc, minutes)
    factors = c["band_factors"]; daily = np.zeros((4, 8, len(names), 61, 3, 3), np.int64)
    source_pins = {}
    for number in range(21):
        expanded = run.prefix(run.OUT, number, minutes)
        base = run.prefix(run.baseout(pc), number, minutes)
        for p in (expanded, base):
            meta = run.read(p.with_suffix(".json")); path = p.with_suffix(".npz")
            assert run.parent.digest(path) == meta["ledger_sha256"]
            source_pins[run.rel(path)] = run.parent.digest(path)
        with np.load(expanded.with_suffix(".npz"), allow_pickle=False) as a, np.load(base.with_suffix(".npz"), allow_pickle=False) as b:
            assert np.array_equal(a["times"], b["times"]) and np.array_equal(a["labels"], b["labels"])
            predictions = [run.tri.predict(p) for p in a["probability"]]+run.fixed.predictions(b)[:9]
            assert len(predictions) == len(names)
            times, returns = a["times"], b["return_pct"]
            days = ((times-run.parent.START)//run.tri.DAY).astype(int)
            for k, factor in enumerate(factors):
                truth = run.tri.labels(returns, c["neutral_bands_pct"][str(minutes)]*factor)
                if k == 0: assert np.array_equal(truth, a["labels"])
                for asset in range(8):
                    for method, pred in enumerate(predictions):
                        codes = days*9+(truth[:, asset]+1)*3+pred[:, asset]+1
                        daily[k, asset, method] += np.bincount(codes, minlength=61*9).reshape(61, 3, 3)
    with np.load(run.OUT / ("daily_confusion_%sm.npz" % minutes), allow_pickle=False) as a: assert np.array_equal(daily[0], a["daily"])
    weights = []
    for seed in pc["bootstrap"]["seeds"]:
        p = run.OUT / ("bootstrap_bank_%s.npz" % seed)
        source_pins[run.rel(p)] = run.parent.digest(p)
        with np.load(p, allow_pickle=False) as a: weights.append(np.stack([np.bincount(row, minlength=61) for row in a["indices"]]))
    rows = []
    for k, factor in enumerate(factors):
        for asset, coin in enumerate(pc["symbols"]):
            for method, name in enumerate(names):
                counts = error_counts(daily[k, asset, method].sum(axis=0))
                raw = error_counts(daily[-1, asset, method].sum(axis=0))
                row = dict(minutes=minutes, symbol=coin, method=name, band_factor=factor,
                    band_pct=c["neutral_bands_pct"][str(minutes)]*factor, **counts,
                    opposite_raw_sign_count=raw["opposite_direction_count"],
                    opposite_raw_sign_per_directional_call=raw["opposite_per_directional_call"],
                    opposite_raw_sign_per_all_intervals=raw["opposite_per_all_intervals"],
                    opposite_inside_band_count=raw["opposite_direction_count"]-counts["opposite_direction_count"],
                    opposite_per_directional_call_ci95_banks=[confidence(daily[k, asset, method], w) for w in weights])
                assert row["opposite_inside_band_count"] >= 0
                assert row["directional_calls"] == raw["directional_calls"]
                rows.append(row)
    with (OUT / ("daily_confusion_%sm.npz" % minutes)).open("xb") as stream: np.savez_compressed(stream, daily=daily, names=np.array(names), factors=np.array(factors))
    run.parent.write(OUT / ("results_%sm.json" % minutes), dict(rows=rows, source_pins=source_pins,
        code_pins=pins(), estimator_fits=0, **{k: False for k in run.fixed.SAFETY}))
    print(json.dumps(dict(minutes=minutes, metric_cells=len(rows), current_band_confusion_matches_parent=True, fits=0)))

def fmt(value, multiplier=100): return "—" if value is None else "%.2f" % (value*multiplier)

def report():
    import matplotlib.pyplot as plt
    c, reg = registered(); pc = run.config()
    rows = sum([run.read(OUT / ("results_%sm.json" % h))["rows"] for h in pc["timeframes_minutes"]], [])
    assert len(rows) == 2496
    medians = []
    for h in pc["timeframes_minutes"]:
        for factor in c["band_factors"]:
            for method in run.methods(pc, h):
                rr = [r for r in rows if r["minutes"] == h and r["band_factor"] == factor and r["method"] == method]
                values = {}
                for key in ("accuracy", "directional_call_fraction", "opposite_per_directional_call", "opposite_raw_sign_per_directional_call", "opposite_per_all_intervals", "neutral_per_directional_call", "same_direction_per_directional_call", "actual_neutral_fraction"):
                    finite = [r[key] for r in rr if r[key] is not None]
                    values[key] = None if not finite else float(np.median(finite))
                medians.append(dict(minutes=h, band_factor=factor, method=method, **values))
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), layout="constrained")
    for ax, h in zip(axes.flat, pc["timeframes_minutes"]):
        names = run.methods(pc, h)
        matrix = np.array([[np.nan if (value := next(r["opposite_raw_sign_per_directional_call"] for r in rows if r["minutes"] == h and r["band_factor"] == 1 and r["method"] == method and r["symbol"] == coin)) is None else 100*value for coin in pc["symbols"]] for method in names])
        im = ax.imshow(matrix, vmin=0, vmax=60, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(8), [s.split('/')[0] for s in pc["symbols"]]); ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_title("%s мин — противоположный знак / вызовы вверх-вниз, %%" % h)
        for i in range(len(names)):
            for j in range(8): ax.text(j, i, "—" if np.isnan(matrix[i,j]) else "%.1f" % matrix[i,j], ha="center", va="center", fontsize=8, color="white" if matrix[i,j] > 40 else "black")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Ошибки знака среди направленных вызовов, %", shrink=.7)
    fig.savefig(OUT / "opposite_raw_sign_by_coin.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(17, 5), layout="constrained", sharey=True)
    for ax, h in zip(axes, pc["timeframes_minutes"]):
        parts = []
        for factor in reversed(c["band_factors"]):
            rr = [r for r in rows if r["minutes"] == h and r["band_factor"] == factor and r["method"] == "boost_uniform"]
            total = np.sum([r["actual_down_neutral_up"] for r in rr], axis=0)
            parts.append(total/total.sum()*100)
        matrix = np.array(parts); bottom = np.zeros(4)
        for j, (name, color) in enumerate(zip(["Вниз", "Нейтрально", "Вверх"], ["#cb4b4b", "#a0a0a0", "#3d9c71"])):
            ax.bar(range(4), matrix[:,j], bottom=bottom, label=name, color=color)
            bottom += matrix[:,j]
        ax.set_xticks(range(4), ["0", "¼", "½", "1"]); ax.set_xlabel("Множитель исходной полосы"); ax.set_title("%s мин" % h)
        for i in range(4): ax.text(i, matrix[i,0]+matrix[i,1]/2, "%.1f%%" % matrix[i,1], ha="center", va="center", fontsize=9)
    axes[0].set_ylabel("Истинные классы, % всех интервалов"); axes[-1].legend(loc="upper right")
    fig.savefig(OUT / "neutral_band_class_mix.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    for ax, h in zip(axes.flat, pc["timeframes_minutes"]):
        names = ["logit_uniform", "boost_uniform", "context_boost", "recency90", "random_uniform_3", "majority_train"]
        for name in names:
            points = [next(r for r in medians if r["minutes"] == h and r["band_factor"] == factor and r["method"] == name) for factor in reversed(c["band_factors"])]
            ax.plot([0, .25, .5, 1], [100*r["accuracy"] for r in points], marker="o", label=name)
        ax.set_xticks([0, .25, .5, 1]); ax.set_title("%s мин — прогнозы неизменны, labels меняются" % h)
        ax.set_xlabel("Множитель исходной полосы"); ax.set_ylabel("Медиана трёхклассовой accuracy, %"); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(OUT / "frozen_prediction_band_sensitivity.png", dpi=150); plt.close(fig)
    lines = ["# Ошибки направления и чувствительность к нейтральной полосе", "", "Период: 1 августа — 30 сентября 2026 UTC. Анализ выполнен 14 октября 2026 без нового обучения.", "",
        "## Итог и смысл проверки", "", "Ниже отдельно измерены противоположный знак движения и движение за пределами нейтральной полосы. Они не тождественны: небольшое падение при прогнозе вверх считается ошибкой знака, но при достаточно широкой полосе относится к neutral. Уменьшение полосы не исправляет такую ошибку — лишь переносит её из категории neutral в opposite.", "",
        "Исходные вероятности и решения всех моделей полностью заморожены. Проверены полосы 1×, ½×, ¼× и 0×. Это sensitivity saved predictions, **не** результаты моделей, заново обученных на более узких полосах. Порог не выбирается по наибольшей тестовой accuracy. Новых estimator fits: 0.", "",
        "## Формулы и знаменатели", "", "Opposite = N(pred=up, true=down) + N(pred=down, true=up). Основная доля = Opposite / N(pred≠neutral). Дополнительная доля = Opposite / N(всех допустимых интервалов). При отсутствии направленных вызовов основная доля не определена (—), а не 0%. Дополнительно приведены ошибки отдельно для up/down.", "",
        "Для frozen направленных вызовов: верное направление + противоположное направление + истинный neutral = 100%. Прогноз neutral при реальном движении учитывается отдельно как пропуск движения. Доля противоположного направления не равна 1−общая accuracy, поскольку есть другие ошибки. Метрика среди направленных вызовов условная; модели прогнозируют на 100% интервалов, но часть прогнозов neutral.", "",
        "Raw-sign opposite учитывает любое r<0 при pred=up и любое r>0 при pred=down, даже внутри полосы. Beyond-band opposite требует r<−δ либо r>δ. Exact r=0 не противоположный знак. При δ=0 эти две метрики совпадают; буквальная неизменность цены сохранена отдельным neutral. Важны обе постановки, не только более удобная для модели.", "",
        "## Базовые модели: медианы долей по восьми валютам", "", "| Мин | Модель | Вызовы вверх/вниз, % всех | Противоположный знак, % вызовов | Противоположно за полосой, % вызовов | Противоположно за полосой, % всех | Neutral-исход, % вызовов |", "|---:|---|---:|---:|---:|---:|---:|"]
    for h in pc["timeframes_minutes"]:
        for name in ("logit_uniform", "boost_uniform", "random_uniform_3", "majority_train"):
            r = next(r for r in medians if r["minutes"] == h and r["band_factor"] == 1 and r["method"] == name)
            values = [fmt(r[k]) for k in ("directional_call_fraction", "opposite_raw_sign_per_directional_call", "opposite_per_directional_call", "opposite_per_all_intervals", "neutral_per_directional_call")]
            lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (h, name, *values))
    lines += ["", "Медианы столбцов считаются отдельно и не обязаны складываться в 100%; для каждой исходной строки partition проверен точно. Все методы и активы — в DETAILS_RU.md.", "",
        "## Уменьшение полосы", "", "| Мин | Множитель | Полоса ±, % | Медиана истинного neutral, % | Frozen boost accuracy, % | Opposite за полосой, % вызовов boost |", "|---:|---:|---:|---:|---:|---:|"]
    for h in pc["timeframes_minutes"]:
        for factor in c["band_factors"]:
            r = next(r for r in medians if r["minutes"] == h and r["band_factor"] == factor and r["method"] == "boost_uniform")
            lines.append("| %s | %s | %.6f | %s | %s | %s |" % (h, factor, c["neutral_bands_pct"][str(h)]*factor, fmt(r["actual_neutral_fraction"]), fmt(r["accuracy"]), fmt(r["opposite_per_directional_call"])))
    lines += ["", "Более узкая полоса делает neutral менее частым, но уменьшает размер движения, которое нужно предсказывать. На минутных интервалах это повышает чувствительность к цене последней сделки, tick size и bid/ask bounce. Этот OHLCV-анализ не доказывает, что эти эффекты устранены, и не определяет экономически значимую полосу через spread/costs.", "",
        "Рекомендуется пока сохранить исходный контракт. Если нужна более чувствительная классификация, следующий отдельный ограниченный опыт — половина исходных полос: 1m ±0,010%, 10m ±0,032%, 30m ±0,0545%, 4h ±0,160%. Это кандидат для заранее зафиксированного обучения и нового временного теста, не выбранный здесь победитель. Четверть и zero оставлены диагностическими границами; уменьшать δ ради получения высокой accuracy нельзя.", "",
        "![Ошибки знака по валютам](opposite_raw_sign_by_coin.png)", "", "Рисунок 1. Реально противоположный знак / все frozen направленные вызовы. Neutral-прогнозы не входят в знаменатель; — означает отсутствие таких вызовов.", "",
        "![Классовые доли](neutral_band_class_mix.png)", "", "Рисунок 2. Сужение полосы перераспределяет истинные классы. Показаны объединённые counts восьми активов; это не изменение прогнозов модели.", "",
        "![Чувствительность frozen прогнозов](frozen_prediction_band_sensitivity.png)", "", "Рисунок 3. Accuracy сохранённых прогнозов при новых labels. Линии не являются кривыми качества переобученных моделей.", "",
        "## Данные, проверки и ограничения", "", "Входы: сохранённые origin_*/tf_*m.npz из direction_ternary_hypotheses_v1_20261014 и direction_ternary_fixed_v1_20261013; future return_pct берётся из фиксированного ledger, а не восстанавливается из новых цен. Обрабатываются все 624 прежние строки метод/актив/горизонт, четыре полосы — 2496 строк. Для исходной полосы daily confusion точно совпали с предыдущим frozen расчётом. Проверены формулы на конечной ручной последовательности, нулевые знаменатели и inclusive edges.", "",
        "Для opposite / directional_calls сохранены marginal 95% calendar block-bootstrap CI в двух исходных banks: 1000 повторов, block 3 дня, 61 календарный день. Интервалы не поправлены по семейству, selection/promotion не выполняется. При нулевом числе calls в bootstrap replicate он исключается из условного CI; valid_replicates указаны явно. Не считать отдельные минуты независимыми наблюдениями. Август–сентябрь раскрыт; независимой валидации нет.", "",
        "Навык научно-технических отчётов применён для явных знаменателей, доказательных артефактов и отделения sensitivity от переобучения. Сеть, runtime, credentials, orders, paper/live и promotion остаются false. Таблицы Markdown/JSON, графики PNG; DOCX/XLSX не создавались."]
    with (OUT / "REPORT_RU.md").open("x", encoding="utf-8") as stream: stream.write("\n".join(lines)+"\n")
    details = ["# Все доли ошибок по моделям, валютам и горизонтам", "", "Все значения в таблицах — проценты. Знаменатель 'вызовы' = только pred вверх/вниз. 'Все' = все допустимые интервалы. Neutral прогнозы не маскируются как верные направленные прогнозы.", ""]
    for h in pc["timeframes_minutes"]:
        for coin in pc["symbols"]:
            details += ["## %s — %s мин, исходная полоса" % (coin, h), "", "| Метод | N | Calls | Calls / все | Raw opposite / calls | Opposite за полосой / calls | Opposite / все | Верное направление / calls | Neutral / calls | Ошибка при up | Ошибка при down | 95% CI opposite / calls bank0; bank1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
            selected = [r for r in rows if r["minutes"] == h and r["symbol"] == coin and r["band_factor"] == 1]
            for r in selected:
                intervals = "; ".join("—" if v["ci95"] is None else "[%s; %s]" % (fmt(v["ci95"][0]), fmt(v["ci95"][1])) for v in r["opposite_per_directional_call_ci95_banks"])
                details.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (r["method"], r["n"], r["directional_calls"], *[fmt(r[k]) for k in ("directional_call_fraction", "opposite_raw_sign_per_directional_call", "opposite_per_directional_call", "opposite_per_all_intervals", "same_direction_per_directional_call", "neutral_per_directional_call", "opposite_when_predict_up", "opposite_when_predict_down")], intervals))
            details += ["", "### Чувствительность всех frozen методов", "", "| Множитель | Метод | Accuracy | Истинный neutral | Opposite / calls | Opposite / все | Neutral / calls |", "|---:|---|---:|---:|---:|---:|---:|"]
            for r in [r for r in rows if r["minutes"] == h and r["symbol"] == coin]:
                values = [fmt(r[k]) for k in ("accuracy", "actual_neutral_fraction", "opposite_per_directional_call", "opposite_per_all_intervals", "neutral_per_directional_call")]
                details.append("| %s | %s | %s | %s | %s | %s | %s |" % (r["band_factor"], r["method"], *values))
            details.append("")
    with (OUT / "DETAILS_RU.md").open("x", encoding="utf-8") as stream: stream.write("\n".join(details)+"\n")
    run.parent.write(OUT / "results.json", dict(rows=rows, medians=medians, estimator_fits=0, independent_validation=False, **{k: False for k in run.fixed.SAFETY}))
    print(json.dumps(dict(rows=len(rows), figures=3, fits=0, report=run.rel(OUT / "REPORT_RU.md"))))

def finalize():
    c, reg = registered(); results = run.read(OUT / "results.json")
    for h in run.config()["timeframes_minutes"]:
        rr = run.read(OUT / ("results_%sm.json" % h))
        for path, sha in rr["source_pins"].items(): assert run.parent.digest(ROOT / path) == sha
        # Opposites only increase when the band shrinks; raw-sign numerators and predictions stay fixed.
        for coin in run.config()["symbols"]:
            for method in run.methods(run.config(), h):
                subset = [r for r in rr["rows"] if r["symbol"] == coin and r["method"] == method]
                assert np.all(np.diff([r["opposite_direction_count"] for r in subset]) >= 0)
                assert len({r["directional_calls"] for r in subset}) == 1
                assert len({r["opposite_raw_sign_count"] for r in subset}) == 1
    frozen = run.read(run.OUT / "delivery_manifest.json")
    for path, sha in frozen["artifact_pins"].items(): assert run.parent.digest(ROOT / path) == sha
    previous = run.read(ROOT / c["previous_registry"]); registry = run.read(REGISTRY)
    assert previous["objectives"] == registry["objectives"][:-1]
    run.parent.write(OUT / "checks.json", dict(pure_checks=direct_checks(), current_band_confusion_parity=True,
        narrowing_opposite_monotonicity=True, raw_sign_numerators_unchanged=True, frozen_parent_pins_verified=len(frozen["artifact_pins"]),
        metric_cells=len(results["rows"]), previous_registry_unchanged=True, estimator_fits=0))
    artifacts = {run.rel(p): run.parent.digest(p) for p in OUT.rglob("*") if p.is_file() and p.name != "delivery_manifest.json"}
    artifacts.update(pins()); artifacts[run.rel(REGISTRY)] = run.parent.digest(REGISTRY)
    run.parent.write(OUT / "delivery_manifest.json", dict(artifact_pins=artifacts, figures_visually_inspected=True, **{k: False for k in run.fixed.SAFETY}))
    print(json.dumps(dict(final_pins=len(artifacts), previous_frozen_pins_unchanged=len(frozen["artifact_pins"]), fits=0)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["init", "score", "report", "finalize"]); parser.add_argument("--minutes", type=int)
    args = parser.parse_args()
    if args.stage == "score": score(args.minutes)
    else: globals()[args.stage]()
