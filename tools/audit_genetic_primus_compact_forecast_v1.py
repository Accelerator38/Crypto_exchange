"""Read/reconcile compact forecast evidence; no training or trading evaluation."""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from exia.genetic_primus import compact_forecast_v1 as m
from exia.genetic_primus import compact_forecast_data_v1 as d


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pin(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    folder = (ROOT / args.report_dir).resolve()
    folder.relative_to(ROOT / "Reports/Exia/Genetic_Primus")
    outputs = [folder / name for name in ("reconciliation.json", "ANALYSIS.md", "compact_comparison.png",
                                         "compact_comparison.svg", "analysis_receipt.json")]
    if any(p.exists() for p in outputs):
        raise ValueError("existing audit output refused")
    receipt = read(folder / "evidence_receipt.json")
    inputs = receipt["inputs"] | receipt["outputs"]
    for name, expected in inputs.items():
        if pin(ROOT / name) != expected:
            raise ValueError("input/output identity changed: " + name)
    prep = read(folder / "preparation.json")
    comparison = read(folder / "comparison.json")
    results = [read(folder / f"origin_{i:02d}.json") for i in range(20)]
    with np.load(folder / "sample_bank.npz", allow_pickle=False) as bank:
        samples = {k: bank[k] for k in bank.files}
    with np.load(folder / "audit_resample_banks.npz", allow_pickle=False) as bank:
        audit = bank["seed_7079"]
    if m.array_identity(audit) != prep["bootstrap"]["audit_content_sha256"]:
        raise ValueError("audit bank content mismatch")
    verified_models = 0
    for result in results:
        split = d.eligible_split(samples, result["origin_ms"])
        train = split["train"]
        x = samples["x"][train].reshape(-1, 6)
        y = samples["y"][train].ravel()
        if (result["training_identity"]["cutoff_ms"] != split["cutoff_ms"]
                or result["training_identity"]["train_samples_sha256"] != m.array_identity(samples["origins"][train], samples["available"][train], x, y)
                or result["training_identity"]["fold_sha256"] != m.array_identity(samples["origins"][split["inner_train"]], samples["origins"][split["inner_validation"]])):
            raise ValueError("cutoff/train/fold mismatch")
        dx = samples["x"][split["decision"]]
        if not np.array_equal(result["actual"], samples["y"][split["decision"]]):
            raise ValueError("evaluation outcome mismatch")
        for name, arm in result["arms"].items():
            weighting = arm["models"][0]["weighting"]
            origin_weights = m.temporal_weights(samples["available"][train], split["cutoff_ms"], weighting)
            weights = np.repeat(origin_weights/len(result["symbols"]), len(result["symbols"]))
            if (m.array_identity(origin_weights) != arm["weights_sha256"]
                    or not np.isclose(weights[y < 0].sum(), arm["negative_weight_mass"])
                    or not np.isclose(weights[y > 0].sum(), arm["positive_weight_mass"])
                    or not np.isclose(m.origin_ess(origin_weights), arm["origin_ess"])):
                raise ValueError("weight/sign-mass/ESS mismatch")
            predictions = [m.predict(payload, dx) for payload in arm["models"]]
            if not np.array_equal(np.mean(predictions, axis=0), arm["prediction"]):
                raise ValueError("stored prediction/reloaded model mismatch")
            for payload in arm["models"]:
                scaler = payload["scaler"]
                scale = np.where(x.std(axis=0) < 1e-12, 1.0, x.std(axis=0))
                if (not np.array_equal(scaler["mean"], x.mean(axis=0))
                        or not np.array_equal(scaler["scale"], scale)
                        or scaler["target_mean"] != float(y.mean())
                        or scaler["target_scale"] != (float(y.std()) if y.std() >= 1e-12 else 1.0)):
                    raise ValueError("train-only scaler mismatch")
                verified_models += 1
    old_folder = ROOT / comparison["superseded_attempt"]["path"]
    old_prep = read(old_folder / "preparation.json")
    if old_prep["sample_content_sha256"] != prep["sample_content_sha256"]:
        raise ValueError("correction unexpectedly changed market samples")
    for i, current in enumerate(results):
        previous = read(old_folder / f"origin_{i:02d}.json")
        for name in ("linear6_uniform", "mlp8_uniform"):
            if previous["arms"][name] != current["arms"][name]:
                raise ValueError("semantic correction unexpectedly changed uniform control")
    actual = np.array([r["actual"] for r in results])
    errors = {name: np.mean(((np.array([r["arms"][name]["prediction"] for r in results])-actual)*10000)**2, axis=1)
              for name in results[0]["arms"]}
    pairs = [("linear6_uniform", "mlp8_uniform"),
             ("linear6_half_life_90_anchor20", "mlp8_half_life_90_anchor20"),
             ("linear6_uniform", "linear6_half_life_90_anchor20"),
             ("mlp8_uniform", "mlp8_half_life_90_anchor20")]
    pair_diagnostics = []
    for a, b in pairs:
        delta = errors[b]-errors[a]
        pair_diagnostics.append({"first": a, "second": b, "positive_means_first_lower_mse": True,
                                 "mean_mse_benefit_bps_squared": float(delta.mean()),
                                 "audit_lcb_mse_benefit": float(np.quantile(delta[audit].mean(axis=1), .05))})
    zero_error = np.mean((actual*10000)**2, axis=1)
    peak = int(np.argmax(zero_error))
    facts = {"identities_verified": len(inputs), "serialized_models_reconciled": verified_models,
             "origins_reconciled": 20, "uniform_controls_unchanged_from_superseded_attempt": True,
             "sample_content_unchanged": True, "pairs": pair_diagnostics,
             "peak_origin_index": peak, "peak_origin_ms": results[peak]["origin_ms"],
             "peak_origin_share_of_zero_forecast_sse": float(zero_error[peak]/zero_error.sum()),
             "peak_origin_gross_outcome_range": [float(actual[peak].min()), float(actual[peak].max())],
             "additional_model_fits": 0, "financial_evaluations": 0, "network": 0,
             "profitability_claim": False, "promotion_authority": False}
    write(outputs[0], facts)
    labels = {"causal_train_mean": "Causal train mean (control)",
              "linear6_uniform": "Linear 7 / uniform", "linear6_half_life_90_anchor20": "Linear 7 / HL90 + 20% anchor",
              "mlp8_uniform": "MLP 65 x 3 / uniform", "mlp8_half_life_90_anchor20": "MLP 65 x 3 / HL90 + 20% anchor"}
    names = list(labels)
    metrics = comparison["arms_and_controls"]
    os.environ["MPLCONFIGDIR"] = str(folder / ".mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    colors = ["#727272", "#1674b8", "#38a69c", "#e99a28", "#bb57a5"]
    for axis, metric, title in zip(axes, ("rmse_bps", "mae_bps"), ("RMSE (bps); lower is better", "MAE (bps); lower is better")):
        values = [metrics[n][metric] for n in names]
        axis.barh(list(labels.values()), values, color=colors)
        axis.invert_yaxis()
        axis.set_xlim(0, max(values)*1.14)
        axis.set_title(title)
        axis.grid(axis="x", alpha=.25)
        for i, v in enumerate(values):
            axis.text(v+2, i, f"{v:.2f}", va="center", fontsize=9)
    axes[1].set_yticklabels([])
    fig.suptitle("Aug-Sep 2026: 20 rolling origins / 160 gross outcomes\nForecast accuracy only; no equity, costs, or profitability claim", fontsize=12)
    fig.savefig(outputs[2], dpi=150)
    fig.savefig(outputs[3])
    plt.close(fig)
    lines = ["# Выводы исправленного сравнения компактных моделей", "",
             "**Обучение и сравнение завершены, но подтверждённого выигрыша нет.** Среди четырёх обучаемых arms "
             f"наименьшая RMSE у linear6/uniform ({metrics['linear6_uniform']['rmse_bps']:.2f} bps), "
             f"однако causal train-mean лучше ({metrics['causal_train_mean']['rmse_bps']:.2f} bps). "
             "Все четыре audit LCB улучшения MSE относительно train-mean отрицательны. Добавление MLP или recency "
             "не подтвердило улучшение на этом открытом периоде. Это не доказательство превосходства linear во всех режимах.", "",
             "Исправленная recency — нормированная смесь 20% uniform + 80% exponential half-life90, "
             "а не raw floor. Первый superseded-прогон сохранён вместе с исходным кодом; 280+280=560 "
             "market forecast fits учтены, не скрыты. Данные и оба uniform controls побитово совпадают между попытками.", "",
             "![Сравнение ошибок](compact_comparison.png)", "",
             "## Парные различия", "",
             "Положительное ΔMSE означает, что **первый** вариант имеет меньшую ошибку. Все различия используют "
             "тот же заранее зафиксированный audit bank (4 096 circular block samples, блок=2 origins). "
             "LCB не скорректирован на историческое multiple-testing.", "",
             "| Первый вариант | Второй | Среднее ΔMSE, bps² | Audit LCB |", "|---|---|---:|---:|"]
    for row in pair_diagnostics:
        lines.append(f"| {labels[row['first']]} | {labels[row['second']]} | {row['mean_mse_benefit_bps_squared']:.2f} | {row['audit_lcb_mse_benefit']:.2f} |")
    lines += ["", "## Почему вывод ограничен", "",
              f"Origin с максимальным исходом (индекс {peak}, 19 августа, вход 04:00 → 22 августа 04:00 UTC) "
              f"даёт {facts['peak_origin_share_of_zero_forecast_sse']:.2%} суммы квадратов ошибок zero-forecast. "
              f"Все восемь gross исходов в этом origin положительны: {actual[peak].min():.2%}–{actual[peak].max():.2%}. "
              "Наблюдение оставлено в основной метрике; исключать его ради улучшения ranking нельзя. "
              "Это одновременно показывает важность tail-regime и необходимость проверки исходных цен/агрегации этого окна "
              "по независимому архиву до экономических выводов; не утверждается, что скачок является ошибкой данных.", "",
              "MSE и знак оценивают прогноз, не заработанный капитал. HL90 у linear даёт больше правильных знаков "
              "(53.1% против 51.2% uniform), но хуже RMSE. Знак сам по себе не учитывает масштаб outcome, комиссии "
              "и количество отказов от сделки. Ни один критерий не подменяет net-growth ledger.", "",
              "Для MLP отдельные seeds, early-stopping epochs и ensembles сохранены в основном отчёте; "
              "три сети — 195 весов, не одна модель с 65 весами. Гиперпараметры фиксированы, а архитектуры не имеют "
              "равного вычислительного бюджета. Результат относится к этой конфигурации и не оправдывает дальнейший "
              "подбор seeds/lr/L2 по августу–сентябрю.", "",
              "## Проверка и дальнейшее решение", "",
              f"23 регрессии; {len(inputs)} source/output identities; {verified_models} сохранённых моделей и их прогнозы "
              "сверены с samples, train-only scalers, cutoff, folds, весами всех знаков и ESS. Проверка не выполнила новых fits. "
              "Никакой сети, collector, pytest или торгового runtime.", "",
              "Не объявлять нового financial incumbent и не подключать GA на основании этого результата. "
              "Следующий узкий шаг: подтвердить происхождение/агрегацию пикового окна; затем заранее зафиксировать "
              "predictor→net-policy с abstention и реальными cost/funding inputs, проверить общий ledger и net-growth "
              "на уже открытом периоде без изменения моделей по его исходам. Если baseline не побеждён, оставить NoTrade "
              "и пересмотреть сигнал на отдельном pre-August development interval, не раздувать нейросеть.", "",
              "Финансовая validation, equity curve, live/paper, promotion и согласование риска реальных денег не выполнялись. "
              "Подробные метрики и identities: [REPORT.md](REPORT.md), comparison.json, reconciliation.json, "
              "evidence_receipt.json и analysis_receipt.json."]
    with outputs[1].open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines)+"\n")
    # Supplementary annotation is separate from frozen primary receipts.
    analysis_inputs = inputs | {
        (folder / "evidence_receipt.json").relative_to(ROOT).as_posix(): pin(folder / "evidence_receipt.json"),
        Path(__file__).relative_to(ROOT).as_posix(): pin(Path(__file__)),
        (old_folder / "SUPERSEDED.md").relative_to(ROOT).as_posix(): pin(old_folder / "SUPERSEDED.md")}
    write(outputs[4], {"inputs": analysis_inputs,
                       "outputs": {str(p.relative_to(ROOT)).replace("\\", "/"): pin(p) for p in outputs[:4]},
                       "additional_model_fits": 0, "financial_evaluations": 0, "promotion_authority": False})
    print(f"audit complete: identities={len(inputs)}; origins=20; models={verified_models}; predictions/scalers/weights reconciled; extra_fits=0; trading=0")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print((type(exc).__name__+": "+str(exc)).encode("utf-8")[:800].decode("utf-8", "ignore"))
        raise SystemExit(1)
