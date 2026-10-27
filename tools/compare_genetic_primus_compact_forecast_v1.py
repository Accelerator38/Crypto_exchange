"""One bounded local step per invocation; gross forecast accuracy only."""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from exia.genetic_primus import compact_forecast_v1 as m
from exia.genetic_primus import compact_forecast_data_v1 as d

BASE = ROOT / "Reports/Exia/Genetic_Primus"
CONTRACT = ROOT / "configs/genetic_primus_compact_forecast_v1.json"
MANIFEST = ROOT / "Reports/Exia/Genetic_Primus/prospective_3d_v1_202609/data_manifest_origin_09.json"
TAIL = ROOT / "Retrodate/genetic_primus_v8_clean_ohlcv_final_20260928_20260930/dataset_seal_v1.json"
SAFETY = dict.fromkeys(("credentials_allowed", "orders_enabled", "execution_allowed", "paper", "live",
                       "promotion_authority", "training_allowed", "financial_evaluation_allowed", "network_allowed"), False)
START = int(dt.datetime(2026, 8, 1, tzinfo=dt.timezone.utc).timestamp()*1000)
END = int(dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc).timestamp()*1000)
SYMBOLS = ("ADA/USDT", "BNB/USDT", "BTC/USDT", "DOGE/USDT", "ETH/USDT", "LINK/USDT", "SOL/USDT", "XRP/USDT")
CODE = (Path(__file__), CONTRACT, ROOT / "src/exia/genetic_primus/compact_forecast_v1.py",
        ROOT / "src/exia/genetic_primus/compact_forecast_data_v1.py",
        ROOT / "tools/check_genetic_primus_compact_forecast_v1.py",
        ROOT / "tests/test_genetic_primus_compact_forecast_v1.py",
        ROOT / "docs/GENETIC_PRIMUS_COMPACT_FORECAST_V1.md")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def pin(path):
    content = path.read_bytes()
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def relative(path):
    return path.relative_to(ROOT).as_posix()


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def iso(value):
    return dt.datetime.fromtimestamp(value/1000, dt.timezone.utc).isoformat()


def validate_contract():
    c = read(CONTRACT)
    expected = {"window": "expanding", "purge_bars": 7, "train_end_offset_bars": -1,
                "inner_days": 60, "scaler": "unweighted_train_only_per_origin_ddof0_clip8",
                "loss": "weighted_mean_squared_error_all_outcomes", "l2": 0.01,
                "bias_regularized": False, "mlp_activation": "tanh", "learning_rate": 0.001,
                "max_epochs": 120, "patience": 12, "mlp_seeds": list(m.SEEDS),
                "seed_aggregation": "mean_not_best_seed", "optimizer": "deterministic_full_batch_adam", "workers": 1}
    if (c["training"] != expected or c["safety"] != SAFETY or c["features"] != list(m.FEATURES)
            or c["architectures"] != m.PARAMETERS or c["weightings"] != list(m.WEIGHTINGS)
            or c["recency"] != {"half_life_days": 90, "uniform_anchor": .2, "age_reference": "purged_train_cutoff",
                                "timestamp": "outcome_available_ms", "all_outcome_signs_weighted": True}
            or not c["benchmark"]["data_already_revealed"] or c["benchmark"]["hyperparameter_selection_allowed"]
            or c["target_kind"] != "gross_open_open_forecast_not_net_profit"
            or c.get("recency_formula") != "0.2/N + 0.8*exp2(-age_days/90)/sum(exp2(-age_days/90))"
            or c["market_net_training_targets_requalified"] or c["promotion_authority"]):
        raise ValueError("contract differs from implemented fixed diagnostic")
    return c


def prepare(folder):
    validate_contract()
    if folder.exists():
        raise ValueError("new report directory required")
    sources = read(MANIFEST)["snapshot_files"]
    tail = [f for f in read(TAIL)["files"] if f["path"].endswith("_240m.parquet")]
    if len(sources) != 11 or len(tail) != 1:
        raise ValueError("exact 11-source bank plus independently sealed final tail required")
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd
    frames, source_pins = [], {}
    required = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    for source in sources + tail:
        path = (ROOT / source["path"]).resolve()
        path.relative_to(ROOT / "Retrodate")
        receipt = pin(path)
        if receipt["sha256"] != source["sha256"] or ("size_bytes" in source and receipt["bytes"] != source["size_bytes"]):
            raise ValueError("source identity mismatch: " + relative(path))
        with pa.BufferReader(path.read_bytes()) as buffer:
            reader = pq.ParquetFile(buffer, pre_buffer=False, memory_map=False,
                                    thrift_string_size_limit=1000000, thrift_container_size_limit=100000)
            if not 0 < reader.metadata.num_rows <= 200000 or not set(required) <= set(reader.schema_arrow.names):
                raise ValueError("bounded physical OHLCV schema required")
            frame = reader.read(columns=required, use_threads=False).to_pandas(use_threads=False)
        if frame.timestamp.dtype.kind not in "iu":
            raise ValueError("integer millisecond timestamps required")
        if not set(frame.symbol) <= set(SYMBOLS) or frame.duplicated(["timestamp", "symbol"]).any():
            raise ValueError("unknown or duplicate source symbols")
        values = frame[["open", "high", "low", "close", "volume"]].to_numpy(float)
        if (not np.isfinite(values).all() or np.any(values[:, :4] <= 0) or np.any(values[:, 4] < 0)
                or np.any(values[:, 1] < np.maximum(values[:, 0], values[:, 3]))
                or np.any(values[:, 2] > np.minimum(values[:, 0], values[:, 3]))):
            raise ValueError("OHLCV integrity failure")
        frames.append(frame)
        source_pins[relative(path)] = receipt
    panel = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"])
    if panel.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("overlapping bank must be resolved before fitting")
    opening = panel.pivot(index="timestamp", columns="symbol", values="open").reindex(columns=SYMBOLS)
    closing = panel.pivot(index="timestamp", columns="symbol", values="close").reindex(columns=SYMBOLS)
    times = opening.index.to_numpy(dtype=np.int64)
    if int(times[0]) != 1640995200000 or int(times[-1])+d.BAR_MS != END:
        raise ValueError("expected frozen Jan2022-to-Oct2026 coverage")
    samples = d.make_samples(times, opening.to_numpy(), closing.to_numpy(), START)
    origins = START + np.arange(20, dtype=np.int64)*d.HORIZON_MS
    if np.any(origins + 20*d.BAR_MS > END):
        raise ValueError("benchmark outcome not mature")
    for origin in origins:
        d.eligible_split(samples, int(origin))
    banks = {}
    for seed in (3039, 7079):
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, 20, size=(4096, 10))
        banks["seed_" + str(seed)] = ((starts[:, :, None] + np.arange(2)) % 20).reshape(4096, 20).astype(np.int32)
    folder.mkdir(parents=True)
    with (folder / "sample_bank.npz").open("xb") as handle:
        np.savez_compressed(handle, **{k: v for k, v in samples.items() if isinstance(v, np.ndarray)})
    with (folder / "audit_resample_banks.npz").open("xb") as handle:
        np.savez_compressed(handle, **banks)
    registration = {"schema": "compact_forecast_registration/1", "registered_before_model_fits": True,
                    "arms": [a + "_" + w for a in m.PARAMETERS for w in m.WEIGHTINGS],
                    "seeds": list(m.SEEDS), "no_best_seed_selection": True,
                    "cutoffs_ms": [int(origin)-8*d.BAR_MS for origin in origins],
                    "outer_origins_sha256": m.array_identity(origins), "data_already_revealed": True,
                    "global_trial_registry_imported": False, "multiplicity_adjusted_claim_allowed": False,
                    "scope": "forecast_accuracy_not_profit", "safety": SAFETY}
    registration["superseded_attempt"] = read(CONTRACT)["superseded_attempt"]
    write(folder / "registration.json", registration)
    preparation = {"schema": "compact_forecast_preparation/1", "sources": source_pins,
                   "code_inputs": {relative(p): pin(p) for p in CODE + (MANIFEST, TAIL)},
                   "outputs": {p.name: pin(p) for p in [folder / "sample_bank.npz", folder / "audit_resample_banks.npz", folder / "registration.json"]},
                   "symbols": list(SYMBOLS), "source_file_count": len(source_pins), "market_rows": len(panel),
                   "timestamp_count": len(times), "feature_definition": d.FEATURE_DEFINITION,
                   "feature_definition_sha256": d.FEATURE_SHA256,
                   "sample_content_sha256": m.array_identity(samples["origins"], samples["available"], samples["x"], samples["y"]),
                   "benchmark_origins_ms": origins.tolist(), "benchmark_start_ms": START, "benchmark_end_exclusive_ms": END,
                   "bootstrap": {"samples": 4096, "block_origins": 2, "selection_seed": 3039, "audit_seed": 7079,
                                 "selection_content_sha256": m.array_identity(banks["seed_3039"]),
                                 "audit_content_sha256": m.array_identity(banks["seed_7079"]), "alpha": .05},
                   "model_fit_count": 0, "financial_evaluation_count": 0, "network_used": False,
                   "safety": SAFETY}
    preparation["sha256"] = m.identity(preparation)
    write(folder / "preparation.json", preparation)
    print(f"prepared: files={len(source_pins)}; market_rows={len(panel)}; eligible_sample_origins={len(samples['origins'])}; benchmark=20; fits=0; network=0")


def verify(folder):
    prep = read(folder / "preparation.json")
    if prep["sha256"] != m.identity({k: v for k, v in prep.items() if k != "sha256"}) or prep["safety"] != SAFETY:
        raise ValueError("preparation identity/safety changed")
    for path, expected in (prep["sources"] | prep["code_inputs"]).items():
        if pin(ROOT / path) != expected:
            raise ValueError("frozen input changed: " + path)
    for path, expected in prep["outputs"].items():
        if pin(folder / path) != expected:
            raise ValueError("prepared output changed: " + path)
    validate_contract()
    return prep


def fit(folder, index):
    prep = verify(folder)
    if index is None or not 0 <= index < 20:
        raise ValueError("one origin index 0..19 required")
    output = folder / f"origin_{index:02d}.json"
    if output.exists():
        raise ValueError("immutable existing origin refused")
    with np.load(folder / "sample_bank.npz", allow_pickle=False) as bank:
        samples = {k: bank[k] for k in bank.files}
    result = d.fit_origin(samples, prep["benchmark_origins_ms"][index])
    result["preparation_sha256"] = prep["sha256"]
    result["symbols"] = list(SYMBOLS)
    result["sha256"] = m.identity({k: v for k, v in result.items() if k != "sha256"})
    write(output, result)
    print(f"origin={index:02d}; asof={iso(result['origin_ms'])}; fits={result['model_fit_count']}; steps={result['gradient_steps']}; origins={result['train_origin_count']}; losses={result['target_negative_count']}; network=0; trading=0")


def accuracy(actual, prediction):
    y, p = np.asarray(actual), np.asarray(prediction)
    error = 10000*(p-y)
    mask = y != 0
    return {"rmse_bps": float(np.sqrt(np.mean(error**2))), "mae_bps": float(np.mean(np.abs(error))),
            "sign_accuracy": float(np.mean(np.sign(y[mask]) == np.sign(p[mask]))) if mask.any() else None,
            "mse_bps_squared": float(np.mean(error**2))}


def report(folder):
    prep = verify(folder)
    if (folder / "REPORT.md").exists() or (folder / "comparison.json").exists():
        raise ValueError("immutable report outputs already exist")
    checks = read(folder / "engineering_checks.json")
    if checks["scope"] != "all" or checks["checks"] != 23 or checks["failures"] or checks["errors"]:
        raise ValueError("all engineering checks required")
    for path, digest in checks["source_sha256"].items():
        if pin(ROOT / path)["sha256"] != digest:
            raise ValueError("engineering-check source changed")
    results = [read(folder / f"origin_{i:02d}.json") for i in range(20)]
    for i, result in enumerate(results):
        if (result["sha256"] != m.identity({k: v for k, v in result.items() if k != "sha256"})
                or result["preparation_sha256"] != prep["sha256"]
                or result["origin_ms"] != prep["benchmark_origins_ms"][i]):
            raise ValueError("origin receipt changed")
        for arm in result["arms"].values():
            for payload in arm["models"]:
                m.predict(payload, np.zeros((1, 6)))
    actual = np.array([r["actual"] for r in results])
    controls = {"zero_forecast_not_NoTrade_equity": np.zeros_like(actual),
                "causal_train_mean": np.array([r["baseline_train_mean"] for r in results])}
    predictions = controls | {a: np.array([r["arms"][a]["prediction"] for r in results]) for a in results[0]["arms"]}
    with np.load(folder / "audit_resample_banks.npz", allow_pickle=False) as banks:
        selection, audit = banks["seed_3039"], banks["seed_7079"]
    comparison = {"schema": "compact_forecast_comparison/1", "scope": "gross_accuracy_not_profit",
                  "origins": 20, "outcomes": int(actual.size), "data_already_revealed": True,
                  "positive_outcomes": int(np.count_nonzero(actual > 0)), "negative_outcomes": int(np.count_nonzero(actual < 0)),
                  "arms_and_controls": {}, "model_fit_count": sum(r["model_fit_count"] for r in results),
                  "gradient_steps": sum(r["gradient_steps"] for r in results),
                  "scaler_fit_count": sum(r["scaler_fit_count"] for r in results),
                  "bootstrap": prep["bootstrap"], "preparation_sha256": prep["sha256"],
                  "financial_evaluation_count": 0, "market_net_training_targets_requalified": False,
                  "profit_ranking_available": False, "promotion_authority": False, "safety": SAFETY}
    comparison["superseded_attempt"] = read(CONTRACT)["superseded_attempt"]
    comparison["total_attempt_market_forecast_fits"] = comparison["model_fit_count"] + read(CONTRACT)["superseded_attempt"]["market_forecast_fits"]
    baseline_error = ((predictions["causal_train_mean"]-actual)*10000)**2
    for name, prediction in predictions.items():
        error = ((prediction-actual)*10000)**2
        delta = (baseline_error-error).mean(axis=1)
        metrics = accuracy(actual, prediction)
        metrics.update({"mse_improvement_vs_causal_mean_bps_squared": float(delta.mean()),
                        "selection_lcb_mse_improvement": float(np.quantile(delta[selection].mean(axis=1), .05)),
                        "audit_lcb_mse_improvement": float(np.quantile(delta[audit].mean(axis=1), .05)),
                        "origins_better_than_mean": int(np.count_nonzero(delta > 0)),
                        "per_origin_rmse_bps": np.sqrt(error.mean(axis=1)).tolist(),
                        "per_origin_mse_improvement_bps_squared": delta.tolist()})
        if name in results[0]["arms"]:
            arms = [r["arms"][name] for r in results]
            metrics.update({"parameters_per_model": arms[0]["parameter_count_per_model"],
                            "stored_parameters_per_origin": arms[0]["stored_parameter_count"],
                            "origin_ess_range": [min(a["origin_ess"] for a in arms), max(a["origin_ess"] for a in arms)],
                            "negative_weight_mass_range": [min(a["negative_weight_mass"] for a in arms), max(a["negative_weight_mass"] for a in arms)],
                            "positive_weight_mass_range": [min(a["positive_weight_mass"] for a in arms), max(a["positive_weight_mass"] for a in arms)]})
            if name.startswith("mlp8"):
                metrics["seed_rmse_bps"] = [accuracy(actual, np.array([a["seed_predictions"][j] for a in arms]))["rmse_bps"] for j in range(3)]
                metrics["selected_epoch_range"] = [min(s["selected_epoch"] for a in arms for s in a["epoch_selection"]), max(s["selected_epoch"] for a in arms for s in a["epoch_selection"])]
        comparison["arms_and_controls"][name] = metrics
    write(folder / "comparison.json", comparison)
    os.environ["MPLCONFIGDIR"] = str(folder / ".mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dates = [dt.datetime.fromtimestamp(r["origin_ms"]/1000, dt.timezone.utc) for r in results]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    labels = {"linear6_uniform": "Linear 7 / uniform", "linear6_half_life_90_anchor20": "Linear 7 / HL90 + anchor20",
              "mlp8_uniform": "MLP 65x3 / uniform", "mlp8_half_life_90_anchor20": "MLP 65x3 / HL90 + anchor20",
              "causal_train_mean": "Causal train mean", "zero_forecast_not_NoTrade_equity": "Zero forecast (not trading equity)"}
    for name, metrics in comparison["arms_and_controls"].items():
        if name == "zero_forecast_not_NoTrade_equity":
            continue
        axes[0].plot(dates, metrics["per_origin_rmse_bps"], label=labels[name], linewidth=1.6)
        if name != "causal_train_mean":
            axes[1].plot(dates, np.cumsum(metrics["per_origin_mse_improvement_bps_squared"]), label=labels[name], linewidth=1.6)
    axes[0].set_title("Open Aug-Sep 2026: causal 3-day gross forecasts, NOT trading profit")
    axes[0].set_ylabel("RMSE (bps)")
    axes[1].set_ylabel("Cumulative paired MSE benefit\nvs causal train mean (bps squared)")
    axes[1].axhline(0, color="black", linewidth=.8)
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend(fontsize=8, loc="best")
    fig.savefig(folder / "forecast_errors.png", dpi=150)
    fig.savefig(folder / "forecast_errors.svg")
    plt.close(fig)
    lines = ["# Компактные прогнозные модели: открытый август–сентябрь 2026", "",
             "## Что именно сравнивается", "",
             "Новый зарегистрированный gross predictor, а не обучение на старых заблокированных net-targets. "
             "20 причинных трёхдневных origins, 160 исходов восьми символов; expanding-переобучение каждые три дня. "
             "20 origins — временные группы, не 160 независимых экспериментов. Период уже раскрыт, это не sealed prospective.", "",
             "Linear: 6→1, 7 параметров. MLP: 6→8 tanh→1, 65 параметров; среднее трёх фиксированных seeds, "
             "195 сохранённых коэффициентов на origin. Scaler/target statistics не входят в число весов сети.", "",
             "Uniform и half-life 90 дней + **20% uniform anchor**. Взвешены положительные, отрицательные и нулевые "
             "исходы; вес зависит только от доступности outcome и purged cutoff. Это не profit-only weighting.", "",
             "## Результаты прогнозов", "",
             "| Вариант | RMSE, bps ↓ | MAE, bps ↓ | Верный знак | LCB ΔMSE vs mean | Audit LCB ΔMSE |", "|---|---:|---:|---:|---:|---:|"]
    for name, metrics in comparison["arms_and_controls"].items():
        lines.append(f"| {labels[name]} | {metrics['rmse_bps']:.3f} | {metrics['mae_bps']:.3f} | {metrics['sign_accuracy']:.1%} | {metrics['selection_lcb_mse_improvement']:.2f} | {metrics['audit_lcb_mse_improvement']:.2f} |")
    best = min(results[0]["arms"], key=lambda a: comparison["arms_and_controls"][a]["rmse_bps"])
    lines += ["", f"Минимальная диагностическая RMSE среди моделей: **{labels[best]}**. Это descriptive ranking прогнозов, "
              "не финансовый победитель, не выбор нового production incumbent. ΔMSE > 0 означает улучшение прогноза относительно "
              "причинного train-mean, а не положительную прибыль. Bootstrap общий, блок 2 origins, 4 096 resamples, seeds 3039/7079; "
              "односторонний 5% quantile. Бюджет тестирования и старые попытки не скорректированы DSR/PBO.", "",
              "![Ошибки прогнозов](forecast_errors.png)", "", "## Причинность и воспроизводимость", "",
              f"12 источников, {prep['market_rows']:,} OHLCV-строк, {prep['timestamp_count']:,} timestamps. "
              f"Источник с января 2022 до 1 октября 2026 UTC. Хеш каждого файла проверен перед каждым origin; "
              f"новый банк данных, feature definition, folds, weights, scaler/model SHA и оба resample banks зафиксированы.", "",
              "Feature-asof origin 00:00, entry следующий open 04:00, exit entry+18 баров, доступность outcome exit+1 бар. "
              "Train_end=origin−1 бар, cutoff=train_end−7 баров; inner validation последние 60 дней доступных origins с отдельным "
              "purge. Scaler обучается заново на каждом origin; inner scaler не видит validation. "
              "Epoch выбирается только на прошлом inner block; L2=0.01, lr=0.001, 120 epochs/patience12 заранее фиксированы. "
              "Нет подбора по августу–сентябрю; предшествующие benchmark outcomes причинно добавляются в последующие fit prefixes.", "",
              f"23 детерминированные проверки пройдены. {comparison['model_fit_count']} market forecast fits "
              f"(40 linear + 120 inner MLP + 120 full-prefix refits), {comparison['gradient_steps']} gradient steps; "
              f"{comparison['scaler_fit_count']} scalers. Никакого pytest, collector, subprocess, ожиданий wall-clock или сети.", "",
              "| Вариант | ESS origins, диапазон | Масса убыточных исходов | Масса прибыльных исходов |", "|---|---:|---:|---:|"]
    for name in results[0]["arms"]:
        a = comparison["arms_and_controls"][name]
        lines.append(f"| {labels[name]} | {a['origin_ess_range'][0]:.1f}–{a['origin_ess_range'][1]:.1f} | {a['negative_weight_mass_range'][0]:.3f}–{a['negative_weight_mass_range'][1]:.3f} | {a['positive_weight_mass_range'][0]:.3f}–{a['positive_weight_mass_range'][1]:.3f} |")
    for name in results[0]["arms"]:
        a = comparison["arms_and_controls"][name]
        if name.startswith("mlp8"):
            lines.append("")
            lines.append(f"{labels[name]}: RMSE отдельных фиксированных seeds {a['seed_rmse_bps']}; выбранные inner epochs {a['selected_epoch_range']}. Seeds не независимые outer slices.")
    lines += ["", "## Ограничения и следующий шаг", "",
              "MSE/верный знак не являются net log-growth. Модели не привязаны к торговым policy thresholds, costs/funding; "
              "нет финансовых прогонов и equity curve, потому что денежный ledger для этих прогнозов не рассчитывался. "
              "Zero forecast — контроль прогноза, не доходность NoTrade. Архивные GA/EMA результаты нельзя ранжировать рядом с этой RMSE.", "",
              "Zero-lag candle availability и мгновенная готовность fit — зарегистрированные research assumptions, не измеренный "
              "execution latency. MLP seeds и четыре arms увеличивают adaptive testing budget; global registry не интегрирован, "
              "DSR/PBO и promotion не вычисляются. Нет подтверждения out-of-sample прибыльности или live readiness.", "",
              "Первый engineering-прогон compact_forecast_v1_20261007 сохранён как superseded: raw floor 0.2 "
              "не сохранял 20% нормированной uniform mass. Формула исправлена до повторных fits: "
              "w=0.2/N+0.8*recency/sum(recency). Признаки, seeds и hyperparameters не менялись. "
              "Дополнительные 280 fits первого прогона не скрыты: суммарный attempt budget 560 market forecast fits.", "",
              "Дальше: зафиксировать predictor→net-policy контракт без настройки на этом benchmark; обеспечить подтверждённые "
              "cost/funding inputs и причинную задержку; сравнить политики через единый shared-capital ledger и текущую net-growth fitness "
              "с исследовательскими DD10/20/30%. До этого не подключать GA и не запускать реальную торговлю.", "",
              "Артефакты: preparation.json, registration.json, sample_bank.npz, audit_resample_banks.npz, "
              "origin_00…19.json (модели, прогнозы, scaler/cutoff/fold/weight identities), comparison.json, engineering_checks.json, evidence_receipt.json."]
    with (folder / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines)+"\n")
    outputs = [p for p in folder.iterdir() if p.is_file() and p.name != "evidence_receipt.json"]
    receipt = {"schema": "compact_forecast_evidence/1", "inputs": prep["sources"] | prep["code_inputs"],
               "outputs": {relative(p): pin(p) for p in outputs}, "safety": SAFETY,
               "model_fit_count": comparison["model_fit_count"], "financial_evaluation_count": 0}
    write(folder / "evidence_receipt.json", receipt)
    print(f"report complete: origins=20; checks=23; fits={comparison['model_fit_count']}; best_forecast={best}; trading=0; promotion=false")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("prepare", "fit", "report"))
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--origin-index", type=int)
    args = parser.parse_args()
    folder = (ROOT / args.report_dir).resolve()
    folder.relative_to(BASE)
    if folder == BASE:
        raise ValueError("report subdirectory required")
    if args.step == "prepare":
        prepare(folder)
    elif args.step == "fit":
        fit(folder, args.origin_index)
    else:
        report(folder)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print((type(exc).__name__ + ": " + str(exc)).encode("utf-8")[:800].decode("utf-8", "ignore"))
        raise SystemExit(1)
