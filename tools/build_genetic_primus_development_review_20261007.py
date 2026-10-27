"""Read archived JSON evidence and draw its equity curves; no strategy execution."""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "Reports/Exia/Genetic_Primus"
DEST = ROOT / BASE / "development_plan_20261007"
INPUTS = {}


def read(relative):
    path = ROOT / relative
    if path.stat().st_size > 3 * 1024 * 1024:
        raise ValueError("archive input exceeds bounded read size")
    data = path.read_bytes()
    INPUTS[relative] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return json.loads(data)


def pin(relative):
    path = ROOT / relative
    if path.stat().st_size > 512_000:
        raise ValueError("code/document input exceeds cap")
    data = path.read_bytes()
    INPUTS[relative] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def curve(returns):
    values = [100.0]
    for value in returns:
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= -1:
            raise ValueError("invalid archived daily return")
        values.append(values[-1] * (1 + value))
    peak = values[0]
    dd = []
    for value in values:
        peak = max(peak, value)
        dd.append(100 * (value / peak - 1))
    return values, dd


def archived_origins(paths):
    series = {cost: {} for cost in ("normal", "stress")}
    first = previous_end = None
    revealed = []
    identities = []
    for path in paths:
        result = read(path)
        start, end = result["origin_start"], result["origin_end_exclusive"]
        if end - start != 3 * 86_400_000 or (previous_end is not None and start != previous_end):
            raise ValueError("archive origins are not consecutive complete 3-day windows")
        if first is None:
            first = start
        previous_end = end
        revealed.append(result.get("data_already_revealed"))
        identities.append({"path": path, "producer_config_identity": {key: result[key] for key in
                           ("config_sha256", "contract_sha256", "base_contract_sha256") if key in result},
                           "model_sha256": {name: item["model_sha256"] for name, item in
                                result["evaluations"]["normal"]["candidates"].items()}})
        for cost in series:
            for policy, data in result["evaluations"][cost]["candidates"].items():
                daily = data["daily_returns"]
                if len(daily) != 3:
                    raise ValueError("origin daily observation count")
                series[cost].setdefault(policy, []).extend(daily)
    return first, previous_end, series, identities, revealed


def summary_rows(series):
    rows = []
    for cost, policies in series.items():
        for policy, returns in policies.items():
            equity, dd = curve(returns)
            rows.append({"policy": policy, "cost": cost, "days": len(returns),
                         "net_return_pct": equity[-1] - 100,
                         "max_drawdown_daily_close_pct": -min(dd),
                         "daily_returns": returns})
    return rows


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("reconstructed equity disagrees with archived summary")


def main():
    if DEST.exists():
        raise ValueError("report output already exists; no overwrite")
    august_paths = [f"{BASE}/revealed_august_diagnostic_v1_20260904/origin_{i:02d}.json" for i in range(1, 11)]
    a_start, a_end, august, a_models, a_revealed = archived_origins(august_paths)
    august_summary = read(f"{BASE}/revealed_august_diagnostic_v1_20260904/summary.json")
    for cost in august:
        for name, returns in august[cost].items():
            close(curve(returns)[0][-1] / 100 - 1,
                  august_summary["costs"][cost]["candidates"][name]["compounded_return"])
    september_paths = [f"{BASE}/prospective_3d_v1_202609/origin_{i:02d}.json" for i in range(1, 10)]
    september_paths.append(f"{BASE}/september_final_diagnostic_20261001/origin_10_diagnostic.json")
    s_start, s_end, september, s_models, s_revealed = archived_origins(september_paths)
    september_summary = read(f"{BASE}/september_final_diagnostic_20261001/verification_and_summary.json")
    for cost in september:
        for name, returns in september[cost].items():
            close(curve(returns)[0][-1] / 100 - 1,
                  september_summary["details"][cost][name]["descriptive_9_plus_1_return"])
    replay = read(f"{BASE}/baseline_economics_audit_v1_20261002/result.json")
    replay_series = {cost: {} for cost in ("normal", "stress")}
    replay_rows = []
    for name, record in replay["policies"].items():
        for cost, evaluation in record["evaluations"].items():
            metrics, economics = evaluation["metrics"], evaluation["economics"]
            returns = metrics["daily_returns"]
            close(curve(returns)[0][-1] - 100, metrics["net_return_pct"])
            replay_series[cost][name] = returns
            replay_rows.append({"policy": name, "cost": cost,
                "net_return_pct": metrics["net_return_pct"], "archived_max_drawdown_pct": metrics["max_drawdown_pct"],
                "trades": metrics["trades"], "gross_return_pct": metrics["gross_return_pct"],
                "mean_capital_exposure_fraction": economics["mean_portfolio_capital_exposure_fraction"],
                "fees_pct_initial_capital": economics["fees_pct_initial_capital"]})
    attribution = read(f"{BASE}/episode_attribution_v3_20261002/result.json")
    warm = read(f"{BASE}/warmstart_august_v1_20260907/summary.json")
    linear = read(f"{BASE}/rolling_linear_matrix_v1_20261002/result.json")
    reversal = read(f"{BASE}/reversal_calibration_pilot_v1_20261005/result.json")
    v4 = read(f"{BASE}/v4_final_20260824/manifest.json")
    v4_strict = read(f"{BASE}/v4_final_20260824/strict_base16_metrics.json")
    model = read(f"{BASE}/research_snapshot_f7a4ae11_20260828/model.json")
    semantics = read(f"{BASE}/accounting_semantics_audit_v1_20261005/summary.json")
    coverage = read(f"{BASE}/market_feature_materialization_v1_20261007/preparation.json")["metadata"]
    tail_manifest = read("Retrodate/genetic_primus_v8_clean_ohlcv_final_20260928_20260930/derived_v1/origin_10_manifest.json")
    tail_seal = read("Retrodate/genetic_primus_v8_clean_ohlcv_final_20260928_20260930/dataset_seal_v1.json")
    current_net = read(f"{BASE}/reference_open_net_inputs_v1_20261007/result.json")
    current_labels = read(f"{BASE}/market_gross_labels_v1_20261007/result.json")
    current_features = read(f"{BASE}/market_feature_materialization_v1_20261007/result.json")
    for path in ["src/exia/genetic_primus/fitness_v4.py", "src/exia/genetic_primus/fitness_recency.py",
                 "src/exia/genetic_primus/evolution_v4.py", "src/exia/genetic_primus/rolling_v4.py",
                 "src/exia/genetic_primus/policy_v4.py", "src/exia/genetic_primus/rolling_linear_retrain_v2.py",
                 "src/exia/genetic_primus/market_relative_reversal_linear_v1.py",
                 "src/exia/genetic_primus/reference_open_net_accounting_v1.py",
                 "configs/genetic_primus_overlay_3d_v6_relative_opportunity_sealed.json",
                 "configs/genetic_primus_rolling_linear_matrix_v1.json",
                 "configs/genetic_primus_reference_open_net_contract_v1.json",
                 "Reports/Exia/Genetic_Primus/development_review_v1_20261005/REPORT.md",
                 "tools/build_genetic_primus_development_review_20261007.py"]:
        pin(path)
    facts = {
        "schema": "exia.genetic_primus.development_review_20261007/1",
        "evidence_kind": "archive_read_and_current_static_code_review_no_new_strategy_run",
        "august": {"start_ms": a_start, "end_exclusive_ms": a_end, "rows": summary_rows(august),
                   "archive_summary": august_summary, "origin_model_contexts": a_models, "revealed_flags": a_revealed},
        "september_9_plus_1": {"start_ms": s_start, "end_exclusive_ms": s_end, "rows": summary_rows(september),
                   "archive_summary": september_summary, "origin_model_contexts": s_models,
                   "same_prospective_protocol_for_all_ten": False},
        "september_replay": {"start_ms": replay["interval_start"], "end_exclusive_ms": replay["interval_end_exclusive"],
                             "metrics": replay_rows, "rows": summary_rows(replay_series)},
        "attribution": {key: attribution[key] for key in ("all_controls_match", "distinct_control_targets",
                         "historical_support_gate", "comparisons", "maximum_changed_decision_rows")},
        "warm_start": {key: warm[key] for key in ("origins", "same_winner_as_cold_origins", "screen_passes",
                                                 "search_evaluations", "warm_plus_cold_evaluations")},
        "linear_matrix": {"selected_candidate": linear["selected_candidate"], "validation_result": linear["validation_result"],
                          "records": [{"features": item["feature_set"], "variant": item["variant"],
                                       "evaluations": {c: {k: v for k, v in e.items() if k != "daily_returns"}
                                            for c, e in item["evaluations"].items()}}
                                       for item in linear["records"]]},
        "reversal_summary": reversal["summary"],
        "v4": {"architecture": v4["architecture"], "parameter_count": v4["parameter_count"], "strict": v4_strict},
        "frozen_ga_model": model["model"], "historical_accounting_audit": semantics,
        "source_coverage": coverage,
        "tail_metadata": {"start_ms": tail_manifest["start_ms"], "end_exclusive_ms": tail_manifest["end_exclusive_ms"],
                          "four_hour_rows": tail_seal["four_hour_rows"], "raw_prices_read_for_this_review": False,
                          "already_has_separate_diagnostic_result": True},
        "latest_features": current_features, "latest_gross_labels": current_labels, "latest_net_inputs": current_net,
        "new_model_fits": 0, "new_strategy_evaluations": 0, "raw_parquet_reads": 0,
        "network_used": False, "trading_runtime_used": False,
        "archived_endpoint_reconstruction_verified": True,
    }
    DEST.mkdir(exist_ok=False)
    (DEST / "figures").mkdir(exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(DEST / ".matplotlib")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.22, "axes.spines.top": False, "axes.spines.right": False})
    names = {"NoTrade": "NoTrade", "GA_overlay_f7a4ae11": "GA f7a4ae11", "EMA_TrendConsensus": "EMA long/short",
             "GA_target_gate": "GA target-gate", "GA_entry_only": "GA entry-only", "EMA_long_only": "EMA long-only"}
    colors = {"NoTrade": "#686e77", "GA_overlay_f7a4ae11": "#2563eb", "GA_target_gate": "#2563eb",
              "GA_entry_only": "#9333ea", "EMA_TrendConsensus": "#e56b2f", "EMA_long_only": "#16856b"}

    def plot(series, start_ms, filename, title, caption):
        figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.0), sharex="col")
        start = datetime.fromtimestamp(start_ms / 1000, timezone.utc)
        legend = []
        for col, cost in enumerate(("normal", "stress")):
            for name, returns in series[cost].items():
                equity, dd = curve(returns)
                dates = [start + timedelta(days=i) for i in range(len(equity))]
                line, = axes[0, col].plot(dates, equity, color=colors[name], label=names[name], linewidth=2)
                axes[1, col].plot(dates, dd, color=colors[name], linewidth=1.8)
                if col == 0:
                    legend.append(line)
            axes[0, col].set_title("Архивный normal: 16 bps round-trip" if col == 0 else "Архивный stress: 24 bps round-trip", fontsize=11)
            axes[0, col].axhline(100, color="#777777", linewidth=0.6, linestyle=":")
            axes[0, col].set_ylabel("Капитал, усл. ед. (старт 100)")
            axes[1, col].set_ylabel("Просадка по дневным закрытиям, %")
            axes[1, col].xaxis.set_major_locator(mdates.DayLocator(interval=5))
            axes[1, col].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
            axes[1, col].set_xlabel("Дата UTC")
        figure.suptitle(title, fontsize=15, fontweight="bold", y=0.985)
        figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=len(legend), frameon=False)
        figure.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=9, color="#424b57")
        figure.tight_layout(rect=(0.01, 0.055, 0.99, 0.9))
        for suffix in ("png", "svg"):
            figure.savefig(DEST / "figures" / f"{filename}.{suffix}", dpi=140)
        plt.close(figure)

    plot(august, a_start, "august_archive", "Август 2026: архивные окна 1–30 августа",
         "Раскрытый исследовательский период; 31 августа в этой серии отсутствует.\nКривые восстановлены из архивных daily returns, без нового запуска стратегии.")
    plot(september, s_start, "september_archive", "Сентябрь 2026: девять окон + отдельная финальная диагностика",
         "Геометрическая цепочка закрытых окон; весь месяц не является единым завершённым prospective-тестом.\nАрхивные costs и исполнение не заменяют новый reference-open контракт.")
    plot(replay_series, replay["interval_start"], "september_comparable_replay", "Сопоставимый replay пяти политик: 1–27 сентября 2026",
         "Один архивный accounting и период; экспозиция, длительность позиций и search budget различаются.\nДневная просадка на графике может отличаться от intraperiod drawdown в архивных метриках.")
    with (DEST / "facts.json").open("x", encoding="utf-8") as output:
        json.dump(facts, output, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        output.write("\n")
    for path, identity in INPUTS.items():
        data = (ROOT / path).read_bytes()
        if len(data) != identity["bytes"] or hashlib.sha256(data).hexdigest() != identity["sha256"]:
            raise ValueError("evidence changed while building review")
    outputs = {}
    for path in [DEST / "facts.json", *sorted((DEST / "figures").glob("*"))]:
        data = path.read_bytes()
        outputs[str(path.relative_to(DEST)).replace("\\", "/")] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    receipt = {"schema": "exia.genetic_primus.development_review_artifact_receipt/1", "inputs": INPUTS,
               "outputs": outputs, "new_fits": 0, "new_strategy_runs": 0, "parquet_reads": 0,
               "report_markdown_is_separate_editorial_artifact": True}
    with (DEST / "evidence_receipt.json").open("x", encoding="utf-8") as output:
        json.dump(receipt, output, ensure_ascii=False, sort_keys=True, indent=2)
        output.write("\n")
    print(f"REVIEW DATA OK: sources={len(INPUTS)}; figures=3 PNG+SVG; archive endpoints reconciled; fits/strategy-runs=0")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("REVIEW BUILD FAILED: " + type(error).__name__ + ": " + str(error)[:180])
        raise SystemExit(1) from None
