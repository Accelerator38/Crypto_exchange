"""One short offline stage for EMA decision clock versus unit retention."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from tools import run_genetic_primus_forecast4h_v1 as b
from tools import run_genetic_primus_forecast4h_positions_v1 as previous
from exia.genetic_primus import ema_units_v1 as e

np = b.np
OUT = b.BASE / "ema_units_v1_20261012"
CONFIG = ROOT / "configs/genetic_primus_ema_units_v1.json"
CODE = (Path(__file__), CONFIG, ROOT / "src/exia/genetic_primus/ema_units_v1.py")


def write(name, value):
    value = b.f.p.c.plain(value)
    value["sha256"] = b.f.m.identity(value)
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return value


def data_inputs(parent):
    data, vectors = b.load_data(), b.vector_bank(parent)
    covariances = {t: e.p.risk_covariance(data["times"], data["closes"], t)[0]
                   for t in range(parent["start_ms"], parent["end_ms"], e.DAY_MS)}
    return data, vectors, covariances, b.sealed(b.MODELED_RATES)["events"]


def checks():
    contract = b.read(CONFIG)
    result = e.direct_checks(contract)
    # A full-day analytical unit-hold example: 100 -> 99, then flat.
    toy = deepcopy(contract)
    toy["costs"] = {name: {"fee_bps": 0, "drag_bps": 0} for name in ("normal", "stress")}
    t0 = 1000 * e.DAY_MS
    times = t0 + np.arange(7, dtype=np.int64) * e.BAR_MS
    prices = np.full((7, 8), 100.)
    prices[1:, 0] = 99.
    features = np.zeros((7, 8, 6))
    features[:, 0, 0] = 1.
    vectors = [{"asof_ms": int(t), "target_ms": int(t + e.BAR_MS), "available_us": int(t * 1000 + 1000000),
                "sha256": "synthetic_vector", "model_sha256": "synthetic_forecast_not_used"} for t in times[:-1]]
    data = {"times": times, "closes": prices, "features": features, "symbols": [f"A{i}" for i in range(8)]}
    covariance = {t0: np.eye(8) * .0001}
    run = e.replay(data, vectors, covariance, [], toy, "normal", retain_units=True)
    if abs(run["metrics"]["final_capital"] - 9990.) > 1e-10 or run["metrics"]["fills"] != 2:
        raise ValueError("analytical fixed-unit capital/fill check failed")
    if any(row["side"] == "BUY" and row["quantity_before"] > 1e-12 for row in run["ledger"] if row["kind"] == "modeled_fill"):
        raise ValueError("held units unexpectedly topped up")
    control = e.replay(data, vectors, covariance, [], toy, "normal", retain_units=False)
    if control["metrics"]["fills"] <= 2:
        raise ValueError("synthetic target rebalance did not exercise the differing path")
    result["checks_passed"] += 4
    OUT.mkdir(parents=True, exist_ok=True)
    write("direct_checks.json", result)
    print(json.dumps(result))


def prepare():
    parent, previous_reg, contract = b.registration(), previous.registration(), b.read(CONFIG)
    check = b.sealed(OUT / "direct_checks.json")
    if any(contract["safety"].values()):
        raise ValueError("offline authorities must remain false")
    parent_risk = parent["contract"]["risk"]
    if {k: v for k, v in contract["risk"].items() if k != "numerical_tolerance"} != parent_risk:
        raise ValueError("risk estimator or budget changed")
    for name in ("normal", "stress"):
        if contract["costs"][name] != parent["contract"]["costs"][name]:
            raise ValueError("costs changed")
    if contract["period"]["start"] != parent["contract"]["period"]["start"] or contract["period"]["end_exclusive"] != parent["contract"]["period"]["end_exclusive"]:
        raise ValueError("evaluation period changed")
    input_pins, parent_artifacts = {}, 0
    for folder in (b.OUT, previous.OUT):
        path = folder / "receipt.json"
        receipt = b.sealed(path)
        input_pins[b.relative(path)] = b.pin(path)
        for name, expected in receipt["artifact_pins"].items():
            if b.pin(folder / name) != expected:
                raise ValueError("frozen parent artifact changed: " + name)
            parent_artifacts += 1
    inputs = [b.DATA, b.MODELED_RATES, b.OUT / "registration.json", b.OUT / "vectors.json", b.OUT / "resample_bank.npz",
              previous.OUT / "registration.json", previous.OUT / "final_verification.json", OUT / "direct_checks.json"]
    inputs += list(sorted(b.OUT.glob("engineering_amendment_*.json")))
    inputs += [b.OUT / f"replay_{name}_{scenario}.json" for name in ("R1_EMA_4h", "R0_EMA_daily") for scenario in ("normal", "stress")]
    input_pins.update({b.relative(path): b.pin(path) for path in inputs})
    sources = CODE + (Path(b.__file__), Path(previous.__file__), ROOT / "src/exia/genetic_primus/forecast4h_v1.py",
                      ROOT / "src/exia/genetic_primus/position_policy_v1.py", ROOT / "src/exia/genetic_primus/compact_forecast_v1.py",
                      ROOT / "src/exia/genetic_primus/compact_forecast_data_v1.py")
    reg = write("registration.json", {"schema": "ema_units_registration/1", "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract, "contract_sha256": e.m.identity(contract), "parent_registration_sha256": parent["sha256"],
        "previous_registration_sha256": previous_reg["sha256"], "start_ms": parent["start_ms"], "end_ms": parent["end_ms"],
        "symbols": parent["symbols"], "code_pins": {b.relative(path): b.pin(path) for path in sources}, "input_pins": input_pins,
        "parent_artifact_pins_verified": parent_artifacts, "new_fits": 0, "new_policy_arms": 1, "new_threshold_trials": 0,
        "successful_policy_replay_budget": 2, "parity_replay_budget": 2,
        "already_seen": "all original forecast/EMA/position diagnostic results on this interval; adaptive research, no independent validation",
        "direct_checks_passed": check["checks_passed"], "safety": contract["safety"]})
    print(json.dumps({"registered": reg["sha256"], "new_fits": 0, "policy_arms": 1, "parent_pins": parent_artifacts}))


def registration():
    b.registration()
    previous.registration()
    reg = b.sealed(OUT / "registration.json")
    for name, expected in (reg["code_pins"] | reg["input_pins"]).items():
        if b.pin(ROOT / name) != expected:
            raise ValueError("registered input/code changed: " + name)
    return reg


def parity(scenario):
    reg = registration()
    args = data_inputs(b.registration())
    run = e.replay(*args, reg["contract"], scenario, retain_units=False)
    original = b.sealed(b.OUT / f"replay_R1_EMA_4h_{scenario}.json")
    for name, expected in original["metrics"].items():
        if not np.isclose(run["metrics"][name], expected, rtol=0, atol=1e-8):
            raise ValueError("legacy rebalance metric parity failed: " + name)
    if not np.array_equal(run["daily_log_returns"], original["daily_log_returns"]):
        raise ValueError("legacy rebalance daily path changed")
    for symbol in reg["symbols"]:
        for name, expected in original["per_asset"][symbol].items():
            if not np.isclose(run["per_asset"][symbol][name], expected, rtol=0, atol=1e-8):
                raise ValueError("legacy asset attribution changed")
    checks = b.audit_replay(run, {run["ema_model_sha256"]})
    write(f"parity_{scenario}.json", {"registration_sha256": reg["sha256"], "scenario": scenario,
          "all_metrics_match": True, "daily_path_bit_exact": True, "asset_attribution_matches": True,
          "ledger_checks": checks, "control_replay_saved_as_financial_candidate": False})
    print(json.dumps({"scenario": scenario, "parity": True, "ledger_checks": checks}))


def run_replay(scenario):
    reg = registration()
    for name in ("normal", "stress"):
        if b.sealed(OUT / f"parity_{name}.json")["registration_sha256"] != reg["sha256"]:
            raise ValueError("both legacy parity checks required")
    run = e.replay(*data_inputs(b.registration()), reg["contract"], scenario, retain_units=True)
    run["registration_sha256"] = reg["sha256"]
    run["source_vectors_pin"] = b.pin(b.OUT / "vectors.json")
    write(f"replay_{scenario}.json", run)
    metric = run["metrics"]
    print(json.dumps({"scenario": scenario, "return_pct": round(metric["net_return"] * 100, 5),
                      "dd_pct": round(metric["max_drawdown"] * 100, 5), "fills": metric["fills"],
                      "turnover": round(metric["turnover_initial_capital"], 4)}))


def analyze():
    reg, parent = registration(), b.registration()
    data, rows, covariances, _ = data_inputs(parent)
    stats, decomposition, total_checks = {}, {}, 0
    for scenario in ("normal", "stress"):
        run = b.sealed(OUT / f"replay_{scenario}.json")
        old = b.sealed(b.OUT / f"replay_R1_EMA_4h_{scenario}.json")
        daily_reference = b.sealed(b.OUT / f"replay_R0_EMA_daily_{scenario}.json")
        if run["registration_sha256"] != reg["sha256"] or run["source_vectors_pin"] != b.pin(b.OUT / "vectors.json"):
            raise ValueError("unit-policy provenance mismatch")
        total_checks += b.audit_replay(run, {run["ema_model_sha256"]})
        calendar = {row["timestamp_ms"]: row for row in run["equity"] if row["phase"] == "calendar_bucket_end"}
        actual_risk = []
        no_fill_decisions, nonflat_no_fill_decisions, max_nav_error = 0, 0, 0.
        fills_at = {row["timestamp_us"] - reg["contract"]["fill_delay_us"] for row in run["ledger"] if row["kind"] == "modeled_fill"}
        for row in run["decisions"]:
            t, point = row["asof_ms"], calendar[row["asof_ms"]]
            covariance = covariances[t - t % e.DAY_MS]
            weight = np.asarray(point["asset_exposure"])
            risk = {"asset_max": float(weight.max()), "gross": float(weight.sum()), "daily_volatility": float(np.sqrt(max(0., weight @ covariance @ weight)))}
            actual_risk.append(risk)
            for name, limit in (("asset_max", .1), ("gross", .8), ("daily_volatility", .005)):
                if risk[name] > limit + 1e-10:
                    raise ValueError("actual post-trade risk cap breached: " + name)
                total_checks += 1
            if row["risk_covariance_sha256"] != e.m.array_identity(covariance) or row["forecast_used_for_trading"]:
                raise ValueError("risk provenance or unused forecast flag failed")
            max_nav_error = max(max_nav_error, abs(point["nav"] - row["projected_post_cost_nav"]))
            if t * 1000 not in fills_at:
                no_fill_decisions += 1
                nonflat_no_fill_decisions += point["exposure"] > 0
        if max_nav_error > 1e-7:
            raise ValueError("projected post-cost NAV differs from ledger")
        topups = sum(row["kind"] == "modeled_fill" and row["side"] == "BUY" and row["quantity_before"] > 1e-12 for row in run["ledger"])
        if topups:
            raise ValueError("retained position was topped up")
        daily = np.asarray(run["daily_log_returns"])
        stats[scenario] = {"U1": run["metrics"], "R1": old["metrics"], "R0": daily_reference["metrics"],
            "fills_by_reason": run["fills_by_reason"], "per_asset": run["per_asset"],
            "monthly_returns": {"2026-08": float(np.expm1(daily[:31].sum())), "2026-09": float(np.expm1(daily[31:].sum()))},
            "no_fill_decisions": no_fill_decisions, "nonflat_no_fill_decisions": int(nonflat_no_fill_decisions),
            "held_position_topups": topups, "retained_asset_decisions": sum(row["retained_assets"] for row in run["decisions"]),
            "risk_trim_asset_decisions": sum(row["risk_trim_assets"] for row in run["decisions"]),
            "post_decision_maxima": {name: max(row[name] for row in actual_risk) for name in actual_risk[0]},
            "max_projected_nav_error": max_nav_error}
        decomposition[scenario] = {"delta_gross_pnl": run["metrics"]["gross_pnl"] - old["metrics"]["gross_pnl"],
            "saved_fees": old["metrics"]["fees"] - run["metrics"]["fees"],
            "saved_slippage": old["metrics"]["slippage"] - run["metrics"]["slippage"],
            "delta_funding": run["metrics"]["funding"] - old["metrics"]["funding"],
            "delta_net_capital": run["metrics"]["final_capital"] - old["metrics"]["final_capital"]}
        component_sum = sum(value for key, value in decomposition[scenario].items() if key != "delta_net_capital")
        if abs(component_sum - decomposition[scenario]["delta_net_capital"]) > 1e-7:
            raise ValueError("incremental PnL decomposition failed")
    new = np.asarray(b.sealed(OUT / "replay_stress.json")["daily_log_returns"])
    old = np.asarray(b.sealed(b.OUT / "replay_R1_EMA_4h_stress.json")["daily_log_returns"])
    alpha = reg["contract"]["bootstrap"]["family_alpha"] / reg["contract"]["bootstrap"]["nominal_current_chain_contrasts"]
    contrasts = {name: {key: b.f.lcb(values * 10000, bank, alpha) for key, bank in b.banks().items()}
                 for name, values in (("U1_minus_R1", new - old), ("U1_minus_NoTrade", new))}
    gate = all(value["lcb"] > 0 for comparison in contrasts.values() for value in comparison.values())
    result = write("analysis.json", {"registration_sha256": reg["sha256"], "scenarios": stats, "decomposition": decomposition,
        "stress_contrasts_daily_log_bps": contrasts, "within_current_chain_alpha": alpha,
        "new_policy_ledger_and_risk_checks": total_checks, "primary_and_absolute_research_gate": gate,
        "promote_to_real_trading": False, "independent_validation": False, "new_training_fits": 0})
    print(json.dumps({"ledger_and_risk_checks": total_checks, "stress_delta_usdt": round(decomposition["stress"]["delta_net_capital"], 4),
                      "gate": result["primary_and_absolute_research_gate"]}))


def charts():
    registration()
    import matplotlib.pyplot as plt
    for scenario in ("normal", "stress"):
        target = OUT / f"capital_drawdown_{scenario}.png"
        if target.exists():
            raise ValueError("chart already exists")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        sources = (("EMA daily", b.OUT / f"replay_R0_EMA_daily_{scenario}.json", "#1f77b4"),
                   ("EMA 4h rebalance", b.OUT / f"replay_R1_EMA_4h_{scenario}.json", "#17becf"),
                   ("EMA 4h hold units", OUT / f"replay_{scenario}.json", "#2ca02c"))
        for name, path, color in sources:
            run = b.sealed(path)
            points = [row for row in run["equity"] if row["phase"] == "calendar_bucket_end"]
            times = [datetime.fromtimestamp(row["timestamp_ms"] / 1000, timezone.utc) for row in points]
            axes[0].plot(times, [row["nav"] for row in points], label=f'{name}: {run["metrics"]["net_return"]*100:+.3f}%', color=color, lw=1.5)
            axes[1].plot(times, [row["drawdown"]*100 for row in points], color=color, lw=1.2)
        axes[0].set(title=f"EMA decision clock vs unit retention | {scenario} | revealed Aug-Sep 2026", ylabel="Capital, USDT (initial 10,000)")
        axes[0].legend(loc="upper left", fontsize=9)
        axes[1].set(ylabel="Drawdown, %", xlabel="UTC date")
        for axis in axes:
            axis.grid(alpha=.2)
        fig.text(.5, .01, "Modeled close execution; frozen rates; no new training; risk-trim-only unit adjustments", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .03, 1, 1))
        fig.savefig(target, dpi=150)
        plt.close(fig)
    print(json.dumps({"charts": 2}))


def report():
    reg, result = registration(), b.sealed(OUT / "analysis.json")
    lines = ["# EMA: четырёхчасовые решения с сохранением units", "", "12 октября 2026. Исследовательский replay 1 августа — 30 сентября 2026 на уже раскрытых данных. Один новый policy arm, два cost-сценария, без обучения и подбора порогов.", "",
        "## Измеряемая гипотеза", "", "Прежний EMA 4h пересчитывал целевые units на каждом decision, поддерживая долю NAV. Новая U1 сохраняет количество монет, пока EMA-направление положительное и фактическая позиция допустима по риску. Новый вход получает те же causal EMA risk targets. Выход — по неположительной EMA. Сокращение — только при превышении asset/gross/volatility constraints. Пополнения ранее открытой позиции ради возврата к доле NAV или после risk trim отсутствуют. В конце периода позиции закрываются с costs.", "",
        "Это одновременно меняет turnover и фактическую экспозицию. Разницу net-результата нельзя целиком приписать экономии costs: отдельно показаны gross PnL, funding и экспозиция.", "",
        "## Общие данные, риск и исполнение", "", "Взяты тот же full8 completed-close price/feature bank, EMA48-направление, daily covariance по 60 доходностям строго до UTC дня решения, shrinkage 20%, individual vol floor 0.2%, daily vol target 0.5%, asset cap 10%, gross cap 80%. Covariance обновляется раз в день; допустимость позиции проверяется каждые четыре часа. Используются frozen reported funding rates, включая terminal-settlement rows.", "",
        "Normal costs — fee 6 + drag 2 bps на сторону; stress — fee 6 + drag 8 bps. Cash ledger начисляет costs по фактическому delta units. Signal delay 2s, modeled fill delay 2.25s, close считается неизменным до fill. Сохранены модельные fractional units; квалификация реальных fills/min lot/marks отсутствует. EMA-model SHA и covariance SHA зафиксированы в каждом решении. Старый прогнозный поток сохранён для provenance, но не используется при торговых решениях U1.", "",
        "При risk breach применяется один ограниченный downscale. Нижняя оценка post-cost NAV = NAV − max cost per reference notional × (old market value + tentative market value). Затем asset, gross и volatility weights только уменьшаются. Когда позиция допустима, этого downscale нет и units сохраняются точно. Проверка применена к фактическим post-fill weights/NAV; внутрисвечной риск не наблюдается.", "",
        "## Результаты", ""]
    for scenario in ("normal", "stress"):
        lines += [f"### {scenario}", "", "| Вариант | Итог, % | Max DD, % | Экспозиция, % | Оборот / капитал | Fills |", "|---|---:|---:|---:|---:|---:|"]
        for name, label in (("R0", "EMA daily"), ("R1", "EMA 4h rebalance"), ("U1", "EMA 4h hold units")):
            metric = result["scenarios"][scenario][name]
            lines.append(f"| {label} | {metric['net_return']*100:+.4f} | {metric['max_drawdown']*100:.4f} | {metric['mean_exposure']*100:.3f} | {metric['turnover_initial_capital']:.3f} | {metric['fills']} |")
        lines += ["", f"![Капитал и просадка, {scenario}](capital_drawdown_{scenario}.png)", ""]
    lines += ["## Разложение приращения U1 − EMA 4h rebalance", "", "Знаки ниже уже отражают влияние на капитал: положительные saved fees/slippage увеличивают net-результат. Сумма компонентов сверена с фактической разницей конечных капиталов.", "",
        "| Costs | Δ Gross PnL, USDT | Saved fees | Saved slippage | Δ Funding | Δ Net capital |", "|---|---:|---:|---:|---:|---:|"]
    for scenario, value in result["decomposition"].items():
        lines.append(f"| {scenario} | {value['delta_gross_pnl']:+.4f} | {value['saved_fees']:+.4f} | {value['saved_slippage']:+.4f} | {value['delta_funding']:+.4f} | {value['delta_net_capital']:+.4f} |")
    stress = result["scenarios"]["stress"]
    lines += ["", "## Причины операций и соблюдение риска", "", "| Причина | Fills | Reference notional, USDT | Fees | Slippage |", "|---|---:|---:|---:|---:|"]
    for reason, value in stress["fills_by_reason"].items():
        lines.append(f"| {reason} | {value['fills']} | {value['notional']:.2f} | {value['fees']:.4f} | {value['slippage']:.4f} |")
    lines += ["", f"Решений без fills: {stress['no_fill_decisions']} из 366; с ненулевой позицией и без fills: {stress['nonflat_no_fill_decisions']}. Retained asset-decisions: {stress['retained_asset_decisions']}; risk-trim asset-decisions: {stress['risk_trim_asset_decisions']}; top-ups уже удерживаемых позиций: {stress['held_position_topups']}.", "",
        f"Фактические максимумы после решений: asset weight {stress['post_decision_maxima']['asset_max']*100:.6f}%, gross {stress['post_decision_maxima']['gross']*100:.6f}%, daily ex-ante volatility {stress['post_decision_maxima']['daily_volatility']*100:.6f}%. Max projected NAV error {stress['max_projected_nav_error']:.3g} USDT. Эти limits не гарантируют bar-internal или будущий drawdown.", "",
        f"U1 по месяцам после stress: август {stress['monthly_returns']['2026-08']*100:+.3f}%, сентябрь {stress['monthly_returns']['2026-09']*100:+.3f}%.", "",
        "## Статистика", "", "Парные daily log-return differences и абсолютный результат против NoTrade проверены на тех же общих primary/audit/sensitivity block banks. Primary/audit block 3 дня, sensitivity 7 дней, 5000 повторов. Audit seed не заменяет новые данные. Для 14 сравнений текущей исследовательской цепочки nominal α=0.05/14. Более ранний глобальный поиск на этих датах этим не исправляется.", "",
        "| Контраст | Mean, bps/day | LCB primary | LCB audit | LCB 7d |", "|---|---:|---:|---:|---:|"]
    for name, value in result["stress_contrasts_daily_log_bps"].items():
        lines.append(f"| {name} | {value['primary']['mean']:+.5f} | {value['primary']['lcb']:+.5f} | {value['audit']['lcb']:+.5f} | {value['sensitivity']['lcb']:+.5f} |")
    lines += ["", f"Совместный research gate положительной добавки и абсолютного результата: **{result['primary_and_absolute_research_gate']}**. Independent validation и admission к real money: False.", "",
        "## Вывод и следующий шаг", ""]
    delta = result["decomposition"]["stress"]["delta_net_capital"]
    if delta > 0:
        lines += [f"Сохранение units улучшило stress net-капитал относительно EMA 4h rebalance на {delta:.4f} USDT. Подтверждение устойчивого преимущества зависит от paired LCB и других временных режимов. Экспозиция и gross PnL также изменились; это не чистый эксперимент с одной только стоимостью исполнения.", ""]
    else:
        lines += [f"Сохранение units изменило stress net-капитал относительно EMA 4h rebalance на {delta:.4f} USDT. Экономия costs, если она есть, не обеспечила лучшего net-итога. Уменьшение оборота само по себе недостаточно; нужно учитывать потерю/изменение gross PnL и экспозиции.", ""]
    lines += ["Один следующий ограниченный этап — определить действие и соответствующий ему target: самостоятельное удержание четыре часа либо incremental управление многобарной EMA-позицией. Зафиксировать costs относительно фактического перехода, созревание labels и контрольную политику; затем зарегистрировать один быстрый feature block при прежней линейной архитектуре. Новый подбор порогов или MLP/GA до этого будет смешивать изменение сигнала, позиции и стоимости оборота.", "",
        "## Атрибуция по активам, stress", "", "| Актив | Gross PnL | Fees | Slippage | Funding | Net PnL |", "|---|---:|---:|---:|---:|---:|"]
    for symbol, value in stress["per_asset"].items():
        lines.append(f"| {symbol} | {value['gross_pnl']:+.4f} | {value['fees']:.4f} | {value['slippage']:.4f} | {value['funding']:+.4f} | {value['net_pnl']:+.4f} |")
    parity_checks = sum(b.sealed(OUT / f"parity_{name}.json")["ledger_checks"] for name in ("normal", "stress"))
    lines += ["", "## Воспроизводимость", "", f"Первоначальные {reg['parent_artifact_pins_verified']} artifact pins проверены. Новый cash engine в control-mode совпал со всеми метриками, per-asset атрибуцией и bit-exact daily path сохранённого R1 при обоих costs. Engineering checks: {reg['direct_checks_passed']}; parity ledger checks: {parity_checks}; новые ledger/risk checks: {result['new_policy_ledger_and_risk_checks']}. Исходные эксперименты и их код не изменялись.", "",
        "Зарегистрирован один новый policy arm до его расчёта. Обучений и подбора порогов — ноль. Все результаты сохранены, включая отрицательные. Команды выполнялись последовательно, foreground, одним потоком; сеть, collector, pytest и trading runtime не использовались.", "",
        "- [Регистрация](registration.json)", "- [Результаты и разложение PnL](analysis.json)", "- [Stress ledger](replay_stress.json)", "- [Предыдущие результаты](../forecast4h_positions_v1_20261012/SUMMARY_RU.md)", ""]
    with (OUT / "REPORT_RU.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    write("receipt.json", {"schema": "ema_units_receipt/1", "registration_sha256": reg["sha256"],
          "artifact_pins": {path.name: b.pin(path) for path in OUT.iterdir() if path.is_file()},
          "successful_new_policy_replays": 2, "engineering_parity_replays": 2, "training_fits": 0,
          "runtime_or_network_started": False})
    print(json.dumps({"report": b.relative(OUT / "REPORT_RU.md"), "stress_delta_usdt": round(delta, 4), "new_training_fits": 0}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("checks", "prepare", "parity", "replay", "analyze", "charts", "report"))
    parser.add_argument("--scenario", choices=("normal", "stress"))
    args = parser.parse_args()
    if args.stage in ("parity", "replay"):
        if args.scenario is None:
            parser.error("scenario required")
        (parity if args.stage == "parity" else run_replay)(args.scenario)
    else:
        globals()[args.stage]()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, AssertionError, FileExistsError, FileNotFoundError, KeyError) as error:
        print(f"ERROR {type(error).__name__}: {str(error)[:500]}", file=sys.stderr)
        raise SystemExit(1) from None
