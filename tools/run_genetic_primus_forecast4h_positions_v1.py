"""Predeclared fixed-rule management diagnostics on frozen 4h forecasts; no fits."""
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
from tools import run_genetic_primus_forecast4h_v1 as b

f, np = b.f, b.np
OUT = b.BASE / "forecast4h_positions_v1_20261012"
NAMES = ("D_exit", "D_size", "D_quarter")


def write(name, value):
    value = f.p.c.plain(value)
    value["sha256"] = f.m.identity(value)
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return value


def target_rule(policy, forecast, base, held, contract):
    """No entry filter: independently test exit, sizing, and its fixed floor."""
    name = contract["diagnostic_policy"]
    prediction, result = np.asarray(forecast, float), np.asarray(base, float).copy()
    if name == "D_exit":
        result[np.asarray(held, bool) & (prediction <= contract["exit_at_or_below_bps"] / 10000)] = 0.
    elif name == "D_size":
        result *= np.clip(prediction / (contract["entry_strictly_above_bps"] / 10000), contract["sizing_minimum_fraction"], 1.)
    elif name == "D_quarter":
        result *= contract["sizing_minimum_fraction"]
    else:
        raise ValueError("unregistered diagnostic")
    return result


def prepare():
    parent = b.registration()
    original = b.sealed(b.OUT / "receipt.json")
    for name, expected in original["artifact_pins"].items():
        if b.pin(b.OUT / name) != expected:
            raise ValueError("original artifact changed: " + name)
    contract = parent["contract"]
    checks = 0
    base, held, pred = np.full(8, .1), np.ones(8, bool), np.zeros(8)
    for name, expectation in (("D_exit", .1), ("D_size", .025), ("D_quarter", .025)):
        variant = dict(contract, diagnostic_policy=name)
        if not np.allclose(target_rule("R1_EMA_4h", pred, base, held, variant), expectation):
            raise ValueError("diagnostic direct rule check failed")
        checks += 1
    variant = dict(contract, diagnostic_policy="D_exit")
    if not np.all(target_rule("R1_EMA_4h", np.full(8, -.002), base, held, variant) == 0):
        raise ValueError("held-position exit check failed")
    checks += 1
    data, rows = b.load_data(), b.vector_bank(parent)
    cov = {t: f.p.risk_covariance(data["times"], data["closes"], t)[0]
           for t in range(parent["start_ms"], parent["end_ms"], f.DAY_MS)}
    unchanged = f.replay(data, rows, cov, b.sealed(b.MODELED_RATES)["events"], contract, "R1_EMA_4h", "stress")
    saved = b.sealed(b.OUT / "replay_R1_EMA_4h_stress.json")
    for key, value in unchanged["metrics"].items():
        if not np.isclose(value, saved["metrics"][key], rtol=0, atol=1e-10):
            raise ValueError("default ledger behavior changed: " + key)
        checks += 1
    if not np.array_equal(unchanged["daily_log_returns"], saved["daily_log_returns"]):
        raise ValueError("default ledger daily path changed")
    checks += 1
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [b.OUT / "receipt.json", b.OUT / "registration.json", b.OUT / "vectors.json", b.OUT / "analysis.json",
              b.OUT / "forecast_quality.json", b.OUT / "resample_bank.npz", b.MODELED_RATES]
    inputs += list(sorted(b.OUT.glob("engineering_amendment_*.json")))
    inputs += [b.OUT / f"replay_R1_EMA_4h_{scenario}.json" for scenario in ("normal", "stress")]
    reg = write("registration.json", {"schema": "forecast4h_position_diagnostics/1",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(), "parent_registration_sha256": parent["sha256"],
        "trigger": "original forecast-entry filter opened zero positions; identify exit and sizing independently on EMA positions",
        "already_seen": "original 4h forecast quality and original 14 replay results; this extension is adaptive research",
        "base_contract": contract, "policies": list(NAMES), "new_training_fits": 0, "new_threshold_trials": 0,
        "additional_policy_replays": 6, "additional_trade_contrasts": 3,
        "within_current_chain_alpha": .05 / 12,
        "exit_semantics": "held EMA position exits at forecast <= -14bps; EMA entries are unfiltered and may reenter at a later decision",
        "sizing_semantics": "EMA targets times clip(forecast/30bps,0.25,1); no forecast entry/exit filter",
        "floor_control_semantics": "EMA targets times constant 0.25, fixed by original sizing floor, not estimated from observed exposure",
        "code_pins": {b.relative(path): b.pin(path) for path in (Path(__file__), Path(b.__file__), ROOT / "src/exia/genetic_primus/forecast4h_v1.py")},
        "input_pins": {b.relative(path): b.pin(path) for path in inputs},
        "direct_checks_passed": checks, "default_parent_path_exact_parity": True,
        "safety": contract["safety"], "independent_validation": False})
    print(json.dumps({"registered": reg["sha256"], "new_fits": 0, "replay_budget": 6, "checks": checks}))


def registration():
    b.registration()
    reg = b.sealed(OUT / "registration.json")
    for name, expected in (reg["code_pins"] | reg["input_pins"]).items():
        if b.pin(ROOT / name) != expected:
            raise ValueError("diagnostic source changed: " + name)
    return reg


def run_replay(name, scenario):
    reg, parent, data = registration(), b.registration(), b.load_data()
    rows = b.vector_bank(parent)
    cov = {t: f.p.risk_covariance(data["times"], data["closes"], t)[0]
           for t in range(parent["start_ms"], parent["end_ms"], f.DAY_MS)}
    contract = dict(deepcopy(reg["base_contract"]), diagnostic_policy=name, diagnostic_target_rule_source_sha256=reg["code_pins"][b.relative(Path(__file__))]["sha256"])
    result = f.replay(data, rows, cov, b.sealed(b.MODELED_RATES)["events"], contract, "R1_EMA_4h", scenario, target_rule=target_rule)
    result["policy"], result["registration_sha256"] = name, reg["sha256"]
    result["vectors_pin"] = b.pin(b.OUT / "vectors.json")
    write(f"replay_{name}_{scenario}.json", result)
    print(json.dumps({"policy": name, "scenario": scenario, "return_pct": round(result["metrics"]["net_return"] * 100, 4), "dd_pct": round(result["metrics"]["max_drawdown"] * 100, 4)}))


def analyze():
    reg = registration()
    rows = b.vector_bank(b.registration())
    models = {row["model_sha256"] for row in rows}
    runs, count = {}, 0
    for scenario in ("normal", "stress"):
        runs[scenario] = {name: b.sealed(OUT / f"replay_{name}_{scenario}.json") for name in NAMES}
        runs[scenario]["R1_EMA_4h"] = b.sealed(b.OUT / f"replay_R1_EMA_4h_{scenario}.json")
        for name in NAMES:
            result = runs[scenario][name]
            if result["registration_sha256"] != reg["sha256"] or result["vectors_pin"] != b.pin(b.OUT / "vectors.json"):
                raise ValueError("diagnostic replay provenance mismatch")
            count += b.audit_replay(result, models)
    contrasts = {}
    for left, right in (("D_exit", "R1_EMA_4h"), ("D_size", "R1_EMA_4h"), ("D_size", "D_quarter")):
        a, c = runs["stress"][left], runs["stress"][right]
        delta = (np.asarray(a["daily_log_returns"]) - np.asarray(c["daily_log_returns"])) * 10000
        contrasts[left + "-" + right] = {"return_difference_pp": (a["metrics"]["net_return"] - c["metrics"]["net_return"]) * 100,
            "daily_log_difference_bps": {key: f.lcb(delta, bank, reg["within_current_chain_alpha"]) for key, bank in b.banks().items()}}
    predictions = np.asarray([row["expected_return_4h"] for row in rows]) * 10000
    result = write("analysis.json", {"registration_sha256": reg["sha256"],
        "policies": {scenario: {name: {"metrics": run["metrics"], "per_asset": run["per_asset"]} for name, run in values.items()} for scenario, values in runs.items()},
        "contrasts": contrasts, "ledger_checks": count,
        "size_multiplier_above_floor_asset_rows": int(np.sum(predictions > 7.5)), "total_asset_forecast_rows": int(predictions.size),
        "exit_only_path_identical_to_ema": np.array_equal(runs["stress"]["D_exit"]["daily_log_returns"], runs["stress"]["R1_EMA_4h"]["daily_log_returns"]),
        "promote_to_real_trading": False, "training_fits": 0, "independent_validation": False})
    print(json.dumps({"ledger_checks": count, "size_above_floor_rows": result["size_multiplier_above_floor_asset_rows"], "exit_path_identical": result["exit_only_path_identical_to_ema"]}))


def report():
    reg, result = registration(), b.sealed(OUT / "analysis.json")
    import matplotlib.pyplot as plt
    for scenario in ("normal", "stress"):
        target = OUT / f"capital_drawdown_{scenario}.png"
        if target.exists():
            raise ValueError("chart already exists")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        for name, color in (("R1_EMA_4h", "#17becf"), ("D_exit", "#d62728"), ("D_size", "#9467bd"), ("D_quarter", "#b58900")):
            source = b.OUT if name == "R1_EMA_4h" else OUT
            run = b.sealed(source / f"replay_{name}_{scenario}.json")
            points = [row for row in run["equity"] if row["phase"] == "calendar_bucket_end"]
            times = [datetime.fromtimestamp(row["timestamp_ms"] / 1000, timezone.utc) for row in points]
            axes[0].plot(times, [row["nav"] for row in points], label=f'{name}: {run["metrics"]["net_return"] * 100:+.3f}%', color=color, lw=1.5)
            axes[1].plot(times, [row["drawdown"] * 100 for row in points], color=color, lw=1.2)
        axes[0].set(title=f"EMA positions + fixed 4h forecast rules | {scenario} | revealed Aug-Sep 2026", ylabel="Capital, USDT (initial 10,000)")
        axes[0].legend(fontsize=9, loc="upper left")
        axes[1].set(ylabel="Drawdown, %", xlabel="UTC date")
        for axis in axes:
            axis.grid(alpha=.2)
        fig.text(.5, .01, "Adaptive research diagnostic; frozen forecasts; modeled execution; zero new training fits", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .03, 1, 1))
        fig.savefig(target, dpi=150)
        plt.close(fig)
    lines = ["# Четырёхчасовой прогноз: итог всех четырёх проверок", "", "12 октября 2026. Исследовательский период 1 августа — 30 сентября 2026 уже раскрыт. Первичный пилот и его инженерные amendments сохранены.", "",
        "## Прогноз, частота решений и вход", "", "Основной [отчёт первого пилота](../forecast4h_v1_20261012/REPORT_RU.md) и [подробная интерпретация](../forecast4h_v1_20261012/CONCLUSIONS_RU.md) содержат 21 причинное переобучение linear6, 2928 прогнозов и 14 исходных replay. RMSE модели 128.946 bps против 128.697 bps у исторического среднего; forecast gate не пройден. EMA daily после stress-costs дал +8.9758%, EMA 4h +7.9292%. Entry threshold 30 bps не достигнут ни разу, поскольку максимальный прогноз +10.24 bps. В исходной цепочке R2/R3/R4 торговля отсутствовала.", "",
        "## Отдельная проверка управления позицией", "", "Для завершения проверки выхода/размера зарегистрирован этот дополнительный diagnostic после просмотра исходного пилота. Это адаптивное исследование с явно увеличенным testing budget, не новая независимая валидация. Ни одного нового обучения или нового подбора порогов. Использованы прежние −14 bps exit и clip(forecast/30bps,0.25,1) sizing. Общие frozen forecasts, costs, lagged daily covariance и cash engine.", "",
        "D_exit применяет forecast exit только к уже открытой EMA-позиции. Последующий EMA-вход допускается без forecast entry filter. D_size применяет только изменение размера к EMA-позициям, без forecast entry/exit filter. D_quarter — заранее заданные постоянные 25% от EMA-targets, соответствующие исходному floor; он показывает, отличается ли sizing от простого снижения экспозиции. Это не точное постфактум выравнивание средних экспозиций.", ""]
    for scenario in ("normal", "stress"):
        lines += [f"## {scenario}", "", "| Вариант | Итог, % | Max DD, % | Экспозиция, % | Оборот / капитал | Fills |", "|---|---:|---:|---:|---:|---:|"]
        for name in ("R1_EMA_4h",) + NAMES:
            metric = result["policies"][scenario][name]["metrics"]
            lines.append(f"| {name} | {metric['net_return']*100:+.4f} | {metric['max_drawdown']*100:.4f} | {metric['mean_exposure']*100:.3f} | {metric['turnover_initial_capital']:.3f} | {metric['fills']} |")
        lines += ["", f"![Капитал и просадка, {scenario}](capital_drawdown_{scenario}.png)", ""]
    lines += ["## Paired stress-проверки", "", "Те же общие 3-day block banks, независимый audit seed и 7-day sensitivity. Для 12 сравнений текущей цепочки nominal α=0.05/12. Это учитывает 4 forecast + 5 исходных trading + 3 дополнительные trading contrasts, но не прошлые многократные исследования на этих датах.", "",
        "| Контраст | Разница итогов, п.п. | LCB primary, bps/day | LCB audit | LCB 7d |", "|---|---:|---:|---:|---:|"]
    for name, value in result["contrasts"].items():
        bounds = value["daily_log_difference_bps"]
        lines.append(f"| {name} | {value['return_difference_pp']:+.5f} | {bounds['primary']['lcb']:+.5f} | {bounds['audit']['lcb']:+.5f} | {bounds['sensitivity']['lcb']:+.5f} |")
    lines += ["", "## Интерпретация", "", f"Совпадение stress-траектории D_exit с EMA: **{result['exit_only_path_identical_to_ema']}**. Прогнозных строк со sizing multiplier выше floor: **{result['size_multiplier_above_floor_asset_rows']} из {result['total_asset_forecast_rows']}**.", "",
        "Если exit совпадает с EMA, это означает отсутствие дополнительного действия в доступных held-состояниях при выбранном фиксированном правиле, а не отсутствие любых потенциально полезных exit-моделей. Если D_size близок к D_quarter, уменьшение drawdown объясняется прежде всего уменьшением экспозиции. Значима добавка к D_quarter, её costs, риск и paired LCB; положительный итог D_size сам по себе не доказывает прогнозный edge.", "",
        "Четырёхчасовой прогноз текущих шести медленных ценовых признаков не подтвердил преимущество над простыми controls. Следующий ограниченный шаг: отдельно измерить эффект мелкого rebalance при 4h decision clock с сохранением units; затем при необходимости зарегистрировать один быстрый feature block и проверить его на нескольких заранее выбранных режимах. Entry threshold следует связывать с фактическим горизонтом удержания/приращением к EMA, прежде чем подбирать значения. Подключение MLP и GA сейчас не обосновано.", "",
        f"В основном пилоте 40712 независимых ledger checks; здесь ещё {result['ledger_checks']}. Engineering/direct checks расширения: {reg['direct_checks_passed']}. Default parent replay после добавления pure target-rule hook сохранил точное совпадение дневной траектории. Original 52 artifact pins проверены до дополнительного расчёта.", "",
        "Исполнение — условный close с модельным drag/fee; funding по reported rates и reference close, наблюдённых fills/marks нет. Базовые risk targets не гарантируют внутрисвечную просадку. Все trials сохранены. Сеть, collector, runtime и торговые разрешения не использовались.", "",
        "- [Регистрация дополнительных сравнений](registration.json)", "- [Результаты и атрибуция по активам](analysis.json)", "- [Исходный прогнозный поток](../forecast4h_v1_20261012/vectors.json)", ""]
    with (OUT / "REPORT_RU.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    write("receipt.json", {"registration_sha256": reg["sha256"], "artifact_pins": {path.name: b.pin(path) for path in OUT.iterdir() if path.is_file()},
                           "new_training_fits": 0, "successful_policy_replays": 6, "runtime_or_network_started": False})
    print(json.dumps({"report": b.relative(OUT / "REPORT_RU.md"), "new_training_fits": 0, "replays": 6}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "replay", "analyze", "report"))
    parser.add_argument("--policy", choices=NAMES)
    parser.add_argument("--scenario", choices=("normal", "stress"))
    args = parser.parse_args()
    if args.stage == "replay":
        if args.policy is None or args.scenario is None:
            parser.error("replay requires policy and scenario")
        run_replay(args.policy, args.scenario)
    else:
        globals()[args.stage]()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, AssertionError, FileExistsError, FileNotFoundError, KeyError) as error:
        print(f"ERROR {type(error).__name__}: {str(error)[:500]}", file=sys.stderr)
        raise SystemExit(1) from None
