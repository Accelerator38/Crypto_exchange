"""Direct synthetic profit-fitness checks and optional equity/audit report."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from decimal import Decimal
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def load_checks():
    spec = importlib.util.spec_from_file_location(
        "profit_fitness_checks", ROOT / "tests/test_genetic_primus_profit_fitness_v1.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def file_identity(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def build_demo(module):
    from exia.genetic_primus.unified_measurement_v1 import (
        BAR_MS, CostScenario, compare_policies, identity, make_decision,
    )
    symbols = ("Safe", "Medium", "High", "Smooth")
    spec, bars, _, funding, _ = module._fixture_module.fixture(
        ({s: "0" for s in symbols}, {s: "0" for s in symbols}), symbols=symbols, tail_bars=5
    )
    profiles = {"Safe": (95, 115), "Medium": (85, 125), "High": (75, 140)}
    for i, row in enumerate(bars):
        for symbol, (minimum, final) in profiles.items():
            value = Decimal(100) if i <= 1 else (100 + Decimal(minimum - 100) * (i - 1) / 17
                                                if i <= 18 else (Decimal(minimum) if i < 37 else Decimal(final)))
            row["prices"][symbol] = str(value)
        row["prices"]["Smooth"] = str(Decimal(100) + Decimal(max(0, i - 1)) * 10 / 36)
    spec = replace(spec, market_inputs_sha256=identity(bars))
    policies = {}
    for name in (*symbols, "NoTrade"):
        decisions = []
        for i in range(2):
            origin = i * 18 * BAR_MS
            decisions.append(make_decision(
                origin_start_ms=origin, decision_asof_ms=origin,
                targets={s: "1" if s == name else "0" for s in symbols},
                model_sha256=identity({"synthetic_rule": name}), scaler_sha256=identity({"synthetic_scaler": "none"}),
                feature_prefix_sha256=identity({"synthetic_prefix": i}),
                training_identity_sha256=identity({"synthetic_training": "none"}),
            ))
        policies[name] = decisions
    scenarios = [CostScenario("normal", "6", "6", "2", "2", "synthetic", identity({"fixture_cost": "normal"})),
                 CostScenario("stress", "8", "8", "4", "4", "synthetic", identity({"fixture_cost": "stress"}))]
    results = compare_policies(spec, price_bars=bars, policies=policies, funding=funding, scenarios=scenarios)
    fixture = {"spec": asdict(spec), "prices": bars, "policies": policies,
               "funding": funding, "costs": [asdict(s) for s in scenarios]}
    return results, fixture


def report(destination, module, count):
    from exia.genetic_primus.profit_fitness_v1 import (
        check_profit_stability, objectives_from_contract, score_net_growth,
    )
    destination = destination.resolve()
    allowed = (ROOT / "Reports/Exia/Genetic_Primus").resolve()
    if destination == allowed or not destination.is_relative_to(allowed) or destination.exists():
        raise ValueError("report requires a new child directory under Reports/Exia/Genetic_Primus")
    sources = [
        "src/exia/genetic_primus/profit_fitness_v1.py", "src/exia/genetic_primus/unified_measurement_v1.py",
        "src/exia/genetic_primus/fitness_v4.py", "src/exia/genetic_primus/evolution_v4.py",
        "src/exia/genetic_primus/measurement.py", "configs/genetic_primus_profit_fitness_v1.json",
        "tests/test_genetic_primus_profit_fitness_v1.py", "tests/test_genetic_primus_unified_measurement_v1.py",
        "tools/check_genetic_primus_profit_fitness_v1.py", "docs/GENETIC_PRIMUS_PROFIT_FITNESS_V1.md",
    ]
    source_ids = {p: file_identity(ROOT / p) for p in sources}
    contract = json.loads((ROOT / "configs/genetic_primus_profit_fitness_v1.json").read_text(encoding="utf-8"))
    objectives = objectives_from_contract(contract)
    results, fixture = build_demo(module)
    matrix, winners = {}, {}
    for objective in objectives:
        budget = str(objective.drawdown_budget_percent)
        matrix[budget] = {name: score_net_growth(result, objective=objective,
                                                 stress_result=results["stress"][name])
                          for name, result in results["normal"].items()}
        winners[budget] = max(matrix[budget], key=lambda name: matrix[budget][name]["ranking_key"])
    folds, selection = module.bank_fixture(results["normal"]["Safe"], 7)
    _, audit = module.bank_fixture(results["normal"]["Safe"], 19)
    stability = {name: check_profit_stability(result, stress=results["stress"][name], folds=folds,
                                              selection_bank=selection, audit_bank=audit)
                 for name, result in results["normal"].items()}
    # Ranking is frozen before any audit and must remain exactly unchanged.
    for objective in objectives:
        budget = str(objective.drawdown_budget_percent)
        for name in results["normal"]:
            repeated = score_net_growth(results["normal"][name], objective=objective,
                                        stress_result=results["stress"][name])
            if repeated != matrix[budget][name]:
                raise ValueError("stability audit changed search fitness")
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "fixture.json", fixture)
    write_json(destination / "ledgers.json", results)
    write_json(destination / "fitness_and_audit.json", {
        "research_matrix": matrix, "search_winners": winners, "stability": stability,
        "folds": folds, "selection_bank": selection.metadata(), "audit_bank": audit.metadata(),
    })
    import numpy as np
    np.savez_compressed(destination / "resample_banks.npz", **{
        prefix + "_" + scope + "_" + kind: values
        for prefix, bank in (("selection", selection), ("audit", audit))
        for kind, scopes in (("timestamps", bank.scope_timestamps), ("indices", bank.scope_indices))
        for scope, values in scopes.items()
    })
    os.environ["MPLCONFIGDIR"] = str(destination / ".matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "profit-fitness-v1"})
    colors = {"Safe": "#2563eb", "Medium": "#d97706", "High": "#dc2626", "Smooth": "#059669", "NoTrade": "#374151"}
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for col, scenario in enumerate(("normal", "stress")):
        for name, result in results[scenario].items():
            days = [row["timestamp"] / 86400000 for row in result["equity"]]
            for axis, field, factor in ((axes[0, col], "equity_quote", 1),
                                        (axes[1, col], "drawdown", 100), (axes[2, col], "exposure_fraction", 100)):
                axis.plot(days, [float(row[field]) * factor for row in result["equity"]],
                          label=name, color=colors[name], linewidth=1.4)
        axes[0, col].set_title(scenario.capitalize() + ": per-fill costs включены в капитал")
        axes[0, col].legend(fontsize=8)
        axes[2, col].set_xlabel("Дни синтетического сценария")
        for i, title in enumerate(("Капитал, quote", "Просадка, %", "Экспозиция / NAV, %")):
            axes[i, col].set_ylabel(title)
            axes[i, col].grid(alpha=0.25)
    fig.suptitle("Проверка profit fitness: искусственные данные, initial capital 100\n"
                 "Одна панель четырёх синтетических активов; модели не обучались", fontsize=12)
    fig.tight_layout(rect=(0, 0.015, 1, 0.94))
    fig.savefig(destination / "equity.png", dpi=140)
    fig.savefig(destination / "equity.svg", metadata={"Date": None})
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharex=True, sharey=True)
    for col, scenario in enumerate(("normal", "stress")):
        for budget in (10, 20, 30):
            axes[col].axvline(budget, color="#64748b", linestyle="--", alpha=0.5)
        for name, result in results[scenario].items():
            x = float(result["metrics"]["max_drawdown"]) * 100
            y = float(result["metrics"]["net_return"]) * 100
            marker = "s" if name == "NoTrade" else ("o" if stability[name]["status"] == "passed" else "x")
            axes[col].scatter(x, y, marker=marker, s=65, color=colors[name])
            axes[col].annotate(name, (x, y), xytext=(5, 6), textcoords="offset points", fontsize=9)
        axes[col].set_title(scenario.capitalize())
        axes[col].set_xlabel("Максимальная reference-open просадка, %")
        axes[col].grid(alpha=0.25)
        axes[col].set_xlim(-2, 34)
    axes[0].set_ylabel("Чистый рост капитала, %")
    fig.suptitle("Синтетический пример: прибыль и risk constraint оцениваются отдельно от LCB\n"
                 "○ audit LCB passed; × audit LCB failed; ■ NoTrade baseline", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(destination / "risk_and_growth.png", dpi=140)
    fig.savefig(destination / "risk_and_growth.svg", metadata={"Date": None})
    plt.close(fig)
    lines = [
        "# Profit fitness v1: инженерный отчёт", "",
        "Реализована цель максимизации чистого роста общего капитала при явном лимите просадки. "
        "Данные этого отчёта искусственные: четыре синтетических актива на одной общей панели. "
        "Политики фиксированы, модели не обучались. Это проверка расчётов, а не рыночное сравнение.", "",
        f"**{count} проверок прошли**: {count - 2} новых profit/audit-проверок и 2 регрессии предыдущего ledger/scorer.", "",
        "`fitness = 10000 × ln(final_NAV / initial_capital) / calendar_days − registered_parsimony_penalties`.", "",
        "Поиск использует hard drawdown gate и penalized net growth. LCB не входит ни в fitness, ни в ranking; "
        "по умолчанию search не исполняет bootstrap. Статистический аудит выполняется отдельно для final candidates.", "",
        "![Капитал, DD и экспозиция](equity.png)", "",
        "![Прибыль, risk budgets и LCB](risk_and_growth.png)", "",
        "| Политика | Normal net, % | Stress net, % | Normal DD, % | Stress DD, % | Audit LCB |", "|---|---:|---:|---:|---:|---|",
    ]
    for name, normal in results["normal"].items():
        n, s = normal["metrics"], results["stress"][name]["metrics"]
        lines.append(f"| {name} | {float(n['net_return']) * 100:.4f} | {float(s['net_return']) * 100:.4f} | "
                     f"{float(n['max_drawdown']) * 100:.4f} | {float(s['max_drawdown']) * 100:.4f} | {stability[name]['status']} |")
    lines.extend(["", "| Исследовательский DD budget | Search winner | Статистический аудит этого winner |",
                  "|---|---|---|"])
    for budget, winner in winners.items():
        lines.append(f"| {budget}% | {winner} | {stability[winner]['status']} |")
    lines.extend([
        "", "В примере больший risk budget допускает больший net рост. Высокодоходные концентрированные "
        "траектории не проходят независимый LCB-аудит; Smooth проходит его при меньшем net росте. "
        "NoTrade имеет fitness=0 и остаётся базовой политикой; LCB=0 не считается доказанной положительной доходностью.", "",
        "Risk constraint проверяет normal и предоставленный stress ledger. Отсутствующий stress не считается "
        "нулевой просадкой; для статистического аудита оба cost scenarios обязательны. Эти 10/20/30% являются "
        "исследовательской матрицей, а не утверждённым риском реальных торгов.", "",
        "Normal: fee 6 bps и adverse drag 2 bps на каждый leg. Stress: fee 8 bps и drag 4 bps. "
        "Costs синтетические; funding coverage в fixture явно полный с пустым settlement set. "
        "Комиссии и turnover повторно не вычитаются из fitness. DD измерен на reference opens, без внутрибара экстремумов.", "",
        "Один общий selection bank: seed 7; независимый audit bank: seed 19. Оба используют 64 resamples, "
        "block=2 дня, одинаковые cutoff и fold hashes. Это малый инженерный банк, не достаточная рыночная мощность проверки. "
        "Все индексы сохранены в resample_banks.npz. Fold summaries являются diagnostic, а не outer slices.", "",
        "В новом scorer отсутствует запрет торговать ноль сделок ради прохождения minimum trades. "
        "Risk pass и LCB pass не дают promotion_authority; temporal_gate_passed остаётся false до отдельной temporal validation.", "",
        "Новая ветка подключена к score_simulation_v4 через явный search_objective и проверенный unified ledger. "
        "Legacy суммирование доходностей не принимается новым objective. Generation log сохраняет objective SHA, "
        "risk budget и validation status; нерассчитанный LCB хранится как null.", "",
        "[Точные ledger](ledgers.json) · [Fitness, аудит и hashes банков](fitness_and_audit.json) · "
        "[Входная fixture](fixture.json) · [Evidence receipt](evidence_receipt.json)", "",
        "Выполнено 10 synthetic ledger replays; обучение=0, market evaluations=0, network=0. "
        "Август–сентябрьские данные и торговый runtime не читались. Следующий этап — модели и причинный rolling evaluator.", "",
    ])
    with (destination / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    if any(file_identity(ROOT / name) != value for name, value in source_ids.items()):
        raise ValueError("source changed during verification")
    outputs = {name: file_identity(destination / name) for name in (
        "fixture.json", "ledgers.json", "fitness_and_audit.json", "resample_banks.npz",
        "equity.png", "equity.svg", "risk_and_growth.png", "risk_and_growth.svg", "REPORT.md",
    )}
    write_json(destination / "evidence_receipt.json", {
        "schema": "exia.genetic_primus.profit_fitness_receipt/1", "checks_passed": count,
        "sources": source_ids, "outputs": outputs,
        "objective_sha256_by_budget": {str(v.drawdown_budget_percent): v.sha256 for v in objectives},
        "synthetic_ledger_replays": 10, "fits": 0, "market_evaluations": 0, "network_calls": 0,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("core", "integration", "all"), default="core")
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    if args.report_dir and args.scope != "all":
        raise ValueError("report requires --scope all")
    module = load_checks()
    suite = unittest.TestSuite()
    if args.scope in {"core", "all"}:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(module.ProfitSearchTests))
    if args.scope in {"integration", "all"}:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(module.ProfitAuditIntegrationTests))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(module._fixture_module.FitnessRankingIntegrationTests))
    outcome = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
    print(f"Checks={outcome.testsRun}; failures={len(outcome.failures)}; errors={len(outcome.errors)}; fits=0; market=0")
    if not outcome.wasSuccessful():
        detail = "\n".join(str(test) + ": " + error.splitlines()[-1] for test, error in outcome.failures + outcome.errors)
        print(detail.encode("utf-8")[:750].decode("utf-8", "ignore"))
        return 1
    if args.report_dir:
        report(args.report_dir, module, outcome.testsRun)
        print("Synthetic report:", args.report_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print((type(exc).__name__ + ": " + str(exc)).encode("utf-8")[:750].decode("utf-8", "ignore"))
        raise SystemExit(1)
