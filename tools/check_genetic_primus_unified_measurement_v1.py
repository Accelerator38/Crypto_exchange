"""Bounded direct synthetic checks and optional accounting report, no pytest."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
        "unified_measurement_checks", ROOT / "tests/test_genetic_primus_unified_measurement_v1.py"
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


def report(destination, module, checks_count):
    from exia.genetic_primus.unified_measurement_v1 import (
        BAR_MS, CostScenario, SEMANTICS, SEMANTIC_SHA256, compare_policies,
        daily_measurements, identity,
    )
    allowed = (ROOT / "Reports/Exia/Genetic_Primus").resolve()
    destination = destination.resolve()
    if not destination.is_relative_to(allowed) or destination == allowed or destination.exists():
        raise ValueError("report must use a new child directory under Reports/Exia/Genetic_Primus")
    source_paths = [
        "src/exia/genetic_primus/unified_measurement_v1.py",
        "src/exia/genetic_primus/research_execution_contract_v1.py",
        "src/exia/genetic_primus/fitness_v4.py",
        "tests/test_genetic_primus_unified_measurement_v1.py",
        "tools/check_genetic_primus_unified_measurement_v1.py",
        "docs/GENETIC_PRIMUS_UNIFIED_MEASUREMENTS_V1.md",
    ]
    source_ids = {name: file_identity(ROOT / name) for name in source_paths}
    spec, bars, carry, funding, _ = module.fixture(tail_bars=3)
    for i, row in enumerate(bars):
        price = 100 if i == 0 else (100 - 0.8 * (i - 1) if i <= 18 else 86.4 + 1.4 * (i - 18))
        row["prices"]["X"] = str(round(price, 6))
    spec = replace(spec, market_inputs_sha256=identity(bars))
    funding["events"] = [
        {"timestamp": row["timestamp"], "symbol": "X", "mark_price": row["prices"]["X"],
         "signed_rate_bps": "-1" if i % 4 == 0 else "2"}
        for i, row in enumerate(bars) if i >= 2 and i % 2 == 0
    ]
    funding["evidence_sha256"] = identity({"synthetic_fixture": funding["events"]})
    policies = {"Carry": carry, "Reduce": module.fixture(("0.5", "0.25"))[2],
                "NoTrade": module.fixture(("0", "0"))[2]}
    scenarios = [
        CostScenario("normal", "6", "6", "2", "2", "synthetic", identity({"synthetic_cost_scenario": "normal"})),
        CostScenario("stress", "8", "8", "4", "4", "synthetic", identity({"synthetic_cost_scenario": "stress"})),
    ]
    results = compare_policies(spec, price_bars=bars, policies=policies, funding=funding, scenarios=scenarios)
    daily = {s: {p: daily_measurements(r) for p, r in values.items()} for s, values in results.items()}
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "result.json", results)
    write_json(destination / "daily.json", daily)
    write_json(destination / "fixture.json", {"spec": asdict(spec), "prices": bars,
                                              "funding": funding, "policies": policies,
                                              "scenarios": [asdict(s) for s in scenarios]})
    os.environ["MPLCONFIGDIR"] = str(destination / ".matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "unified-measurement-v1"})
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    colors = {"Carry": "#2563eb", "Reduce": "#d97706", "NoTrade": "#374151"}
    labels = {"Carry": "Сохранение позиции", "Reduce": "Снижение веса", "NoTrade": "NoTrade"}
    for col, scenario in enumerate(("normal", "stress")):
        for policy, result in results[scenario].items():
            rows = result["equity"]
            days = [row["timestamp"] / 86400000 for row in rows]
            for axis, field, factor in ((axes[0, col], "equity_quote", 1),
                                        (axes[1, col], "drawdown", 100),
                                        (axes[2, col], "exposure_fraction", 100)):
                axis.plot(days, [float(row[field]) * factor for row in rows],
                          color=colors[policy], label=labels[policy], linewidth=1.6)
        axes[0, col].set_title("Normal: 6 bps fee + 2 bps drag на leg" if col == 0
                               else "Stress: 8 bps fee + 4 bps drag на leg")
        axes[0, col].legend(fontsize=8)
        axes[2, col].set_xlabel("Дни синтетического сценария")
        for row, title in enumerate(("Капитал, quote", "Просадка, %", "Экспозиция / NAV, %")):
            axes[row, col].set_ylabel(title)
            axes[row, col].grid(alpha=0.25)
    figure.suptitle("Единый ledger: синтетическая проверка, начальный капитал 100\n"
                    "Искусственные цены и funding; это не результат модели на рынке", fontsize=12)
    figure.tight_layout(rect=(0, 0.025, 1, 0.94))
    figure.text(0.5, 0.008, "Неизменный target сохраняет quantity. После общего terminal — cash. Borrow/leverage отсутствуют.",
                ha="center", fontsize=8)
    figure.savefig(destination / "equity.png", dpi=140)
    figure.savefig(destination / "equity.svg", metadata={"Date": None})
    plt.close(figure)
    lines = [
        "# Единые измерения: инженерная проверка v1", "",
        "Это синтетическая проверка бухгалтерии, горизонтов и ranking. Цены, funding и costs искусственные; "
        "результаты не являются доходностью обученной модели. Market replay и обучение не выполнялись.", "",
        f"Проверок: **{checks_count}**, все прошли. Шесть synthetic ledger replay; fits=0, market evaluations=0.", "",
        "![Капитал, просадка и экспозиция](equity.png)", "",
        "| Costs | Политика | Конечный капитал | Net, % | DD, % | Fills | Funding, quote |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, values in results.items():
        for policy, value in values.items():
            m = value["metrics"]
            lines.append(f"| {scenario} | {policy} | {float(m['final_capital_quote']):.6f} | "
                         f"{100 * float(m['net_return']):.4f} | {100 * float(m['max_drawdown']):.4f} | "
                         f"{m['fills']} | {float(m['funding_quote']):.6f} |")
    lines.extend([
        "", "Весь universe использует один initial capital и один cash account. При изменении target "
        "сначала выполняются sells, затем buys; цена, fee и adverse drag фиксируются на каждом fill. "
        "Funding в момент входа не относится к новой позиции, в момент выхода относится к старой.", "",
        "Число дней сценария включает cash после terminal. DD измерен на reference opens четырёхчасовой сетки "
        "и не оценивает внутрибара экстремумы. Fee/drag не являются наблюдёнными fills.", "",
        "Для августа–сентября 2026 общий календарь содержит 20 полных origins: первый вход 1 августа 04:00 UTC, "
        "последний terminal 30 сентября 04:00 UTC; cash до 1 октября 00:00 UTC. Реальные данные этим checker не читаются.", "",
        "Ranking теперь строится один раз: прохождение всех gates → меньше нарушений → penalized fitness → "
        "DD → turnover → complexity. Проверена фактическая paired/constrained-overlay ветка scoring.", "",
        "Gross labels принимаются только open→open/18 bars и проходят новый cutoff каждого origin с purge 7 bars. "
        "Legacy open→last-close targets отклоняются. Изменение задержки между origins также отклоняется.", "",
        "Проверка не включает обучение, сеть, collector, runtime, paper/live или promotion. Все девять safety-флагов false.", "",
        "[Дневные измерения](daily.json) · [Непрерывные ledger и decisions](result.json) · "
        "[Точная fixture](fixture.json) · [Контрольные суммы](evidence_receipt.json)", "",
        f"Measurement semantic SHA256: `{SEMANTIC_SHA256}`.", "",
    ])
    with (destination / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    if any(file_identity(ROOT / name) != value for name, value in source_ids.items()):
        raise ValueError("source changed during verification")
    outputs = {name: file_identity(destination / name) for name in (
        "result.json", "daily.json", "fixture.json", "equity.png", "equity.svg", "REPORT.md")}
    write_json(destination / "evidence_receipt.json", {
        "schema": "exia.genetic_primus.unified_measurement_receipt/1",
        "measurement_semantic_sha256": SEMANTIC_SHA256, "semantics": SEMANTICS,
        "checks_passed": checks_count, "sources": source_ids, "outputs": outputs,
        "synthetic_ledger_replays": 6, "market_evaluations": 0, "fits": 0, "network_calls": 0,
        "price_inputs_sha256": spec.market_inputs_sha256,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("core", "ranking", "all"), default="core")
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    if args.report_dir and args.scope != "all":
        raise ValueError("complete report requires --scope all")
    module = load_checks()
    suite = unittest.TestSuite()
    if args.scope in {"core", "all"}:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(module.UnifiedMeasurementTests))
    if args.scope in {"ranking", "all"}:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(module.FitnessRankingIntegrationTests))
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log).run(suite)
    print(f"Checks={result.testsRun}; failures={len(result.failures)}; errors={len(result.errors)}; fits=0; market=0")
    if not result.wasSuccessful():
        detail = "\n".join(str(test) + ": " + error.splitlines()[-1] for test, error in result.failures + result.errors)
        print(detail.encode("utf-8")[:750].decode("utf-8", "ignore"))
        return 1
    if args.report_dir:
        report(args.report_dir, module, result.testsRun)
        print("Synthetic report:", args.report_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print((type(exc).__name__ + ": " + str(exc)).encode("utf-8")[:750].decode("utf-8", "ignore"))
        raise SystemExit(1)
