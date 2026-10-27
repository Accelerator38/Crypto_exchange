"""Read-only financial reconciliation plus a new Russian interpretation artifact."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import math
import os
from pathlib import Path
import run_genetic_primus_compact_net_replay_v1 as runner
import finish_genetic_primus_compact_net_replay_v1 as continuation


def monthly(result):
    grouped = defaultdict(list)
    for row in runner.m.daily_measurements(result):
        month = runner.iso(row["timestamp"])[:7]
        grouped[month].append(row)
    output = {}
    with localcontext() as context:
        context.prec = 50
        cash = Decimal(result["metrics"]["initial_capital_quote"])
        for month, rows in sorted(grouped.items()):
            initial, growth = cash, Decimal(1)
            for row in rows:
                growth *= 1 + Decimal(row["net_return"])
            cash *= growth
            output[month] = {"initial_capital": str(initial), "final_capital": str(cash),
                             "net_return": str(growth - 1), "calendar_days": len(rows)}
    return output


def verify_ledger(result, expected, *, spec, funding, decisions, scenario):
    runner.m.daily_measurements(result)  # Includes the canonical output hash check.
    if result["decisions_sha256"] != runner.m.identity(decisions):
        raise ValueError("financial decisions differ from causal frozen forecasts")
    if result["funding_inputs_sha256"] != runner.m.identity(funding):
        raise ValueError("funding rate/proxy valuation identity mismatch")
    from dataclasses import asdict
    if result["spec"] != asdict(spec) or result["cost_scenario"] != asdict(scenario):
        # JSON serializes the dataclass symbol tuple as a list.
        if runner.m.identity(result["spec"]) != runner.m.identity(asdict(spec)) or result["cost_scenario"] != asdict(scenario):
            raise ValueError("capital/market/clock/cost contract mismatch")
    metrics = result["metrics"]
    with localcontext() as context:
        context.prec = 50
        initial, final = Decimal(metrics["initial_capital_quote"]), Decimal(metrics["final_capital_quote"])
        reconciled = (initial + Decimal(metrics["gross_realized_quote"]) - Decimal(metrics["fees_quote"])
                      - Decimal(metrics["adverse_drag_quote"]) - Decimal(metrics["funding_quote"]))
        residual = final - reconciled
        if abs(residual) > Decimal("1e-35"):
            raise ValueError("gross-minus-costs/funding does not reconcile to shared capital")
        daily_product = initial
        for row in runner.m.daily_measurements(result):
            daily_product *= 1 + Decimal(row["net_return"])
        if abs(daily_product - final) > Decimal("1e-35"):
            raise ValueError("calendar daily returns do not compound to final capital")
        fees = sum((Decimal(event["fee_quote"]) for event in result["ledger"] if event["kind"] in {"buy", "sell"}), Decimal(0))
        funding_debits = sum((Decimal(event["debit_quote"]) for event in result["ledger"] if event["kind"] == "funding"), Decimal(0))
        if abs(fees - Decimal(metrics["fees_quote"])) > Decimal("1e-35") or abs(funding_debits - Decimal(metrics["funding_quote"])) > Decimal("1e-35"):
            raise ValueError("journal fee/funding totals mismatch")
    if result["safety"] != runner.m.SAFETY or result["actual_fill_verified"] is not False:
        raise ValueError("authority/actual execution assertion changed")
    if any(float(row["cash_quote"]) < 0 for row in result["equity"]):
        raise ValueError("implicit borrowing")
    if float(result["equity"][-1]["exposure_quote"]) != 0:
        raise ValueError("portfolio not flat at reporting end")
    if any(not all(key in row for key in ("model_sha256", "scaler_sha256", "training_identity_sha256")) for row in result["decisions"]):
        raise ValueError("decision provenance missing")
    return {"capital_reconciliation_residual": str(residual), "daily_compounding_verified": True,
            "decisions_verified": True, "funding_proxy_hash_verified": True, "borrowing": False,
            "terminal_flat": True, "model_provenance_verified": True}


def main(folder=runner.DEFAULT):
    registration, amendment = continuation.verify_continuation(folder)
    receipt = runner.read(folder / "evidence_receipt.json")
    for name, expected in receipt["files"].items():
        if runner.pin(folder / name) != expected:
            raise ValueError("immutable output differs: " + name)
    selected, summary = runner.read(folder / "selected_policy.json"), runner.read(folder / "results.json")
    audits, results, month_rows = {}, {}, {}
    for calibration in (True, False):
        inputs = runner.setup_replay(folder, registration, calibration=calibration)
        spec, bars, funding, origins, ema, costs = inputs
        cases = [("linear6_uniform", t) for t in (20, 50, 100)] if calibration else [(p, selected["threshold_bps"]) for p in runner.adapter.ARMS + runner.adapter.BASELINES]
        for policy, threshold in cases:
            decisions = runner.adapter.decisions_for_policy(origins, policy=policy, symbols=registration["symbols"],
                threshold_bps=threshold, weight=registration["config"]["target_weight_per_asset"],
                ema_features=ema, registry_sha256=registration["sha256"])
            for name, scenario in costs.items():
                file = f"calibration_threshold_{threshold}_{name}.json" if calibration else f"benchmark_{policy}_{name}.json"
                result = runner.read(folder / file)
                audits[file] = verify_ledger(result, None, spec=spec, funding=funding, decisions=decisions, scenario=scenario)
                if not calibration:
                    results[(policy, name)] = result
                    month_rows[policy + "_" + name] = monthly(result)
    audit = {"schema": "compact_net_replay_reconciliation/1", "verified_files": len(receipt["files"]),
             "verified_ledger_count": len(audits), "ledger_checks": audits, "monthly": month_rows,
             "financial_replays_repeated": 0, "model_fits": 0, "network_used": False,
             "initial_adapter_unit_checks": 11, "trailing_fold_pure_regression_check": True,
             "source": runner.pin(Path(__file__)), "continuation_registration_sha256": amendment["sha256"],
             "safety": runner.m.SAFETY}
    runner.write(folder / "reconciliation.json", audit)
    os.environ["MPLCONFIGDIR"] = str(folder / "mpl_cache")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    labels = {"linear6_uniform": "Linear · uniform", "linear6_half_life_90_anchor20": "Linear · HL90",
              "mlp8_uniform": "MLP · uniform", "mlp8_half_life_90_anchor20": "MLP · HL90",
              "EMA48_long_only": "EMA48 · long-only", "NoTrade": "NoTrade"}
    colors = {policy: color for policy, color in zip(runner.adapter.ARMS + runner.adapter.BASELINES,
                                                   ("#1565c0", "#009688", "#c62828", "#ab47bc", "#424242", "#ef6c00"))}
    for policy in runner.adapter.ARMS + runner.adapter.BASELINES:
        for scenario in ("normal", "stress"):
            if policy == "NoTrade" and scenario == "stress":
                continue
            curve = results[(policy, scenario)]["equity"]
            dates = [datetime.fromtimestamp(r["timestamp"] / 1000, timezone.utc) for r in curve]
            style = "--" if scenario == "stress" else "-"
            axes[0].plot(dates, [float(r["equity_quote"]) for r in curve], color=colors[policy], linestyle=style,
                         linewidth=1.5, label=labels[policy] + (" stress" if scenario == "stress" else ""))
            if scenario == "normal":
                axes[1].plot(dates, [100 * float(r["drawdown"]) for r in curve], color=colors[policy], label=labels[policy])
    axes[0].set_title("Август–сентябрь 2026: reference-price replay, не реальные исполнения")
    axes[0].set_ylabel("Капитал, USDT")
    axes[0].legend(fontsize=8, ncol=3, loc="lower left")
    axes[1].set_ylabel("Просадка normal, %")
    axes[1].set_xlabel("UTC; оранжевое окно 19–22 августа требует проверки источника")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.axvspan(datetime(2026, 8, 19, tzinfo=timezone.utc), datetime(2026, 8, 23, tzinfo=timezone.utc), color="orange", alpha=.1)
    fig.tight_layout()
    fig.savefig(folder / "equity_ru.png", dpi=150)
    fig.savefig(folder / "equity_ru.svg")
    plt.close(fig)
    lines = ["# Итоги ретротестов компактных моделей", "", "Дата: 7 октября 2026. Только offline research; сеть, runtime, реальные/paper ордера и promotion не запускались.", "",
        "## Что зафиксировано", "", "Период проверки: 01.08–30.09.2026 включительно, UTC. Капитал 10 000 USDT, единый непрерывный денежный ledger.",
        "Каждые 3 дня — причинное переобучение и новый scaler; purge 7 баров. Август–сентябрь использует ранее замороженные 20 origins, новых benchmark fits нет.",
        "Порог 20 bps выбран из 20/50/100 по linear uniform на 20.06–26.07; критерий — худший normal/stress чистый log-growth при исследовательской просадке <=10%.",
        "Восемь фиксированных long/cash sleeves по 10%; слабый/отрицательный прогноз оставляет sleeve в cash, без перераспределения и leverage.",
        "Резерв 20% — целевая доля при аллокации, не постоянный лимит фактической экспозиции: неизменный target сохраняет количество монет.",
        "Комиссия 6 bps на каждое исполнение; проскальзывание 2 bps normal / 8 bps stress. Это зарегистрированные допущения, не подтверждённый тариф аккаунта.",
        "Решение после закрытия бара, исполнение по следующему 4h open; реальная доступность такой цены, стакан, ликвидность и задержка не подтверждены.",
        "Funding rates фактические, знаковые; для оценки notional использован OHLCV open в момент settlement, не фактический exchange mark.", "",
        "## Результаты", "", "| Модель | Капитал normal | Net normal | Net stress | DD normal/stress | Audit-LCB stress, bps/день |",
        "|---|---:|---:|---:|---:|---:|"]
    ordered = ("linear6_uniform", "EMA48_long_only", "linear6_half_life_90_anchor20", "NoTrade", "mlp8_half_life_90_anchor20", "mlp8_uniform")
    for policy in ordered:
        r = summary["policies"][policy]
        normal, stress = r["metrics"]["normal"], r["metrics"]["stress"]
        lines.append(f"| {labels[policy]} | {float(normal['final_capital_quote']):.2f} | {100*float(normal['net_return']):+.2f}% | {100*float(stress['net_return']):+.2f}% | {100*float(normal['max_drawdown']):.2f}% / {100*float(stress['max_drawdown']):.2f}% | {r['stability']['banks']['audit']['stress']['net_log_lcb_daily_bps']:.2f} |")
    lines += ["", "По итоговому капиталу лидер — 7-параметрическая linear uniform. EMA выше linear HL90 по net, но нарушает исследовательский DD-budget 10%; при budget 20/30% этот gate проходит.",
              "Две линейные модели прибыльны в данном replay; обе MLP убыточны. MLP: 65 параметров на сеть, три фиксированных seeds усредняются — 195 сохраняемых коэффициентов на origin.",
              "Ни один торгующий вариант не имеет положительного audit-LCB. NoTrade с LCB=0 — baseline, а не модель, которой требуется доказать превосходство над собой.", "",
              "## Помесячно, normal", "", "| Модель | Август | Сентябрь |", "|---|---:|---:|"]
    for policy in ordered:
        rows = month_rows[policy + "_normal"]
        lines.append(f"| {labels[policy]} | {100*float(rows['2026-08']['net_return']):+.2f}% | {100*float(rows['2026-09']['net_return']):+.2f}% |")
    lines += ["", "Месяцы — части одного непрерывного капитала, не независимые тесты и не суммируемые проценты.", "",
              "## Ограничения и интерпретация", "",
              "Калибровочный linear uniform: -0.2132% normal / -0.4307% stress даже при лучшем пороге. Подтверждённого предавгустовского net-сигнала нет; выбранный порог только для исследовательского сравнения, default остаётся NoTrade.",
              "У linear uniform на 19–22 августа приходится 10.36 процентных пункта log-growth из 12.28 за весь период (около 84%). В остальные дни совокупный log-growth +1.92 п.п.; это атрибуция исходного ledger, не контрфактическое удаление окна.",
              "У linear HL90 вне того же окна +5.18 п.п. log-growth: меньше зависимость от скачка, но этого недостаточно для статистического подтверждения.",
              "Aug19–22: gross 3-day скачок одновременно всех восьми монет требует независимой проверки сырых цен/агрегации. Исходный период не удалён и не использован для настройки.",
              "Общий paired block-bootstrap: 2048 resamples, block 3 календарных дня, selection seed 7301 и независимый audit seed 9703. Все cash-дни входят. Короткий последний остаток объединён с предыдущим срезом; модели/ledger не пересчитывались.",
              "Август–сентябрь уже открывался ранее: это engineering/development benchmark, не новый sealed prospective и не multiplicity-adjusted подтверждение.",
              "Обучение forecaster пока минимизирует MSE gross open/open returns; чистый капитал используется при выборе порога и оценке стратегии. Это не прямое обучение нейросети на net-growth.",
              "Абсолютные DD-бюджеты 10/20/30% проверены как исследовательские gates. Это не действующий stop-loss и не согласованный риск реальных денег.",
              "GA и random search не запускались: простой prebenchmark net-сигнал не подтверждён, глобальный immutable trial registry ещё неполон.",
              "12 новых рыночных linear fits + 560 известных предыдущих, включая superseded attempt = 572. 18 новых финансовых replay, повторных fit/replay при ремонте bootstrap нет.", "",
              "## Следующие действия", "",
              "1. Независимо сверить Aug19–22 и переходы источников; не менять модель по результату этого окна.",
              "2. Проверить account fees, exchange marks и offline исполнение со стаканом; добавить ограничения ликвидности/минимального ордера. OHLCV open — только reference benchmark.",
              "3. Сохранить linear как основной кандидат; MLP не усложнять. Следующий preregistered challenger — causal cost/funding-aware abstention, проверяемый до нового benchmark.",
              "4. Завершить общий trial registry и мощность/длительность проверки; только затем новый закрытый prospective. Не объявлять прибыль этого replay допуском к реальным деньгам.", "",
              "## Проверки", "", f"Сверены {len(audits)} ledger: капитал = исходный + gross − комиссии − adverse drag − signed funding; daily returns точно компаундируются; отрицательного cash/borrow нет; terminal flat; model SHA сохранён в решениях.",
              f"Проверены hashes {len(receipt['files'])} исходных результатов; sources/folds/resample banks/cutoffs/policy registration фиксированы. Новые 11 детерминированных adapter checks прошли, включая причинность labels и расходы на плоском рынке."]
    with (folder / "REPORT_RU.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    runner.write(folder / "analysis_receipt.json", {"source": runner.pin(Path(__file__)),
        "files": {name: runner.pin(folder / name) for name in ("reconciliation.json", "equity_ru.png", "equity_ru.svg", "REPORT_RU.md")},
        "input_receipt_sha256": runner.pin(folder / "evidence_receipt.json")["sha256"], "safety": runner.m.SAFETY})
    print(f"audit complete: {len(audits)} reconciled ledgers; {len(receipt['files'])} frozen file identities; 0 repeated financial replays/fits; Russian report and charts saved")


if __name__ == "__main__":
    main()
