"""Short named tests or immutable preflight packaging, never financial replay."""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import datetime as dt
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exia.genetic_primus import rolling_replay_protocol_v1 as p

SOURCE = ROOT / "Reports/Exia/Genetic_Primus/compact_forecast_v1_20261007_r2"
CONTRACT = ROOT / "configs/genetic_primus_rolling_replay_protocol_v1.json"
FUNDING = ROOT / "Retrodate/bitget_funding_history_v1_20260727/integrity_manifest.json"
NET_GATE = ROOT / "Reports/Exia/Genetic_Primus/reference_open_net_inputs_v1_20261007/result.json"
TEST = ROOT / "tests/test_genetic_primus_rolling_replay_protocol_v1.py"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def pin(path):
    content = path.read_bytes()
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def checks():
    spec = importlib.util.spec_from_file_location("rolling_protocol_tests", TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.ProtocolTests)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    summary = {"checks": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "model_fits": 0, "financial_replays": 0, "network": 0, "collector": 0}
    if not result.wasSuccessful():
        print(stream.getvalue().encode("utf-8")[:720].decode("utf-8", "ignore"))
        raise ValueError("pure regression failure")
    return summary


def package(folder, check_result):
    if folder.exists():
        raise ValueError("new output directory required")
    contract = read(CONTRACT)
    if (contract["safety"] != p.SAFETY or contract["clock"] != {
            "bar_ms": p.BAR_MS, "horizon_bars": 18, "retrain_cadence_days": 3, "purge_bars": 7,
            "train_end_offset_bars": -1, "outcome_availability": "exit_plus_one_completed_bar",
            "incomplete_final_origin": "exclude_and_hold_cash_after_terminal"}
            or contract["arms"] != list(p.ARMS) or contract["period"]["hyperparameter_selection_allowed"]
            or any(v is not None for v in contract["pending_inputs"].values())
            or contract["ga"]["global_trial_registry_complete"] is not False):
        raise ValueError("only fixed pending-input diagnostic contract supported")
    receipt = read(SOURCE / "analysis_receipt.json")
    inputs = receipt["inputs"] | receipt["outputs"]
    for path, expected in inputs.items():
        if pin(ROOT / path) != expected:
            raise ValueError("frozen source/output changed: " + path)
    funding = read(FUNDING)
    funding_pins = {}
    for file in funding["files"]:
        path = (FUNDING.parent / file["path"]).resolve()
        path.relative_to(FUNDING.parent)
        value = pin(path)
        if value != {"bytes": file["bytes"], "sha256": file["sha256"]}:
            raise ValueError("funding source identity mismatch")
        funding_pins[path.relative_to(ROOT).as_posix()] = value
    coverage = funding["observed_common_window"]
    if coverage["end_timestamp_ms"] >= p.START_MS:
        raise ValueError("funding coverage changed: require renewed explicit registration")
    prep = read(SOURCE / "preparation.json")
    results = [read(SOURCE / f"origin_{i:02d}.json") for i in range(20)]
    import numpy as np
    with np.load(SOURCE / "sample_bank.npz", allow_pickle=False) as bank:
        origins, available = bank["origins"].tolist(), bank["available"].tolist()
    ledger = p.build_checkpoints(results, origins, available, prep["sha256"], prep["symbols"])
    prerequisites = {"policy_registered": False, "capital_and_allocation_registered": False,
                     "normal_costs_registered": False, "stress_costs_registered": False,
                     "funding_complete_for_interval": False, "input_evidence_verified": True}
    gates = p.stage_gates(prerequisites=prerequisites, signal=None, trial_registry_complete=False)
    budget = p.equal_search_budget(contract["ga"]["per_origin_unique_candidate_cap"])
    old_gate = read(NET_GATE)
    status = {"schema": "rolling_replay_preflight/1", "causal_forecasts_audited": True,
              "financial_replay_status": "blocked_missing_registered_policy_costs_funding",
              "financial_replay_completed": False, "stability_of_net_policy_evaluated": False,
              "model_fit_count_this_step": 0, "market_forecast_fit_count_prior_valid_attempt": 280,
              "market_forecast_fit_count_including_superseded": 560,
              "data_already_revealed": True, "gates": gates, "search_budget_plan": budget,
              "funding_observed_common_window": coverage, "funding_allowed_use": funding["allowed_use"],
              "legacy_net_inputs_gate_preserved": old_gate["gate"],
              "safety": p.SAFETY}
    requirements = {"financial_replay_scope": "reference_open_research_not_observed_execution",
                    "policy": ["gross_forecast_to_net_action_rule", "decision_threshold_and_abstention",
                               "capital_usdt_and_equal_allocation_rule", "cash_reserve_and_no_borrow_policy",
                               "fixed_execution_delay_ms", "unchanged_position_carry_and_terminal_close"],
                    "cost_scenarios": {"normal": None, "stress": None,
                                       "fields": ["buy_fee_bps", "sell_fee_bps", "buy_adverse_bps", "sell_adverse_bps", "provenance", "evidence_sha256"],
                                       "registered_assumptions_do_not_qualify_live": True},
                    "funding": {"required_start_ms": p.START_MS, "required_end_exclusive_ms": p.END_MS,
                                "symbols": prep["symbols"], "events": "settlement_ms, signed_rate, settlement_mark_or_registered_reference_basis",
                                "completeness_gaps_and_interval_changes_verified": False,
                                "zero_funding_fallback_allowed": False},
                    "data_quality": {"independent_aug19_aug22_crosscheck": "pending_not_excluded_from_benchmark"},
                    "prospective": {"not_started": True, "start_and_duration": None,
                                    "only_after_net_signal_stability_final_contract_and_unseen_data": True}}
    folder.mkdir(parents=True)
    write(folder / "forecast_checkpoint_ledger.json", ledger)
    write(folder / "status.json", status)
    write(folder / "input_requirements.json", requirements)
    write(folder / "engineering_checks.json", check_result)
    def iso(value):
        return dt.datetime.fromtimestamp(value/1000, dt.timezone.utc).isoformat()
    lines = ["# Причинный rolling replay: пакет и состояние этапов", "",
             "## Подтверждено", "",
             "Проверены завершённые gross forecast fits, а не выполнено новое обучение. "
             "20 origins, четыре фиксированных arms, 80 checkpoint-записей, 160 сохранённых моделей. "
             "Для каждого решения связаны model/ensemble SHA, scaler SHA, training/fold/weight/prefix hashes, "
             "train_end, cutoff и фактическая последняя доступная тренировочная метка. "
             "Персональные scaler identities различаются на всех 20 origins, общие внутри каждого origin.", "",
             "Train_end=origin−1 четырёхчасовой бар; cutoff=train_end−7 баров. "
             "Labels созревают после завершения exit-бара; availability≤cutoff. "
             "Переобучение каждые три дня; entry следующий open после decision, hold 18 баров.", "",
             f"Фиксированный интервал: **{iso(p.START_MS)} — {iso(p.END_MS)}, конец exclusive**. "
             f"Полностью включает 30 сентября. Последний terminal {iso(ledger['final_terminal_ms'])}; "
             "последние 20 часов — cash, без неполного origin и октябрьских outcomes.", "",
             "## Не выполнено — не подменять результат", "",
             "Финансовый replay, net-устойчивость и equity curve **не рассчитаны**. Forecast-checkpoint ledger "
             "не является position/fill/capital ledger. Нет подтверждённого net-сигнала; RMSE не является его заменой. "
             "GA и random search не запускались. Prospective не запечатан и не начат.", "",
             f"Текущий проверенный funding archive покрывает {coverage['start_at']} — {coverage['end_at']}, "
             "имеет allowed_use=recent_diagnostic_screen_only и не покрывает ни один август–сентябрьский settlement. "
             "Нулевой funding не подставлен. Также не зарегистрированы net policy, капитал/аллокация, "
             "normal/stress costs и execution delay. Старые net-target gates сохранены, не сняты задним числом.", "",
             "## Условный GA против random search", "",
             "Сначала подтвердить net-сигнал простой модели на причинном pre-August development interval: "
             "normal/stress absolute и paired LCB>0, независимый audit bank, risk/stability gates и полный trial registry. "
             "Все денежные метрики вычислять через unified shared-capital ledger; fitness — net log-growth под research DD10/20/30%, "
             "LCB — проверка устойчивости. Открытый август–сентябрь не использовать для выбора GA-параметров.", "",
             "У обоих методов один зарегистрированный linear feature-mask/L2 search space, folds и bootstrap banks. "
             "На origin: 64 unique objective calls, 128 fits, 64 inner-fold evaluations, 0 gradient steps. "
             "Budget verifier проверяет реальные counters и уникальные candidate hashes, завершение одинакового бюджета "
             "и идентичность data/space/folds/bank. Дубликаты не считаются новыми evaluations; incumbent считается. "
             "Space SHA пока null — это план, не выполненное сравнительное исследование.", "",
             "## Следующий необходимый ввод", "",
             "1. Полная funding history full8 за 1 августа — 30 сентября 2026 UTC с settlement basis и проверкой пропусков/интервалов.",
             "2. Зафиксированные normal/stress fees и adverse execution costs с происхождением; research assumptions "
             "отдельно от наблюдаемого исполнения.",
             "3. Registered predictor→net-policy: threshold/abstention, капитал, exposure/аллокация и cash/borrow, задержка.",
             "4. Независимая проверка пикового окна 19–22 августа, без его исключения для улучшения метрик.", "",
             "После устранения входных блокеров — один фиксированный replay всех arms + NoTrade, непрерывные net equity/DD curves, "
             "cost/funding/turnover/exposure/carry отчёт и common/audit stability. До этого нельзя заявить, что этап replay завершён. "
             "После результата — окончательный контракт/registry, заранее выбранная длительность и genuinely unseen future interval "
             "с единственным финальным анализом. Никакого автоматического запуска live/paper.", "",
             f"Проверки: {check_result['checks']} новых pure unittest regressions; {len(inputs)} текущих source/output identities "
             "и восемь funding file identities сверены. Этот шаг: model fits=0, финансовые replay=0, network=0, "
             "collector=0, pytest=0. Старые 560 market forecast fits (включая superseded 280) не скрыты.", "",
             "Артефакты: forecast_checkpoint_ledger.json, status.json, input_requirements.json, engineering_checks.json, evidence_receipt.json."]
    with (folder / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines)+"\n")
    extra = (CONTRACT, FUNDING, NET_GATE, TEST, Path(__file__),
             ROOT / "src/exia/genetic_primus/rolling_replay_protocol_v1.py",
             ROOT / "docs/GENETIC_PRIMUS_ROLLING_REPLAY_PROTOCOL_V1.md", SOURCE / "analysis_receipt.json")
    all_inputs = inputs | funding_pins | {path.relative_to(ROOT).as_posix(): pin(path) for path in extra}
    write(folder / "evidence_receipt.json", {
        "schema": "rolling_replay_preflight_evidence/1", "inputs": all_inputs,
        "outputs": {path.relative_to(ROOT).as_posix(): pin(path) for path in folder.iterdir() if path.is_file()},
        "safety": p.SAFETY, "financial_replay_completed": False, "model_fits_this_step": 0})
    print(f"package complete: checks={check_result['checks']}; checkpoints=80; origins=20; replay=blocked; ga=not_started; prospective=not_started; fits=0; network=0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("check", "package"))
    parser.add_argument("--report-dir")
    args = parser.parse_args()
    result = checks()
    if args.step == "check":
        print(f"checks={result['checks']}; failures=0; errors=0; fits=0; network=0; collector=0")
    else:
        if not args.report_dir:
            raise ValueError("new report-dir required")
        folder = (ROOT / args.report_dir).resolve()
        folder.relative_to(ROOT / "Reports/Exia/Genetic_Primus")
        if folder == ROOT / "Reports/Exia/Genetic_Primus":
            raise ValueError("report subdirectory required")
        package(folder, result)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print((type(exc).__name__+": "+str(exc)).encode("utf-8")[:800].decode("utf-8", "ignore"))
        raise SystemExit(1)
