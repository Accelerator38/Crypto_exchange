"""Deterministic assertions on tiny artificial accounting inputs only."""
from __future__ import annotations

import copy
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    builder = runpy.run_path(str(ROOT / "tools/build_genetic_primus_synthetic_accounting_probe_v1.py"))
    api, artifact, environment, summary, _, verify = builder["prepare"]()
    verify()
    probe = runpy.run_path(str(builder["MODULE"]))
    module = sys.modules["_gp_synthetic_accounting_probe_v1"]
    count = 0
    def check(condition):
        nonlocal count
        if not condition:
            raise AssertionError(f"synthetic accounting check {count + 1}")
        count += 1
    def rejects(fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except (ValueError, AssertionError, TypeError, ArithmeticError):
            check(True)
        else:
            check(False)
    check(len(artifact["results"]) == 6)
    check(summary["actual_synthetic_simulator_calls"] == 8)
    check(summary["all_consistent"] is True)
    check(summary["maximum_absolute_error"] <= summary["tolerance"])
    check(summary["new_model_fits"] == summary["new_market_strategy_evaluations"] == 0)
    check(summary["current_source_synthetic_behavior_verified"] is True)
    check(summary["full_contract_integration_verified"] is False)
    check(summary["per_decision_entry_sha_directly_available"] is False)
    check(environment["source_identity_verified"] is True)
    check(environment["runtime_dependency_graph_verified"] is False)
    check(all(x == "socket.gethostname" for x in environment["blocked_dependency_import_events"]))
    check(environment["hostname_request_allowed"] is False)
    check(not any(x == "exia" or x.startswith("exia.") for x in sys.modules))
    check(sys.dont_write_bytecode is True)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        check(os.environ[name] == "1")
    for result in artifact["results"]:
        check(result["checks"]["consistent"] is True)
        check(result["oracle_final_flat"] is True)
        check(result["financial_strategy_result"] is False)
        check(result["forced_open_resets"] == [])
        rows = result["actual"]["decisions"]
        check(len(rows) == 108)
        check(all(x["forced_reset"] is False for x in rows))
        check(all("entry_model_sha256" not in x for x in rows))
        check(all(x["exit_timestamp"] <= result["actual"]["portfolio"][-1]["timestamp"] for x in result["actual"]["trades"]))
        if result["case"] == "abstain":
            check(result["actual_metrics"]["final_equity"] == 1.0)
            check(result["actual_metrics"]["trades"] == result["actual_metrics"]["fills"] == 0)
            check(result["actual_metrics"]["total_fees"] == result["actual_metrics"]["max_drawdown"] == 0.0)
        elif result["case"] == "carry":
            check(result["actual_metrics"]["trades"] == 2)
            check(result["actual_metrics"]["fills"] == 4)
            for symbol in ("SYNTH_A", "SYNTH_B"):
                selected = [x for x in rows if x["symbol"] == symbol]
                check(all(x["quantity_after"] == selected[0]["quantity_after"] for x in selected))
                check(selected[-1]["model_sha256"] != selected[0]["model_sha256"])
                trade = next(x for x in result["actual"]["trades"] if x["symbol"] == symbol)
                check(trade["entry_model_sha256"] == selected[0]["model_sha256"])
                check(trade["exit_reason"] == "interval_end")
        else:
            check(result["actual_metrics"]["trades"] == 4)
            check(result["actual_metrics"]["fills"] == 8)
        terminal = result["actual"]["events"][-1]
        check(terminal["phase"] == "terminal_fills")
        check(abs(terminal["cash"] - terminal["equity"]) <= probe["TOLERANCE"])
        compounded = 1.0
        for row in result["actual"]["portfolio"]:
            compounded *= 1 + row["net_return"]
        check(abs(compounded - result["actual_metrics"]["final_equity"]) <= probe["TOLERANCE"])
        # Corrupt a copy of a stored actual result; comparison must not certify it.
        holder = type("ResultCopy", (), {})()
        for name in ("portfolio", "trades", "decisions", "events"):
            setattr(holder, name, module.pd.DataFrame(copy.deepcopy(result["actual"][name])))
        holder.metrics = copy.deepcopy(result["actual_metrics"])
        holder.metrics["net_return"] += 0.001
        expected = {key: copy.deepcopy(result["actual"][key]) for key in ("portfolio", "trades", "decisions", "events")}
        expected["metrics"] = {"net_return": result["actual_metrics"]["net_return"]}
        rejects(probe["compare_result"], holder, expected)
    control = artifact["forced_open_reset_negative_control"]
    check(control["actual_metrics"]["trades"] == 4)
    check(len(artifact["open_vs_previous_close"]) == 2)
    check(all(x["actual_reset_exit_price"] == x["origin_open"] != x["previous_close"] for x in artifact["open_vs_previous_close"]))
    check(artifact["negative_control_is_not_registered_deployment_policy"] is True)
    for price in (0.0, -1.0, float("nan"), float("inf")):
        rejects(module.CashSleeve._price, price)
    check(all(flag is False for flag in summary["safety"].values()))
    verify()
    print(f"synthetic accounting regressions: {count} assertions passed; {summary['oracle_comparison_checks']} oracle comparisons")
    print("8 fixed artificial cases; no market strategy evaluation/model training/network/runtime")


if __name__ == "__main__":
    main()
