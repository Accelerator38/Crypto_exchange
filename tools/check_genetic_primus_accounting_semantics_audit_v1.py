"""Deterministic arithmetic/AST regressions; no trading simulator is executed."""

from hashlib import sha256
from pathlib import Path
import runpy


def main():
    root = Path(__file__).resolve().parents[1]
    api = runpy.run_path(str(root / "src/exia/genetic_primus/accounting_semantics_audit_v1.py"))
    checks = 0

    def check(condition):
        nonlocal checks
        assert condition
        checks += 1

    def rejects(call):
        try:
            call()
        except ValueError:
            return True
        return False

    check(abs(api["compound"]([0.1, -0.1]) + 0.01) < 1e-12)
    check(api["compound"]([0, 0]) == 0)
    for invalid in (None, [], [True], ["0"], [float("nan")], [float("inf")], [-1], [10 ** 1000]):
        check(api["compound"](invalid) is None)
    check(api["equality"]("x", None, 0, ["x"])["status"] == "unknown")
    check(api["equality"]("x", False, 0, ["x"])["status"] == "unknown")
    check(api["equality"]("x", 0, 0, ["x"])["status"] == "consistent")
    check(api["equality"]("x", 0.1, 0.2, ["x"])["status"] == "inconsistent")
    no_trade = {"net_return_pct": 0, "gross_return_pct": 0, "fees_pct": 0, "daily_returns": [0, 0]}
    check(all(c["status"] == "consistent" for c in api["stored_checks"](no_trade, percent=True, expected_days=2)))
    inconsistent = {"net_return_pct": 2, "gross_return_pct": 5, "fees_pct": 1, "daily_returns": [0, 0]}
    check(any(c["status"] == "inconsistent" for c in api["stored_checks"](inconsistent, percent=True)))
    unknown_gross = api["stored_checks"]({"net_return": 0, "fees": 0, "daily_returns": [0]})
    check(unknown_gross[-1]["status"] == "unknown" and "no_cross_scenario" in unknown_gross[-1]["unknown_reason"])
    check(api["sum_fields"]({}, "x") is None)
    check(api["sum_fields"]({"a": {"x": 1}, "b": {"x": 2}}, "x", scale=100) == 0.03)
    check(api["sum_fields"]([{ "x": 1}, {}], "x") is None)

    def blob(text, *, bound=False):
        raw = text.encode()
        return api["source"]("synthetic.py", raw, sha256(raw).hexdigest(), historically_bound=bound)

    text = "from exia.genetic_primus.accounting_v2 import simulate_cash_targets\ndef run():\n return simulate_cash_targets([], [], start_timestamp=0, end_timestamp=1, round_trip_cost_bps=16)\n"
    call = api["simulator_call"](blob(text, bound=True), "run")
    check(call["explicit_force_flat_argument"] is False)
    check(call["dependency_runtime_identity_verified"] is False)
    check(call["proof"]["binding"] == "result_recorded_file_hash")
    explicit = text.replace("round_trip_cost_bps=16)", "round_trip_cost_bps=16, force_flat_timestamps=(1,))")
    check(api["simulator_call"](blob(explicit), "run")["explicit_force_flat_argument"] is True)
    opaque = text.replace("round_trip_cost_bps=16)", "round_trip_cost_bps=16, **options)")
    check(rejects(lambda: api["simulator_call"](blob(opaque), "run")))
    ambiguous = text.replace("def run():", "def run():") + text[text.index("def run():"):]
    check(rejects(lambda: api["simulator_call"](blob(ambiguous), "run")))
    check(rejects(lambda: api["source"]("x.py", text.encode(), "0" * 64)))
    toy_profile = {"observed_current_rules": {"mandatory_terminal": "whole_simulation_interval_final_close"}}
    finding = api["terminal_finding"]("origin_final_candle_close", call, toy_profile)
    check(finding["status"] == "potential_mismatch" and finding["historical_execution_mismatch_proven"] is False)
    finding_explicit = api["terminal_finding"]("origin_final_candle_close", api["simulator_call"](blob(explicit), "run"), toy_profile)
    check(finding_explicit["status"] == "not_established")  # Does not certify correctly chosen reset timestamps.

    builder = runpy.run_path(str(root / "tools/build_genetic_primus_accounting_semantics_audit_v1.py"))
    _, audit, summary, inputs, _, _, _ = builder["prepare"]()
    check(summary["selected_evaluations"] == 25)
    check(summary["arithmetic_check_statuses"] == {"consistent": 100, "unknown": 15})
    check(summary["potential_terminal_scope_mismatches"] == 2)
    check(summary["strict_historical_accounting_verified_evaluations"] == 0)
    check(summary["historical_accounting_unknown_in_original_context_unchanged"] == 42)
    check(audit["historical_dependency_graph_verified"] is audit["all_costs_execution_verified"] is False)
    check(audit["new_strategy_evaluation_runs_started"] == audit["new_model_fits"] == 0)
    check(all(audit[flag] is False for flag in ("execution_allowed", "credentials_allowed", "orders_enabled", "paper", "live", "promotion_authority", "cross_source_comparison_authority")))
    check(audit["current_simulator_profile"]["observed_current_rules"]["intrabar_high_low_risk_verified"] is False)
    check(audit["current_simulator_profile"]["observed_current_rules"]["funding_borrow_real_fills_verified"] is False)
    for producer in audit["producer_profiles"]:
        check(producer["profile_sha256"] == api["digest"]({k: v for k, v in producer.items() if k != "profile_sha256"}))
        check(producer["strict_historical_accounting_verified"] is False)
        check(inputs[producer["producer_callsite"]["proof"]["path"]]["byte_sha256"] == producer["producer_callsite"]["proof"]["byte_sha256"])
    accounting_path = "src/exia/genetic_primus/accounting_v2.py"
    raw = (root / accounting_path).read_bytes()
    current = api["source"](accounting_path, raw, builder["REVIEWED_CURRENT"][accounting_path])
    # Mutated source exists only in memory; original implementation remains untouched.
    changed = current.text.replace("force_flat_timestamps: tuple[int, ...] = ()", "force_flat_timestamps: tuple[int, ...] = (1,)")
    check(rejects(lambda: api["current_simulator_profile"](blob(changed))))
    changed = current.text.replace("self.cash / (1.0 + self.fee_rate)", "self.cash")
    check(rejects(lambda: api["current_simulator_profile"](blob(changed))))
    check(all(e["strict_historical_accounting_verified"] is False for e in audit["evaluation_checks"]))
    print(f"accounting audit regressions: {checks} passed; stored numbers/AST only, no simulator/runtime")


if __name__ == "__main__":
    main()
