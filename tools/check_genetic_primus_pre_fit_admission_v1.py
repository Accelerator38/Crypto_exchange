"""Direct metadata admission/weight/timing regressions; no fitter or pytest."""
from __future__ import annotations

import argparse
import ast
import copy
from fractions import Fraction
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-only", action="store_true")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    builder = runpy.run_path(str(ROOT / "tools/build_genetic_primus_pre_fit_admission_v1.py"))
    api, guard, origin, proposal, result, summary, bank = builder["prepare"]()
    checks = 0

    def check(condition):
        nonlocal checks
        if not condition:
            raise AssertionError("pre-fit admission regression")
        checks += 1

    def reject(call, fragment=None):
        try:
            call()
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            if fragment and fragment not in str(exc):
                raise AssertionError("wrong rejection: " + str(exc)) from exc
            check(True)
        else:
            raise AssertionError("invalid admission input accepted")

    def reseal(value):
        value.pop("payload_sha256", None)
        value["payload_sha256"] = api["sha"](value)

    def bad(mutate, fragment, inner=False, deploy=False):
        altered = copy.deepcopy(origin)
        mutate(altered)
        if inner:
            reseal(altered["training_inputs"])
        if deploy:
            reseal(altered["deploy_features"])
        reseal(altered)
        reject(lambda: api["admit"](altered, proposal, guard), fragment)

    if not args.artifacts_only:
        check(api["admit"](origin, proposal, guard) == result)
        check(result["fit_allowed"] is False and len(result["plans"]) == 2)
        check(all(p["fit_allowed"] is False and p["sample_gates_passed"] is False for p in result["plans"]))
        check(all(value is False for value in result["safety"].values()))
        check(result["minimum_trade_count_forcing"] is False)
        check(result["new_model_fits"] == result["new_market_strategy_evaluations"] == 0)
        check(result["scaler_actual_fit_verified"] is False)
        check(result["timing"]["declared_timing_gate_passed"] is False)
        check(set(result["timing"]["denial_reasons"]) == {"training_latency_budget_unregistered", "execution_delay_unregistered"})
        for plan in result["plans"]:
            check(plan["opportunities"] == 7 and plan["minimum_opportunities"] == 64)
            check(plan["kish_weight_ess"] <= 7 + 1e-12 and plan["minimum_weight_ess"] == 40)
            check(plan["weight_bank"]["negative_outcome_weight"] > 0)
            check(len(plan["weight_bank"]["entries"]) == 7)
            check(abs(sum(e["weight"] for e in plan["weight_bank"]["entries"]) - 1) <= 1e-12)
            check(plan["scaler_candidate_rows_sha256"] == origin["training_inputs"]["scaler_candidate_input"]["rows_sha256"])
            check(plan["scaler_candidate_rows_sha256"] != origin["training_inputs"]["training_guard"]["scaler_fit_rows_sha256"])
            check(set(plan["denial_reasons"]) >= {"proposal_not_executable", "training_authority_disabled", "insufficient_completed_opportunities", "insufficient_kish_weight_ess"})
        dt, day = api["DT"], 86_400_000
        asof = 250 * day
        ends = [asof - 8 * dt, asof - 8 * dt - 90 * day]
        toy = [{"row_id": "toy_" + str(i), "symbol": "SYN_A", "label_end_exclusive_ms": end,
                "label_available_at_ms": end, "gross_return_3d": 0.1 if i == 0 else -0.1} for i, end in enumerate(ends)]
        weighted = api["weight_bank"](toy, "half_life_90d_anchor_20pct", asof, guard)
        expected = [Fraction(19, 30), Fraction(11, 30)]
        for entry, reference in zip(weighted["entries"], expected):
            check(abs(entry["weight"] - float(reference)) <= 1e-12)
        expected_ess = Fraction(1) / sum(w * w for w in expected)
        check(abs(weighted["kish_weight_ess"] - float(expected_ess)) <= 1e-12)
        check(abs(weighted["negative_outcome_weight"] - float(Fraction(11, 30))) <= 1e-12)
        flipped = copy.deepcopy(toy)
        for row in flipped:
            row["gross_return_3d"] *= -1
        check([e["weight"] for e in api["weight_bank"](flipped, "half_life_90d_anchor_20pct", asof, guard)["entries"]] == [e["weight"] for e in weighted["entries"]])
        equal = [{"row_id": "equal_" + str(i), "symbol": "SYN_A", "label_end_exclusive_ms": ends[0],
                  "label_available_at_ms": ends[0], "gross_return_3d": -0.1 if i % 2 else 0.1} for i in range(64)]
        for variant in api["VARIANTS"]:
            control = api["weight_bank"](equal, variant, asof, guard)
            check(abs(control["kish_weight_ess"] - 64) <= 1e-10)
            check(abs(control["negative_outcome_weight"] - 0.5) <= 1e-12)
        ancient = copy.deepcopy(toy)
        ancient[0].update({"label_end_exclusive_ms": 100 * day, "label_available_at_ms": 100 * day})
        ancient[1].update({"label_end_exclusive_ms": 10 * day, "label_available_at_ms": 10 * day})
        robust = api["weight_bank"](ancient, "half_life_90d_anchor_20pct", 2**53 - 1, guard)
        check(all(e["weight"] > 0 for e in robust["entries"]) and abs(robust["weight_sum"] - 1) <= 1e-12)
        for variant in ("unknown", "half_life_30d"):
            reject(lambda v=variant: api["weight_bank"](toy, v, asof, guard), "unregistered")
        reject(lambda: api["weight_bank"]([], "uniform", asof, guard), "budget")
        reject(lambda: api["weight_bank"](toy + [copy.deepcopy(toy[0])], "uniform", asof, guard), "duplicate")
        future = copy.deepcopy(toy)
        future[0]["label_available_at_ms"] = asof
        reject(lambda: api["weight_bank"](future, "uniform", asof, guard), "future or unavailable")
        malformed = copy.deepcopy(toy)
        malformed[0]["gross_return_3d"] = True
        reject(lambda: api["weight_bank"](malformed, "uniform", asof, guard), "numeric scalar")
        reject(lambda: api["weight_bank"](toy, "uniform", True, guard), "integer UTC")
        ctx = origin["context"]
        window_seconds = (ctx["decision_asof_ms"] - ctx["training_asof_ms"]) / 1000
        check(api["timing_gate"](ctx, window_seconds, 1.0, guard)["declared_timing_gate_passed"] is True)
        check(api["timing_gate"](ctx, window_seconds + 1, 1.0, guard)["declared_timing_gate_passed"] is False)
        reject(lambda: api["timing_gate"](ctx, 0, 1, guard), "positive registered")
        reject(lambda: api["timing_gate"](ctx, True, 1, guard), "numeric scalar")
        reject(lambda: api["timing_gate"](ctx, 1, -1, guard), "nonnegative")
        # Passing pure numerical controls cannot change the draft's fit authority.
        check(result["fit_allowed"] is False)
        bad(lambda o: o["safety"].update({"training_allowed": True}), "authority")
        bad(lambda o: o["safety"].update({"orders_enabled": 0}), "authority")
        bad(lambda o: o.update({"model_ready": True}), "cannot earn actual")
        bad(lambda o: o.update({"new_model_fits": True}), "no fits")
        bad(lambda o: o.update({"minimum_opportunities_64_passed": True}), "unearned opportunity")
        bad(lambda o: o.update({"future_labels": []}), "v2 origin inputs")
        bad(lambda o: o["training_inputs"]["scaler_candidate_input"].update({"opportunities": 8}), "wrong opportunity", inner=True)
        bad(lambda o: o["training_inputs"]["scaler_candidate_input"].update({"negative_opportunities": 0}), "wrong opportunity", inner=True)
        bad(lambda o: o["training_inputs"]["scaler_candidate_input"].update({"actual_scaler_fitted": 0}), "typed unfitted", inner=True)
        bad(lambda o: o["training_inputs"]["scaler_candidate_input"].update({"rows_sha256": o["training_inputs"]["training_guard"]["scaler_fit_rows_sha256"]}), "wrong opportunity", inner=True)
        bad(lambda o: o["training_inputs"]["scaler_candidate_input"]["rows"][0].update({"efficiency_12": True}), "numeric scalar", inner=True)
        bad(lambda o: o["deploy_features"]["safety"].update({"orders_enabled": True}), "deploy authority", deploy=True)
        bad(lambda o: o["deploy_features"].update({"future_labels": []}), "asof deploy", deploy=True)
        modified_proposal = copy.deepcopy(proposal)
        modified_proposal["models"]["minimum_completed_opportunities"] = 1
        reject(lambda: api["admit"](origin, modified_proposal, guard), "pinned separate draft")
        check(result["payload_sha256"] == api["sha"]({k: v for k, v in result.items() if k != "payload_sha256"}))
        tree = ast.parse(builder["read_bounded"](builder["HELPER"]).decode("utf-8"))
        imports = {n.module if isinstance(n, ast.ImportFrom) else alias.name for n in ast.walk(tree)
                   if isinstance(n, (ast.Import, ast.ImportFrom)) for alias in n.names}
        check(imports <= {"__future__", "hashlib", "json", "math"})
    else:
        output, read = builder["OUTPUT"], builder["read_bounded"]
        receipt = guard["strict_json"](read(output / "completion_receipt.json").decode("utf-8"))
        check(receipt["schema"] == "exia.genetic_primus.pre_fit_admission_receipt/1")
        check(receipt["inputs"] == bank)
        check(receipt["inputs_verified_unchanged_before_and_after"] is True)
        check(receipt["admission_semantic_sha256"] == result["payload_sha256"])
        check(receipt["safety"] == result["safety"])
        check(receipt["market_research_launch_ready"] is False and receipt["real_money_ready"] is False)
        check(set(receipt["output_byte_sha256"]) == {"admission.json", "summary.json"})
        for name, value in (("admission.json", result), ("summary.json", summary)):
            data = read(output / name)
            check(hashlib.sha256(data).hexdigest() == receipt["output_byte_sha256"][name])
            check(guard["strict_json"](data.decode("utf-8")) == value)
    builder["verify_inputs"](bank)
    check("numpy" not in sys.modules and "pandas" not in sys.modules and "websockets" not in sys.modules)
    check(not any(n == "exia" or n.startswith("exia.") for n in sys.modules))
    print(f"{checks} pre-fit assertions passed; 2 variants denied; fits=0, network/runtime=off")


if __name__ == "__main__":
    main()
