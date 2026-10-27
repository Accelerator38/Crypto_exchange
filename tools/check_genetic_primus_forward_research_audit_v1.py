"""Small deterministic metadata checks; never pytest or model execution."""
from __future__ import annotations

import argparse
import copy
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
    builder = runpy.run_path(str(ROOT / "tools/build_genetic_primus_forward_research_audit_v1.py"))
    api, proposal, audit, summary, bank = builder["prepare"]()
    checks = 0

    def check(condition):
        nonlocal checks
        if not condition:
            raise AssertionError("metadata regression")
        checks += 1

    def reject(call):
        try:
            call()
        except (ValueError, KeyError, TypeError, SyntaxError):
            check(True)
        else:
            raise AssertionError("invalid metadata accepted")

    if not args.artifacts_only:
        check(api["validate_proposal"](proposal) == api["PROPOSAL_SHA"])
        check(audit["observation_count"] == 16)
        check(len({o["id"] for o in audit["observations"]}) == 16)
        check(all(o["status"] == "source_anchor_observed_not_execution_verified" for o in audit["observations"]))
        check(not audit["market_data_read"] and not audit["audited_code_executed"])
        check(not summary["market_research_launch_ready"] and not summary["real_money_ready"])
        check(summary["new_model_fits"] == summary["new_market_strategy_evaluations"] == 0)
        check(len(summary["blocking_requirements"]) == 10)
        check(proposal["measurement"]["economic_floor_bps_per_day"] == 0.0)
        check(api["semantic_sha"]({"a": 1, "b": 2}) == api["semantic_sha"]({"b": 2, "a": 1}))
        for key in api["SAFETY_KEYS"]:
            for value in (True, 0):
                altered = copy.deepcopy(proposal)
                altered["safety"][key] = value
                reject(lambda p=altered: api["validate_proposal"](p))
        for section, key, value in (
            ("clock", "purge_bars", True), ("clock", "retrain_days", 7),
            ("clock", "training_latency_budget_seconds", 0),
            ("models", "ga_enabled", True), ("models", "variants", ["uniform"]),
            ("labels", "new_executable_label_definition", "legacy_assumed_realistic"),
            ("accounting", "cost_adapter", "synthetic_is_live"),
            ("measurement", "economic_floor_bps_per_day", 1),
            ("testing_budget", "authorized_market_runs", 1),
            ("testing_budget", "historical_global_trial_count", 0),
            ("sealed", "peek_allowed", True), ("sealed", "automatic_old_interval_reuse", True),
            ("readiness", "real_money_ready", True),
            ("lineage", "all_bindings_verified_before_launch", True),
        ):
            altered = copy.deepcopy(proposal)
            altered[section][key] = value
            reject(lambda p=altered: api["validate_proposal"](p))
        reject(lambda: api["strict_json"]('{"a":1,"a":2}'))
        reject(lambda: api["strict_json"]('{"a":NaN}'))
        reject(lambda: api["strict_json"]('{"a":Infinity}'))
        fixture = "def chosen(x):\n    cutoff = x - 7\n    return cutoff\n"
        check(api["anchor"](fixture, "chosen", "cutoff = x - 7")["line"] == 2)
        reject(lambda: api["anchor"](fixture, "chosen", "cutoff = x + 7"))
        reject(lambda: api["anchor"](fixture, "missing", "cutoff = x - 7"))
        reject(lambda: api["anchor"](fixture + fixture, "chosen", "cutoff = x - 7"))
        reject(lambda: api["audit"](proposal, {}))
    else:
        output = builder["OUTPUT"]
        read = builder["read_bounded"]
        receipt = api["strict_json"](read(output / "completion_receipt.json").decode("utf-8"))
        check(receipt["schema"] == "exia.genetic_primus.forward_research_audit_receipt/1")
        check(receipt["inputs"] == bank)
        check(receipt["inputs_verified_unchanged_before_and_after"] is True)
        check(receipt["proposal_semantic_sha256"] == api["PROPOSAL_SHA"])
        check(receipt["market_research_launch_ready"] is False and receipt["real_money_ready"] is False)
        check(receipt["safety"] == proposal["safety"])
        check(set(receipt["output_byte_sha256"]) == {"proposal.json", "audit.json", "summary.json"})
        for name, expected in (("proposal.json", proposal), ("audit.json", audit), ("summary.json", summary)):
            data = read(output / name)
            check(hashlib.sha256(data).hexdigest() == receipt["output_byte_sha256"][name])
            check(api["strict_json"](data.decode("utf-8")) == expected)
    builder["verify_inputs"](bank)
    print(f"{checks} metadata assertions passed; static only; fits=0; no market data/network/runtime")


if __name__ == "__main__":
    main()
