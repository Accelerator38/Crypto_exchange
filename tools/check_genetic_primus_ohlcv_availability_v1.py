"""Bounded pure causal regressions; no parquet, fitter, network or pytest."""
from __future__ import annotations

import copy
import hashlib
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PURE = "src/exia/genetic_primus/ohlcv_availability_v1.py"
SELF = "tools/check_genetic_primus_ohlcv_availability_v1.py"
CONFIG = "configs/genetic_primus_ohlcv_availability_contract_v1.json"
DOC = "docs/GENETIC_PRIMUS_OHLCV_AVAILABILITY_V1.md"
PREVIOUS = "Reports/Exia/Genetic_Primus/offline_ohlcv_quality_v1_20261007"
DEST = "Reports/Exia/Genetic_Primus/ohlcv_availability_v1_20261007"
prior = runpy.run_path(str(ROOT / "tools/check_genetic_primus_offline_ohlcv_quality_v1.py"))
old, api = prior["old"], prior["api"]
pure = runpy.run_path(str(ROOT / PURE))
ASSERTIONS = 0


def check(ok):
    global ASSERTIONS
    if not ok:
        raise AssertionError("causal regression")
    ASSERTIONS += 1


def rejected(call):
    try:
        call()
    except ValueError:
        check(True)
    else:
        raise AssertionError("unsafe timing accepted")


def regressions(contract):
    b = pure["BAR_MS"]
    row, prefix = pure["row_available_at"], pure["closed_prefix_plan"]
    feature, label = pure["feature_gate"], pure["label_gate"]
    assessed = pure["assess"](contract)
    check(len(assessed["missing_registration"]) == 3 and not assessed["fit_allowed"])
    # Synthetic lags, not chosen/calibrated market assumptions.
    for lag in (0, 1, 120000, b, b + 1):
        start = 10 * b
        ready = row(start, lag)
        check(ready == start + b + lag)
        for asof, expected in ((start, False), (ready - 1, False), (ready, True), (ready + 1, True)):
            gate = feature([start - b, start], asof, lag, 0, 100 * b)
            check(gate["scenario_eligible"] is expected and not gate["fit_allowed"])
        for asof in (0, ready - 1, ready, ready + 1, 110 * b):
            plan = prefix(asof, lag, 0, 100 * b)
            # Independent bounded calendar enumeration checks arithmetic plan.
            expected = [s for s in range(0, 100 * b, b) if s + b + lag <= asof]
            check(plan["bar_count"] == len(expected))
            check(plan["last_bar_start_ms"] == (expected[-1] if expected else None))
        entry, exit_open = 10 * b, 28 * b
        maturity = exit_open + b + lag
        for train, expected in ((maturity - 1, False), (maturity, False),
                                (maturity + 7 * b - 1, False),
                                (maturity + 7 * b, True), (maturity + 7 * b + 1, True)):
            gate = label(entry, exit_open, train, lag, 0, 100 * b)
            check(gate["scenario_eligible"] is expected and not gate["fit_allowed"])
            previous = api["completed_label_gate"](
                api["execution_window"](9 * b, 9 * b, 0), row(entry, lag), row(exit_open, lag), train)
            check(gate["label_available_at_ms"] == previous["label_available_at_ms"]
                  and gate["completed_and_purged"] == previous["completed_and_purged"])
    for delay in (0, 1, b - 1):
        for decision in (10 * b, 11 * b - 1):
            window = api["execution_window"](10 * b, decision, delay)
            entry, exit_open = window["planned_entry_open_ms"], window["planned_exit_open_ms"]
            check(entry > decision + delay and exit_open - entry == 18 * b)
            gate = label(entry, exit_open, 100 * b, 0, 0, 100 * b)
            check(gate["label_available_at_ms"] == exit_open + b)
    check(not feature([0, b], 10 * b, 0, b, 100 * b)["scenario_eligible"])
    check(not feature([99 * b, 100 * b], 110 * b, 0, 0, 100 * b)["scenario_eligible"])
    check(not label(82 * b, 100 * b, 110 * b, 0, 0, 100 * b)["scenario_eligible"])
    # Late labels must not become eligible simply because nominal exit already occurred.
    check(not label(10 * b, 28 * b, 35 * b, 0, 0, 100 * b)["scenario_eligible"])
    for start, lag in ((True, 0), (-b, 0), (1, 0), (0, None), (0, -1),
                       (0, False), (0, 0.5), (0, 2**63 - 1)):
        rejected(lambda start=start, lag=lag: row(start, lag))
    for deps in ([], [0, 0], [b, 0], [0, 2 * b], [1], [False], "0", [0] * 20001):
        rejected(lambda deps=deps: feature(deps, 100 * b, 0, 0, 100 * b))
    for args in ((0, 0), (b, 0), (1, b), (0, 1)):
        rejected(lambda args=args: prefix(10 * b, 0, *args))
    rejected(lambda: feature([0], True, 0, 0, 100 * b))
    rejected(lambda: label(10 * b, 27 * b, 100 * b, 0, 0, 100 * b))
    rejected(lambda: api["execution_window"](10 * b, 10 * b, b))
    for key in pure["SAFETY"]:
        changed = copy.deepcopy(contract)
        changed["safety"][key] = True
        rejected(lambda changed=changed: pure["assess"](changed))
    for key, value in (("bar_ms", True), ("purge_bars", 6), ("label_horizon_bars", 17),
                       ("timestamp_semantics", "bar_end"), ("historical_results_requalified", True)):
        changed = copy.deepcopy(contract)
        changed[key] = value
        rejected(lambda changed=changed: pure["assess"](changed))
    for value in (False, -1, "0"):
        changed = copy.deepcopy(contract)
        changed["availability"]["ingestion_lag_ms"] = value
        rejected(lambda changed=changed: pure["assess"](changed))
    for value in ("x" * 64, True, "a" * 63):
        changed = copy.deepcopy(contract)
        changed["availability"]["evidence_sha256"] = value
        rejected(lambda changed=changed: pure["assess"](changed))
    claimed = copy.deepcopy(contract)
    claimed["availability"] = {"ingestion_lag_ms": 0, "evidence_sha256": "a" * 64,
                                "late_revision_policy_sha256": "b" * 64}
    check(not pure["assess"](claimed)["fit_allowed"])
    check(not pure["assess"](claimed)["observed_arrival_verified"])
    rejected(lambda: pure["assess"]({**contract, "unexpected": 0}))
    return assessed


def build():
    previous = prior["read"](PREVIOUS + "/completion_receipt.json")
    bank = dict(previous["inputs"])
    prior["unchanged"](bank)
    for name, expected in previous["output_identities"].items():
        path = PREVIOUS + "/" + name
        if old["identity"](path) != expected:
            raise ValueError("previous result identity changed")
        bank[path] = expected
    for path in (PREVIOUS + "/completion_receipt.json", PURE, SELF, CONFIG, DOC):
        bank[path] = old["identity"](path)
    contract = prior["read"](CONFIG)
    assessed = regressions(contract)
    if any(k.split(".")[0] in old["BLOCKED"] for k in sys.modules):
        raise ValueError("forbidden dependency loaded")
    prior["unchanged"](bank)
    result = {"schema": "exia.genetic_primus.ohlcv_availability_check/1", "inputs": bank,
              "contract_identity": old["identity"](CONFIG), "contract_semantic_sha256": api["sha"](contract),
              "direct_assertions": ASSERTIONS, "assessment": assessed,
              "parquet_reads": 0, "fit_count": 0, "financial_evaluation_count": 0,
              "synthetic_scenarios_only": True, "safety": contract["safety"],
              "historical_results_requalified": False, "market_research_launch_ready": False,
              "real_money_ready": False}
    body = api["canonical"](result) + b"\n"
    receipt = {"schema": "exia.genetic_primus.ohlcv_availability_receipt/1", "inputs": bank,
               "output_identities": {"result.json": {"bytes": len(body),
                   "sha256": hashlib.sha256(body).hexdigest()}}, "safety": contract["safety"]}
    return result, receipt


def main():
    if sys.argv[1:] not in (["--run"], ["--verify"]):
        raise ValueError("use --run or --verify")
    result, receipt = build()
    if sys.argv[1:] == ["--run"]:
        old["local"](DEST + "/result.json").parent.mkdir(exist_ok=False)
        for name, value in (("result.json", result), ("completion_receipt.json", receipt)):
            old["emit"](DEST + "/" + name, value, api)
        print(f"COMPLETE: assertions={ASSERTIONS}; inputs={len(result['inputs'])}; parquet=0; fit=0")
    else:
        if prior["read"](DEST + "/result.json") != result or prior["read"](DEST + "/completion_receipt.json") != receipt:
            raise ValueError("result/receipt mismatch")
        print(f"VERIFIED: assertions={ASSERTIONS}; receipt OK; parquet=0; fit=0")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("STOPPED: " + type(exc).__name__ + "; no training/network")
        raise SystemExit(1)
