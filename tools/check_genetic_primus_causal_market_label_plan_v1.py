"""Metadata + fabricated clocks only; no parquet/native/feature-value parsing."""
from __future__ import annotations

from pathlib import Path
import hashlib
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
PURE = "src/exia/genetic_primus/causal_market_label_plan_v1.py"
SELF = "tools/check_genetic_primus_causal_market_label_plan_v1.py"
DOC = "docs/GENETIC_PRIMUS_CAUSAL_MARKET_LABEL_PLAN_V1.md"
PREVIOUS = "Reports/Exia/Genetic_Primus/market_feature_materialization_v1_20261007"
DEST = "Reports/Exia/Genetic_Primus/causal_market_label_plan_v1_20261007"
PROTOCOL = "configs/genetic_primus_offline_market_engineering_protocol_v1.json"
KERNEL = "src/exia/genetic_primus/ohlcv_availability_v1.py"
previous = runpy.run_path(str(ROOT / "tools/materialize_genetic_primus_market_features_v1.py"))
old, api = previous["old"], previous["api"]
pure, kernel = runpy.run_path(str(ROOT / PURE)), runpy.run_path(str(ROOT / KERNEL))


def check(ok):
    if not ok:
        raise AssertionError("causal label scheduling regression")


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid scheduling input accepted")


def unchanged(bank):
    # New local cap; frozen earlier-stage validators remain unchanged.
    if not isinstance(bank, dict) or not bank or len(bank) > 256:
        raise ValueError("metadata bank cap")
    for path, expected in bank.items():
        if old["identity"](path) != expected:
            raise ValueError("metadata bank changed")


def run(verify):
    previous["finalize"](verify=True)  # derived bytes/metadata only; no decoder
    prep = previous["read"](PREVIOUS + "/preparation.json")
    receipt = previous["read"](PREVIOUS + "/completion_receipt.json")
    bank = dict(prep["inputs"])
    for path in (PURE, SELF, DOC, KERNEL, PROTOCOL,
                 PREVIOUS + "/preparation.json", PREVIOUS + "/result.json",
                 PREVIOUS + "/completion_receipt.json"):
        bank[path] = old["identity"](path)
    for path, expected in receipt["output_identities"].items():
        if not path.startswith("data/"):
            full = PREVIOUS + "/" + path
            if old["identity"](full) != expected:
                raise ValueError("prior metadata identity")
            bank[full] = expected
    unchanged(bank)
    protocol = previous["read"](PROTOCOL)
    scenario = prep["scenario"]
    check(scenario["lag_bars"] == 0 and scenario["ingestion_lag_ms"] == 0)
    check(all(value is False for value in prep["safety"].values()))
    args = (scenario["events"], protocol["clock"], prep["metadata"], 0)
    blocked = pure["build"](*args, None, kernel)
    check(not blocked["scenario_selected"] and not blocked["label_value_read_allowed"])
    hypothetical = pure["build"](*args, 0, kernel)
    check(len(hypothetical["eligible"]) + len(hypothetical["excluded"]) == blocked["candidate_events"])
    B = pure["BAR_MS"]
    op = pure["opportunity"]
    def fixture(train=36 * B, lag=0, delay=0, end=100 * B):
        return op(9 * B, train, lag, delay, 0, end, kernel)
    r = fixture()
    check(r["entry_open_ms"] == 10 * B and r["exit_open_ms"] == 28 * B)
    check(r["label_available_at_ms"] == 29 * B and r["training_cutoff_ms"] == 29 * B)
    check(r["scenario_eligible"] and not fixture(train=36 * B - 1)["scenario_eligible"])
    check(not fixture(lag=1)["scenario_eligible"])
    check(fixture(train=36 * B + 1, lag=1)["scenario_eligible"])
    check(fixture(delay=1)["entry_open_ms"] == 10 * B)
    check(fixture(delay=B)["entry_open_ms"] == 11 * B)
    check(fixture(delay=B)["exit_open_ms"] == 29 * B)
    check(not fixture(end=28 * B)["scenario_eligible"])
    for invalid in (-1, True, 0.0, None):
        rejected(lambda invalid=invalid: fixture(delay=invalid))
    rejected(lambda: pure["build"](*args, B + 1, kernel))
    wrong_clock = {**protocol["clock"], "purge_bars": 6}
    rejected(lambda: pure["build"](scenario["events"], wrong_clock, prep["metadata"], 0, 0, kernel))
    wrong_events = [*scenario["events"], scenario["events"][-1]]
    rejected(lambda: pure["build"](wrong_events, protocol["clock"], prep["metadata"], 0, 0, kernel))
    check(not any(name.split(".")[0] in {"numpy", "pandas", "exia", "pytest", "pyarrow",
                                         "websockets", "aiohttp", "httpx", "requests"}
                  for name in sys.modules))
    unchanged(bank)
    previous["finalize"](verify=True)
    result = {"schema": "exia.genetic_primus.causal_market_label_plan_result/1",
              "inputs": bank, "scenario_selected": False, "execution_delay_ms": None,
              "registered_ingestion_lag_ms": 0, "default_delay_is_not_zero": True,
              "hypothetical_zero_delay_schedule": hypothetical,
              "boundary_regressions_passed": True, "parquet_reads": 0,
              "label_values_computed": 0, "fit_count": 0, "safety": prep["safety"]}
    completion = {"schema": "exia.genetic_primus.causal_market_label_plan_receipt/1",
                  "inputs": bank, "output_identities": {"result.json": {
                      "bytes": len(api["canonical"](result) + b"\n"),
                      "sha256": hashlib.sha256(api["canonical"](result) + b"\n").hexdigest()}},
                  "safety": prep["safety"]}
    if verify:
        check(previous["read"](DEST + "/result.json") == result)
        check(previous["read"](DEST + "/completion_receipt.json") == completion)
        check(old["identity"](DEST + "/result.json") == completion["output_identities"]["result.json"])
    else:
        (ROOT / DEST).mkdir(exist_ok=False)
        old["emit"](DEST + "/result.json", result, api)
        old["emit"](DEST + "/completion_receipt.json", completion, api)
    print("LABEL PLAN OK: boundaries/purge; execution delay unset; parquet/labels/fit=0")


if __name__ == "__main__":
    if sys.argv[1:] not in (["--check"], ["--verify"]):
        raise ValueError("use --check or --verify; no label materialization mode")
    try:
        run(sys.argv[1] == "--verify")
    except Exception as error:
        print("LABEL PLAN FAILED: " + type(error).__name__ + ": " + str(error)[:160])
        raise SystemExit(1) from None
