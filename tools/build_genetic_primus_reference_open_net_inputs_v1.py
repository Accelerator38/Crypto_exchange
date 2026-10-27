"""Fixture net arithmetic + derived market input bridge, never market net evaluation."""
from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
PURE = "src/exia/genetic_primus/reference_open_net_accounting_v1.py"
SELF = "tools/build_genetic_primus_reference_open_net_inputs_v1.py"
DOC = "docs/GENETIC_PRIMUS_REFERENCE_OPEN_NET_INPUTS_V1.md"
CONFIG = "configs/genetic_primus_reference_open_net_contract_v1.json"
PREVIOUS = "Reports/Exia/Genetic_Primus/market_gross_labels_v1_20261007"
CODE = "Reports/Exia/Genetic_Primus/reference_open_net_code_v1_20261007"
DEST = "Reports/Exia/Genetic_Primus/reference_open_net_inputs_v1_20261007"
prior = runpy.run_path(str(ROOT / "tools/materialize_genetic_primus_market_gross_labels_v1.py"))
old, api, read = prior["old"], prior["api"], prior["read"]
pure = runpy.run_path(str(ROOT / PURE))


def metadata():
    prep, result, receipt = (read(PREVIOUS + "/" + name) for name in
                             ("preparation.json", "result.json", "completion_receipt.json"))
    if prep["inputs"] != receipt["inputs"] or result["labels_are_gross_diagnostics_only"] is not True:
        raise ValueError("gross input roles/bank")
    bank = dict(prep["inputs"])
    prior["checker"]["unchanged"](bank)
    for name, expected in receipt["output_identities"].items():
        path = PREVIOUS + "/" + name
        current = prior["data_identity"](path, 16 * 1024 * 1024) if name.startswith("data/") else old["identity"](path)
        if current != expected:
            raise ValueError("gross artifact byte identity")
        if not name.startswith("data/"):
            bank[path] = expected
    for path in (PURE, SELF, DOC, CONFIG, PREVIOUS + "/completion_receipt.json"):
        bank[path] = old["identity"](path)
    prior["checker"]["unchanged"](bank)
    contract = read(CONFIG)
    gate = pure["assess"](contract)
    if contract["safety"] != prep["safety"] or result["safety"] != prep["safety"]:
        raise ValueError("safety context changed")
    prior["assert_guard"]()
    return bank, result, prep, contract, gate


def fixtures():
    B = pure["BAR_MS"]
    trade = {"comparison_key": "fabricated_only", "symbol": "SYN_REFERENCE_USDT",
             "period_start_ms": 10 * B, "period_end_ms": 28 * B, "action": "Long",
             "normalization_quote": "1000", "allocation_fraction": "0.25",
             "reference_price_kind": pure["FIXED"]["reference_price_kind"],
             "reference_entry": "100", "reference_exit": "110", "funding_events": [],
             "funding_events_complete": True, "borrow_quote": "0"}
    zero = {side: {"fee_bps": "0", "adverse_price_bps": "0"} for side in ("entry", "exit")}
    asymmetric = {"entry": {"fee_bps": "10", "adverse_price_bps": "15"},
                  "exit": {"fee_bps": "20", "adverse_price_bps": "25"}}
    stress = {"entry": {"fee_bps": "15", "adverse_price_bps": "20"},
              "exit": {"fee_bps": "25", "adverse_price_bps": "30"}}
    check, reject = prior["checker"]["check"], prior["checker"]["rejected"]
    result = pure["account"](trade, zero)
    check(result["net_quote"] == "25" and result["net_return"] == "0.025")
    check(result["cash_after_entry"] == "750" and result["final_cash"] == "1025")
    flat = {**trade, "action": "NoTrade", "allocation_fraction": "0", "reference_entry": None, "reference_exit": None}
    check(pure["account"](flat, asymmetric)["net_return"] == "0")
    funded = {**trade, "borrow_quote": "1", "funding_events": [
        {"settled_at_ms": 11 * B, "mark_price": "100", "signed_rate_bps": "5"},
        {"settled_at_ms": 12 * B, "mark_price": "110", "signed_rate_bps": "-2"}]}
    accounted = pure["account"](funded, asymmetric)
    check(Decimal(accounted["net_return"]) < Decimal(result["net_return"]))
    check(Decimal(pure["account"](funded, stress)["net_return"]) < Decimal(accounted["net_return"]))
    pair = pure["paired"](funded, flat, asymmetric)
    check(pair["incremental_net_return"] == accounted["net_return"])
    half = {**funded, "allocation_fraction": "0.125", "borrow_quote": "0.5"}
    delta = pure["paired"](funded, half, asymmetric)
    with localcontext() as context:
        context.prec = 50
        check(abs(Decimal(delta["incremental_net_return"]) - Decimal(accounted["net_return"]) / 2) < Decimal("1e-45"))
    with localcontext() as context:
        context.prec = 6
        check(pure["account"](funded, asymmetric) == accounted)
    reject(lambda: pure["account"]({**trade, "funding_events_complete": False}, zero))
    reject(lambda: pure["account"]({**trade, "borrow_quote": None}, zero))
    reject(lambda: pure["account"](trade, {"entry": {"fee_bps": None, "adverse_price_bps": "0"}, "exit": zero["exit"]}))
    reject(lambda: pure["account"]({**trade, "normalization_quote": None}, zero))
    reject(lambda: pure["account"]({**trade, "reference_price_kind": "mid"}, zero))
    reject(lambda: pure["account"]({**trade, "period_end_ms": 27 * B}, zero))
    reject(lambda: pure["paired"](trade, {**flat, "normalization_quote": "2000"}, zero))
    boundary = {**funded, "funding_events": [{"settled_at_ms": 10 * B, "mark_price": "100", "signed_rate_bps": "1"}]}
    reject(lambda: pure["account"](boundary, zero))
    duplicate = {**funded, "funding_events": [funded["funding_events"][0], funded["funding_events"][0]]}
    reject(lambda: pure["account"](duplicate, zero))
    reject(lambda: pure["account"]({**flat, "reference_entry": "100"}, zero))
    check(Decimal(pure["account"]({**trade, "reference_exit": "90"}, zero)["net_return"]) < 0)
    contract = {**pure["FIXED"], "parameters": {name: None for name in pure["PARAMETERS"]},
                "safety": {name: False for name in pure["SAFETY"]}}
    sample = {"symbol": "SYN_REFERENCE_USDT", "decision_asof_ms": 9 * B,
              "entry_open_ms": 10 * B, "exit_open_ms": 28 * B,
              "label_available_at_ms": 29 * B, "training_cutoff_ms": 29 * B,
              "entry_observation": {"timestamp": 10 * B, "symbol": "SYN_REFERENCE_USDT", "open": 100},
              "exit_observation": {"timestamp": 28 * B, "symbol": "SYN_REFERENCE_USDT", "open": 110},
              "gross_open_return_decimal": "0.1", "proposal_long": False,
              "target_kind": "gross_open_open_diagnostic_not_net_training_target",
              "fit_allowed": False, "net_objective_allowed": False, "outcome_filter_applied": False,
              "feature_row_sha256": "c" * 64}
    sample["sample_id"] = pure["sha"]({"symbol": sample["symbol"], "decision_asof_ms": 9 * B})
    pending = pure["blocked_request"](sample, {"fixture": True}, contract)
    check(pending["net_target"] is None and pending["stress_net_target"] is None
          and pending["paired_incremental_target"] is None and pending["proposal_long"] is False)
    reject(lambda: pure["account"](pending["long_input_pending_registration"], zero))
    reject(lambda: pure["blocked_request"]({**sample, "gross_open_return_decimal": "0.2"}, {}, contract))
    reject(lambda: pure["blocked_request"]({**sample, "training_cutoff_ms": 29 * B - 1}, {}, contract))
    reject(lambda: pure["blocked_request"]({**sample, "sample_id": "d" * 64}, {}, contract))


def byte_identity(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def code_check(verify):
    bank, _, prep, _, gate = metadata()
    fixtures()
    prior["assert_guard"]()
    prior["checker"]["unchanged"](bank)
    result = {"schema": "exia.genetic_primus.reference_open_net_code_result/1", "inputs": bank,
              "fixture_cash_quantity_attribution_and_paired_regressions_passed": True,
              "fixture_rates_are_not_market_cost_estimates": True, "gate": gate,
              "parquet_reads": 0, "market_net_targets": 0, "fit_count": 0, "safety": prep["safety"]}
    receipt = {"inputs": bank, "result_identity": byte_identity(api["canonical"](result) + b"\n"),
               "safety": prep["safety"]}
    if verify:
        if (read(CODE + "/result.json") != result or read(CODE + "/completion_receipt.json") != receipt
                or old["identity"](CODE + "/result.json") != receipt["result_identity"]):
            raise ValueError("net code receipt mismatch")
    else:
        (ROOT / CODE).mkdir(exist_ok=False)
        old["emit"](CODE + "/result.json", result, api)
        old["emit"](CODE + "/completion_receipt.json", receipt, api)
    print("CODE OK: cash/quantity/fees/funding/paired; costs unset; market-net/fit=0")


def build(verify):
    code_check(True)
    bank, source, prep, contract, gate = metadata()
    path = source["label_path"]
    source_identity = prior["data_identity"](path, 16 * 1024 * 1024)
    if source_identity != source["label_identity"]:
        raise ValueError("label source identity")
    records, keys = [], set()
    for line_no, sample in prior["lines"](path, 16 * 1024 * 1024):
        key = (sample["decision_asof_ms"], sample["symbol"])
        if key in keys or sample["symbol"] not in prep["symbols"]:
            raise ValueError("duplicate/substituted net input sample")
        keys.add(key)
        reference = {"path": path, "line_1based": line_no, "file_identity": source_identity}
        records.append(pure["blocked_request"](sample, reference, contract))
    expected = {(r["decision_asof_ms"], symbol) for r in prep["schedule"]["eligible"] for symbol in prep["symbols"]}
    if keys != expected or len(records) != source["sample_rows"]:
        raise ValueError("net input sample bank completeness")
    body = b"".join(api["canonical"](record) + b"\n" for record in records)
    if len(body) > 12 * 1024 * 1024:
        raise ValueError("net input artifact cap")
    data_path = DEST + "/data/blocked_net_inputs.jsonl"
    result = {"schema": "exia.genetic_primus.reference_open_net_inputs_result/1", "sample_rows": len(records),
              "cost_blocked_rows": len(records), "net_target_values": 0, "market_net_account_calls": 0,
              "source_label_identity": source_identity, "data_path": data_path, "data_identity": byte_identity(body),
              "contract_sha256": pure["sha"](contract), "gate": gate, "parquet_reads": 0,
              "fit_count": 0, "model_financial_evaluations": 0, "safety": prep["safety"]}
    receipt = {"schema": "exia.genetic_primus.reference_open_net_inputs_receipt/1", "inputs": bank,
               "output_identities": {"data/blocked_net_inputs.jsonl": result["data_identity"],
                                      "result.json": byte_identity(api["canonical"](result) + b"\n")},
               "safety": prep["safety"]}
    prior["assert_guard"]()
    prior["checker"]["unchanged"](bank)
    if prior["data_identity"](path, 16 * 1024 * 1024) != source_identity:
        raise ValueError("label source changed during net input build")
    # Recheck all prior output bytes as well as the metadata bank.
    after_bank, _, _, _, _ = metadata()
    if after_bank != bank:
        raise ValueError("net input bank changed")
    if verify:
        item = (ROOT / data_path).resolve()
        item.relative_to(ROOT / DEST / "data")
        if item.stat().st_size > 12 * 1024 * 1024 or byte_identity(item.read_bytes()) != result["data_identity"]:
            raise ValueError("net input data identity")
        if (read(DEST + "/result.json") != result or read(DEST + "/completion_receipt.json") != receipt
                or old["identity"](DEST + "/result.json") != receipt["output_identities"]["result.json"]):
            raise ValueError("net input receipt mismatch")
    else:
        (ROOT / DEST).mkdir(exist_ok=False)
        (ROOT / DEST / "data").mkdir(exist_ok=False)
        with (ROOT / data_path).open("xb") as output:
            output.write(body)
        old["emit"](DEST + "/result.json", result, api)
        old["emit"](DEST + "/completion_receipt.json", receipt, api)
    print(f"{'VERIFIED' if verify else 'BUILT'}: inputs={len(records)}; unknown costs block every net target; parquet/fit=0")


if __name__ == "__main__":
    try:
        args = sys.argv[1:]
        if args in (["--check-code"], ["--verify-code"]):
            code_check(args == ["--verify-code"])
        elif args in (["--build"], ["--verify"]):
            build(args == ["--verify"])
        else:
            raise ValueError("use check/verify-code, build or verify")
    except Exception as error:
        print("NET INPUT FAILED: " + type(error).__name__ + ": " + str(error)[:160])
        raise SystemExit(1) from None
