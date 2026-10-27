"""Bounded deterministic synthetic accounting regressions; no market fit/data."""
from __future__ import annotations

import copy
from decimal import Decimal, localcontext
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
PURE = "src/exia/genetic_primus/synthetic_net_accounting_v1.py"
SELF = "tools/check_genetic_primus_synthetic_net_accounting_v1.py"
CONFIG = "configs/genetic_primus_synthetic_net_accounting_contract_v1.json"
DOC = "docs/GENETIC_PRIMUS_SYNTHETIC_NET_ACCOUNTING_V1.md"
PREVIOUS = "Reports/Exia/Genetic_Primus/synthetic_label_matrix_v1_20261007"
DEST = "Reports/Exia/Genetic_Primus/synthetic_net_accounting_v1_20261007"
prior = runpy.run_path(str(ROOT / "tools/check_genetic_primus_synthetic_label_matrix_v1.py"))
old, api, quality = prior["old"], prior["api"], prior["quality"]
pure = runpy.run_path(str(ROOT / PURE))
ASSERTIONS = 0


def check(ok):
    global ASSERTIONS
    if not ok:
        raise AssertionError("accounting regression")
    ASSERTIONS += 1


def rejected(call):
    try:
        call()
    except ValueError:
        check(True)
    else:
        raise AssertionError("unsupported accounting accepted")


def scenarios():
    def item(name, entry, exit_values):
        fields = ("fee_bps", "half_spread_bps", "slippage_bps")
        return {"schema": "exia.genetic_primus.synthetic_cost_scenario/1", "name": name,
                "entry": dict(zip(fields, entry)), "exit": dict(zip(fields, exit_values))}
    return [item("fixture_zero", ("0", "0", "0"), ("0", "0", "0")),
            item("fixture_base", ("2", "0.5", "1"), ("3", "0.75", "2")),
            item("fixture_stress", ("4", "1", "4"), ("5", "2", "6"))]


def trade(action="Long", exit_price="110"):
    return {"schema": "exia.genetic_primus.synthetic_cost_trade/1", "instrument": "SYN_LINEAR_USDT",
            "comparison_key": "fixture_origin_01", "period_start_ms": 14400000,
            "period_end_ms": 19 * 14400000, "normalization_quote": "1000",
            "allocation_fraction": "0.5" if action == "Long" else "0", "action": action,
            "reference_price_kind": "synthetic_mid", "reference_entry": "100" if action == "Long" else None,
            "reference_exit": exit_price if action == "Long" else None,
            "funding_events": [], "funding_events_complete": action == "Long", "borrow_quote": "0"}


def regressions(contract):
    check(not pure["assess"](contract)["costs_calibrated"])
    zero, base, stress = scenarios()
    account, pair = pure["account"], pure["paired"]
    examples = []
    with localcontext() as ctx:
        ctx.prec = 50
        check(pure["bps_fraction"]("2.5") == Decimal("0.00025"))
        for value in (None, True, -1, "NaN", float("inf"), 10000, "0.0000000000001"):
            rejected(lambda value=value: pure["bps_fraction"](value))
        for price, direction in (("90", "down"), ("100", "flat"), ("110", "up")):
            long, idle = trade(exit_price=price), trade("NoTrade")
            results = [account(long, s, contract) for s in (zero, base, stress)]
            check(Decimal(results[0]["net_quote"]) == Decimal("5") * (Decimal(price) - Decimal("100")))
            check(Decimal(results[2]["net_return"]) < Decimal(results[1]["net_return"]) < Decimal(results[0]["net_return"]))
            for scenario, result in zip((zero, base, stress), results):
                idle_result = account(idle, scenario, contract)
                check(all(Decimal(idle_result[k]) == 0 for k in ("quantity", "fees_quote", "execution_drag_quote", "funding_debit_quote", "net_quote", "net_return")))
                identical = pair(long, copy.deepcopy(long), scenario, contract)
                check(identical["delta_net_return"] == "0")
                delta = pair(long, idle, scenario, contract)
                check(delta["delta_net_return"] == result["net_return"])
                check(not delta["promotion_allowed"])
                # Independent cash-proceeds oracle, rather than the decomposition formula.
                rates = scenario
                buy = Decimal("100") * (1 + (Decimal(rates["entry"]["half_spread_bps"]) + Decimal(rates["entry"]["slippage_bps"])) / 10000)
                sell = Decimal(price) * (1 - (Decimal(rates["exit"]["half_spread_bps"]) + Decimal(rates["exit"]["slippage_bps"])) / 10000)
                qty = Decimal("500") / (buy * (1 + Decimal(rates["entry"]["fee_bps"]) / 10000))
                oracle = qty * sell * (1 - Decimal(rates["exit"]["fee_bps"]) / 10000) - Decimal("500")
                check(abs(Decimal(result["net_quote"]) - oracle) < Decimal("1e-40"))
                check(abs(Decimal(result["entry_fee_quote"]) - qty * buy * Decimal(rates["entry"]["fee_bps"]) / 10000) < Decimal("1e-40"))
                check(abs(Decimal(result["exit_fee_quote"]) - qty * sell * Decimal(rates["exit"]["fee_bps"]) / 10000) < Decimal("1e-40"))
            examples.append({"synthetic_direction": direction, "scenario_result_hashes": [pure["sha"](r) for r in results],
                             "notrade_zero_checked": True, "stress_order_checked_without_funding": True})
        funded = trade(exit_price="100")
        funded["funding_events"] = [{"settled_at_ms": 9 * 14400000, "mark_price": "100", "signed_rate_bps": "2"}]
        debit = account(funded, zero, contract)
        check(Decimal(debit["funding_debit_quote"]) == Decimal("0.1"))
        check(Decimal(debit["net_quote"]) == Decimal("-0.1"))
        funded["funding_events"][0]["signed_rate_bps"] = "-2"
        rebate = account(funded, zero, contract)
        check(Decimal(rebate["net_quote"]) == Decimal("0.1"))
        funded["borrow_quote"] = "0.3"
        check(Decimal(account(funded, zero, contract)["net_quote"]) == Decimal("-0.2"))
        for boundary in (funded["period_start_ms"], funded["period_end_ms"], 0, funded["period_end_ms"] + 1):
            altered = copy.deepcopy(funded)
            altered["funding_events"][0]["settled_at_ms"] = boundary
            rejected(lambda altered=altered: account(altered, base, contract))
        for mutation in ("duplicate", "unknown"):
            altered = copy.deepcopy(funded)
            if mutation == "duplicate":
                altered["funding_events"] *= 2
            else:
                altered["funding_events_complete"] = False
            rejected(lambda altered=altered: account(altered, base, contract))
        for key, value in (("allocation_fraction", "1.1"), ("allocation_fraction", "0"), ("borrow_quote", None),
                           ("reference_price_kind", "ohlcv_open"), ("reference_price_kind", "observed_fill"),
                           ("funding_events", None), ("instrument", "BTCUSDT"), ("action", "Short")):
            altered = trade()
            altered[key] = value
            rejected(lambda altered=altered: account(altered, base, contract))
        for key, value in (("allocation_fraction", "0.1"), ("reference_entry", "100"), ("borrow_quote", "1")):
            altered = trade("NoTrade")
            altered[key] = value
            rejected(lambda altered=altered: account(altered, base, contract))
        for key, value in (("normalization_quote", "1001"), ("comparison_key", "other"), ("reference_exit", "111")):
            altered = trade()
            altered[key] = value
            rejected(lambda altered=altered: pair(trade(), altered, base, contract))
        smaller = trade()
        smaller["allocation_fraction"] = "0.25"
        check(Decimal(pair(trade(), smaller, zero, contract)["delta_net_return"]) == Decimal("0.025"))
        for side in ("entry", "exit"):
            for field in ("fee_bps", "half_spread_bps", "slippage_bps"):
                altered = copy.deepcopy(base)
                altered[side][field] = None
                rejected(lambda altered=altered: account(trade(), altered, contract))
        altered = copy.deepcopy(base)
        altered["exit"].update(half_spread_bps="6000", slippage_bps="4000")
        rejected(lambda: account(trade(), altered, contract))
        aggregate = {"round_trip_cost_bps": 16}
        rejected(lambda: account(trade(), aggregate, contract))
        for key in pure["SAFETY"]:
            altered = copy.deepcopy(contract)
            altered["safety"][key] = True
            rejected(lambda altered=altered: pure["assess"](altered))
        altered = copy.deepcopy(contract)
        altered["market_calibration"] = "claimed"
        rejected(lambda: pure["assess"](altered))
        baseline_result = account(trade(), base, contract)
        with localcontext() as other:
            other.prec = 3
            check(account(trade(), base, contract) == baseline_result)
            check(pure["bps_fraction"]("12.345678") == Decimal("0.0012345678"))
            rejected(lambda: pure["number"]("1000000000000000.00001"))
    return examples, [pure["sha"](s) for s in (zero, base, stress)]


def build():
    previous = quality["read"](PREVIOUS + "/completion_receipt.json")
    bank = dict(previous["inputs"])
    quality["unchanged"](bank)
    for name, expected in previous["output_identities"].items():
        path = PREVIOUS + "/" + name
        if old["identity"](path) != expected:
            raise ValueError("previous identity changed")
        bank[path] = expected
    for path in (PREVIOUS + "/completion_receipt.json", PURE, SELF, CONFIG, DOC):
        bank[path] = old["identity"](path)
    contract = quality["read"](CONFIG)
    examples, scenario_hashes = regressions(contract)
    if any(name.split(".")[0] in old["BLOCKED"] for name in sys.modules):
        raise ValueError("forbidden module loaded")
    quality["unchanged"](bank)
    result = {"schema": "exia.genetic_primus.synthetic_net_accounting_check/1", "inputs": bank,
              "direct_assertions": ASSERTIONS, "contract_sha256": pure["sha"](contract),
              "scenario_sha256": scenario_hashes, "examples": examples, "assessment": pure["assess"](contract),
              "legacy_roundtrip_budgets_not_mapped": True, "gross_label_matrix_not_requalified_as_mid_prices": True,
              "parquet_reads": 0, "scaler_fit_count": 0, "model_fit_count": 0,
              "market_financial_evaluation_count": 0, "synthetic_only": True,
              "market_research_launch_ready": False, "real_money_ready": False, "safety": contract["safety"]}
    body = api["canonical"](result) + b"\n"
    receipt = {"schema": "exia.genetic_primus.synthetic_net_accounting_receipt/1", "inputs": bank,
               "output_identities": {"result.json": {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}},
               "safety": contract["safety"]}
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
        if quality["read"](DEST + "/result.json") != result or quality["read"](DEST + "/completion_receipt.json") != receipt:
            raise ValueError("result/receipt mismatch")
        print(f"VERIFIED: assertions={ASSERTIONS}; receipt OK; parquet=0; fit=0")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("STOPPED: " + type(exc).__name__ + "; no training/network")
        raise SystemExit(1)
