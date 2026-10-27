"""Bounded, local, immutable cash-sleeve comparison of frozen ledgers."""
from __future__ import annotations
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLBACKEND"] = "Agg"
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from tools import run_genetic_primus_robust_linear_v1 as robust
from exia.genetic_primus import causal_capital_controls_v1 as kernel

c, base = robust.c, robust.base
OUT = base.BASE / "causal_capital_controls_v1_20261012"
CONTRACT = ROOT / "configs/genetic_primus_causal_capital_controls_v1.json"
CODE = (Path(__file__), CONTRACT, ROOT / "src/exia/genetic_primus/causal_capital_controls_v1.py")
SCENARIOS = ("normal", "stress")


def sealed(path):
    return base.verify_body(base.read(path))


def write(name, value):
    value = c.plain(value)
    result = value | {"sha256": c.m.identity(value)}
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return result


def source_path(period, policy, scenario):
    root = robust.OUT if policy == "H" else base.OUT
    return root / f"replay_{period}_{policy}_{scenario}.json"


def source(period, policy, scenario):
    return sealed(source_path(period, policy, scenario))


def load():
    reg = sealed(OUT / "registration.json")
    for path, expected in (reg["code_pins"] | reg["input_pins"]).items():
        if base.pin(ROOT / path) != expected:
            raise ValueError("registered input/code changed: " + path)
    return reg


def checks():
    OUT.mkdir(parents=True, exist_ok=True)
    result = kernel.direct_checks()
    write("direct_checks.json", result)
    print(json.dumps({"checks_passed": result["checks_passed"], "fits": 0, "replays": 0}))


def prepare():
    parent_reg, phase_reg, old_reg, _, _, all_banks, _ = robust.load()
    robust.verify_parents(phase_reg, old_reg)
    receipt = sealed(robust.OUT / "receipt.json")
    if receipt["registration_sha256"] != parent_reg["sha256"]:
        raise ValueError("robust parent seal mismatch")
    for name, expected in receipt["artifact_pins"].items():
        if base.pin(robust.OUT / name) != expected:
            raise ValueError("robust parent artifact changed: " + name)
    contract = base.read(CONTRACT)
    if contract["safety"] != c.SAFETY or len(contract["controls"]) != 4:
        raise ValueError("unsafe or unbounded contract")
    paths = [source_path(period, policy, scenario) for period in ("development", "report")
             for policy in ("H", "B", "EMA1") for scenario in SCENARIOS]
    paths += [robust.OUT / "registration.json", robust.OUT / "receipt.json",
              robust.phase.OUT / "registration.json", robust.phase.OUT / "receipt.json",
              base.OUT / "registration.json", base.OUT / "receipt.json"]
    reg = write("registration.json", {"schema": "causal_capital_registration/1", "contract": contract,
        "periods": parent_reg["periods"], "symbols": parent_reg["symbols"],
        "before_calibration_and_evaluation_transforms": True,
        "parent_registration_sha256": parent_reg["sha256"],
        "code_pins": {base.relative(p): base.pin(p) for p in CODE},
        "input_pins": {base.relative(p): base.pin(p) for p in paths},
        "bank_hashes": {k: c.m.array_identity(v) for k, v in all_banks.items() if k.startswith("report_")},
        "safety": c.SAFETY})
    write("objective_registry_entry.json", {"schema": "local_append_only_research_entry/1",
        "registration_sha256": reg["sha256"], "controls": contract["controls"],
        "objective": "solve two prior-period budget constraints; no return optimization",
        "adaptive_history_not_reset": True, "not_a_global_registry_completeness_claim": True,
        "new_fits": 0, "new_execution_replays": 0, "safety": c.SAFETY})
    print(json.dumps({"input_pins": len(reg["input_pins"]), "controls": 4, "frozen_parents_verified": 3}))


def calibrate():
    reg = load()
    start = reg["periods"]["report"]["start_ms"]
    reference = kernel.budget_metrics(source("development", "H", "stress"), 1)["metrics"]
    controls = {}
    for name, item in reg["contract"]["controls"].items():
        prior = source("development", item["source"], "stress")
        fitted = kernel.calibrate(prior, reference[item["metric"]], item["metric"], start)
        controls[name] = item | fitted | {
            "all_calibration_metrics": kernel.budget_metrics(prior, fitted["fraction"])["metrics"],
            "source_pin": base.pin(source_path("development", item["source"], "stress"))}
    value = write("calibration.json", {"registration_sha256": reg["sha256"], "controls": controls,
        "reference_calibration_metrics": reference, "no_report_values_used": True, "safety": c.SAFETY})
    print(json.dumps({k: {"fraction": round(v["fraction"], 8), "capped": v["capped_unreachable"]}
                      for k, v in value["controls"].items()}))


def derive():
    reg, calibration = load(), sealed(OUT / "calibration.json")
    if calibration["registration_sha256"] != reg["sha256"]:
        raise ValueError("calibration registration mismatch")
    for name, item in calibration["controls"].items():
        policy_sha = c.m.identity({"registration_sha256": reg["sha256"],
            "calibration_sha256": calibration["sha256"], "control": name, "fraction": item["fraction"]})
        for scenario in SCENARIOS:
            path = source_path("report", item["source"], scenario)
            replay = kernel.transform(sealed(path), item["fraction"], policy_sha, base.pin(path))
            replay.update(control=name, source_policy=item["source"], scenario=scenario,
                          registration_sha256=reg["sha256"], calibration_sha256=calibration["sha256"])
            write(f"ledger_report_{name}_{scenario}.json", replay)
    print(json.dumps({"ledger_transforms": 8, "fits": 0, "execution_replays": 0, "network": False}))


def audit():
    reg, calibration = load(), sealed(OUT / "calibration.json")
    count = 0
    def check(condition, label):
        nonlocal count
        if not condition:
            raise AssertionError(label)
        count += 1
    for period in ("development", "report"):
        for policy in ("H", "B", "EMA1"):
            for scenario in SCENARIOS:
                old = source(period, policy, scenario)
                identity = kernel.budget_metrics(old, 1)
                for metric in old["metrics"]:
                    check(abs(identity["metrics"][metric] - old["metrics"][metric]) < 1e-8, "full-capital identity " + metric)
                check(np.allclose(identity["daily_log_returns"], old["daily_log_returns"], atol=1e-14, rtol=0), "daily identity")
    for name, item in calibration["controls"].items():
        prior = source("development", item["source"], "stress")
        expected = kernel.calibrate(prior, calibration["reference_calibration_metrics"][item["metric"]],
                                    item["metric"], reg["periods"]["report"]["start_ms"])
        check(expected["fraction"] == item["fraction"], "prior-only calibration reproducible")
        for scenario in SCENARIOS:
            old = source("report", item["source"], scenario)
            derived = sealed(OUT / f"ledger_report_{name}_{scenario}.json")
            k, initial = item["fraction"], old["metrics"]["initial_capital"]
            reserve = (1 - k) * initial
            check(derived["safety"] == c.SAFETY, "fail-closed safety")
            check(derived["capital_fraction"] == k, "frozen fraction both costs")
            check(derived["calibration_sha256"] == calibration["sha256"], "calibration seal")
            check(len(derived["ledger"]) == len(old["ledger"]), "same event schedule")
            check(len(derived["decisions"]) == len(old["decisions"]), "same decision schedule")
            check(abs(derived["metrics"]["net_return"] - k * old["metrics"]["net_return"]) < 1e-12, "terminal capital identity")
            check(abs(derived["metrics"]["reconciliation_error"]) < 1e-7, "cash conservation")
            for before, after in zip(old["ledger"], derived["ledger"]):
                check(before["timestamp_us"] == after["timestamp_us"], "event time")
                check(before["model_sha256"] == after["model_sha256"], "model SHA retained")
                for key in ("quantity", "fee", "slippage", "cash_delta", "gross_pnl"):
                    if key in before:
                        check(abs(after[key] - k * before[key]) < 1e-10, "homogeneous cashflow " + key)
                for key in ("price", "reference_close", "side", "symbol", "kind"):
                    if key in before:
                        check(before[key] == after[key], "unchanged source " + key)
            peak = initial
            for before, after in zip(old["equity"], derived["equity"]):
                check(abs(after["nav"] - reserve - k * before["nav"]) < 1e-8, "NAV sleeve")
                check(abs(after["cash"] - reserve - k * before["cash"]) < 1e-8, "cash sleeve")
                check(after["cash"] >= reserve - 1e-7, "reserve untouched")
                peak = max(peak, after["nav"])
                check(abs(after["drawdown"] - (1 - after["nav"] / peak)) < 1e-12, "recomputed drawdown")
            for before, after in zip(old["decisions"], derived["decisions"]):
                for key in ("origin_ms", "model_sha256", "scaler_sha256", "prediction"):
                    if key in before:
                        check(before[key] == after[key], "frozen decision " + key)
                check(before["targets"] == after["source_targets_relative_to_active_sleeve"], "active target semantics")
                check(np.allclose(after["targets"], np.asarray(before["targets"]) * after["active_sleeve_share_before_decision"],
                                  rtol=0, atol=1e-14), "whole-capital new-entry targets")
    write("audit.json", {"checks_passed": count, "pure_checks_passed": sealed(OUT / "direct_checks.json")["checks_passed"],
        "fits": 0, "execution_replays": 0, "safety": c.SAFETY})
    print(json.dumps({"checks_passed": count, "new_fits": 0, "network": False}))


def compare():
    reg, calibration = load(), sealed(OUT / "calibration.json")
    _, _, _, _, _, all_banks, _ = robust.load()
    banks = {k.removeprefix("report_"): v for k, v in all_banks.items() if k.startswith("report_")}
    if {"report_" + k: c.m.array_identity(v) for k, v in banks.items()} != reg["bank_hashes"]:
        raise ValueError("bootstrap bank changed")
    result = {"registration_sha256": reg["sha256"], "calibration_sha256": calibration["sha256"],
        "scenarios": {}, "paired_orientation": "H minus control", "primary_block_days": 7,
        "family_adjusted_percentile": .0125, "multiplicity_caveat": "four contrasts only; adaptive history not corrected",
        "new_fits": 0, "execution_replays": 0, "ledger_transforms": 8, "safety": c.SAFETY}
    for scenario in SCENARIOS:
        reference = source("report", "H", scenario)
        ref_metrics = kernel.budget_metrics(reference, 1)["metrics"]
        policies = {"H": {"metrics": ref_metrics,
            "bootstrap_absolute": c.bootstrap(reference["daily_log_returns"], banks), "fraction": 1}}
        for name, item in calibration["controls"].items():
            derived = sealed(OUT / f"ledger_report_{name}_{scenario}.json")
            daily = np.asarray(derived["daily_log_returns"])
            paired = np.asarray(reference["daily_log_returns"]) - daily
            adjusted = {str(block): {role: float(np.quantile(paired[banks[f"b{block}_{role}"]].mean(axis=1) * 10000, .0125))
                         for role in ("selection", "audit")} for block in (3, 7, 14)}
            per_asset = {symbol: v["gross_pnl"] - v["fees"] - v["slippage"] + v["funding"]
                         for symbol, v in derived["per_symbol"].items()}
            nav = 10000 * np.exp(np.r_[0, np.cumsum(daily)])
            positive = sum(max(v, 0) for v in per_asset.values())
            policies[name] = {"metrics": derived["metrics"], "fraction": item["fraction"],
                "calibration_metric": item["metric"], "calibration_capped_unreachable": item["capped_unreachable"],
                "bootstrap_absolute": c.bootstrap(daily, banks),
                "H_minus_control": c.bootstrap(reference["daily_log_returns"], banks, paired=daily),
                "H_minus_control_family_adjusted_LCB": adjusted,
                "report_to_H_exposure_ratio": derived["metrics"]["mean_exposure"] / ref_metrics["mean_exposure"],
                "report_to_H_volatility_ratio": derived["metrics"]["daily_log_volatility"] / ref_metrics["daily_log_volatility"],
                "per_asset_net_USDT": per_asset,
                "largest_asset_share_of_positive_asset_net": max(per_asset.values()) / positive if positive else None,
                "largest_daily_gain_USDT": float(max(np.diff(nav))),
                "largest_daily_loss_USDT": float(min(np.diff(nav)))}
        for policy in ("B", "EMA1"):
            raw = source("report", policy, scenario)
            policies[policy + "_raw_context"] = {"metrics": kernel.budget_metrics(raw, 1)["metrics"],
                                                 "fraction": 1, "not_an_equal_risk_comparison": True}
        result["scenarios"][scenario] = {"policies": policies}
    write("comparison.json", result)
    values = result["scenarios"]["stress"]["policies"]
    print(json.dumps({k: {"net_pct": round(v["metrics"]["net_return"] * 100, 4),
                           "DD_pct": round(v["metrics"]["max_drawdown"] * 100, 4)}
                      for k, v in values.items()}))


def draw():
    load()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for scenario in SCENARIOS:
        path = OUT / f"capital_drawdown_{scenario}.png"
        if path.exists():
            raise FileExistsError(path)
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        colors = {"H": "#128045", "B_E": "#235fc0", "B_R": "#65a5df", "EMA_E": "#e4781e", "EMA_R": "#b23a32"}
        for name in colors:
            item = source("report", "H", scenario) if name == "H" else sealed(OUT / f"ledger_report_{name}_{scenario}.json")
            dates = [datetime.fromtimestamp(row["timestamp_ms"] / 1000, timezone.utc) for row in item["equity"]]
            axes[0].plot(dates, [r["nav"] for r in item["equity"]], label=name, color=colors[name], lw=1.4)
            axes[1].plot(dates, [-100 * r["drawdown"] for r in item["equity"]], color=colors[name], lw=1.2)
        axes[0].axhline(10000, color="black", ls=":", label="NoTrade")
        axes[0].set_ylabel("Total capital, USDT")
        axes[1].set_ylabel("Drawdown, %")
        axes[1].set_xlabel("UTC; 1 Aug - 30 Sep 2026, terminal close 1 Oct")
        axes[0].set_title(f"Frozen prior-period capital controls / {scenario} modeled costs")
        axes[0].legend(ncol=3)
        for axis in axes:
            axis.grid(alpha=.2)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
    print(json.dumps({"plots": 2, "new_fits": 0}))


def finalize():
    reg = load()
    parent_reg, phase_reg, old_reg, _, _, _, _ = robust.load()
    robust.verify_parents(phase_reg, old_reg)
    parent_receipt = sealed(robust.OUT / "receipt.json")
    for name, expected in parent_receipt["artifact_pins"].items():
        if base.pin(robust.OUT / name) != expected:
            raise ValueError("parent artifact changed: " + name)
    required = ["direct_checks.json", "registration.json", "objective_registry_entry.json", "calibration.json",
                "audit.json", "comparison.json", "capital_drawdown_normal.png", "capital_drawdown_stress.png", "REPORT_RU.md"]
    required += [f"ledger_report_{name}_{scenario}.json" for name in reg["contract"]["controls"] for scenario in SCENARIOS]
    for name in required:
        if not (OUT / name).is_file():
            raise ValueError("missing artifact: " + name)
    for path in OUT.glob("*.json"):
        if path.name != "receipt.json":
            sealed(path)
    receipt = write("receipt.json", {"registration_sha256": reg["sha256"],
        "parent_registration_sha256": parent_reg["sha256"], "frozen_parent_chains_verified": 3,
        "code_pins": reg["code_pins"], "artifact_pins": {name: base.pin(OUT / name) for name in required},
        "documentation_pin": base.pin(ROOT / "docs/GENETIC_PRIMUS_CAUSAL_CAPITAL_CONTROLS_V1.md"),
        "audit_checks_passed": sealed(OUT / "audit.json")["checks_passed"], "safety": c.SAFETY})
    print(json.dumps({"sealed_artifacts": len(receipt["artifact_pins"]), "parents_unchanged": True,
                      "network": False, "fits": 0, "execution_replays": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("checks", "prepare", "calibrate", "derive", "audit", "compare", "draw", "finalize"))
    arguments = parser.parse_args()
    globals()[arguments.stage]()
