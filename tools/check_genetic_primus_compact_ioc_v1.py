"""One bounded synthetic IOC engineering check; no pytest or market replay."""
from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exia.genetic_primus.compact_ioc_execution_v1 import (
    CompactIOC, binding_from_compact_origin, identity,
)
from exia.genetic_primus.unified_measurement_v1 import SAFETY

OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/compact_ioc_engineering_v1_20261008"
ORIGIN = "Reports/Exia/Genetic_Primus/compact_forecast_v1_20261007_r2/origin_00.json"
SOURCES = (
    "configs/genetic_primus_compact_ioc_v1.json",
    "src/exia/genetic_primus/compact_ioc_execution_v1.py",
    "src/exia/genetic_primus/execution_replay_v1.py",
    "src/exia/genetic_primus/release_contract_v1.py",
    "src/exia/genetic_primus/unified_measurement_v1.py",
    "src/exia/genetic_primus/compact_net_replay_v1.py",
    "tests/test_genetic_primus_compact_ioc_v1.py",
    "tools/check_genetic_primus_compact_ioc_v1.py",
    "docs/GENETIC_PRIMUS_COMPACT_IOC_V1.md",
    ORIGIN,
)


def file_sha(path):
    if path.stat().st_size > 2_000_000:
        raise ValueError("bounded engineering input exceeds 2 MB")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, body):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(body, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_checks():
    spec = importlib.util.spec_from_file_location("compact_ioc_explicit_checks", ROOT / SOURCES[6])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_bank(checks):
    args = checks.inputs()
    shared = [checks.mark(), checks.intent(args, "A", quantity=".05"),
              checks.intent(args, "B", quantity=".05"), checks.book(), checks.book(500, sid="B2", qty=".1")]
    reservation = [checks.mark(), checks.intent(args, quantity="2"), checks.book(),
                   checks.book(500, sid="B2", qty="100")]
    reservation += [checks.intent(args, name, 600, quantity="2") for name in ("P1", "P2", "P3")]
    reservation += [checks.book(700, sid="B3"), checks.funding(800, rate=".9"), checks.funding(900, rate=".9")]
    return {
        "NoTrade": [checks.mark(), {"kind": "clock", "ts": checks.BASE + 1300}],
        "flat_price_round_trip": checks.round_trip_events(args),
        "partial_ioc": [checks.mark(), checks.intent(args), checks.book(), checks.book(500, sid="B2", qty=".05")],
        "shared_liquidity": shared,
        "price_protection": [checks.mark(), checks.intent(args), checks.book(),
                             checks.book(500, sid="B2", bid="100.9", ask="101")],
        "exclusive_expiry": [checks.mark(), checks.intent(args), checks.book(), checks.book(2200, sid="B2")],
        "below_minimum_reject": [checks.mark(), checks.intent(args, quantity=".0019"), checks.book()],
        "funding_reservation_fault_stress": reservation,
    }


def figure(results, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    for name, label, style in (("NoTrade", "NoTrade", "--"),
                               ("flat_price_round_trip", "Круговая сделка: spread + fees + funding", "-")):
        points = results[name]["equity"]
        times = [0] + [(p["ts"] - base) / 1000 for p in points]
        values = [1000.0] + [float(p["equity_quote"]) for p in points]
        axes[0].plot(times, values, style, label=label, marker=".")
    round_trip = results["flat_price_round_trip"]
    axes[1].plot([(p["ts"] - base) / 1000 for p in round_trip["equity"]],
                 [float(p["cash_quote"]) for p in round_trip["equity"]], marker=".", color="tab:orange")
    axes[0].set(title="СИНТЕТИЧЕСКАЯ инженерная проверка — не результат стратегии", ylabel="NAV, USDT")
    axes[0].legend(fontsize=9)
    axes[1].set(xlabel="Время событий от условного close, секунды", ylabel="Свободный cash, USDT")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.savefig(OUTPUT / "synthetic_equity.png", dpi=140)
    fig.savefig(OUTPUT / "synthetic_equity.svg")
    plt.close(fig)


def main():
    if OUTPUT.exists():
        raise FileExistsError("evidence root exists; never overwrite engineering evidence")
    source_hashes = {path: file_sha(ROOT / path) for path in SOURCES}
    implementation_sha = identity(source_hashes)
    checks = load_checks()
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log, verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(checks.CompactIOCTests))
    if not tests.wasSuccessful() or tests.testsRun != 37:
        print(log.getvalue().encode("utf-8")[:920].decode("utf-8", "ignore"))
        raise RuntimeError("explicit bounded engineering regressions failed")
    bank = fixture_bank(checks)
    results, inputs = {}, {}
    for name, events in bank.items():
        args = checks.inputs()
        args["implementation_sha256"] = implementation_sha
        args["data_sha256"] = identity({"events": events, "rules": args["rules"],
                                       "fees": args["fee_schedule"], "bindings": args["bindings"]})
        results[name] = checks.run(events, args).result()
        inputs[name] = {"arguments": args, "events": events}
    flat = results["flat_price_round_trip"]
    with localcontext() as context:
        context.prec = 50
        expected = Decimal("1000") - Decimal(flat["fees_quote"]) - Decimal(flat["signed_funding_quote"]) - Decimal(".201") * Decimal(".2")
        if Decimal(flat["equity_quote"]) != expected or Decimal(flat["positions"]["X"]) != 0:
            raise ValueError("flat-market cash/spread/fee/funding reconciliation failed")
    args, events = inputs["flat_price_round_trip"]["arguments"], bank["flat_price_round_trip"]
    prefix = checks.run(events[:4], args)
    checkpoint = prefix.checkpoint()
    resumed = CompactIOC.restore(checkpoint, **args)
    for event in events[4:]:
        resumed.ingest(event)
    if resumed.result() != flat:
        raise ValueError("deterministic resume differs from uninterrupted execution")
    origin = json.loads((ROOT / ORIGIN).read_text(encoding="utf-8"))
    bridge = binding_from_compact_origin(origin, "linear6_uniform", signal_available_ms=origin["origin_ms"] + 100)
    if bridge["new_clock_labels_qualified"] is not False:
        raise ValueError("old labels must remain unqualified for post-close clock")
    if source_hashes != {path: file_sha(ROOT / path) for path in SOURCES}:
        raise ValueError("source changed during verification")
    # Exclusive root creation only after all pure checks pass. No old file is modified.
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(OUTPUT / "fixture_bank.json", inputs)
    write_json(OUTPUT / "synthetic_results.json", results)
    write_json(OUTPUT / "checkpoint.json", checkpoint)
    write_json(OUTPUT / "resumed_result.json", resumed.result())
    write_json(OUTPUT / "frozen_origin_provenance_only.json", bridge)
    figure(results, checks.BASE)
    with (OUTPUT / "checks.txt").open("x", encoding="utf-8") as stream:
        stream.write(log.getvalue())
    receipt = {
        "schema": "compact_ioc_engineering_receipt/1", "evidence_kind": "synthetic_only",
        "unit_checks_passed": tests.testsRun, "fixture_count": len(bank), "new_model_fits": 0,
        "market_strategy_replays": 0, "network_requests": 0,
        "checkpoint_resume_exact_match": True, "flat_market_reconciliation_exact": True,
        "source_hashes": source_hashes, "implementation_sha256": implementation_sha,
        "resample_bank": None, "folds": None, "reason": "no training or financial inference in this engineering check",
        "fixture_bank_sha256": identity(inputs), "result_sha256": {k: v["sha256"] for k, v in results.items()},
        "frozen_origin_bridge_sha256": bridge["sha256"], "old_clock_labels_qualified": False,
        "actual_fill_verified": False, "financial_strategy_qualified": False, "safety": SAFETY,
    }
    receipt["sha256"] = identity(receipt)
    write_json(OUTPUT / "receipt.json", receipt)
    artifacts = {path.name: file_sha(path) for path in sorted(OUTPUT.iterdir()) if path.is_file()}
    write_json(OUTPUT / "artifact_hashes.json", {"schema": "artifact_hashes/1", "files": artifacts,
                                               "source_binding_sha256": implementation_sha, "sha256": identity(artifacts)})
    print(json.dumps({"checks_passed": tests.testsRun, "synthetic_fixtures": len(bank),
                      "flat_round_trip_equity": flat["equity_quote"], "resume_exact": True,
                      "market_replays": 0, "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
