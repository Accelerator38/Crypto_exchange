"""Bounded price-free clocks + local metadata preflight; never train/replay."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exia.genetic_primus import postclose_label_plan_v1 as plan

OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/postclose_label_preflight_v1_20261008"
CONFIG = "configs/genetic_primus_postclose_label_plan_v1.json"
TESTS = "tests/test_genetic_primus_postclose_label_plan_v1.py"
FUNDING_ROOT = "Retrodate/bitget_full8_funding_20260801_20260930_v1_20261007"
IOC_ROOT = "Reports/Exia/Genetic_Primus/compact_ioc_engineering_v1_20261008"
SOURCES = (CONFIG, TESTS, "src/exia/genetic_primus/postclose_label_plan_v1.py",
           "tools/check_genetic_primus_postclose_label_plan_v1.py", "docs/GENETIC_PRIMUS_POSTCLOSE_LABEL_PLAN_V1.md",
           "configs/genetic_primus_compact_ioc_v1.json", "src/exia/genetic_primus/unified_measurement_v1.py",
           "src/exia/genetic_primus/compact_forecast_data_v1.py", "src/exia/genetic_primus/causal_market_label_plan_v1.py",
           FUNDING_ROOT + "/completion_manifest.json", IOC_ROOT + "/receipt.json", IOC_ROOT + "/artifact_hashes.json")
DOWNLOADS = (
    "bitget-futures_book_ticker_2025-01-01_BTCUSDT.csv.gz",
    "bitget-futures_incremental_book_L2_2024-12-01_BTCUSDT.csv.gz",
    "bitget-futures_trades_2024-12-01_BTCUSDT.csv.gz",
    "bitget-futures_derivative_ticker_2024-12-01_BTCUSDT.csv.gz",
)


def file_sha(path):
    if path.stat().st_size > 2_000_000:
        raise ValueError("metadata/code input bound exceeds 2 MB")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    file_sha(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, body):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(body, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def preserve_prior_evidence():
    receipt = read_json(ROOT / IOC_ROOT / "receipt.json")
    manifest = read_json(ROOT / IOC_ROOT / "artifact_hashes.json")
    for name, digest in receipt["source_hashes"].items():
        if file_sha(ROOT / name) != digest:
            raise ValueError("prior IOC source changed")
    for name, digest in manifest["files"].items():
        if file_sha(ROOT / IOC_ROOT / name) != digest:
            raise ValueError("prior IOC evidence changed")
    return {"source_hashes_verified": len(receipt["source_hashes"]), "artifact_hashes_verified": len(manifest["files"])}


def main():
    if OUTPUT.exists():
        raise FileExistsError("immutable preflight root already exists")
    source_hashes = {name: file_sha(ROOT / name) for name in SOURCES}
    c = read_json(ROOT / CONFIG)
    target_sha = plan.validate_contract(c)
    if source_hashes["configs/genetic_primus_compact_ioc_v1.json"] != c["engine_contract_file_sha256"]:
        raise ValueError("engineering execution contract hash changed")
    engine = read_json(ROOT / "configs/genetic_primus_compact_ioc_v1.json")
    for key in ("signal_compute_latency_ms", "order_latency_ms", "reference_wait_ms", "order_ttl_ms", "max_quote_age_ms"):
        if engine[key] != c[key]:
            raise ValueError("label-plan and IOC clock disagree")
    prior_integrity = preserve_prior_evidence()
    spec = importlib.util.spec_from_file_location("postclose_explicit_checks", ROOT / TESTS)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log, verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(checks.PostclosePlanTests))
    if not tests.wasSuccessful() or tests.testsRun != 21:
        print(log.getvalue().encode("utf-8")[:920].decode("utf-8", "ignore"))
        raise ValueError("named price-free checks failed")
    synthetic_pair = checks.pair(c)
    origin = c["validation_start_ms"] + plan.HORIZON_MS + 9 * plan.BAR_MS
    ctx = plan.training_context(origin, c)
    maturity = {"at_cutoff": plan.dependency_maturity(synthetic_pair, [ctx["label_cutoff_ms"]], origin, c),
                "late_cost_source": plan.dependency_maturity(synthetic_pair, [ctx["label_cutoff_ms"] + 1], origin, c)}
    bundle_path = (ROOT / c["expected_execution_bundle_manifest"]).resolve()
    bundle_path.relative_to(ROOT.resolve())  # Workspace-only registered metadata target.
    bundle = read_json(bundle_path) if bundle_path.is_file() else None
    declaration_issues = plan.bundle_declaration_issues(bundle, c)
    if bundle is not None:
        source_hashes[c["expected_execution_bundle_manifest"]] = file_sha(bundle_path)
    funding = read_json(ROOT / FUNDING_ROOT / "completion_manifest.json")
    audits = funding["symbol_audits"]
    rates = {symbol: {"requested_rows": audits[symbol]["requested_rows"],
                       "left_guard_ms": audits[symbol]["left_guard_ms"], "right_guard_ms": audits[symbol]["right_guard_ms"],
                       "boundary_bracketed": audits[symbol]["boundary_bracketed"]} for symbol in c["symbols"]}
    inventory = []
    for filename in DOWNLOADS:
        path = Path("C:/Users/anton/Downloads") / filename
        inventory.append({"path": str(path), "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else None,
                          "evidence_level": "filename_and_stat_only_not_contents_or_coverage",
                          "full8_aug_sep_2026_qualification": False})
    blockers = declaration_issues + [
        "component_content_hashes_and_independent_quality_receipts_not_verified",
        "historical_L2_sequence_gap_reconstruction_adapter_not_qualified",
        "postclose_training_labels_and_causal_training_scope_not_registered",
        "historical_settlement_mark_and_cost_availability_not_qualified",
        "terminal_settlement_inclusion_not_verified",
        "latency_and_liquidity_parameters_not_empirically_calibrated",
    ]
    preflight = {"schema": "postclose_input_preflight/1", "target_clock_sha256": target_sha,
        "registered_manifest": str(bundle_path), "registered_manifest_present": bundle is not None,
        "declaration_issues": declaration_issues, "declaration_structurally_complete": not declaration_issues,
        "required_validation_start_ms": c["validation_start_ms"],
        "required_terminal_tail_end_ms_exclusive": c["validation_end_ms_exclusive"] + 7100,
        "required_components": list(plan.COMPONENTS), "symbols": c["symbols"],
        "known_funding_rate_archive": {"manifest_sha256": source_hashes[FUNDING_ROOT + "/completion_manifest.json"],
            "symbols": rates, "requested_rate_rows": sum(row["requested_rows"] for row in rates.values()),
            "availability_and_observed_marks_qualified": False,
            "terminal_rate_may_exist_in_old_raw_not_checked_here": True},
        "known_download_inventory": inventory, "inventory_exhaustive": False,
        "financial_replay_ready": False, "training_ready": False, "production_qualified": False,
        "blockers": blockers, "safety": plan.SAFETY}
    template = {"schema": "postclose_execution_bundle_declaration/1", "target_clock_sha256": target_sha,
                "venue": "bitget", "product": "USDT-FUTURES", "declaration_is_not_qualification": True,
                "components": {kind: [{"symbol": symbol, "coverage_start_ms": None, "coverage_end_ms_exclusive": None,
                    "evidence_kind": None, "quality_status": "missing", "source_manifest_sha256": None,
                    "quality_receipt_sha256": None} for symbol in c["symbols"]] for kind in plan.COMPONENTS}}
    if source_hashes != {name: file_sha(ROOT / name) for name in source_hashes}:
        raise ValueError("sources changed during preflight")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(OUTPUT / "synthetic_clock_pair.json", synthetic_pair)
    write_json(OUTPUT / "synthetic_maturity_boundaries.json", maturity)
    write_json(OUTPUT / "input_preflight.json", preflight)
    write_json(OUTPUT / "bundle_template_not_qualified.json", template)
    with (OUTPUT / "checks.txt").open("x", encoding="utf-8") as stream:
        stream.write(log.getvalue())
    report = f"""# Post-close clocks и preflight — 8 октября 2026

Новый контракт времени реализован и проверен: **21/21** price-free unittest.
Deployment/retrain каждые 3 дня; выход — следующий scheduled origin,
а не entry fill + 3 дня. Entry после готовности закрытого бара и compute;
reference → latency → отдельный свежий post-arrival snapshot.
Неполные clocks не дают нулевой label. Training cutoff = origin − 8 баров:
один бар до training asof и ещё purge 7 баров после созревания исхода.
Max availability включает late funding/cost/mark dependencies.

Синтетический пример: signal +100 ms, reference +200 ms, arrival +450 ms,
endpoint +500 ms; exit на close +3 дня с той же политикой clocks.
Источник ровно на cutoff проходит time gate; на cutoff +1 ms — не проходит.
В обоих случаях training_ready=false: clocks не доказывают реальный fill.

## Данные и реальный блокер

Registered full8 execution bundle present: **{bundle is not None}**.
Финансовый replay ready: **False**. Требуются verified L2/depth/sequence,
marks, funding, historical rules/fees и closed-feature prefix для full8.
Есть локальный funding manifest на **{preflight['known_funding_rate_archive']['requested_rate_rows']}**
rates за запрошенный период, но rates без исторических availability/observed
marks не закрывают execution qualification. Terminal funding на 1 октября
не следует терять: он может присутствовать в raw, здесь это не проверялось.

Четыре ранее указанных Download-файла найдены по stat; их имена относятся
к BTC в 2024/2025. Содержимое gzip не читалось, coverage не доказана;
это не substitute для full8 за август–сентябрь 2026. Inventory ограничен
известными источниками: отсутствие registered bundle не доказывает отсутствие
всех подходящих файлов на диске. Полный verification L2/marks ещё не выполнен.

Replay data window: 1 августа — 1 октября exclusive UTC, плюс engineering
tail 7.1 s для terminal close. Минимум 49 баров для six features не заменяет
full causal EMA prefix. Для training нужны отдельные зрелые execution labels
по зарегистрированному историческому scope, не ретроспективная замена SHA
старых next-4h-open labels. Latency/participation пока не откалиброваны.

## Следующая работа

1. Подготовить bounded offline parser/reconstruction smoke на имеющихся
   BTC L2/trades/derivative файлах 2024 года; это только проверка формата.
2. Зарегистрировать реальные источники full8 для нужного validation и
   training scope. Шаблон bundle сохранён, все evidence поля незаполнены.
3. Проверить coverage, gaps, sequence и source availability; затем построить
   execution/cost labels, новые hashes и causal scalers. Только после этого
   новый financial replay с непрерывным ledger и equity curve.

Старые прибыльности, objective, validation history и model rankings не
изменены. Новый period не называется prospective. Предыдущие IOC evidence:
{prior_integrity['source_hashes_verified']} source hashes и {prior_integrity['artifact_hashes_verified']}
artifact hashes сверены без изменений. New fits/market replays/network
requests = 0. Все девять authority flags false, runtime не открывался.
"""
    with (OUTPUT / "REPORT_RU.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report)
    receipt = {"schema": "postclose_label_preflight_receipt/1", "source_hashes": source_hashes,
        "target_clock_sha256": target_sha, "unit_checks_passed": tests.testsRun,
        "synthetic_clock_pair_sha256": synthetic_pair["sha256"], "preflight_sha256": plan.identity(preflight),
        "prior_IOC_integrity": prior_integrity, "new_fits": 0, "market_replays": 0, "network_requests": 0,
        "economic_labels_constructed": 0, "no_large_tapes_read": True, "financial_replay_ready": False,
        "training_ready": False, "safety": plan.SAFETY}
    receipt["sha256"] = plan.identity(receipt)
    write_json(OUTPUT / "receipt.json", receipt)
    artifacts = {path.name: file_sha(path) for path in sorted(OUTPUT.iterdir())}
    write_json(OUTPUT / "artifact_hashes.json", {"files": artifacts, "sha256": plan.identity(artifacts)})
    print(json.dumps({"checks_passed": tests.testsRun, "funding_rate_rows_in_manifest": preflight["known_funding_rate_archive"]["requested_rate_rows"],
                      "bundle_present": bundle is not None, "financial_replay_ready": False, "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
