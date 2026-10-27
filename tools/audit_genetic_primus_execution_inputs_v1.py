"""Bounded local input/provenance audit, never a financial replay or downloader."""
import hashlib
import json
import os
import re
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exia.genetic_primus.market_fragment_v1 import number, object_sha
from exia.genetic_primus.market_timebase_fragment_v1 import SAFETY, seal, verify
from exia.genetic_primus.postclose_label_plan_v1 import COMPONENTS, bundle_declaration_issues, validate_contract

OUT = ROOT / "Reports/Exia/Genetic_Primus/execution_input_audit_v1_20261008"
PARENT = "configs/genetic_primus_postclose_label_plan_v1.json"
FUND = "Retrodate/bitget_full8_funding_20260801_20260930_v1_20261007"
PRICE = "Reports/Exia/Genetic_Primus/price_execution_audit_v1_20261008"
PREVIOUS = "Reports/Exia/Genetic_Primus/postclose_clock_us_smoke_v1_20261008"
OHLCV = "Retrodate/simple_research_reset_v1/bitget_full8_1m.manifest.json"
SOURCES = ("tools/audit_genetic_primus_execution_inputs_v1.py",
           "docs/GENETIC_PRIMUS_EXECUTION_INPUT_AUDIT_V1.md", PARENT,
           "src/exia/genetic_primus/market_fragment_v1.py",
           "src/exia/genetic_primus/market_timebase_fragment_v1.py",
           "src/exia/genetic_primus/postclose_label_plan_v1.py",
           "src/exia/genetic_primus/unified_measurement_v1.py")
DOWNLOADS = (
    "bitget-futures_incremental_book_L2_2024-12-01_BTCUSDT.csv.gz",
    "bitget-futures_trades_2024-12-01_BTCUSDT.csv.gz",
    "bitget-futures_derivative_ticker_2024-12-01_BTCUSDT.csv.gz",
    "bitget-futures_book_ticker_2025-01-01_BTCUSDT.csv.gz",
)


def read_small(path, cap=2_097_152):
    # Only selected source/receipt files. No Parquet, archives, secrets or runtime.
    if path.suffix.lower() not in (".py", ".json", ".md", ".txt", ".png", ".svg"):
        raise ValueError("unsupported audit content type")
    if re.search(r"(^|[_.-])(key|keys|secret|secrets|credentials)([_.-]|$)", path.name.lower()):
        raise ValueError("sensitive file not in scope")
    with path.open("rb") as source:
        data = source.read(cap + 1)
    if len(data) > cap:
        raise ValueError("small-file byte bound exceeded: " + path.name)
    return data


def file_sha(path):
    return hashlib.sha256(read_small(path)).hexdigest()


def read_json(path, cap=524288):
    return json.loads(read_small(path, cap).decode("utf-8"))


def relative(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("source path outside workspace")
    return path.relative_to(ROOT).as_posix()


def pin(name, pins, expected=None):
    path = (ROOT / name).resolve()
    name = relative(path)
    data = read_small(path)
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None and (digest != expected["sha256"] or len(data) != expected["bytes"]):
        raise ValueError("frozen evidence pin mismatch: " + name)
    if name in pins and pins[name] != digest:
        raise ValueError("source changed during audit: " + name)
    pins[name] = digest
    return data


def utc(ms):
    return (datetime.fromtimestamp(ms // 1000, timezone.utc)
            + timedelta(milliseconds=ms % 1000)).isoformat()


def reported_row(row):
    """Parse a reported rate, NOT an observed settlement or cashflow."""
    if not isinstance(row, dict) or set(row) != {"symbol", "fundingRate", "fundingTime"}:
        raise ValueError("unexpected funding row")
    if not isinstance(row["symbol"], str) or not re.fullmatch(r"[A-Z]{2,12}USDT", row["symbol"]):
        raise ValueError("invalid symbol")
    value = row["fundingTime"]
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{13}", value):
        raise ValueError("funding timestamp must preserve millisecond units")
    number(row["fundingRate"], positive=False)
    return {"symbol": row["symbol"], "reported_funding_time_ms": int(value),
            "rate_fraction": row["fundingRate"]}


def terminal_reports(pages, target_ms, symbols):
    if (not isinstance(pages, list) or not 1 <= len(pages) <= 24
            or type(target_ms) is not int or not 10**12 <= target_ms < 10**13
            or not 1 <= len(symbols) <= 8 or len(set(symbols)) != len(symbols)):
        raise ValueError("invalid bounded report request")
    bank = {symbol: [] for symbol in symbols}
    parsed = 0
    for page in pages:
        body = page["body"]
        rows = body.get("data")
        if body.get("code") != "00000" or not isinstance(rows, list) or not 1 <= len(rows) <= 100:
            raise ValueError("invalid/big raw response")
        request_time = body.get("requestTime")
        if type(request_time) is not int or not 10**12 <= request_time < 10**13:
            raise ValueError("invalid reported response time")
        for ordinal, row in enumerate(rows):
            value = reported_row(row)
            parsed += 1
            if value["symbol"] not in bank:
                raise ValueError("unexpected symbol in raw funding response")
            if value["reported_funding_time_ms"] == target_ms:
                bank[value["symbol"]].append({**value, "source_path": page["path"],
                                             "source_file_sha256": page["sha256"], "row_ordinal": ordinal,
                                             "source_row_sha256": object_sha(row),
                                             "response_reported_request_ms": request_time})
    result = []
    for symbol, reports in bank.items():
        rates = sorted(set(report["rate_fraction"] for report in reports))
        result.append({"symbol": symbol, "reported_funding_time_ms": target_ms,
                       "status": "UNIQUE_REPORTED_RATE" if len(rates) == 1 else "MISSING" if not rates else "CONFLICTING_REPORTS",
                       "rate_fraction": rates[0] if len(rates) == 1 else None,
                       "reported_rate_variants": rates, "sources": reports,
                       "strategy_available_us": None, "settlement_mark": None,
                       "is_funding_settlement_verified": False, "consumable_as_funding_cashflow": False})
    return seal({"schema": "terminal_funding_report_candidates/1", "raw_rows_inspected": parsed,
                 "target_ms": target_ms, "target_utc": utc(target_ms), "records": result,
                 "financial_replay_ready": False, "safety": SAFETY})


def pure_checks():
    row = {"symbol": "BTCUSDT", "fundingRate": "-0.0001", "fundingTime": "1790812800000"}
    page = {"path": "synthetic.json", "sha256": "a" * 64,
            "body": {"code": "00000", "requestTime": 1791396397550, "data": [row]}}
    checks = []
    def check(condition, name):
        if not condition:
            raise ValueError("pure check failed: " + name)
        checks.append(name)
    check(reported_row(row)["rate_fraction"] == "-0.0001", "signed_rate_preserved")
    before = deepcopy(page)
    result = terminal_reports([page], 1790812800000, ["BTCUSDT", "ETHUSDT"])
    check(page == before, "raw_input_immutable")
    check(result["records"][1]["status"] == "MISSING", "missing_asset_explicit")
    check(result["records"][0]["strategy_available_us"] is None, "request_time_not_availability")
    check(result["records"][0]["consumable_as_funding_cashflow"] is False, "no_cashflow_authority")
    duplicate = deepcopy(page)
    duplicate["body"]["data"].append(deepcopy(row))
    record = terminal_reports([duplicate], 1790812800000, ["BTCUSDT"])["records"][0]
    check(len(record["sources"]) == 2 and record["status"] == "UNIQUE_REPORTED_RATE", "duplicates_preserve_provenance")
    duplicate["body"]["data"][1]["fundingRate"] = "0.0002"
    record = terminal_reports([duplicate], 1790812800000, ["BTCUSDT"])["records"][0]
    check(record["rate_fraction"] is None and record["status"] == "CONFLICTING_REPORTS", "conflict_never_first_wins")
    for field, value, name in (("fundingTime", "1790812800", "seconds_not_milliseconds"),
                               ("fundingRate", "NaN", "nonfinite_rejected")):
        try:
            reported_row(row | {field: value})
        except ValueError:
            checks.append(name)
        else:
            raise ValueError("missing rejection: " + name)
    check(all(flag is False for flag in result["safety"].values()), "all_authority_false")
    return checks


def inventory(root):
    """Bounded metadata-only filename snapshot. NEVER asserts content coverage."""
    root = root.resolve()
    stack, entries, directories, candidates, skipped = [root], 0, 0, [], 0
    stop = None
    while stack and stop is None:
        folder = stack.pop()
        directories += 1
        if directories > 512:
            stop = "directory_budget"
            break
        with os.scandir(folder) as iterator:
            for entry in iterator:
                entries += 1
                if entries > 8192:
                    stop = "entry_budget"
                    break
                if entry.is_symlink():
                    skipped += 1
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stat = entry.stat(follow_symlinks=False)
                    if getattr(stat, "st_file_attributes", 0) & 1024:  # Windows reparse point.
                        skipped += 1
                        continue
                    if len(stack) >= 512:
                        stop = "pending_directory_budget"
                        break
                    stack.append(Path(entry.path))
                    continue
                name = entry.name.lower()
                if any(word in name for word in ("secret", "credential", "api_key", "orders", "runtime")):
                    continue
                relevant = name.endswith((".parquet", ".csv", ".csv.gz")) or (
                    name.endswith(".json") and any(word in str(Path(entry.path).relative_to(root)).lower()
                                                  for word in ("funding", "book", "depth", "mark", "manifest")))
                if not relevant:
                    continue
                path = Path(entry.path).resolve()
                if not path.is_relative_to(root):
                    skipped += 1
                    continue
                stat = entry.stat(follow_symlinks=False)
                candidates.append({"path": relative(path), "bytes": stat.st_size,
                                   "content_read": False, "content_coverage_verified": False})
                if len(candidates) >= 2048:
                    stop = "candidate_budget"
                    break
    candidates.sort(key=lambda item: item["path"])
    return {"root": relative(root), "entries_seen": min(entries, 8192),
            "directories_seen": min(directories, 512), "stop_reason": stop,
            "filename_scan_complete_within_scope": stop is None,
            "symlinks_or_reparse_points_skipped": skipped, "candidates": candidates,
            "data_coverage_is_not_inferred_from_filenames": True}


def verify_previous(pins):
    receipt = json.loads(pin(PREVIOUS + "/receipt.json", pins))
    verify(receipt)
    manifest = json.loads(pin(PREVIOUS + "/artifact_hashes.json", pins))
    if object_sha(manifest["files"]) != manifest["sha256"]:
        raise ValueError("previous artifact seal mismatch")
    for name, expected in receipt["source_hashes"].items():
        if hashlib.sha256(pin(name, pins)).hexdigest() != expected:
            raise ValueError("previous source changed")
    for name, expected in manifest["files"].items():
        if hashlib.sha256(pin(PREVIOUS + "/" + name, pins)).hexdigest() != expected:
            raise ValueError("previous artifact changed")
    return {"sources_verified": len(receipt["source_hashes"]), "artifacts_verified": len(manifest["files"])}


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        output.write("\n")


def main():
    if OUT.exists():
        raise ValueError("immutable output root already exists")
    pins = {}
    for name in SOURCES:
        pin(name, pins)
    contract = json.loads(pin(PARENT, pins))
    validate_contract(contract)
    prior_integrity = verify_previous(pins)
    checks = pure_checks()
    bundle_path = (ROOT / contract["expected_execution_bundle_manifest"]).resolve()
    relative(bundle_path)
    bundle = json.loads(pin(relative(bundle_path), pins)) if bundle_path.exists() else None
    issues = bundle_declaration_issues(bundle, contract)
    funding_receipt = json.loads(pin(FUND + "/evidence_receipt.json", pins))
    fund_pin_count = 0
    for section in ("inputs", "outputs"):
        if len(funding_receipt[section]) > 128:
            raise ValueError("funding receipt pin budget")
        for name, expected in funding_receipt[section].items():
            pin(name, pins, expected)
            fund_pin_count += 1
    completion = json.loads(pin(FUND + "/completion_manifest.json", pins))
    verify(completion)
    normalized = json.loads(pin(FUND + "/funding_events.json", pins, completion["events_output_pin"]))
    if not isinstance(normalized["events"], list) or len(normalized["events"]) > 1600:
        raise ValueError("normalized funding row bound")
    pages = []
    if len(completion["raw_inputs"]) > 48:
        raise ValueError("raw funding input budget")
    for name, expected in sorted(completion["raw_inputs"].items()):
        path = (ROOT / name).resolve()
        if not path.is_relative_to((ROOT / FUND / "raw").resolve()):
            raise ValueError("raw funding path outside registered root")
        data = pin(name, pins, expected)
        if not name.endswith(".receipt.json"):
            pages.append({"path": name, "sha256": expected["sha256"], "body": json.loads(data)})
    tail = terminal_reports(pages, contract["validation_end_ms_exclusive"], contract["symbols"])
    normalized_counts = dict(Counter(event["symbol"] for event in normalized["events"]))
    terminal_old = sum(event["settlement_ms"] == contract["validation_end_ms_exclusive"] for event in normalized["events"])
    price_receipt = json.loads(pin(PRICE + "/event_analysis_receipt.json", pins))
    event_pin = price_receipt["files"]["event_audit_results.json"]
    price = json.loads(pin(PRICE + "/event_audit_results.json", pins, event_pin))
    ohlcv = json.loads(pin(OHLCV, pins))
    snapshots = [inventory(ROOT / "Retrodate"), inventory(ROOT / PRICE)]
    downloads = []
    for name in DOWNLOADS:
        path = Path("C:/Users/anton/Downloads") / name
        downloads.append({"path": str(path), "exists": path.exists(),
                          "bytes": path.stat().st_size if path.exists() else None,
                          "content_read": False, "target_period_matches": False})
    expected_end = contract["validation_end_ms_exclusive"] + sum(contract[key] for key in
                      ("signal_compute_latency_ms", "reference_wait_ms", "order_ttl_ms"))
    gaps = []
    for component in COMPONENTS:
        for symbol in contract["symbols"]:
            candidate = []
            if component == "funding":
                status = "RATES_PRESENT_BUT_SETTLEMENT_MARK_AVAILABILITY_UNQUALIFIED"
                candidate = [FUND + "/funding_events.json", "terminal_funding_reports.json"]
            elif component == "marks" and symbol[:-4] in price["marks"]:
                status = "PARTIAL_4H_MARK_SAMPLE_NOT_FULL_PERIOD"
                candidate = [PRICE + "/event_audit_results.json"]
            elif component == "fees":
                status = "CURRENT_DEFAULT_FEE_PROXY_NOT_HISTORICAL_ACCOUNT_PROOF"
                candidate = [PRICE + "/event_audit_results.json"]
            elif component == "closed_features":
                status = "CAUSAL_FEATURE_HISTORY_AND_STRATEGY_AVAILABILITY_UNREGISTERED"
            else:
                status = "NO_REGISTERED_QUALIFIED_COMPONENT"
            gaps.append({"component": component, "symbol": symbol,
                         "qualified_for_postclose_replay": False, "status": status,
                         "candidate_sources_not_qualified": candidate,
                         "minimum_declared_start_ms": contract["validation_start_ms"] - (
                             contract["feature_min_history_bars"] * contract["bar_ms"]
                             if component == "closed_features" else contract["max_quote_age_ms"]),
                         "minimum_declared_end_ms_exclusive": expected_end})
    unique_tail = sum(row["status"] == "UNIQUE_REPORTED_RATE" for row in tail["records"])
    summary = {"direct_pure_checks_passed": len(checks), "execution_bundle_exists": bundle is not None,
               "bundle_structural_issues": issues, "components_reviewed": len(gaps),
               "qualified_components": 0, "normalized_reported_rate_rows": len(normalized["events"]),
               "normalized_rate_rows_per_symbol": normalized_counts, "old_normalized_terminal_rows": terminal_old,
               "raw_terminal_assets_with_unique_report": unique_tail, "raw_funding_pages_read": len(pages),
               "raw_funding_rows_inspected": tail["raw_rows_inspected"], "funding_receipt_pins_verified": fund_pin_count,
               "funding_settlement_marks_present": completion["settlement_marks_present"],
               "funding_schedule_independently_verified": completion["settlement_schedule_independently_verified"],
               "sampled_mark_assets": sorted(price["marks"]),
               "sampled_mark_candles": {symbol: record["common_4h_mark_candles"] for symbol, record in price["marks"].items()},
               "current_default_taker_fee_bps_not_account_proof": price["current_taker_fee_bps"],
               "minimum_terminal_tail_end_utc_exclusive": utc(expected_end),
               "training_history_status": contract["training_label_history_status"],
               "filename_inventory_complete_by_root": {snap["root"]: snap["filename_scan_complete_within_scope"] for snap in snapshots},
               "financial_replay_ready": False, "safety": SAFETY}
    for name, expected in pins.items():
        if file_sha(ROOT / name) != expected:
            raise ValueError("input changed during audit: " + name)
    OUT.mkdir(parents=True, exist_ok=False)
    write_json(OUT / "terminal_funding_reports.json", tail)
    write_json(OUT / "execution_input_gaps.json", seal({"schema": "postclose_execution_input_gap/1",
                                                      "components": gaps, "structural_issues": issues,
                                                      "financial_replay_ready": False, "safety": SAFETY}))
    write_json(OUT / "filename_inventory.json", seal({"schema": "bounded_filename_inventory/1", "roots": snapshots,
                                                    "provided_archives_stat_only": downloads,
                                                    "scope_is_not_entire_filesystem": True,
                                                    "contents_or_market_coverage_verified": False}))
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "direct_checks.json", {"checks_passed": checks, "pytest_or_collector_started": False})
    report = f"""# Execution input audit: full8 августа–сентября 2026

Дата: 2026-10-08. Локальная проверка входов; не обучение и не финансовый replay.

## Результат

Зарегистрированный bundle: `{contract['expected_execution_bundle_manifest']}`.
Наличие: {bundle is not None}. Structural issues: `{issues}`.
48 component/symbol требований рассмотрено; **0 квалифицированных для нового post-close replay**.
Это статус registered input gate, не утверждение об отсутствии всех данных на диске.
{len(checks)} прямых pure checks прошли без pytest/collector.

## Funding: полезная находка без новых загрузок

Исходный normalized файл не изменён: {len(normalized['events'])} reported-rate строк,
по `{normalized_counts}`; terminal timestamp 1 октября в нём отсутствует ({terminal_old} строк).
По frozen pins проверены {fund_pin_count} funding receipt inputs/outputs и 48 raw file pins.
В {len(pages)} сохранённых raw pages ({tail['raw_rows_inspected']} строк) найдены unique
reported rates на 2026-10-01 00:00 UTC для {unique_tail}/8 активов.
Создана отдельная immutable выборка `terminal_funding_reports.json` со source file/row hashes.
Старые rates/ledger/результаты не исправлялись задним числом.

Записи — **reported funding rates, не наблюдаемые денежные settlements**.
Settlement marks present={completion['settlement_marks_present']}; independent schedule
verified={completion['settlement_schedule_independently_verified']}.
У каждой новой записи strategy_available_us=null и settlement_mark=null;
consumable_as_funding_cashflow=false. October requestTime не подменяет историческую
доступность ставки в August/September. Имя settlement_ms старого normalized
формата тоже само по себе не доказывает debit/credit или contemporaneous mark.

## Что имеется и чего не хватает

- Funding: rates для 8/8 и terminal reports есть, но marks/settlements/availability не квалифицированы.
- Marks: старый audit содержит только `{summary['sampled_mark_candles']}` common 4h sample candles,
  не continuous full8 marks всего 61-дневного периода и terminal хвоста.
- Fees: current default taker proxy `{price['current_taker_fee_bps']}` bps не доказывает
  historical/account fee tier. Current-rule feasibility не является historical rule provenance.
- L2: recorded BTC архивы декабря 2024 и book ticker января 2025 вне целевого периода;
  не подменяют full8 2026 depth/sequence/coverage и volume replay.
- Closed features: prices/Parquet на диске не доказывают full causal EMA state,
  maturity labels, отдельный scaler и consumer availability. Training history status:
  `{contract['training_label_history_status']}`.

Отдельно прочитана только старая 1m OHLCV manifest декларация:
{ohlcv['rows']} строк, {utc(ohlcv['first_timestamp'])} — {utc(ohlcv['last_timestamp'])}.
Её {ohlcv['output_bytes']} byte Parquet **не читался и не хешировался сейчас**.
Этот пример — не доказательство отсутствия более новых OHLCV в других кешах.
Filename inventory ограничен Retrodate и price audit; максимум 8192 entries,
512 directories, 2048 candidate names на root. Coverage по имени не выводится.
Другие directories/Downloads, runtime, ключи и ордера не исследовались.
У четырёх предоставленных gzip проверены только существование/size; содержимое не открывалось.

## Calendar и terminal boundary

Период ledger: 2026-08-01 00:00 UTC — 2026-10-01 00:00 UTC exclusive.
Минимальный зарегистрированный terminal tail: до {utc(expected_end)} exclusive.
Funding timestamp ровно на Oct 1 boundary относится к terminal ledger перед fills:
он не должен исчезнуть из-за exclusive normalization end.

Согласно старому postclose plan, terminal closure — отдельное решение;
не новый 3-дневный label и не перенос выхода ledger на Oct 3.
49 баров — минимум features, не разрешение пересоздать EMA только на 49 барах:
нужен зарегистрированный full causal prefix. Training history scope остаётся
отдельным незавершённым входом, не покрывается одной validation выборкой.

## Следующие работы по приоритету

1. Зарегистрировать источники target-period L2/sequence/coverage и contemporaneous marks;
   L2 нужна для decision/terminal execution windows, marks — для непрерывной NAV/DD.
   Границы L2 windows зависят от доказанной/отдельно зарегистрированной availability,
   нельзя считать номинальный bar close реальным временем получения features.
2. Сопоставить найденные rates/terminal reports с settlement marks, signed exposure,
   schedule и availability provenance. Пока не включать их в финансовый ledger.
3. Квалифицировать historical rules/fees и causal training-history/feature clocks.
   Если используется latency/fee model вместо наблюдений — отдельный заранее
   зарегистрированный исследовательский режим, не observed execution qualification.
4. Только после input gate — один ограниченный реалистичный replay, затем полный
   период с непрерывным капиталом/ledger, графиком NAV и costs attribution.
   Не запускать новое обучение на старых next-open labels ради ещё одного ranking.

Все 9 authority flags false: 0 fits, 0 financial labels, 0 market strategy replays,
0 network requests, 0 runtime/orders. Результаты прибыли этого этапа не существуют.
"""
    with (OUT / "REPORT_RU.md").open("x", encoding="utf-8", newline="\n") as output:
        output.write(report)
    receipt = seal({"schema": "bounded_execution_input_audit_receipt/1", "source_hashes": pins,
                    "prior_evidence_integrity": prior_integrity, "summary_sha256": object_sha(summary),
                    "terminal_reports_sha256": tail["sha256"], "direct_checks_passed": len(checks),
                    "economic_labels_constructed": 0, "new_model_fits": 0, "market_strategy_replays": 0,
                    "network_requests": 0, "financial_replay_ready": False, "safety": SAFETY})
    write_json(OUT / "receipt.json", receipt)
    artifacts = {path.name: file_sha(path) for path in sorted(OUT.iterdir()) if path.is_file()}
    write_json(OUT / "artifact_hashes.json", {"files": artifacts, "sha256": object_sha(artifacts)})
    print(json.dumps({"ok": True, "direct_checks": len(checks), "terminal_rate_assets": unique_tail,
                      "qualified_components": 0, "execution_bundle_exists": bundle is not None,
                      "filename_scan_complete": summary["filename_inventory_complete_by_root"],
                      "financial_replay_ready": False, "out": str(OUT)}).encode("utf-8")[:920].decode("utf-8", "ignore"))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)[:650]}, ensure_ascii=False)
              .encode("utf-8")[:920].decode("utf-8", "ignore"))
        sys.exit(1)
