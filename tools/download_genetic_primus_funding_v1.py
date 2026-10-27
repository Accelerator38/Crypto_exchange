"""One short public funding GET per CLI invocation; local prepare/check/seal."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exia.genetic_primus import funding_history_download_v1 as f

CONTRACT = ROOT / "configs/genetic_primus_funding_download_v1.json"
TEST = ROOT / "tests/test_genetic_primus_funding_history_download_v1.py"
SOURCES = (Path(__file__), CONTRACT, TEST, ROOT / "src/exia/genetic_primus/funding_history_download_v1.py",
           ROOT / "docs/GENETIC_PRIMUS_FUNDING_DOWNLOAD_V1.md")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def pin(path):
    content = path.read_bytes()
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def iso(value):
    return dt.datetime.fromtimestamp(value/1000, dt.timezone.utc).isoformat() if value else None


def check():
    spec = importlib.util.spec_from_file_location("funding_download_tests", TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(module.FundingTests))
    if not result.wasSuccessful():
        print(stream.getvalue().encode("utf-8")[:720].decode("utf-8", "ignore"))
        raise ValueError("pure funding tests failed")
    return {"checks": result.testsRun, "failures": 0, "errors": 0, "network_calls": 0}


def verify_contract():
    c = read(CONTRACT)
    if (c["endpoint"] != f.ENDPOINT or c["method"] != "GET" or c["symbols"] != list(f.SYMBOLS)
            or c["product_type"] != "USDT-FUTURES" or c["page_size"] != f.PAGE_SIZE
            or c["max_pages_per_symbol"] != f.MAX_PAGES or c["requests_per_command"] != 1
            or c["timeout_seconds"] != 5 or c["max_response_bytes"] != f.MAX_RESPONSE_BYTES
            or c["automatic_retries"] != 0 or c["redirects_allowed"] is not False
            or c["cookies_or_authorization_headers_allowed"] is not False or c["safety"] != f.SAFETY
            or c["period"] != {"start_utc": "2026-08-01T00:00:00Z", "end_exclusive_utc": "2026-10-01T00:00:00Z"}):
        raise ValueError("only fixed public readonly funding contract supported")
    return c


def prepare(folder):
    verify_contract()
    if folder.exists():
        raise ValueError("new output directory required")
    tests = check()
    folder.mkdir(parents=True)
    (folder / "raw").mkdir()
    plan = {"schema": "public_funding_download_plan/1", "start_ms": f.START_MS, "end_exclusive_ms": f.END_MS,
            "symbols": list(f.SYMBOLS), "endpoint": f.ENDPOINT, "page_size": f.PAGE_SIZE,
            "max_pages_per_symbol": f.MAX_PAGES, "user_authorized_public_historical_GET": True,
            "authorization_basis": "human_yes_to_public_rest_funding_only_20261007",
            "source_pins": {path.relative_to(ROOT).as_posix(): pin(path) for path in SOURCES},
            "engineering_checks": tests, "safety": f.SAFETY}
    plan["sha256"] = f.identity(plan)
    write(folder / "plan.json", plan)
    print(f"prepared: tests={tests['checks']}; symbols=8; period=Aug-Sep2026; network=0; existing data unchanged")


def verify(folder):
    plan = read(folder / "plan.json")
    if plan["sha256"] != f.identity({k: v for k, v in plan.items() if k != "sha256"}) or plan["safety"] != f.SAFETY:
        raise ValueError("plan identity/safety mismatch")
    for path, expected in plan["source_pins"].items():
        if pin(ROOT / path) != expected:
            raise ValueError("frozen downloader source changed")
    verify_contract()
    return plan


def page_bank(folder, symbol):
    receipts = sorted((folder / "raw").glob(symbol + "_page_*.receipt.json"))
    pages = []
    for expected_page, path in enumerate(receipts, 1):
        receipt = read(path)
        if (receipt["status"] != "success" or receipt["symbol"] != symbol
                or receipt["page_no"] != expected_page or receipt["plan_sha256"] != read(folder / "plan.json")["sha256"]):
            raise ValueError("failed or noncontiguous page receipt; do not silently skip")
        raw = folder / receipt["raw_path"]
        if pin(raw) != receipt["raw_pin"]:
            raise ValueError("raw page identity mismatch")
        pages.append({"page_no": expected_page, "payload": f.strict_json(raw.read_bytes())})
    return pages


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(request.full_url, code, "redirect disallowed", headers, fp)


def fetch(folder, symbol, page_no):
    plan = verify(folder)
    if symbol not in f.SYMBOLS or type(page_no) is not int or not 1 <= page_no <= f.MAX_PAGES:
        raise ValueError("one allowed symbol/page required")
    receipt_path = folder / "raw" / f"{symbol}_page_{page_no:03d}.receipt.json"
    raw_path = folder / "raw" / f"{symbol}_page_{page_no:03d}.json"
    if receipt_path.exists() or raw_path.exists():
        raise ValueError("existing page refused; no overwrite or automatic retry")
    existing = page_bank(folder, symbol)
    if page_no != len(existing)+1:
        raise ValueError("exact next page required")
    if existing:
        audit = f.audit_symbol(existing, symbol)
        if not audit["more_pages_useful"]:
            raise ValueError("additional older pages are not useful for fixed interval")
    params = {"symbol": symbol, "productType": "USDT-FUTURES", "pageSize": str(f.PAGE_SIZE), "pageNo": str(page_no)}
    url = f.ENDPOINT + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "GeneticPrimusOfflineFundingResearch/1",
                                                 "Accept": "application/json", "Accept-Encoding": "identity"}, method="GET")
    receipt = {"schema": "public_funding_page_receipt/1", "symbol": symbol, "page_no": page_no,
               "url": url, "plan_sha256": plan["sha256"], "status": "pending", "public_GET_attempts": 1,
               "local_request_started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "raw_path": raw_path.relative_to(folder).as_posix(), "safety": f.SAFETY}
    body = b""
    try:
        opener = urllib.request.build_opener(NoRedirect())
        with opener.open(request, timeout=5) as response:
            receipt["http_status"] = response.status
            receipt["http_date"] = response.headers.get("Date")
            body = response.read(f.MAX_RESPONSE_BYTES+1)
        if len(body) > f.MAX_RESPONSE_BYTES:
            raise ValueError("response exceeded hard cap")
        with raw_path.open("xb") as handle:
            handle.write(body)
        parsed = f.parse_page(f.strict_json(body), symbol)
        receipt.update({"status": "success", "raw_pin": pin(raw_path), "parsed": {k: v for k, v in parsed.items() if k != "rows"}})
    except Exception as exc:
        if isinstance(exc, urllib.error.HTTPError):
            receipt["http_status"] = exc.code
            try:
                body = exc.read(f.MAX_RESPONSE_BYTES+1)
            except Exception:
                body = b""
            finally:
                exc.close()
        receipt["status"] = "failed"
        receipt["failure_type"] = type(exc).__name__
        receipt["failure_message"] = str(exc)[:300]
        receipt["response_truncated"] = len(body) > f.MAX_RESPONSE_BYTES
        if body and not raw_path.exists():
            with raw_path.open("xb") as handle:
                handle.write(body[:f.MAX_RESPONSE_BYTES])
        if raw_path.exists():
            receipt["raw_pin"] = pin(raw_path)
    receipt["local_request_finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write(receipt_path, receipt)
    if receipt["status"] != "success":
        print(f"GET failed: symbol={symbol}; page={page_no}; type={receipt['failure_type']}; attempts=1; receipt preserved; no retry")
        return False
    pages = page_bank(folder, symbol)
    audit = f.audit_symbol(pages, symbol)
    print(f"GET success: {symbol}; page={page_no}; rows={parsed['row_count']}; newest={iso(parsed['newest_ms'])}; oldest={iso(parsed['oldest_ms'])}; interval_rows={audit['requested_rows']}; more_pages={audit['more_pages_useful']}; network_calls=1")
    return True


def inspect(folder, symbol):
    verify(folder)
    if symbol not in f.SYMBOLS:
        raise ValueError("known symbol required")
    audit = f.audit_symbol(page_bank(folder, symbol), symbol)
    print(json.dumps({k: v for k, v in audit.items() if k != "events"}, ensure_ascii=False).encode("utf-8")[:900].decode("utf-8", "ignore"))


def finalize(folder, complete):
    plan = verify(folder)
    output = folder / ("completion_manifest.json" if complete else "blocked_status.json")
    if output.exists():
        raise ValueError("existing final status refused")
    audits, failures, raw_pins = {}, [], {}
    for symbol in f.SYMBOLS:
        try:
            pages = page_bank(folder, symbol)
            if not pages:
                failures.append(symbol + ":not_requested")
                continue
            audit = f.audit_symbol(pages, symbol)
            audits[symbol] = audit
            if not audit["api_enumeration_ready"]:
                failures.append(symbol + ":interval_not_ready")
        except ValueError as exc:
            failures.append(symbol + ":" + str(exc)[:120])
    for path in (folder / "raw").iterdir():
        if path.is_file():
            raw_pins[path.relative_to(ROOT).as_posix()] = pin(path)
    if complete and (failures or len(audits) != 8):
        raise ValueError("cannot seal incomplete interval: " + ";".join(failures)[:550])
    events = [row for symbol in sorted(audits) for row in audits[symbol]["events"]]
    if complete:
        write(folder / "funding_events.json", {"schema": "observed_public_funding_rates/1", "events": events,
              "start_ms": f.START_MS, "end_exclusive_ms": f.END_MS,
              "units": "rate_fraction_per_settlement_and_exact_bps", "settlement_marks_present": False,
              "missing_data_imputed": False, "safety": f.SAFETY})
    status = {"schema": "public_funding_download_completion/1", "requested_interval_enumerated": complete,
              "source_endpoint": f.ENDPOINT, "source_docs": f.DOC_URL, "plan_sha256": plan["sha256"],
              "symbol_audits": {s: {k: v for k, v in a.items() if k != "events"} for s, a in audits.items()},
              "failures": failures, "requested_interval_rows": len(events), "raw_inputs": raw_pins,
              "latest_funding_timestamp_actually_seen_ms": max((a["newest_observed_ms"] or 0 for a in audits.values()), default=0),
              "settlement_schedule_independently_verified": False, "settlement_marks_present": False,
              "financial_replay_completed": False, "model_fits": 0, "live_or_paper_started": False,
              "safety": f.SAFETY}
    if complete:
        status["events_output_pin"] = pin(folder / "funding_events.json")
    status["sha256"] = f.identity(status)
    write(output, status)
    report_path = folder / ("REPORT.md" if complete else "BLOCKED.md")
    lines = ["# Funding Bitget: август–сентябрь 2026", "",
             "Статус: " + ("интервал перечислен публичным REST для full8" if complete else "не завершено, данные/запросы неполны"), "",
             "Сырые responses и receipts сохранены, без перезаписи. Один GET на команду, без ключей, retries, WS, "
             "collector, фоновых процессов, моделей или торгов. Funding-rate units проверены: fraction×10000=bps; "
             "оба знака и нулевые ставки сохранены точно, без fill-forward и подстановки нулей.", "",
             "| Symbol | Pages | Rows Aug–Sep | Left guard | Right guard | API enumeration |", "|---|---:|---:|---|---|---|"]
    for symbol, audit in audits.items():
        lines.append(f"| {symbol} | {audit['pages']} | {audit['requested_rows']} | {iso(audit['left_guard_ms'])} | {iso(audit['right_guard_ms'])} | {audit['api_enumeration_ready']} |")
    lines += ["", "Незаполненные условия: " + str(failures), "",
              "Это архив ставок, а не рассчитанный funding cash flow или законченный net replay. "
              "Нет settlement marks, зарегистрированного funding valuation/cash policy, normal/stress costs "
              "и predictor→net-policy. Исторический settlement schedule не подтверждён независимым источником; "
              "observed gaps и полнота REST enumeration перечислены в manifest. Current instruments не заменяют эту проверку.", "",
              "Август–сентябрь остаётся открытым benchmark. Funding timestamps вне интервала, полученные для "
              "pagination guards, сохранены в raw и latest_funding_timestamp_actually_seen_ms; учитывать это при будущей prospective seal.", "",
              "Источник: [официальная документация Bitget]("+f.DOC_URL+"). Старый июльский funding snapshot, "
              "модели и preflight не изменены. Все runtime safety flags и promotion authority остаются false."]
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines)+"\n")
    write(folder / ("evidence_receipt.json" if complete else "blocked_evidence_receipt.json"), {
        "inputs": plan["source_pins"] | raw_pins | {(folder / "plan.json").relative_to(ROOT).as_posix(): pin(folder / "plan.json")},
        "outputs": {path.relative_to(ROOT).as_posix(): pin(path) for path in (output, report_path) + ((folder / "funding_events.json",) if complete else ())},
        "safety": f.SAFETY, "financial_replay_completed": False})
    print(f"final status: complete={complete}; symbols_audited={len(audits)}; interval_rows={len(events)}; failed_or_unrequested={len(failures)}; financial_replay=false; fits=0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("check", "prepare", "fetch", "inspect", "seal", "status"))
    parser.add_argument("--output-dir")
    parser.add_argument("--symbol", choices=f.SYMBOLS)
    parser.add_argument("--page", type=int)
    args = parser.parse_args()
    if args.step == "check":
        print(check())
        return
    if not args.output_dir:
        raise ValueError("output directory required")
    folder = (ROOT / args.output_dir).resolve()
    folder.relative_to(ROOT / "Retrodate")
    if folder == ROOT / "Retrodate":
        raise ValueError("dataset subdirectory required")
    if args.step == "prepare":
        prepare(folder)
    elif args.step == "fetch":
        if not fetch(folder, args.symbol, args.page):
            raise SystemExit(1)
    elif args.step == "inspect":
        inspect(folder, args.symbol)
    else:
        finalize(folder, args.step == "seal")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print((type(exc).__name__+": "+str(exc)).encode("utf-8")[:800].decode("utf-8", "ignore"))
        raise SystemExit(1)
