"""One-shot bounded offline engineering report; never a trading/replay runner."""
import io
import json
import hashlib
import sys
import unittest
from collections import Counter
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from exia.genetic_primus.bounded_gzip_csv_v1 import read_prefix
from exia.genetic_primus.market_fragment_v1 import mark_events, object_sha
from exia.genetic_primus.market_timebase_fragment_v1 import (
    SAFETY, TICKER_HEADER, TRADE_HEADER, inspect_timebase, seal, ticker_events,
    trade_events, verify,
)
import test_genetic_primus_market_timebase_v1 as tests

CONFIG = "configs/genetic_primus_market_timebase_v1.json"
OUT = ROOT / "Reports/Exia/Genetic_Primus/market_timebase_smoke_v1_20261008"
NEW_SOURCES = (CONFIG, "src/exia/genetic_primus/market_timebase_fragment_v1.py",
               "tests/test_genetic_primus_market_timebase_v1.py",
               "tools/check_genetic_primus_market_timebase_v1.py",
               "docs/GENETIC_PRIMUS_MARKET_TIMEBASE_V1.md",
               "src/exia/genetic_primus/bounded_gzip_csv_v1.py",
               "src/exia/genetic_primus/market_fragment_v1.py")


def file_sha(path):
    # Called only for small code/evidence files, never entire input archives.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        output.write("\n")


def verify_prior(prior):
    receipt = read_json(prior / "receipt.json")
    verify(receipt)
    manifest = read_json(prior / "artifact_hashes.json")
    if object_sha(manifest["files"]) != manifest["sha256"]:
        raise ValueError("prior manifest seal mismatch")
    for name, expected in receipt["source_hashes"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or file_sha(path) != expected:
            raise ValueError("prior source mismatch: " + name)
    for name, expected in manifest["files"].items():
        path = (prior / name).resolve()
        if not path.is_relative_to(prior) or file_sha(path) != expected:
            raise ValueError("prior artifact mismatch: " + name)
    return {"sources_verified": len(receipt["source_hashes"]),
            "artifacts_verified": len(manifest["files"])}


def stat_value(path):
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_prefix(path, limits, header):
    before = stat_value(path)
    with path.open("rb") as raw:
        prefix = read_prefix(raw, **limits)
    after = stat_value(path)
    if before != after or prefix["header"] != list(header):
        raise ValueError("changed archive or unexpected header")
    if prefix["row_bank_sha256"] != object_sha(prefix["rows"]):
        raise ValueError("row bank mismatch")
    return prefix, {"path": str(path), "stat_before": before, "stat_after": after,
                    "unchanged_stat_during_read": True,
                    "full_archive_hash_verified": False,
                    "entire_archive_read_and_crc_verified": prefix["entire_archive_read_and_crc_verified"]}


def build_plot(books, trades, tickers, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    first, last = books[0]["received_ts_us"], books[-1]["received_ts_us"]
    selected = {"L2": books,
                "trades": [e for e in trades["events"] if first <= e["received_ts_us"] <= last],
                "ticker": [e for e in tickers["events"] if first <= e["received_ts_us"] <= last]}
    seconds = lambda e: (e["received_ts_us"] - first) / 1_000_000
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    x = [seconds(e) for e in books]
    axes[0].step(x, [float(e["bids"][0][0]) for e in books], where="post", label="Observed best bid")
    axes[0].step(x, [float(e["asks"][0][0]) for e in books], where="post", label="Observed best ask")
    for name, price, marker in (("trades", "price", "o"), ("ticker", "mark_price", "x")):
        events = [e for e in selected[name] if e[price] is not None]
        axes[0].scatter([seconds(e) for e in events], [float(e[price]) for e in events],
                        marker=marker, label="Trade prints" if name == "trades" else "Reported mark")
    for name, events in selected.items():
        axes[1].plot([seconds(e) for e in events],
                     [(e["received_ts_us"] - e["exchange_ts_us"]) / 1000 for e in events],
                     marker=".", linestyle="none", label=name)
    axes[0].set_ylabel("USDT; not fills")
    axes[1].set_ylabel("Provider receive - exchange, ms\nNOT order latency")
    axes[1].set_xlabel("Seconds from first L2 provider receipt (UTC microseconds preserved in JSON)")
    axes[0].set_title("BTCUSDT 2024-12-01: bounded format/clock diagnostic, NOT NAV or strategy replay")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "timebase_fragment.png", dpi=140)
    fig.savefig(out / "timebase_fragment.svg")
    plt.close(fig)


def main():
    if OUT.exists():
        raise ValueError("output root already exists; immutable evidence must not be overwritten")
    config = read_json(ROOT / CONFIG)
    if config["safety"] != SAFETY or config["financial_replay_ready"] is not False:
        raise ValueError("invalid safety contract")
    prior = (ROOT / config["prior_l2_evidence"]).resolve()
    if not prior.is_relative_to(ROOT):
        raise ValueError("invalid prior root")
    prior_integrity = verify_prior(prior)
    sources = {name: file_sha(ROOT / name) for name in NEW_SOURCES}
    for name in ("receipt.json", "artifact_hashes.json", "recorded_prefix.json", "reconstruction.json"):
        relative = str((prior / name).relative_to(ROOT)).replace("\\", "/")
        sources[relative] = file_sha(prior / name)
    check_log = io.StringIO()
    result = unittest.TextTestRunner(stream=check_log, verbosity=0).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(tests.TimebaseTests))
    if not result.wasSuccessful() or result.testsRun != 42:
        raise ValueError("bounded unit checks failed: " + check_log.getvalue()[:600])
    trade_prefix, trade_stat = load_prefix(Path(config["sources"]["trades"]),
                                          config["prefix_limits"], TRADE_HEADER)
    ticker_prefix, ticker_stat = load_prefix(Path(config["sources"]["ticker"]),
                                            config["prefix_limits"], TICKER_HEADER)
    trades = trade_events(trade_prefix["rows"], trade_prefix["uncompressed_prefix_sha256"])
    tickers = ticker_events(ticker_prefix["rows"], ticker_prefix["uncompressed_prefix_sha256"])
    reconstruction = read_json(prior / "reconstruction.json")
    verify(reconstruction)
    books = reconstruction["events"]
    diagnostic = inspect_timebase(books, trades, tickers, max_age_us=config["max_age_us"])
    if diagnostic != inspect_timebase(books, trades, tickers, max_age_us=config["max_age_us"]):
        raise ValueError("nondeterministic diagnostic")
    # Independent older pure parser: compare only published, fully delimited rows.
    old_marks = mark_events([ticker_prefix["rows"][e["source_row_ordinal"]]
                            for e in tickers["events"]], ticker_prefix["uncompressed_prefix_sha256"])
    keys = ("received_ts_us", "exchange_ts_us", "mark_price", "reported_funding_rate",
            "reported_funding_timestamp_us", "is_funding_settlement")
    if [[e[k] for k in keys] for e in old_marks] != [[e[k] for k in keys] for e in tickers["events"]]:
        raise ValueError("legacy mark parser disagreement")
    status = {kind: dict(Counter(j[kind]["status"] for j in diagnostic["joins"]))
              for kind in ("ticker", "trades")}
    summary = {"tests_passed": result.testsRun, "book_events": len(books),
               "trade_rows_read": trade_prefix["row_count"],
               "ticker_rows_read": ticker_prefix["row_count"],
               "complete_trade_events": len(trades["events"]),
               "complete_ticker_events": len(tickers["events"]),
               "dropped_final_trade_events": trades["dropped_final_events"],
               "dropped_final_ticker_events": tickers["dropped_final_events"],
               "duplicate_trade_rows": len(trades["duplicate_rows"]),
               "exchange_time_regressions": trades["exchange_time_regressions"],
               "same_receive_trade_pairs": sum(a["received_ts_us"] == b["received_ts_us"]
                                                for a, b in zip(trades["events"], trades["events"][1:])),
               "book_prefix_span_us": diagnostic["book_prefix_span_us"],
               "join_status_counts": status,
               "trade_quote_comparisons": len(diagnostic["trade_quote_comparisons"]),
               "outside_observed_spread": sum(c["outside_observed_spread"] is True
                                              for c in diagnostic["trade_quote_comparisons"]),
               "legacy_mark_agreements": len(old_marks),
               "reported_funding_times_us": sorted(set(e["reported_funding_timestamp_us"]
                                                       for e in tickers["events"])),
               "reported_funding_rates": sorted(set(e["reported_funding_rate"] for e in tickers["events"])),
               "funding_time_relation_counts": dict(Counter(e["funding_time_relation_to_receipt"]
                                                            for e in tickers["events"])),
               "missing_predicted_rate_rows": sum(r["predicted_funding_rate"] == ""
                                                   for r in ticker_prefix["rows"])}
    synthetic = inspect_timebase(*tests.fixtures(price="101"))
    if not synthetic["trade_quote_comparisons"][0]["outside_observed_spread"]:
        raise ValueError("synthetic diagnostic did not expose out-of-spread print")
    verify_prior(prior)
    if any(file_sha(ROOT / name) != expected for name, expected in sources.items()):
        raise ValueError("sources changed during checks")
    OUT.mkdir(parents=True, exist_ok=False)
    write_json(OUT / "trades_prefix.json", trade_prefix)
    write_json(OUT / "ticker_prefix.json", ticker_prefix)
    write_json(OUT / "trades_series.json", trades)
    write_json(OUT / "ticker_series.json", tickers)
    write_json(OUT / "timebase_diagnostic.json", diagnostic)
    write_json(OUT / "synthetic_out_of_spread.json", synthetic)
    write_json(OUT / "summary.json", summary)
    with (OUT / "checks.txt").open("x", encoding="utf-8") as output:
        output.write(check_log.getvalue())
    plot_log = io.StringIO()
    with redirect_stderr(plot_log):
        build_plot(books, trades, tickers, OUT)
    if plot_log.getvalue():
        with (OUT / "plot_diagnostics.txt").open("x", encoding="utf-8") as output:
            output.write(plot_log.getvalue()[:4096])
    report = f"""# Trades / derivative ticker: bounded offline timebase smoke

Дата запуска: 2026-10-08. Исходные данные: BTCUSDT, 2024-12-01.

## Результат

- {result.testsRun}/{result.testsRun} явных unittest проверок; deterministic повтор равен.
- Прочитано по {trade_prefix['row_count']} trade и {ticker_prefix['row_count']} ticker строк.
- По архивам прочитано {trade_prefix['compressed_bytes_read']} / {ticker_prefix['compressed_bytes_read']}
  сжатых и {trade_prefix['uncompressed_bytes_inflated']} / {ticker_prefix['uncompressed_bytes_inflated']}
  распакованных байт. Полный архив/hash/CRC НЕ проверялись.
- Опубликовано {len(trades['events'])} уникальных trade events и {len(tickers['events'])} ticker events.
  Последние недоказанно завершённые пакеты: {trades['dropped_final_events']} / {tickers['dropped_final_events']} events исключены.
- {summary['same_receive_trade_pairs']} соседних trade pairs с одинаковым receive timestamp:
  никакого вымышленного within-batch порядка; duplicate IDs: {summary['duplicate_trade_rows']}.
- Переиспользованы {len(books)} замороженных L2 states, span {diagnostic['book_prefix_span_us'] / 1e6:.6f} s.
  Старые source/artifact hashes проверены: {prior_integrity['sources_verified']} / {prior_integrity['artifacts_verified']}.
- Сопоставлений trade с предшествующим свежим BBO: {len(diagnostic['trade_quote_comparisons'])};
  вне наблюдаемого spread: {summary['outside_observed_spread']}. Это не доказательство fill или ошибки данных.
- Старый pure mark parser согласен в {len(old_marks)} случаях.

## Часы и funding

Все timestamps сохранены в UTC микросекундах. As-of: provider receive строго
раньше cutoff; equal-clock boundary ambiguous, оба возраста <=500000 us.
Статусы joins: `{json.dumps(status, sort_keys=True)}`.
Это некалиброванный engineering freshness budget, не latency ордеров.

Reported funding rate: `{summary['reported_funding_rates']}`;
reported funding timestamps, us: `{summary['reported_funding_times_us']}`;
отношение к времени receipt: `{summary['funding_time_relation_counts']}`.
Пустой predicted rate в {summary['missing_predicted_rate_rows']} raw строках сохранён как missing/null.
Ни один ticker не превращён в settlement или cashflow. Будущее/прошлое/rate=0
не доказывает final rate, фактическое списание или settlement mark.

## Ограничения и следующий шаг

Это только короткое пересечение двух префиксов с ранее проверенными 15 levels/side
L2. Не подтверждены venue sequence, полнота глубины, доступность стратегии,
реальное исполнение, account/historical fees/rules и funding settlements.
Архив book ticker января 2025 не присоединялся к декабрю 2024.
График — engineering цены/часы, не накопленный капитал или доходность.

0 fits, 0 economic labels, 0 market strategy replays, 0 network requests.
Все 9 authority flags false; financial_replay_ready=false.
Положительный PnL предыдущих ретротестов здесь не перепроверен и не квалифицирован.
BTC 2024 не заменяет отсутствующий verified full8 execution bundle августа–сентября 2026.

Далее: детерминированный microsecond-to-postclose clock adapter с полной
provenance и явным fail-closed на недоказанные available/funding fields.
Новый financial replay/обучение/сеть не разрешены этим отчётом.
"""
    with (OUT / "REPORT_RU.md").open("x", encoding="utf-8", newline="\n") as output:
        output.write(report)
    receipt = seal({"schema": "market_timebase_smoke_receipt/1", "source_hashes": sources,
                    "input_archives": {"trades": trade_stat, "ticker": ticker_stat},
                    "prefix_hashes": {kind: {key: prefix[key] for key in
                                             ("compressed_prefix_sha256", "uncompressed_prefix_sha256", "row_bank_sha256")}
                                      for kind, prefix in (("trades", trade_prefix), ("ticker", ticker_prefix))},
                    "summary_sha256": object_sha(summary), "diagnostic_sha256": diagnostic["sha256"],
                    "prior_evidence_integrity": prior_integrity, "unit_checks_passed": result.testsRun,
                    "financial_replay_ready": False, "economic_labels_constructed": 0,
                    "new_model_fits": 0, "market_strategy_replays": 0, "network_requests": 0,
                    "safety": SAFETY})
    write_json(OUT / "receipt.json", receipt)
    artifacts = {path.name: file_sha(path) for path in sorted(OUT.iterdir()) if path.is_file()}
    write_json(OUT / "artifact_hashes.json", {"files": artifacts, "sha256": object_sha(artifacts)})
    print(json.dumps({"ok": True, "tests": result.testsRun, "trade_events": len(trades['events']),
                      "ticker_events": len(tickers['events']), "books": len(books),
                      "financial_replay_ready": False, "out": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)[:650]}, ensure_ascii=False)
              .encode("utf-8")[:920].decode("utf-8", "ignore"))
        sys.exit(1)
