"""Bounded read-only diagnosis: historical prices and reference execution gaps.

Each fetch command makes at most ONE public historical/configuration GET, no
retries, credentials, orders, WS, collector, model fitting or financial replay.
Existing sources/results are immutable. New evidence uses exclusive creation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exia.genetic_primus.unified_measurement_v1 import SAFETY, identity, BAR_MS

BASE = ROOT / "Reports/Exia/Genetic_Primus/compact_net_replay_v1_20261007"
DEFAULT = ROOT / "Reports/Exia/Genetic_Primus/price_execution_audit_v1_20261008"
RAW = ROOT / "Retrodate/bitget_full8_1m_prospective_v1_20260818_20260903"
FOUR = ROOT / "Retrodate/exia_full8_prospective_timeframes_v1_20260818_20260903"
SYMBOLS = ("ADA", "BNB", "BTC", "DOGE", "ETH", "LINK", "SOL", "XRP")
DOCS = ("https://www.bitget.com/docs/catalog/market/market-data",
        "https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data",
        "https://www.bitget.com/docs/catalog/classic-contract-market/classic-contract-market")


def ms(value):
    return int(datetime.fromisoformat(value).timestamp() * 1000)


START, END = ms("2026-08-18T00:00:00+00:00"), ms("2026-08-23T00:00:00+00:00")
ENTRY, EXIT = ms("2026-08-19T04:00:00+00:00"), ms("2026-08-22T04:00:00+00:00")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def pin(path):
    content = path.read_bytes()
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def aggregate_minutes(frame):
    """No fill-forward, zero imputation, incomplete bars or timestamp relabeling."""
    import numpy as np
    if frame.duplicated("timestamp").any() or not np.all(np.diff(frame["timestamp"]) == 60000):
        raise ValueError("duplicate/missing/reordered minute")
    columns = ["open", "high", "low", "close", "volume"]
    values = frame[columns].to_numpy()
    if not np.isfinite(values).all() or (frame[columns[:4]] <= 0).any().any() or (frame.volume < 0).any():
        raise ValueError("invalid OHLCV")
    if ((frame.low > frame[["open", "close"]].min(axis=1)) | (frame.high < frame[["open", "close"]].max(axis=1)) | (frame.low > frame.high)).any():
        raise ValueError("OHLC bounds invalid")
    frame = frame.assign(timestamp=frame.timestamp // BAR_MS * BAR_MS)
    group = frame.groupby("timestamp", sort=True)
    counts = group.size()
    if not (counts == 240).all():
        raise ValueError("incomplete four-hour source bar")
    return group.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                     close=("close", "last"), volume=("volume", "sum"))


def parquet(path):
    import pyarrow.parquet as pq
    file = pq.ParquetFile(io.BytesIO(path.read_bytes()), pre_buffer=False,
                         thrift_string_size_limit=1_000_000, thrift_container_size_limit=100_000)
    if file.metadata.num_rows > 200_000:
        raise ValueError("bounded parquet read exceeded")
    return file.read(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"],
                     use_threads=False).to_pandas(use_threads=False)


def prepare(folder):
    paths = [Path(__file__), BASE / "registration.json", BASE / "selected_policy.json", BASE / "results.json",
             BASE / "evidence_receipt.json", BASE / "analysis_receipt.json", RAW / "integrity_manifest.json",
             FOUR / "index.json", FOUR / "parts/full8_240m.parquet", FOUR / "parts/full8_240m.manifest.json",
             ROOT / "Retrodate/exia_full8_holdout_timeframes_v1_20260715_20260817/parts/full8_240m.parquet",
             ROOT / "src/exia/genetic_primus/unified_measurement_v1.py", ROOT / "configs/genetic_primus_release_candidate_v1.json"]
    raw_manifest = read(RAW / "integrity_manifest.json")
    paths += [ROOT / item["path"] for item in raw_manifest["files"] if item["path"].endswith("2026-08.parquet")]
    paths += [BASE / f"benchmark_{policy}_{scenario}.json" for policy in
              ("linear6_uniform", "linear6_half_life_90_anchor20", "mlp8_uniform", "mlp8_half_life_90_anchor20", "NoTrade", "EMA48_long_only")
              for scenario in ("normal", "stress")]
    cases = []
    for symbol in SYMBOLS:
        cases.append({"kind": "bitget_trade", "symbol": symbol, "endpoint": "https://api.bitget.com/api/v3/market/history-candles",
                      "params": {"category": "USDT-FUTURES", "symbol": symbol + "USDT", "interval": "4H",
                                 "startTime": str(START - 1), "endTime": str(END - 1), "limit": "100"}})
    for symbol in SYMBOLS:
        cases.append({"kind": "binance_trade", "symbol": symbol, "endpoint": "https://fapi.binance.com/fapi/v1/klines",
                      "params": {"symbol": symbol + "USDT", "interval": "4h", "startTime": str(START), "endTime": str(END - 1), "limit": "100"}})
    for symbol in ("BTC", "ETH", "SOL"):
        cases.append({"kind": "bitget_mark", "symbol": symbol, "endpoint": "https://api.bitget.com/api/v2/mix/market/history-mark-candles",
                      "params": {"symbol": symbol + "USDT", "productType": "USDT-FUTURES", "granularity": "4H",
                                 "startTime": str(START - 1), "endTime": str(END - 1), "limit": "100"}})
    cases.append({"kind": "bitget_contracts", "symbol": "FULL8_current_rules_only",
                  "endpoint": "https://api.bitget.com/api/v2/mix/market/contracts", "params": {"productType": "USDT-FUTURES"}})
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "raw").mkdir()
    registration = {"schema": "price_execution_audit_plan/1", "client_date": "2026-10-08",
        "period_ms": [START, END], "event_ms": [ENTRY, EXIT], "max_public_GETs": len(cases),
        "max_response_bytes": 1_000_000, "timeout_seconds": 10, "automatic_retries": 0,
        "permission": "user_explicit_historical_public_REST_Bitget_and_Binance_only",
        "cases": cases, "sources": {str(path.relative_to(ROOT)).replace("\\", "/"): pin(path) for path in paths},
        "primary_docs": DOCS, "live_or_account_access": False, "model_fits_or_financial_replays": 0,
        "price_confirmation_criteria": {"Bitget_same_venue_OHLC_exact": True,
            "Binance_return_difference_bps_abs_lt": 100, "Binance_open_price_basis_bps_abs_lt": 200,
            "purpose": "independent_confirmation_of_large_movement_not_fill_price_equivalence"},
        "existing_model_configuration_changed": False, "safety": SAFETY}
    registration["sha256"] = identity(registration)
    write(folder / "registration.json", registration)
    print("audit registered: 8 Bitget + 8 Binance + 3 historical mark + 1 public config GET maximum; no retries/runtime")


def verify(folder):
    registration = read(folder / "registration.json")
    if registration["sha256"] != identity({k: v for k, v in registration.items() if k != "sha256"}) or registration["safety"] != SAFETY:
        raise ValueError("registration/authority changed")
    for name, expected in registration["sources"].items():
        if pin(ROOT / name) != expected:
            raise ValueError("immutable input changed: " + name)
    return registration


def local(folder):
    import numpy as np
    registration = verify(folder)
    manifest = read(RAW / "integrity_manifest.json")
    compiled = parquet(FOUR / "parts/full8_240m.parquet")
    output, panel = {}, []
    for symbol in SYMBOLS:
        item = next(row for row in manifest["files"] if row["symbol"] in (symbol, symbol + "/USDT") and row["path"].endswith("2026-08.parquet"))
        path = ROOT / item["path"]
        if pin(path)["sha256"] != item["sha256"]:
            raise ValueError("original minute source pin mismatch")
        source = parquet(path)
        minute = source[(source.timestamp >= START) & (source.timestamp < END)].sort_values("timestamp")
        if len(minute) != 5 * 1440:
            raise ValueError("exact five-day minute window required")
        bars = aggregate_minutes(minute)
        archive = compiled[(compiled.symbol == symbol + "/USDT") & (compiled.timestamp >= START) & (compiled.timestamp < END)].set_index("timestamp").sort_index()
        if not bars.index.equals(archive.index):
            raise ValueError("minute/4h clock mismatch")
        error = {column: float(np.max(np.abs(bars[column].to_numpy() - archive[column].to_numpy()))) for column in ("open", "high", "low", "close", "volume")}
        if any(error[k] != 0 for k in ("open", "high", "low", "close")):
            raise ValueError("four-hour prices not reproduced from raw minute data")
        relative_volume_error = float(np.max(np.abs((bars.volume - archive.volume) / np.maximum(1, archive.volume))))
        if relative_volume_error > 1e-12:
            raise ValueError("four-hour source volume mismatch")
        event = minute[(minute.timestamp >= ENTRY) & (minute.timestamp <= EXIT)]
        open_returns = np.diff(np.log(event.open.to_numpy()))
        worst = int(np.argmax(np.abs(open_returns)))
        growth = float(archive.loc[EXIT, "open"] / archive.loc[ENTRY, "open"] - 1)
        output[symbol] = {"raw_rows": len(minute), "four_hour_rows": len(bars), "OHLC_max_absolute_error": error,
            "volume_max_relative_error": relative_volume_error, "entry_open": float(archive.loc[ENTRY, "open"]),
            "exit_open": float(archive.loc[EXIT, "open"]), "three_day_return": growth,
            "largest_minute_open_move_fraction": float(np.expm1(open_returns[worst])),
            "largest_minute_move_at_ms": int(event.iloc[worst + 1].timestamp),
            "nonpositive_volume_minute_count": int((minute.volume == 0).sum())}
        panel.extend({"timestamp": int(t), "symbol": symbol, **{k: float(row[k]) for k in ("open", "high", "low", "close", "volume")}}
                     for t, row in archive.iterrows())
    # Check the join immediately preceding this source; no model/data modification.
    previous = parquet(ROOT / "Retrodate/exia_full8_holdout_timeframes_v1_20260715_20260817/parts/full8_240m.parquet")
    boundary = {}
    for symbol in SYMBOLS:
        last = previous[(previous.symbol == symbol + "/USDT") & (previous.timestamp == START - BAR_MS)].iloc[0]
        first = next(row for row in panel if row["symbol"] == symbol and row["timestamp"] == START)
        boundary[symbol] = 10000 * (first["open"] / float(last.close) - 1)
    result = {"registration_sha256": registration["sha256"], "assets": output, "source_join_gap_bps": boundary,
        "all_four_hour_prices_reproduced_exactly": True, "minute_rows": 57600,
        "source": manifest["source"], "external_authenticity_confirmed": False, "safety": SAFETY}
    write(folder / "local_audit.json", result)
    write(folder / "local_four_hour_panel.json", panel)
    print("local audit: 57,600 minutes -> 240 four-hour candles; OHLC exact; complete grid; source join checked; no fits/replays")


def fetch(folder, index):
    registration = verify(folder)
    if index is None or not 0 <= index < len(registration["cases"]):
        raise ValueError("one registered case index required")
    case = registration["cases"][index]
    receipt_path = folder / "raw" / f"case_{index:02d}.receipt.json"
    body_path = folder / "raw" / f"case_{index:02d}.response.json"
    if receipt_path.exists() or body_path.exists():
        raise ValueError("immutable existing public request refused")
    receipt = {"case": case, "registration_sha256": registration["sha256"], "requests": 0,
               "authentication": "none", "endpoint_scope": "historical_market_or_public_contract_config"}
    if case["kind"] == "binance_trade" and index > 8:
        first = read(folder / "raw/case_08.receipt.json")
        if first.get("http_status") in (403, 451) or first.get("transport_error"):
            receipt["skipped_reason"] = "first Binance public request inaccessible; no repeated blocked calls"
            write(receipt_path, receipt)
            print(f"case={index:02d}; Binance skipped after first inaccessible request; requests=0")
            return
    url = case["endpoint"] + "?" + urlencode(case["params"])
    receipt.update({"url": url, "requests": 1, "fetched_at_host_utc": datetime.now(timezone.utc).isoformat()})
    request = Request(url, headers={"User-Agent": "GeneticPrimusOfflineHistoricalAudit/1", "Accept": "application/json"})
    body = b""
    try:
        with urlopen(request, timeout=registration["timeout_seconds"]) as response:
            receipt["http_status"] = response.status
            receipt["server_date"] = response.headers.get("Date")
            body = response.read(registration["max_response_bytes"] + 1)
    except HTTPError as error:
        receipt["http_status"] = error.code
        receipt["server_date"] = error.headers.get("Date")
        body = error.read(registration["max_response_bytes"] + 1)
    except (URLError, TimeoutError, OSError) as error:
        receipt["transport_error"] = type(error).__name__ + ": " + str(error)[:180]
    if len(body) > registration["max_response_bytes"]:
        receipt["response_truncated"] = True
        body = body[:registration["max_response_bytes"]]
    if body:
        with body_path.open("xb") as stream:
            stream.write(body)
        receipt["response"] = pin(body_path)
        try:
            payload = json.loads(body)
            receipt["api_code"] = payload.get("code") if isinstance(payload, dict) else None
            data = payload.get("data", []) if isinstance(payload, dict) else payload
            receipt["rows"] = len(data) if isinstance(data, list) else None
        except (ValueError, TypeError):
            receipt["non_json_response"] = True
    write(receipt_path, receipt)
    print(f"case={index:02d}; {case['kind']} {case['symbol']}; HTTP={receipt.get('http_status')}; rows={receipt.get('rows')}; requests=1; no retry")


def candle_map(folder, index):
    receipt = read(folder / "raw" / f"case_{index:02d}.receipt.json")
    if receipt.get("http_status") != 200 or receipt.get("transport_error"):
        return None
    path = folder / "raw" / f"case_{index:02d}.response.json"
    if pin(path) != receipt["response"]:
        raise ValueError("public raw response changed")
    payload = read(path)
    if isinstance(payload, dict):
        if payload.get("code") != "00000":
            return None
        payload = payload["data"]
    result = {}
    for row in payload:
        timestamp = int(row[0])
        if START <= timestamp < END:
            if timestamp in result:
                raise ValueError("duplicate public candle")
            values = [Decimal(str(x)) for x in row[1:5]]
            if timestamp % BAR_MS or not all(x.is_finite() and x > 0 for x in values):
                raise ValueError("public clock/price invalid")
            result[timestamp] = dict(zip(("open", "high", "low", "close"), values))
    if set(result) != set(range(START, END, BAR_MS)):
        return None
    return result


def compare(folder):
    import numpy as np
    registration = verify(folder)
    local_result, local_panel = read(folder / "local_audit.json"), read(folder / "local_four_hour_panel.json")
    comparisons = {}
    for i, symbol in enumerate(SYMBOLS):
        local_map = {row["timestamp"]: row for row in local_panel if row["symbol"] == symbol}
        observations = {}
        for kind, index in (("bitget_refetch", i), ("binance_independent", i + 8)):
            public = candle_map(folder, index)
            if public is None:
                observations[kind] = {"complete": False, "confirmed": False}
                continue
            changes = [float(public[t]["open"] / Decimal(str(local_map[t]["open"])) - 1) * 10000 for t in sorted(public)]
            growth = float(public[EXIT]["open"] / public[ENTRY]["open"] - 1)
            growth_error = 10000 * (growth - local_result["assets"][symbol]["three_day_return"])
            exact_ohlc = all(public[t][k] == Decimal(str(local_map[t][k])) for t in public for k in ("open", "high", "low", "close"))
            observations[kind] = {"complete": True, "candle_count": len(public), "three_day_return": growth,
                "return_difference_bps": growth_error, "median_open_basis_bps": float(np.median(changes)),
                "max_abs_open_basis_bps": max(abs(x) for x in changes), "OHLC_exact": exact_ohlc,
                "confirmed": exact_ohlc if kind == "bitget_refetch" else abs(growth_error) < 100 and max(abs(x) for x in changes) < 200}
        comparisons[symbol] = observations
    marks = {}
    for i, symbol in enumerate(("BTC", "ETH", "SOL"), 16):
        public = candle_map(folder, i)
        if public is None:
            marks[symbol] = {"available": False}
            continue
        local_map = {row["timestamp"]: row for row in local_panel if row["symbol"] == symbol}
        delta = {str(t): 10000 * float(public[t]["open"] / Decimal(str(local_map[t]["open"])) - 1) for t in public}
        funding_delta = [v for t, v in delta.items() if int(t) % (8 * 3_600_000) == 0]
        marks[symbol] = {"available": True, "max_abs_open_basis_bps": max(abs(x) for x in delta.values()),
                         "max_abs_settlement_open_basis_bps": max(abs(x) for x in funding_delta), "observed_mark_4h_only_not_tick": True}
    config_receipt = read(folder / "raw/case_19.receipt.json")
    contracts = {}
    if config_receipt.get("http_status") == 200:
        payload = read(folder / "raw/case_19.response.json")
        if payload.get("code") == "00000":
            contracts = {row["symbol"]: row for row in payload["data"] if row["symbol"] in {s + "USDT" for s in SYMBOLS}}
    feasibility = {}
    for policy in ("linear6_uniform", "linear6_half_life_90_anchor20", "mlp8_uniform", "mlp8_half_life_90_anchor20", "EMA48_long_only"):
        result = read(BASE / f"benchmark_{policy}_normal.json")
        fills = [row for row in result["ledger"] if row["kind"] in ("buy", "sell")]
        off_grid, below_min, rounding_loss = 0, 0, Decimal(0)
        with localcontext() as context:
            context.prec = 60
            for event in fills:
                symbol = event["symbol"].replace("/", "")
                rules = contracts.get(symbol)
                if rules is None:
                    continue
                quantity, price = Decimal(event["quantity"]), Decimal(event["fill_price"])
                step = Decimal(rules["sizeMultiplier"])
                rounded = (quantity // step) * step
                off_grid += quantity % step != 0
                below_min += quantity < Decimal(rules["minTradeNum"]) or quantity * price < Decimal(rules["minTradeUSDT"])
                rounding_loss += (quantity - rounded) * price
        delays = [row["entry_timestamp"] - row["window"]["decision_asof_ms"] for row in result["decisions"]]
        feasibility[policy] = {"reference_fills": len(fills), "rules_available": len(contracts) == 8,
            "off_current_quantity_grid": off_grid, "below_current_minimum": below_min,
            "total_floor_rounding_dust_quote_diagnostic_not_pnl": str(rounding_loss),
            "decision_to_entry_delay_ms": sorted(set(delays)), "historical_rules_not_verified": True,
            "partial_fills_or_rejections_modeled": False, "observed_historical_depth_available": False,
            "intrabar_mark_drawdown_measured": False}
    requests = sum(read(path).get("requests", 0) for path in (folder / "raw").glob("*.receipt.json"))
    evidence = {"registration_sha256": registration["sha256"], "price_comparisons": comparisons,
        "same_venue_refetch_confirmed_assets": sum(v["bitget_refetch"]["confirmed"] for v in comparisons.values()),
        "independent_venue_confirmed_assets": sum(v["binance_independent"]["confirmed"] for v in comparisons.values()),
        "mark_proxy_diagnostic": marks, "execution_feasibility": feasibility,
        "current_taker_fee_bps": {s: str(Decimal(v["takerFeeRate"]) * 10000) for s, v in contracts.items()},
        "fee_scope": "public_current_contract_default_not_account_or_historical_tariff",
        "historical_price_check_not_execution_qualification": True, "observed_IOC_execution_confirmed": False,
        "public_requests": requests, "model_fits": 0, "financial_replays": 0, "safety": SAFETY}
    write(folder / "audit_results.json", evidence)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(folder / "mpl_cache")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for symbol in SYMBOLS:
        rows = sorted([r for r in local_panel if r["symbol"] == symbol], key=lambda x: x["timestamp"])
        anchor = next(r["open"] for r in rows if r["timestamp"] == ENTRY)
        axes[0].plot([datetime.fromtimestamp(r["timestamp"] / 1000, timezone.utc) for r in rows],
                     [100 * (r["open"] / anchor - 1) for r in rows], label=symbol)
    for name, index, style in (("Bitget re-fetch BTC", 2, "-"), ("Binance BTC", 10, "--"), ("Bitget mark BTC", 16, ":")):
        data = candle_map(folder, index)
        if data:
            axes[1].plot([datetime.fromtimestamp(t / 1000, timezone.utc) for t in sorted(data)],
                         [float(data[t]["open"]) for t in sorted(data)], style, label=name)
    axes[0].set_title("Проверка цен 18–22 августа 2026; локальная 1m → 4h агрегация")
    axes[0].set_ylabel("Изменение относительно 19 августа 04:00, %")
    axes[1].set_ylabel("BTC/USDT, public historical reference")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, ncol=4 if axis is axes[0] else 1)
    fig.tight_layout()
    fig.savefig(folder / "price_check.png", dpi=150)
    fig.savefig(folder / "price_check.svg")
    plt.close(fig)
    lines = ["# Проверка скачка цен и исполнения — 8 октября 2026", "",
        f"Сверены 57 600 локальных минут: OHLC всех 240 четырёхчасовых свечей воспроизведены точно; пропусков/дубликатов нет. Склейка источников перед 18 августа проверена.",
        f"Повторный публичный Bitget подтвердил {evidence['same_venue_refetch_confirmed_assets']}/8 активов; независимый Binance — {evidence['independent_venue_confirmed_assets']}/8 в пределах зарегистрированного диагностического критерия (difference return <100 bps, price basis <200 bps). Это проверка наличия крупного движения, не тождественности цен бирж.", "",
        "## Скачок", "", "| Актив | Локальный open/open 3d | Bitget return diff, bps | Binance return diff, bps |", "|---|---:|---:|---:|"]
    for symbol in SYMBOLS:
        row = comparisons[symbol]
        values = [row[name].get("return_difference_bps") for name in ("bitget_refetch", "binance_independent")]
        lines.append(f"| {symbol} | {100*local_result['assets'][symbol]['three_day_return']:.2f}% | " + " | ".join("нет подтверждения" if x is None else f"{x:.3f}" for x in values) + " |")
    lines += ["", "## Реалистичность исполнения", "",
        "OHLCV open — цена первой сделки бара, не гарантированная цена IOC-ордера, bid/ask или доступная ликвидность. Наличие такого движения не подтверждает исполнимость полученной прибыли.",
        "Decision→entry в текущем benchmark — 4 часа, не сразу после закрытия. Labels/replay согласованы между собой, но это иной deployment clock, чем исполнение сразу после закрытия.",
        "Модель исполнения не округляет quantity/price по историческим grids, не проверяет min notional, не моделирует partial fill, reject, глубину/участие, latency и рыночное воздействие.",
        "Публичные текущие contract rules проверены только диагностически; они не являются исторической спецификацией августа–сентября или account fee evidence.",
        "Normal/slippage 2 bps и stress 8 bps — assumptions, не измеренные из стакана данного периода. Полноценная квалификация требует исторического L2/trades/mark с event/receive-time и общего IOC/liquidity adapter.",
        "DD по 4h trade-open NAV не равен непрерывной просадке по mark; budgets 10/20/30% — research gates, не stop-loss и не real-money authorization.", "",
        "## Диагностика quantity", "", "| Модель | Fills | Off current grid | Below current min | Decision→entry |", "|---|---:|---:|---:|---:|"]
    for policy, row in feasibility.items():
        lines.append(f"| {policy} | {row['reference_fills']} | {row['off_current_quantity_grid']} | {row['below_current_minimum']} | {row['decision_to_entry_delay_ms']} ms |")
    lines += ["", "## Вывод и следующие действия", "",
        "Сначала отделить источник движения от модели исполнения. Даже подтверждённый скачок остаётся концентрированным единичным событием; исходные отрицательные audit-LCB не изменяются.",
        "Не удалять это окно и не перенастраивать модель на нём. Следующий engineering change — общий исторический IOC-adapter с quantization, latency, bounded depth, reject/partial fill, signed funding и observed mark. Запуск финансового пересчёта после отдельной регистрации сценариев, не автоматически из этой проверки.",
        f"Публичных запросов: {requests}; retries=0; credentials/orders/WS/runtime=0; fits/replays=0. Все старые модели, ledger, hashes и risk authority сохранены.", "",
        "Документация: " + ", ".join(f"[официальный источник {i+1}]({url})" for i, url in enumerate(DOCS))]
    with (folder / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    receipts = {str(p.relative_to(folder)).replace("\\", "/"): pin(p) for p in folder.rglob("*") if p.is_file() and "mpl_cache" not in p.parts}
    write(folder / "evidence_receipt.json", {"files": receipts, "source": pin(Path(__file__)), "safety": SAFETY})
    print(f"price audit done: Bitget={evidence['same_venue_refetch_confirmed_assets']}/8; Binance={evidence['independent_venue_confirmed_assets']}/8; GETs={requests}; execution qualified=false; fits/replays=0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("prepare", "local", "fetch", "compare"))
    parser.add_argument("--folder", type=Path, default=DEFAULT)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.step == "fetch":
        fetch(args.folder, args.index)
    else:
        {"prepare": prepare, "local": local, "compare": compare}[args.step](args.folder)
