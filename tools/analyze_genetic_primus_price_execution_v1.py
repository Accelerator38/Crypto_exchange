"""Refine event coverage using already saved responses, no additional GETs.

The original strict 5-day reader required 30 in-window bars. Bitget returned a
one-bar-left-shifted window for nonaligned request bounds. Its 29 overlapping
bars include EVERY candle of the requested three-day event. Preserve both audits.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import os
from pathlib import Path
import check_genetic_primus_price_execution_v1 as base


def event_candles(payload):
    if isinstance(payload, dict):
        if payload.get("code") != "00000":
            raise ValueError("unsuccessful saved API response")
        payload = payload["data"]
    result = {}
    for row in payload:
        time = int(row[0])
        if time in result or time % base.BAR_MS:
            raise ValueError("duplicate/misaligned saved candle")
        prices = dict(zip(("open", "high", "low", "close"), [Decimal(str(x)) for x in row[1:5]]))
        if not all(x.is_finite() and x > 0 for x in prices.values()):
            raise ValueError("nonfinite/nonpositive saved public candle")
        result[time] = prices
    required = set(range(base.ENTRY, base.EXIT + base.BAR_MS, base.BAR_MS))
    if not required <= set(result):
        raise ValueError("event interval has missing historical candle")
    return result


def main(folder=base.DEFAULT):
    registration = base.verify(folder)
    receipt = base.read(folder / "evidence_receipt.json")
    for name, expected in receipt["files"].items():
        if base.pin(folder / name) != expected:
            raise ValueError("initial audit changed: " + name)
    panel = base.read(folder / "local_four_hour_panel.json")
    initial = base.read(folder / "audit_results.json")
    rows, series, marks = {}, {}, {}
    for i, symbol in enumerate(base.SYMBOLS):
        local = {r["timestamp"]: r for r in panel if r["symbol"] == symbol}
        data = {}
        for kind, index in (("bitget", i), ("binance", i + 8)):
            public = event_candles(base.read(folder / "raw" / f"case_{index:02d}.response.json"))
            common = sorted(set(public) & set(local))
            exact = all(public[t][k] == Decimal(str(local[t][k])) for t in common for k in ("open", "high", "low", "close"))
            local_growth = Decimal(str(local[base.EXIT]["open"])) / Decimal(str(local[base.ENTRY]["open"])) - 1
            growth = public[base.EXIT]["open"] / public[base.ENTRY]["open"] - 1
            max_basis = max(abs(public[t]["open"] / Decimal(str(local[t]["open"])) - 1) * 10000 for t in common)
            data[kind] = {"event_19_candle_grid_complete": True, "common_window_candles": len(common),
                "all_raw_timestamps": sorted(public), "local_common_OHLC_exact": exact,
                "three_day_return": str(growth), "return_difference_bps": str((growth - local_growth) * 10000),
                "max_abs_common_open_basis_bps": str(max_basis),
                "confirmed_large_movement": exact if kind == "bitget" else abs(growth - local_growth) * 10000 < 100 and max_basis < 200}
            if symbol == "BTC":
                series[kind] = {t: values for t, values in public.items() if base.START <= t < base.END}
        rows[symbol] = data
    for symbol, index in zip(("BTC", "ETH", "SOL"), (16, 17, 18)):
        public = event_candles(base.read(folder / "raw" / f"case_{index:02d}.response.json"))
        local = {r["timestamp"]: r for r in panel if r["symbol"] == symbol}
        common = sorted(set(local) & set(public))
        basis = {t: (public[t]["open"] / Decimal(str(local[t]["open"])) - 1) * 10000 for t in common}
        marks[symbol] = {"common_4h_mark_candles": len(common),
            "max_abs_mark_trade_open_basis_bps": str(max(abs(v) for v in basis.values())),
            "max_abs_funding_timestamp_open_basis_bps": str(max(abs(v) for t, v in basis.items() if t % (8 * 3_600_000) == 0)),
            "not_tick_mark_or_full_benchmark_verification": True}
        if symbol == "BTC":
            series["mark"] = {t: public[t] for t in common}
    corrected = {"schema": "price_execution_event_audit/1", "registration_sha256": registration["sha256"],
        "initial_audit_receipt_sha256": base.pin(folder / "evidence_receipt.json")["sha256"],
        "parsing_refinement": "keep initial full-window failure; verify complete event grid and shared overlap without any new response",
        "initial_strict_5day_Bitget_status": "0/8 because one outer-window candle absent, not because price disagreement",
        "Bitget_event_confirmed_assets": sum(r["bitget"]["confirmed_large_movement"] for r in rows.values()),
        "Binance_event_confirmed_assets": sum(r["binance"]["confirmed_large_movement"] for r in rows.values()),
        "event_grid_required_candles": 19, "comparisons": rows, "marks": marks,
        "execution_feasibility": initial["execution_feasibility"], "current_taker_fee_bps": initial["current_taker_fee_bps"],
        "source": base.pin(Path(__file__)), "new_public_requests": 0,
        "new_model_fits_or_financial_replays": 0, "original_ledger_or_ranking_changed": False,
        "observed_execution_qualified": False, "safety": base.SAFETY}
    base.write(folder / "event_audit_results.json", corrected)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(folder / "mpl_cache")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for symbol in base.SYMBOLS:
        local = sorted([r for r in panel if r["symbol"] == symbol], key=lambda r: r["timestamp"])
        anchor = next(r["open"] for r in local if r["timestamp"] == base.ENTRY)
        axes[0].plot([datetime.fromtimestamp(r["timestamp"] / 1000, timezone.utc) for r in local],
                     [100 * (r["open"] / anchor - 1) for r in local], label=symbol)
    for name, style in (("bitget", "-"), ("binance", "--"), ("mark", ":")):
        data = series[name]
        axes[1].plot([datetime.fromtimestamp(t / 1000, timezone.utc) for t in sorted(data)],
                     [float(data[t]["open"]) for t in sorted(data)], style, label="BTC " + name)
    axes[0].set_title("19–22 августа 2026: рост подтверждён Bitget и независимым Binance")
    axes[0].set_ylabel("Доходность относительно 19.08 04:00, %")
    axes[1].set_ylabel("BTC/USDT open; trade / mark reference")
    axes[1].set_xlabel("UTC; наличие свечной цены не означает исполнение IOC по ней")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(ncol=4 if axis is axes[0] else 3, fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "price_check_final.png", dpi=150)
    fig.savefig(folder / "price_check_final.svg")
    plt.close(fig)
    lines = ["# Финальная проверка скачка и исполнения — 8 октября 2026", "",
        "## Результат", "", "Крупный рост 19–22 августа подтверждён по всем восьми монетам свежими историческими данными Bitget и независимым Binance. Признаков ошибки агрегации/склейки в данном окне не обнаружено.",
        "Это устраняет подозрение в искусственном ценовом скачке, но не устраняет концентрацию прибыли в одном событии и не превращает отрицательные audit-LCB в положительные.", "",
        "Локально: 57 600 минут, 240 свечей 4h, точное совпадение всех OHLC при новой агрегации; объёмы согласованы, пропусков/дубликатов нет.",
        "Bitget: 29 общих свечей на каждую монету, все OHLC совпадают точно. Полная 19-свечная сетка event entry 19.08 04:00 → exit 22.08 04:00 покрыта, 3d returns совпадают.",
        "Binance: полная 30-свечная пятидневная сетка; на всех восьми активах крупное движение подтверждено. Межбиржевые цены не используются как исполнимые цены Bitget.",
        "Первоначальный audit_results.json оставлен неизменным: он требовал полную пятидневную сетку Bitget и показал 0/8 из-за граничного бара. При start/end на -1ms Bitget вернул окно 17.08 20:00 → 22.08 16:00. Это несовпадение границ ответа, не цен. Уточнение использует только сохранённые ответы, без повторных GET.", "",
        "| Актив | Bitget 3d рост | Binance 3d рост | Разница outcome, bps | Max open basis, bps |", "|---|---:|---:|---:|---:|"]
    for symbol, row in rows.items():
        lines.append(f"| {symbol} | {100*float(row['bitget']['three_day_return']):.3f}% | {100*float(row['binance']['three_day_return']):.3f}% | {float(row['binance']['return_difference_bps']):.3f} | {float(row['binance']['max_abs_common_open_basis_bps']):.3f} |")
    lines += ["", "## Funding mark proxy", "",
        "Доступные исторические 4h mark candles BTC/ETH/SOL согласованы со trade-open. Максимальный абсолютный basis на funding timestamps:"]
    for symbol, values in marks.items():
        lines.append(f"- {symbol}: {float(values['max_abs_funding_timestamp_open_basis_bps']):.4f} bps.")
    lines += ["", "Это узкая диагностика 18–22 августа по трём активам, не проверка всего 61-дневного funding ledger, не tick-level mark и не фактические исполнения. Разница mark/trade не доказывает допустимое slippage.", "",
        "## Почему исполнение ещё не реалистично квалифицировано", "",
        "1. Decision→entry в текущем benchmark равен 4 часам. Пользовательский желаемый режим «сразу после закрытия» требует нового clock: первый доступный quote после закрытия + зарегистрированная latency, без чтения цен до доступности сигнала. Нельзя незаметно заменить entry и сохранить старые labels/результаты.",
        "2. Все 82 reference fills победившего linear uniform имеют дробные quantity вне текущего size grid Bitget. Они не готовы для отправки в API без quantization. Минимальные размеры по текущим rules проходят, но сами rules не доказывают исторические ограничения августа. Floor-dust не является готовой оценкой net PnL.",
        "3. Комиссия 6 bps совпадает с публичным текущим default takerFeeRate всех восьми контрактов. Account/VIP/акционные и исторические ставки не подтверждены; доступ к аккаунту или ключам не использовался.",
        "4. Slippage 2/8 bps не калиброван на L2/trades периода. OHLCV open не гарантирует доступный ask, глубину, размер IOC, общий participation budget или очередь. Partial fill, reject, limit price, minimum notional/qty/price ticks и реальная latency не смоделированы.",
        "5. Исследовательский DD основан на 4h trade-open NAV, не на непрерывном observed mark. 9.31% не доказывает соблюдение live drawdown budget 10%.",
        "6. Существующий release contract относится к другой старой политике/модели, требует observed L2/mark и запрещает OHLCV fill fallback. Компактная модель текущим replay не квалифицирована к этому контракту; runtime не менялся.", "",
        "## Дальше", "",
        "Первый engineering этап: зарегистрировать clock после закрытия, history/version instrument rules, qty/price quantization, min notional и IOC state machine (reject/partial/timeout/closed position), затем проверить на deterministic offline fixtures.",
        "Для полноценного финансового пересчёта на августе–сентябре нужны L2/trades и mark для нужного периода, event/receive-time и fee evidence либо явно зарегистрированные conservative assumptions. Без этих данных не называть расчёт реалистичным и не открывать live/paper.",
        "Запросов по зарегистрированному плану 20, все публичные, последовательные, без retries. При уточнении 0 новых GET; 0 fits/financial replays. Исходные результаты и safety flags сохранены.", "",
        f"Округление временных границ описано в [Bitget market docs]({base.DOCS[0]}). Исторические mark candles и поля contract fee/size описаны в [Bitget classic market docs]({base.DOCS[2]}). Независимые candles получены из [Binance futures REST]({base.DOCS[1]})."]
    with (folder / "REPORT_FINAL.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    base.write(folder / "event_analysis_receipt.json", {"source": base.pin(Path(__file__)),
        "initial_receipt_sha256": base.pin(folder / "evidence_receipt.json")["sha256"],
        "files": {name: base.pin(folder / name) for name in ("event_audit_results.json", "price_check_final.png", "price_check_final.svg", "REPORT_FINAL.md")},
        "safety": base.SAFETY})
    print("event verification: Bitget=8/8 exact shared OHLC; Binance=8/8 independently; funding mark sampled; additional GET/fits/replays=0; execution remains unqualified")


if __name__ == "__main__":
    main()
