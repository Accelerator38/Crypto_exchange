from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_panteon_flash_profitability_matrix import _extract_metrics


BEST_RUN = (
    ROOT
    / "Results"
    / "PanteonFlashTargetedLcbDenyRegimePullbackTrxFull2022_2026_20260525"
    / "RETRODATE_MARKET"
    / "2026-05-25_07-12-52_retrodate_market_v2"
)
SYMBOL_GUARD_RUN = (
    ROOT
    / "Results"
    / "PanteonFlashPreLiveSymbolGuard_Full2022_2026_20260524"
    / "RETRODATE_MARKET"
    / "2026-05-24_00-20-33_retrodate_market_v2"
)
ITERATION_METRICS = (
    ROOT
    / "Reports"
    / "PanteonFlashFinalTradingReadiness_20260523"
    / "panteon_flash_iteration_metrics.json"
)
OUT = ROOT / "Reports" / "PanteonFlashPreLive_20260524"
CHARTS = OUT / "charts"


COLORS = {
    "blue": (42, 95, 140),
    "green": (55, 128, 86),
    "red": (176, 74, 74),
    "orange": (214, 138, 55),
    "gray": (100, 110, 120),
    "light": (245, 247, 250),
    "grid": (214, 221, 229),
    "text": (30, 36, 42),
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 2) -> str:
    if isinstance(value, bool):
        return "да" if value else "нет"
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _bar_chart(
    filename: str,
    title: str,
    rows: list[tuple[str, float]],
    *,
    unit: str = "%",
    width: int = 1400,
    height: int = 820,
    colors: list[tuple[int, int, int]] | None = None,
) -> Path:
    CHARTS.mkdir(parents=True, exist_ok=True)
    path = CHARTS / filename
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = _font(34, True)
    label_font = _font(22)
    small_font = _font(18)
    draw.text((55, 35), title, font=title_font, fill=COLORS["text"])
    left, right, top, bottom = 120, width - 60, 120, height - 140
    values = [value for _, value in rows]
    max_abs = max(1.0, max(abs(value) for value in values))
    min_value = min(0.0, min(values))
    max_value = max(0.0, max(values))
    span = max(1.0, max_value - min_value)

    for i in range(6):
        y = top + (bottom - top) * i / 5
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)
        value = max_value - span * i / 5
        draw.text((20, y - 12), f"{value:.1f}{unit}", font=small_font, fill=COLORS["gray"])

    zero_y = bottom - (0.0 - min_value) / span * (bottom - top)
    draw.line((left, zero_y, right, zero_y), fill=(90, 100, 110), width=2)
    gap = 22
    bar_w = max(24, int((right - left - gap * (len(rows) + 1)) / max(1, len(rows))))
    palette = colors or [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["red"]]
    for idx, (label, value) in enumerate(rows):
        x0 = left + gap + idx * (bar_w + gap)
        x1 = x0 + bar_w
        y = bottom - (value - min_value) / span * (bottom - top)
        y0, y1 = (y, zero_y) if value >= 0 else (zero_y, y)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=7, fill=palette[idx % len(palette)])
        draw.text((x0, min(y0, y1) - 28), f"{value:.2f}{unit}", font=small_font, fill=COLORS["text"])
        words = label.split()
        y_label = bottom + 18
        if len(words) > 1:
            for word in words:
                draw.text((x0, y_label), word, font=small_font, fill=COLORS["text"])
                y_label += 22
        else:
            draw.text((x0, y_label), label, font=small_font, fill=COLORS["text"])
    img.save(path)
    return path


def _horizontal_bar_chart(
    filename: str,
    title: str,
    rows: list[tuple[str, float]],
    *,
    unit: str = " USD",
    width: int = 1500,
    height: int = 900,
) -> Path:
    CHARTS.mkdir(parents=True, exist_ok=True)
    path = CHARTS / filename
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = _font(34, True)
    label_font = _font(20)
    small_font = _font(18)
    draw.text((55, 35), title, font=title_font, fill=COLORS["text"])
    left, right, top = 360, width - 70, 115
    row_h = max(34, int((height - top - 60) / max(1, len(rows))))
    max_abs = max(1.0, max(abs(v) for _, v in rows))
    zero_x = left + (right - left) / 2
    draw.line((zero_x, top - 10, zero_x, height - 55), fill=COLORS["grid"], width=2)
    for idx, (label, value) in enumerate(rows):
        y = top + idx * row_h
        draw.text((35, y + 4), label[:34], font=label_font, fill=COLORS["text"])
        length = abs(value) / max_abs * ((right - left) / 2 - 20)
        if value >= 0:
            x0, x1, color = zero_x, zero_x + length, COLORS["green"]
        else:
            x0, x1, color = zero_x - length, zero_x, COLORS["red"]
        draw.rounded_rectangle((x0, y + 4, x1, y + row_h - 7), radius=6, fill=color)
        value_text = f"{value:.1f}{unit}"
        text_w = draw.textlength(value_text, font=small_font)
        if value >= 0:
            text_x = min(x1 + 8, width - text_w - 24)
        else:
            text_x = max(x0 - text_w - 10, left - text_w - 16)
        draw.text((text_x, y + 5), value_text, font=small_font, fill=COLORS["text"])
    img.save(path)
    return path


def _symbol_pnl(run: Path) -> dict[str, dict[str, float]]:
    data = _load_json(run / "flash_attribution_summary.json")
    out: dict[str, dict[str, float]] = defaultdict(lambda: {"pnl": 0.0, "closed": 0, "signals": 0})
    for row in data["rows"]:
        bucket = out[str(row["symbol"])]
        bucket["pnl"] += float(row.get("realized_pnl_usd") or 0.0)
        bucket["closed"] += int(row.get("closed_trades") or 0)
        bucket["signals"] += int(row.get("executable_selected_signals") or 0)
    return dict(out)


def _regime_rows(run: Path) -> list[tuple[str, float]]:
    data = _load_json(run / "walk_forward_report.json")
    return [
        (regime, float(row.get("net_pnl") or 0.0))
        for regime, row in data["by_regime"].items()
    ]


def _actor_rows(run: Path, limit: int = 12) -> list[tuple[str, float]]:
    data = _load_json(run / "flash_attribution_summary.json")
    rows = [
        (key, float(value.get("realized_pnl_usd") or 0.0))
        for key, value in data["actor_summary"].items()
    ]
    return sorted(rows, key=lambda item: item[1])[:limit // 2] + sorted(rows, key=lambda item: item[1], reverse=True)[:limit // 2]


def _symbol_guard_diagnostics() -> dict[str, Any]:
    path = SYMBOL_GUARD_RUN / "causal_entry_decisions.jsonl"
    symbol_bars: Counter[str] = Counter()
    reject_symbols: Counter[str] = Counter()
    reject_actor: Counter[str] = Counter()
    degraded_nonempty = 0
    bars = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            bars += 1
            row = json.loads(line)
            degraded = row.get("flash_degraded_open_symbols") or []
            if degraded:
                degraded_nonempty += 1
                for symbol in degraded:
                    symbol_bars[str(symbol)] += 1
            for decision in row.get("flash_decisions") or []:
                counts = decision.get("candidate_rejection_counts") or {}
                if counts.get("flash_symbol_degraded"):
                    symbol = str(decision.get("symbol") or "")
                    reject_symbols[symbol] += int(counts.get("flash_symbol_degraded") or 0)
                    for cand in decision.get("top_rejected_candidates") or []:
                        if cand.get("reason") == "flash_symbol_degraded":
                            key = f"{cand.get('actor_key')}|{symbol}|{cand.get('action')}"
                            reject_actor[key] += 1
    return {
        "bars": bars,
        "degraded_nonempty_bars": degraded_nonempty,
        "degraded_symbol_bars": symbol_bars.most_common(),
        "rejected_candidates_by_symbol": reject_symbols.most_common(),
        "top_rejected_actor_symbol_actions": reject_actor.most_common(20),
    }


def _config_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, dict):
        return len(value)
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return 1 if str(value).strip() else 0


def _live_profile_summary(run_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "execution_leader": "Panteon_Flash",
        "run_profile": "Round3 Deny8 EntryRegime",
        "flash_enabled": bool(run_summary.get("flash_enabled")),
        "flash_min_score_to_trade": run_summary.get("flash_min_score_to_trade"),
        "flash_min_closed_trades_to_trade": run_summary.get("flash_min_closed_trades_to_trade"),
        "flash_shadow_confirmation_enabled": bool(run_summary.get("flash_shadow_confirmation_enabled")),
        "flash_symbol_shadow_confirmation_enabled": bool(run_summary.get("flash_symbol_shadow_confirmation_enabled")),
        "flash_shadow_min_closed_trades": run_summary.get("flash_shadow_min_closed_trades"),
        "flash_max_signals_per_actor": run_summary.get("flash_max_signals_per_actor"),
        "max_new_opens_per_bar": run_summary.get("max_new_opens_per_bar"),
        "risk_max_open_positions": run_summary.get("risk_max_open_positions"),
        "flash_degradation_guard_enabled": bool(run_summary.get("flash_degradation_guard_enabled")),
        "flash_degradation_actor_guard_enabled": bool(run_summary.get("flash_degradation_actor_guard_enabled")),
        "flash_degradation_actor_scope": run_summary.get("flash_degradation_actor_scope"),
        "flash_degradation_signal_cooldown_bars": run_summary.get("flash_degradation_signal_cooldown_bars"),
        "flash_degradation_actor_cooldown_bars": run_summary.get("flash_degradation_actor_cooldown_bars"),
        "flash_promotion_manifest_enabled": bool(run_summary.get("flash_promotion_manifest_enabled")),
        "flash_denied_signal_keys_count": _config_count(run_summary.get("flash_denied_signal_keys")),
        "flash_terminal_denied_signal_keys_count": _config_count(run_summary.get("flash_terminal_denied_signal_keys")),
        "flash_portfolio_actor_keys_count": _config_count(run_summary.get("flash_portfolio_actor_keys")),
    }


def build_data() -> dict[str, Any]:
    best_metrics = _extract_metrics(BEST_RUN)
    symbol_metrics = _extract_metrics(SYMBOL_GUARD_RUN)
    component = _load_json(BEST_RUN / "component_benchmark_report.json")
    standalone = _load_json(BEST_RUN / "standalone_vs_flash_selected_report.json")
    attribution = _load_json(BEST_RUN / "flash_attribution_summary.json")
    iterations = _load_json(ITERATION_METRICS)
    run_summary = _load_json(BEST_RUN / "run_summary.json")
    symbol_pnl = _symbol_pnl(BEST_RUN)
    data = {
        "best_run": str(BEST_RUN),
        "symbol_guard_run": str(SYMBOL_GUARD_RUN),
        "live_profile": _live_profile_summary(run_summary),
        "best_metrics": best_metrics,
        "symbol_guard_metrics": symbol_metrics,
        "component_benchmark": component,
        "standalone_vs_flash": standalone,
        "attribution_summary": attribution["summary"],
        "actor_summary": attribution["actor_summary"],
        "symbol_pnl": symbol_pnl,
        "regime_pnl_best": _regime_rows(BEST_RUN),
        "regime_pnl_symbol_guard": _regime_rows(SYMBOL_GUARD_RUN),
        "iteration_metrics": iterations,
        "symbol_guard_diagnostics": _symbol_guard_diagnostics(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report_summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def build_charts(data: dict[str, Any]) -> list[Path]:
    best = data["best_metrics"]
    guard = data["symbol_guard_metrics"]
    charts = [
        _bar_chart(
            "01_full_pnl_vs_component.png",
            "Полный 2022-2026: Flash против лучшего компонента",
            [
                ("Best component", float(best["best_component_pnl_pct"])),
                ("Round3 Deny8", float(best["pnl_pct"])),
                ("Symbol guard", float(guard["pnl_pct"])),
            ],
            colors=[COLORS["gray"], COLORS["green"], COLORS["orange"]],
        ),
        _bar_chart(
            "02_full_quality_gates.png",
            "Качество полного прогона: alpha, drawdown, churn",
            [
                ("Alpha Round3", float(best["panteon_alpha_pct"])),
                ("Alpha symbol guard", float(guard["panteon_alpha_pct"])),
                ("Max DD Round3", -float(best["max_drawdown_pct"])),
                ("Switches/day", float(best["actor_switches_per_day"])),
            ],
            colors=[COLORS["green"], COLORS["orange"], COLORS["red"], COLORS["blue"]],
        ),
    ]
    regime = dict(data["regime_pnl_best"])
    charts.append(
        _bar_chart(
            "03_entry_regime_pnl_usd.png",
            "Round3 Deny8: PnL по режиму входа",
            [(key, regime[key]) for key in ("bearish", "bullish", "neutral", "crash")],
            unit=" USD",
            colors=[COLORS["green"], COLORS["blue"], COLORS["orange"], COLORS["red"]],
        )
    )
    sym = data["symbol_pnl"]
    symbol_rows = sorted(((k, v["pnl"]) for k, v in sym.items()), key=lambda item: item[1])
    charts.append(_horizontal_bar_chart("04_symbol_pnl_usd.png", "Round3 Deny8: вклад символов", symbol_rows, unit=" USD"))
    actor_rows = _actor_rows(BEST_RUN, limit=12)
    charts.append(_horizontal_bar_chart("05_actor_contribution_usd.png", "Round3 Deny8: вклад агентов/игроков", actor_rows, unit=" USD"))
    iterations = [
        row
        for row in data["iteration_metrics"]
        if row["name"] in {
            "base_cap10",
            "generated_selected_deny",
            "round2",
            "shadow_round2only_fail",
            "atom_deny_wrong_base",
            "round2_plus_terminal_atom",
            "round2_open2_fail",
        }
    ]
    charts.append(
        _bar_chart(
            "06_iteration_alpha.png",
            "Итерации 2025/2026 H1: alpha к лучшему компоненту",
            [(f"{row['period']} {row['name']}", float(row["panteon_alpha_pct"])) for row in iterations],
            unit=" п.п.",
            width=1700,
            height=900,
        )
    )
    funnel = data["attribution_summary"]
    charts.append(
        _bar_chart(
            "07_signal_funnel.png",
            "Round3 Deny8: воронка сигналов Flash",
            [
                ("selected", float(funnel["selected_signals"])),
                ("executable", float(funnel["executable_selected_signals"])),
                ("filled", float(funnel["filled_signals"])),
                ("closed", float(funnel["closed_trades"])),
            ],
            unit="",
            colors=[COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["gray"]],
        )
    )
    return charts


def _add_table(doc: Document, headers: list[str], rows: Iterable[Iterable[Any]]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for cell, header in zip(table.rows[0].cells, headers):
        cell.text = str(header)
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)


def _md_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    def cell(value: Any) -> str:
        return str(value).replace("\n", " ").replace("|", "/")

    out = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(cell(value) for value in row) + " |")
    out.append("")
    return out


def _top_actor_summary_rows(data: dict[str, Any], limit: int = 8) -> list[list[Any]]:
    actor_rows = sorted(
        data["actor_summary"].items(),
        key=lambda item: float(item[1].get("realized_pnl_usd") or 0.0),
        reverse=True,
    )
    rows: list[list[Any]] = []
    for key, value in actor_rows[:limit]:
        rows.append([
            key,
            value.get("actor_type", ""),
            value.get("selected_signals", 0),
            value.get("executable_selected_signals", 0),
            value.get("closed_trades", 0),
            f"{float(value.get('realized_pnl_usd') or 0.0):.2f}",
        ])
    return rows


def _save_docx(doc: Document, path: Path) -> Path:
    try:
        doc.save(path)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_expanded{path.suffix}")
        doc.save(fallback)
        return fallback


def build_markdown(data: dict[str, Any], charts: list[Path]) -> Path:
    best = data["best_metrics"]
    guard = data["symbol_guard_metrics"]
    component = data["component_benchmark"]["summary"]
    standalone_report = data["standalone_vs_flash"]
    standalone = standalone_report["summary"]
    attr = data["attribution_summary"]
    diag = data["symbol_guard_diagnostics"]
    live = data["live_profile"]
    md = OUT / "Panteon_Flash_PreLive_Report.md"
    lines = [
        "# Panteon Flash pre-live отчет",
        "",
        "Дата: 2026-05-24.",
        "",
        "## Резюме",
        "",
        "Лучший кандидат для ограниченного pilot/live-shadow: `Round3 Deny8 EntryRegime`.",
        f"Полный 2022-2026 PnL: **{_fmt(best['pnl_pct'])}%**, max DD: **{_fmt(best['max_drawdown_pct'])}%**, alpha к лучшему компоненту: **{_fmt(best['panteon_alpha_pct'])} п.п.**",
        f"Лучший компонент: `{component['best_component_label']}` ({_fmt(component['best_component_pnl_pct'])}%).",
        "",
        "Запуск реальных торгов допустим только как ограниченный пилот: малый капитал, `max_new_opens_per_bar=1`, ежедневный attribution-контроль и kill-switch. Полный production без ограничений не рекомендован.",
        "",
        "## Как читать отчет: что здесь является самим Пантеоном",
        "",
        "В этом отчете самим реально торгующим Пантеоном является агрегат `Round3 Deny8 EntryRegime` / `Panteon_Flash`. "
        "Это не отдельный агент и не отдельный игрок, а контур выбора и исполнения: `FlashAllocator` выбирает лучшего актора по каждому символу, затем execution-слой пропускает выбранные сигналы через лимиты позиций и отправляет их в executor.",
        "",
        f"Именно этот контур дал **{_fmt(best['pnl_pct'])}%** PnL: {attr['selected_signals']} выбранных сигналов -> "
        f"{attr['executable_selected_signals']} executable -> {attr['filled_signals']} filled -> {attr['closed_trades']} closed trades, "
        f"realized PnL **{_fmt(attr['realized_pnl_usd'])} USD**.",
        "",
    ]
    lines.extend(_md_table(
        ["Строка в отчете", "Что это", "Это сам Пантеон?"],
        [
            [
                "`Round3 Deny8 EntryRegime` / `Panteon_Flash`",
                "боевой Flash-контур выбора и исполнения",
                "да, это агрегат реально исполненных сделок Пантеона",
            ],
            [
                f"`{component['best_component_label']}`",
                "лучший отдельный компонент-бенчмарк",
                "нет, это игрок-кандидат; его сделки внутри Flash учитываются как вклад выбранного актора",
            ],
            [
                "`Solo_MomentumScalper`, `LiveOIBreakout`, другие actor rows",
                "агенты или ensemble-игроки, выбранные на отдельных символах",
                "нет как самостоятельная система; да как вклад в сделки, которые исполнил `Panteon_Flash`",
            ],
            [
                "Standalone vs Flash-selected",
                "сравнение полного самостоятельного поведения актора с subset, выбранным Пантеоном",
                "standalone - нет; Flash-selected - часть реальных сделок Пантеона",
            ],
            [
                "`NoTrade`",
                "явное решение не торговать по символу",
                "часть decision layer, но не сделка",
            ],
        ],
    ))
    lines.extend([
        "## Как работают агенты, игроки и Пантеон",
        "",
        "Агент - это источник торговой логики с интерфейсом `act(market) -> dict[symbol, Action]`. Он предлагает действие по монете, но сам не решает, можно ли это действие реально исполнять.",
        "",
        "Игрок - это ансамблевый актор (`actor_type=\"ensemble\"`). `EnsemblePlayer` вызывает несколько агентов, агрегирует их голоса через voting policy, назначает основного contributor и возвращает сигналы. В Flash игрок конкурирует с одиночными агентами как обычный кандидат, а не выбирается глобальным лидером на весь рынок.",
        "",
        "Пантеон в режиме Flash - это decision/execution слой. На каждом баре он собирает real-executable агентов и ensemble-игроков, строит кандидатов, оценивает их по каждому символу, выбирает одного победителя или `NoTrade`, затем передает только прошедшие сигналы в execution.",
        "",
        "Схема решения:",
        "",
        "```text",
        "MarketSnapshot -> agents + ensemble players -> FlashAllocator",
        "    -> one FlashDecision per symbol -> risk/position filters",
        "    -> TradeExecutor -> EventLog + attribution",
        "```",
        "",
        "Фильтры перед исполнением: карантин, наличие статистики, минимум закрытых сделок, PnL-порог, shadow confirmation, degradation/deny keys, score gate, лимит новых открытий на бар и лимит открытых позиций.",
        "",
        "## Профиль реально торгующего контура",
        "",
    ])
    lines.extend(_md_table(
        ["Параметр", "Значение", "Смысл"],
        [
            ["Leader", live["execution_leader"], "имя, под которым сделки проходят как Пантеон"],
            ["Run profile", live["run_profile"], "выбранная pre-live конфигурация"],
            ["Flash enabled", _fmt(live["flash_enabled"]), "используется per-symbol Flash selection"],
            ["Min score", _fmt(live["flash_min_score_to_trade"]), "минимальная оценка кандидата"],
            ["Min closed trades", live["flash_min_closed_trades_to_trade"], "минимальная история для допуска"],
            ["Shadow confirmation", _fmt(live["flash_shadow_confirmation_enabled"]), "проверка через shadow-статистику"],
            ["Symbol shadow confirmation", _fmt(live["flash_symbol_shadow_confirmation_enabled"]), "подтверждение качества по символу"],
            ["Max signals per actor", live["flash_max_signals_per_actor"], "защита от концентрации в одном акторе"],
            ["Max new opens per bar", live["max_new_opens_per_bar"], "боевой лимит новых открытий"],
            ["Risk max open positions", live["risk_max_open_positions"], "общий лимит открытых позиций"],
            ["Degradation guard", _fmt(live["flash_degradation_guard_enabled"]), "временное отключение деградировавших signal keys"],
            ["Actor degradation guard", f"{_fmt(live['flash_degradation_actor_guard_enabled'])}, scope={live['flash_degradation_actor_scope']}", "контроль деградации актора по режиму"],
            ["Deny keys", live["flash_denied_signal_keys_count"], "ручные запреты плохих actor|symbol|action"],
            ["Terminal deny keys", live["flash_terminal_denied_signal_keys_count"], "жесткие запреты, ведущие к NoTrade"],
        ],
    ))
    lines.extend([
        "## Работа Пантеона на полном прогоне",
        "",
        f"Flash обработал {attr['flash_decisions']} решений по символам. Из них {attr['no_trade_decisions']} закончились `NoTrade`, поэтому активная торговля была редкой и отфильтрованной: no-trade share **{_fmt(best['no_trade_decision_share_pct'])}%**.",
        "",
    ])
    lines.extend(_md_table(
        ["Показатель", "Значение"],
        [
            ["Selected signals", attr["selected_signals"]],
            ["Executable selected", attr["executable_selected_signals"]],
            ["Filtered before execution", attr["selected_filtered_before_execution"]],
            ["Filled signals", attr["filled_signals"]],
            ["Blocked signals", attr["blocked_signals"]],
            ["Closed trades", attr["closed_trades"]],
            ["Winning / losing trades", f"{attr['winning_trades']} / {attr['losing_trades']}"],
            ["Win rate", f"{attr['winning_trades'] / max(1, attr['closed_trades']) * 100:.2f}%"],
            ["Realized PnL", f"{attr['realized_pnl_usd']:.2f} USD"],
        ],
    ))
    lines.extend([
        "## Вклад агентов и игроков внутри Пантеона",
        "",
        "Следующая таблица не показывает отдельные самостоятельные торговые системы. Она показывает, какие акторы были выбраны самим `Panteon_Flash` и какой вклад дали в его реальные/ретро-реальные сделки.",
        "",
    ])
    lines.extend(_md_table(
        ["Actor", "Type", "Selected", "Executable", "Closed", "PnL USD"],
        _top_actor_summary_rows(data, limit=8),
    ))
    lines.extend([
        "## Финальный тест symbol guard",
        "",
        f"Новый symbol guard был проверен на полном 2022-2026 прогоне: PnL **{_fmt(guard['pnl_pct'])}%** против **{_fmt(best['pnl_pct'])}%** у Round3.",
        "Итог: symbol guard ухудшил результат на 4.92 п.п. и не включается в live-профиль. Код оставлен отключенным по умолчанию как инструмент будущих экспериментов.",
        f"Деградированные символы появлялись на {diag['degraded_nonempty_bars']} барах; чаще всего: {diag['degraded_symbol_bars'][:5]}.",
        "",
        "## Standalone vs Flash",
        "",
        f"Для Solo_MomentumScalper и LiveOIBreakout standalone-суммарно дал {_fmt(standalone['total_standalone_pnl_pct'])}%, а Flash-selected subset дал {_fmt(standalone['total_flash_selected_pnl_pct'])}%.",
        "Это значит, что Пантеон прибыльнее полного набора компонентов за счет других игроков/фильтров, но selection subset еще требует улучшения.",
        "",
    ])
    lines.extend(_md_table(
        ["Actor", "Standalone", "Flash-selected", "Selected", "Executable", "Closed", "Selection alpha"],
        [
            [
                row["label"],
                f"{row['standalone_pnl_pct']:.2f}%",
                f"{row['flash_selected_pnl_pct']:.2f}%",
                row["flash_selected_signals"],
                row["flash_executable_selected_signals"],
                row["flash_closed_trades"],
                f"{row['selection_alpha_pct']:.2f} п.п.",
            ]
            for row in standalone_report["actors"]
        ],
    ))
    lines.extend([
        "## Визуализации",
        "",
    ]
    )
    for chart in charts:
        lines.append(f"![{chart.stem}]({chart.as_posix()})")
        lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")
    return md


def build_docx(data: dict[str, Any], charts: list[Path]) -> Path:
    best = data["best_metrics"]
    guard = data["symbol_guard_metrics"]
    component = data["component_benchmark"]["summary"]
    standalone = data["standalone_vs_flash"]
    attr = data["attribution_summary"]
    live = data["live_profile"]
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.7)
        section.bottom_margin = Inches(0.7)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
    for style in ("Normal", "Body Text"):
        if style in doc.styles:
            doc.styles[style].font.name = "Arial"
            doc.styles[style].font.size = Pt(10)

    title = doc.add_paragraph()
    title.style = "Title"
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("Panteon Flash: pre-live отчет по агентам, игрокам и Пантеону")
    subtitle = doc.add_paragraph("Дата: 2026-05-24 | Ветка анализа: Panteon_Flash")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("Executive Summary", level=1)
    doc.add_paragraph(
        "Лучший кандидат для ограниченного pilot/live-shadow: Round3 Deny8 EntryRegime. "
        "Новая проверка symbol guard на полном 2022-2026 ухудшила результат, поэтому в live-профиль он не включается."
    )
    doc.add_paragraph(
        "Реальные торги можно начинать только как ограниченный пилот: малый капитал, max_new_opens_per_bar=1, "
        "ежедневный attribution-контроль и kill-switch. Полный production без ограничений пока не рекомендован."
    )

    doc.add_heading("Как читать отчет: что здесь является самим Пантеоном", level=1)
    doc.add_paragraph(
        "Самим реально торгующим Пантеоном в этом отчете является агрегат Round3 Deny8 EntryRegime / Panteon_Flash. "
        "Это не отдельный агент и не отдельный игрок, а контур выбора и исполнения: FlashAllocator выбирает лучшего актора "
        "по каждому символу, после чего execution-слой применяет лимиты позиций и отправляет прошедшие сигналы в executor."
    )
    doc.add_paragraph(
        f"Именно этот контур дал {best['pnl_pct']:.2f}% PnL: {attr['selected_signals']} выбранных сигналов -> "
        f"{attr['executable_selected_signals']} executable -> {attr['filled_signals']} filled -> {attr['closed_trades']} closed trades, "
        f"realized PnL {attr['realized_pnl_usd']:.2f} USD."
    )
    _add_table(
        doc,
        ["Строка в отчете", "Что это", "Это сам Пантеон?"],
        [
            [
                "Round3 Deny8 EntryRegime / Panteon_Flash",
                "боевой Flash-контур выбора и исполнения",
                "да, агрегат реально исполненных сделок Пантеона",
            ],
            [
                component["best_component_label"],
                "лучший отдельный компонент-бенчмарк",
                "нет, это игрок-кандидат; внутри Flash он учитывается как выбранный актор",
            ],
            [
                "Solo_MomentumScalper, LiveOIBreakout, actor rows",
                "агенты или ensemble-игроки, выбранные на отдельных символах",
                "нет как самостоятельная система; да как вклад в сделки Panteon_Flash",
            ],
            [
                "Standalone vs Flash-selected",
                "сравнение самостоятельного актора с subset, выбранным Пантеоном",
                "standalone - нет; Flash-selected - часть реальных сделок Пантеона",
            ],
            [
                "NoTrade",
                "явное решение не торговать по символу",
                "часть decision layer, но не сделка",
            ],
        ],
    )

    doc.add_heading("Как работают агенты, игроки и Пантеон", level=1)
    for text in [
        "Агент - источник торговой логики с интерфейсом act(market) -> dict[symbol, Action]. Агент предлагает действие, но не решает сам, можно ли его исполнять реальными деньгами.",
        "Игрок - ensemble actor. EnsemblePlayer вызывает несколько агентов, агрегирует голоса через voting policy, назначает основного contributor и возвращает сигналы. В Flash игрок конкурирует с одиночными агентами как кандидат по конкретному символу.",
        "Пантеон Flash - decision/execution слой. Он собирает real-executable агентов и ensemble-игроков, ранжирует кандидатов по каждому символу, выбирает одного победителя или NoTrade и передает только прошедшие сигналы в execution.",
        "Фильтры перед исполнением: карантин, наличие статистики, минимум закрытых сделок, PnL-порог, shadow confirmation, degradation/deny keys, score gate, max_new_opens_per_bar и risk_max_open_positions.",
    ]:
        doc.add_paragraph(text, style="List Bullet")

    doc.add_heading("Профиль реально торгующего контура", level=1)
    _add_table(
        doc,
        ["Параметр", "Значение", "Смысл"],
        [
            ["Leader", live["execution_leader"], "имя, под которым сделки проходят как Пантеон"],
            ["Run profile", live["run_profile"], "выбранная pre-live конфигурация"],
            ["Flash enabled", _fmt(live["flash_enabled"]), "per-symbol Flash selection"],
            ["Min score", _fmt(live["flash_min_score_to_trade"]), "минимальная оценка кандидата"],
            ["Min closed trades", live["flash_min_closed_trades_to_trade"], "минимальная история для допуска"],
            ["Shadow confirmation", _fmt(live["flash_shadow_confirmation_enabled"]), "проверка через shadow-статистику"],
            ["Symbol shadow confirmation", _fmt(live["flash_symbol_shadow_confirmation_enabled"]), "подтверждение качества по символу"],
            ["Max signals per actor", live["flash_max_signals_per_actor"], "защита от концентрации в одном акторе"],
            ["Max new opens per bar", live["max_new_opens_per_bar"], "боевой лимит новых открытий"],
            ["Risk max open positions", live["risk_max_open_positions"], "общий лимит открытых позиций"],
            ["Degradation guard", _fmt(live["flash_degradation_guard_enabled"]), "временное отключение деградировавших signal keys"],
            [
                "Actor degradation guard",
                f"{_fmt(live['flash_degradation_actor_guard_enabled'])}, scope={live['flash_degradation_actor_scope']}",
                "контроль деградации актора по режиму",
            ],
            ["Deny keys", live["flash_denied_signal_keys_count"], "ручные запреты плохих actor|symbol|action"],
            ["Terminal deny keys", live["flash_terminal_denied_signal_keys_count"], "жесткие запреты, ведущие к NoTrade"],
        ],
    )

    doc.add_heading("Ключевые метрики", level=1)
    _add_table(
        doc,
        ["Версия", "PnL", "Max DD", "Best component", "Alpha", "LCB dominance", "Churn/day", "Exec signals"],
        [
            [
                "Round3 Deny8 EntryRegime",
                f"{_fmt(best['pnl_pct'])}%",
                f"{_fmt(best['max_drawdown_pct'])}%",
                f"{_fmt(best['best_component_pnl_pct'])}%",
                f"{_fmt(best['panteon_alpha_pct'])} п.п.",
                _fmt(best["panteon_lcb_dominance"]),
                _fmt(best["actor_switches_per_day"]),
                int(best["flash_executable_selected_signals"]),
            ],
            [
                "Symbol guard test",
                f"{_fmt(guard['pnl_pct'])}%",
                f"{_fmt(guard['max_drawdown_pct'])}%",
                f"{_fmt(guard['best_component_pnl_pct'])}%",
                f"{_fmt(guard['panteon_alpha_pct'])} п.п.",
                _fmt(guard["panteon_lcb_dominance"]),
                _fmt(guard["actor_switches_per_day"]),
                int(guard["flash_executable_selected_signals"]),
            ],
        ],
    )

    doc.add_heading("Почему выбрана Round3 Deny8 EntryRegime", level=1)
    for text in [
        f"Пантеон дал {best['pnl_pct']:.2f}% против {best['best_component_pnl_pct']:.2f}% у лучшего компонента.",
        f"PanteonAdvantage = {best['panteon_advantage']:.2f}; dominance equity ratio = {best['dominance_equity_ratio']:.3f}.",
        "Churn-budget пройден: 1.16 switches/day при лимите 24.",
        "Regime-floor формально пройден; crash помечен как insufficient sample, потому что закрытых сделок только 14.",
        "Сохранение жесткого open-limit оказалось полезным: open2 ухудшал OOS, значит текущий лимит защищает от плохой экспозиции.",
    ]:
        doc.add_paragraph(text, style="List Bullet")

    doc.add_heading("Работа Пантеона", level=1)
    doc.add_paragraph(
        "Эта таблица описывает не отдельного актора, а весь Panteon_Flash execution path: от per-symbol решений до исполненных и закрытых сделок."
    )
    _add_table(
        doc,
        ["Показатель", "Значение"],
        [
            ["Flash decisions", attr["flash_decisions"]],
            ["NoTrade decisions", attr["no_trade_decisions"]],
            ["Selected signals", attr["selected_signals"]],
            ["Executable selected", attr["executable_selected_signals"]],
            ["Filled", attr["filled_signals"]],
            ["Blocked", attr["blocked_signals"]],
            ["Closed trades", attr["closed_trades"]],
            ["Winning / losing trades", f"{attr['winning_trades']} / {attr['losing_trades']}"],
            ["Win rate", f"{attr['winning_trades'] / max(1, attr['closed_trades']) * 100:.2f}%"],
            ["Realized PnL USD", f"{attr['realized_pnl_usd']:.2f}"],
            ["Filtered before execution", attr["selected_filtered_before_execution"]],
        ],
    )

    doc.add_heading("Работа игроков и агентов", level=1)
    doc.add_paragraph(
        "Строки ниже - это attribution внутри сделок Пантеона. Они показывают, какой агент или ensemble-игрок был выбран для конкретных символов, но не являются отдельной строкой production PnL."
    )
    actor_rows = sorted(
        data["actor_summary"].items(),
        key=lambda item: float(item[1].get("realized_pnl_usd") or 0.0),
        reverse=True,
    )
    _add_table(
        doc,
        ["Actor", "Type", "Selected", "Executable", "Closed", "PnL USD"],
        [
            [
                key,
                value.get("actor_type", ""),
                value.get("selected_signals", 0),
                value.get("executable_selected_signals", 0),
                value.get("closed_trades", 0),
                f"{float(value.get('realized_pnl_usd') or 0.0):.2f}",
            ]
            for key, value in actor_rows[:8]
        ],
    )
    doc.add_paragraph(
        "Главный положительный вклад дали ensemble-игроки: Antonius_conservative, Solo_MomentumScalper и Optimal_StaticRotator. "
        "LiveOIBreakout как agent полезен точечно: standalone 9.28%, а Flash-selected subset 11.29%, то есть Пантеон выбирал его лучше, чем полный standalone. "
        "Сам Panteon_Flash находится не в этой actor-таблице, а в агрегатной строке Round3 Deny8 EntryRegime."
    )

    doc.add_heading("Standalone vs Flash-selected", level=1)
    _add_table(
        doc,
        ["Actor", "Standalone", "Flash-selected", "Capture", "Selection alpha"],
        [
            [
                row["label"],
                f"{row['standalone_pnl_pct']:.2f}%",
                f"{row['flash_selected_pnl_pct']:.2f}%",
                f"{row['flash_capture_ratio_pct']:.2f}%",
                f"{row['selection_alpha_pct']:.2f} п.п.",
            ]
            for row in standalone["actors"]
        ],
    )
    doc.add_paragraph(
        "Solo_MomentumScalper все еще теряет существенную часть standalone edge при selection: это основной резерв улучшения. "
        "Приоритет следующих работ: LCB/PNL gate именно для selected subset и per-key дифф перед добавлением deny-правил."
    )

    doc.add_heading("Почему symbol guard не включен", level=1)
    diag = data["symbol_guard_diagnostics"]
    doc.add_paragraph(
        f"Symbol guard ухудшил полный PnL с {best['pnl_pct']:.2f}% до {guard['pnl_pct']:.2f}% и немного увеличил max DD "
        f"с {best['max_drawdown_pct']:.2f}% до {guard['max_drawdown_pct']:.2f}%."
    )
    _add_table(
        doc,
        ["Символ", "Баров в cooldown"],
        [[symbol, bars] for symbol, bars in diag["degraded_symbol_bars"][:8]],
    )
    doc.add_paragraph(
        "Причина ухудшения: coarse symbol-level cooldown отключал не только плохие ключи, но и полезные замещающие сделки. "
        "Отключение непредсказуемых монет нужно делать не по символу целиком, а через actor|symbol|action и selected-subset PnL LCB."
    )

    doc.add_heading("Визуализации", level=1)
    for chart in charts:
        doc.add_picture(str(chart), width=Inches(6.8))
        caption = doc.add_paragraph(chart.stem.replace("_", " "))
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("Production readiness", level=1)
    for text in [
        "Разрешен только ограниченный pilot/live-shadow; полный production без ограничений не запускать.",
        "Начальный real capital не более 5-10% планового лимита.",
        "max_new_opens_per_bar=1 и risk_max_open_positions=8 оставить без расширения.",
        "Kill-switch: остановка real execution при rolling PanteonAdvantage < 0 или equity drawdown > 1.5-2.0%.",
        "Daily CI/report gates: panteon_lcb_dominance, churn_budget, ensemble_lift, regime_floor, standalone_vs_flash_selected.",
    ]:
        doc.add_paragraph(text, style="List Bullet")

    doc.add_heading("Артефакты", level=1)
    for path in [BEST_RUN, SYMBOL_GUARD_RUN, OUT / "report_summary.json"]:
        doc.add_paragraph(str(path), style="List Bullet")

    out = OUT / "Panteon_Flash_PreLive_Report.docx"
    return _save_docx(doc, out)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    data = build_data()
    charts = build_charts(data)
    md = build_markdown(data, charts)
    docx = build_docx(data, charts)
    print(f"markdown={md}")
    print(f"docx={docx}")
    print(f"summary={OUT / 'report_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
