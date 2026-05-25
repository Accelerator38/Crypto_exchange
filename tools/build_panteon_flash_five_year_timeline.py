from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_YEARS = {2022, 2023, 2024, 2025, 2026}
MIN_FULL_BARS = 38_000
MIN_LONG_RUN_BARS = 30_000
FULL_WINDOW_BARS = 38_375
RUN_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})")


@dataclass(frozen=True)
class RunArtifacts:
    run_summary: Path
    output_dir: Path
    label: str
    run_datetime: datetime


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _parse_run_datetime(path: Path) -> datetime | None:
    for part in reversed(path.parts):
        match = RUN_TS_RE.search(part)
        if match:
            date, hour, minute, second = match.groups()
            return datetime.strptime(f"{date} {hour}:{minute}:{second}", "%Y-%m-%d %H:%M:%S")
    return None


def _experiment_label(path: Path) -> str:
    parts = list(path.parts)
    if "Results" in parts:
        parts = parts[parts.index("Results") + 1 :]
    if "RETRODATE_MARKET" in parts:
        parts = parts[: parts.index("RETRODATE_MARKET")]
    return "/".join(parts) or path.parent.name


def _parse_percent_from_report(text: str, label: str) -> float | None:
    match = re.search(rf"{re.escape(label)}:\s*([-+]?\d+(?:\.\d+)?)%", text)
    return float(match.group(1)) if match else None


def _parse_usd_from_report(text: str, label: str) -> float | None:
    match = re.search(rf"{re.escape(label)}:\s*\$?([-+]?\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def _full_run_artifacts(results_dir: Path) -> tuple[list[RunArtifacts], list[dict[str, Any]]]:
    included: list[RunArtifacts] = []
    excluded: list[dict[str, Any]] = []
    for summary_path in results_dir.rglob("run_summary.json"):
        summary = _load_json(summary_path)
        years = set(summary.get("executed_years") or summary.get("requested_years") or [])
        bars = int(summary.get("bars_processed") or 0)
        first = str(summary.get("first_timestamp") or "")
        last = str(summary.get("last_timestamp") or "")
        output_dir = Path(summary.get("output_dir") or summary_path.parent)
        if not output_dir.exists():
            output_dir = summary_path.parent
        run_datetime = _parse_run_datetime(summary_path)

        is_full = (
            REQUIRED_YEARS.issubset(years)
            and bars >= MIN_FULL_BARS
            and first.startswith("2022-01-01")
            and last.startswith("2026-05-18")
            and run_datetime is not None
        )
        if not is_full:
            excluded.append(
                {
                    "path": str(summary_path),
                    "executed_years": sorted(years),
                    "bars_processed": bars,
                    "first_timestamp": first,
                    "last_timestamp": last,
                    "reason": "not_full_2022_2026",
                }
            )
            continue

        included.append(
            RunArtifacts(
                run_summary=summary_path,
                output_dir=output_dir,
                label=_experiment_label(summary_path),
                run_datetime=run_datetime,
            )
        )
    included.sort(key=lambda item: (item.run_datetime, item.label))
    return included, excluded


def _row_from_artifacts(artifacts: RunArtifacts) -> dict[str, Any]:
    summary = _load_json(artifacts.run_summary)
    output_dir = artifacts.output_dir if artifacts.output_dir.exists() else artifacts.run_summary.parent
    benchmark = _load_json(output_dir / "component_benchmark_report.json")
    attribution = _load_json(output_dir / "flash_attribution_summary.json")
    analysis_path = output_dir / "analysis_report.md"
    analysis_text = analysis_path.read_text(encoding="utf-8", errors="ignore") if analysis_path.exists() else ""

    benchmark_summary = benchmark.get("summary") or {}
    attribution_summary = attribution.get("summary") or {}

    pnl_pct = benchmark_summary.get("panteon_pnl_pct")
    pnl_usd = benchmark_summary.get("panteon_pnl_usd")
    max_dd_pct = _parse_percent_from_report(analysis_text, "Panteon max drawdown")
    realized_max_dd_pct = _parse_percent_from_report(analysis_text, "Panteon realized max drawdown")

    if pnl_pct is None:
        pnl_pct = _parse_percent_from_report(analysis_text, "Panteon owned PnL")
    if pnl_usd is None:
        pnl_usd = attribution_summary.get("realized_pnl_usd")
    if pnl_usd is None:
        pnl_usd = _parse_usd_from_report(analysis_text, "Panteon realized PnL USD")

    best_component_pct = benchmark_summary.get("best_component_pnl_pct")
    best_component_usd = benchmark_summary.get("best_component_pnl_usd")
    alpha_pct = benchmark_summary.get("panteon_alpha_pct")
    beats_best = benchmark_summary.get("panteon_beats_best_component")

    return {
        "run_datetime": artifacts.run_datetime.isoformat(sep=" "),
        "run_date": artifacts.run_datetime.date().isoformat(),
        "version_label": artifacts.label,
        "output_dir": str(output_dir),
        "bars_processed": summary.get("bars_processed"),
        "panteon_pnl_pct": pnl_pct,
        "panteon_pnl_usd": pnl_usd,
        "panteon_max_dd_pct": max_dd_pct,
        "panteon_realized_max_dd_pct": realized_max_dd_pct,
        "selected_signals": attribution_summary.get("selected_signals"),
        "filled_signals": attribution_summary.get("filled_signals"),
        "closed_trades": attribution_summary.get("closed_trades"),
        "best_component": benchmark_summary.get("best_component_label"),
        "best_component_type": benchmark_summary.get("best_component_type"),
        "best_component_pnl_pct": best_component_pct,
        "best_component_pnl_usd": best_component_usd,
        "panteon_alpha_pct": alpha_pct,
        "beats_best_component": beats_best,
        "initial_capital": summary.get("initial_capital"),
        "risk_capital_fraction": summary.get("risk_capital_fraction"),
        "flash_min_score_to_trade": summary.get("flash_min_score_to_trade"),
        "flash_max_signals_per_actor": summary.get("flash_max_signals_per_actor"),
        "flash_denied_signal_keys_count": len(summary.get("flash_denied_signal_keys") or []),
        "flash_selected_subset_score_boosts_count": len(summary.get("flash_selected_subset_score_boosts") or []),
        "analysis_report": str(analysis_path) if analysis_path.exists() else "",
    }


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number


def _scaled_pnl_to_full_window(pnl_pct: float | None, bars_processed: int | None) -> float | None:
    if pnl_pct is None or bars_processed is None or bars_processed <= 0:
        return None
    capital_ratio = 1.0 + float(pnl_pct) / 100.0
    if capital_ratio <= 0.0:
        return None
    return (capital_ratio ** (FULL_WINDOW_BARS / float(bars_processed)) - 1.0) * 100.0


def _long_run_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in results_dir.rglob("run_summary.json"):
        summary = _load_json(summary_path)
        run_datetime = _parse_run_datetime(summary_path)
        if run_datetime is None:
            continue

        bars = int(summary.get("bars_processed") or 0)
        if bars < MIN_LONG_RUN_BARS:
            continue

        years = set(summary.get("executed_years") or summary.get("requested_years") or [])
        first = str(summary.get("first_timestamp") or "")
        last = str(summary.get("last_timestamp") or "")
        output_dir = Path(summary.get("output_dir") or summary_path.parent)
        if not output_dir.exists():
            output_dir = summary_path.parent

        analysis_path = output_dir / "analysis_report.md"
        if not analysis_path.exists():
            continue
        analysis_text = analysis_path.read_text(encoding="utf-8", errors="ignore")
        pnl_pct = _parse_percent_from_report(analysis_text, "Panteon owned PnL")
        max_dd_pct = _parse_percent_from_report(analysis_text, "Panteon max drawdown")
        if pnl_pct is None:
            continue

        is_full = (
            REQUIRED_YEARS.issubset(years)
            and bars >= MIN_FULL_BARS
            and first.startswith("2022-01-01")
            and last.startswith("2026-05-18")
        )
        is_legacy = (
            {2022, 2023, 2024, 2025}.issubset(years)
            and not REQUIRED_YEARS.issubset(years)
            and first.startswith("2022-01-01")
            and last.startswith("2025-12-31")
        )
        if not is_full and not is_legacy:
            continue

        scaled_pnl = pnl_pct if is_full else _scaled_pnl_to_full_window(pnl_pct, bars)
        rows.append(
            {
                "run_datetime": run_datetime.isoformat(sep=" "),
                "run_date": run_datetime.date().isoformat(),
                "period_class": "2022-2026 full" if is_full else "2022-2025 legacy scaled",
                "executed_years": ",".join(str(year) for year in sorted(years)),
                "bars_processed": bars,
                "scale_factor_to_full_window": FULL_WINDOW_BARS / float(bars),
                "first_timestamp": first,
                "last_timestamp": last,
                "panteon_pnl_pct_raw": pnl_pct,
                "panteon_pnl_pct_scaled_to_full_window": scaled_pnl,
                "panteon_max_dd_pct_raw": max_dd_pct,
                "version_label": _experiment_label(summary_path),
                "output_dir": str(output_dir),
                "run_summary": str(summary_path),
                "analysis_report": str(analysis_path),
            }
        )
    rows.sort(key=lambda item: (item["run_datetime"], item["version_label"]))
    return rows


def _annotate_key_points(ax: plt.Axes, df: pd.DataFrame, y_col: str) -> None:
    valid = df.dropna(subset=[y_col])
    if valid.empty:
        return
    key_indices = {
        "first": valid.index[0],
        "best": valid[y_col].idxmax(),
        "worst": valid[y_col].idxmin(),
        "latest": valid.index[-1],
    }
    for name, idx in key_indices.items():
        row = df.loc[idx]
        label = {
            "first": "first",
            "best": "best",
            "worst": "worst",
            "latest": "latest",
        }[name]
        ax.annotate(
            f"{label}: {row[y_col]:.1f}%",
            xy=(row["run_dt"], row[y_col]),
            xytext=(8, 10 if name != "worst" else -22),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#555555"},
        )


def _save_pnl_timeline(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 7))
    colors = ["#208a4b" if bool(v) else "#b23b3b" for v in df["beats_best_component"].fillna(False)]
    ax.scatter(df["run_dt"], df["panteon_pnl_pct"], s=44, c=colors, zorder=3, label="5-year run")
    ax.plot(df["run_dt"], df["panteon_pnl_pct"], color="#2c5aa0", lw=1.3, alpha=0.65, label="Panteon PnL")
    ax.plot(df["run_dt"], df["best_so_far_pnl_pct"], color="#111111", lw=1.2, ls="--", label="Best so far")
    if df["best_component_pnl_pct"].notna().any():
        ax.plot(
            df["run_dt"],
            df["best_component_pnl_pct"],
            color="#9a7b22",
            lw=1.1,
            alpha=0.75,
            label="Best standalone component",
        )
    _annotate_key_points(ax, df, "panteon_pnl_pct")
    ax.set_title("Panteon Flash: полные 5-летние прогоны 2022-2026 по дате версии")
    ax.set_ylabel("PnL, % от начального капитала")
    ax.set_xlabel("Дата/время версии прогона")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_dd_timeline(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df["run_dt"], df["panteon_max_dd_pct"], color="#b23b3b", marker="o", lw=1.2, label="MaxDD")
    ax.set_title("Panteon Flash: риск полных 5-летних прогонов")
    ax.set_ylabel("Max drawdown, %")
    ax.set_xlabel("Дата/время версии прогона")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_alpha_timeline(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.axhline(0.0, color="#444444", lw=1.0)
    ax.bar(
        df["run_dt"],
        df["panteon_alpha_pct"],
        width=0.025,
        color=["#208a4b" if _safe_float(v) >= 0 else "#b23b3b" for v in df["panteon_alpha_pct"]],
        alpha=0.85,
    )
    ax.set_title("Panteon Flash: alpha против лучшего standalone-компонента")
    ax.set_ylabel("Alpha, п.п.")
    ax.set_xlabel("Дата/время версии прогона")
    ax.grid(True, axis="y", alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_scaled_legacy_timeline(long_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 7))
    styles = {
        "2022-2025 legacy scaled": {"color": "#b65f00", "marker": "^", "label": "2022-2025 legacy, CAGR-scaled"},
        "2022-2026 full": {"color": "#208a4b", "marker": "o", "label": "2022-2026 full"},
    }
    for period_class, group in long_df.groupby("period_class"):
        style = styles.get(period_class, {"color": "#555555", "marker": "o", "label": period_class})
        ax.scatter(
            group["run_dt"],
            group["panteon_pnl_pct_scaled_to_full_window"],
            s=48,
            color=style["color"],
            marker=style["marker"],
            alpha=0.92,
            label=style["label"],
            zorder=3,
        )
        ax.plot(
            group["run_dt"],
            group["panteon_pnl_pct_scaled_to_full_window"],
            color=style["color"],
            lw=1.0,
            alpha=0.35,
        )

    legacy = long_df[long_df["period_class"] == "2022-2025 legacy scaled"].copy()
    if not legacy.empty:
        ax.scatter(
            legacy["run_dt"],
            legacy["panteon_pnl_pct_raw"],
            s=18,
            color="#d9a35f",
            alpha=0.45,
            label="legacy raw",
            zorder=2,
        )

    ax.axhline(0.0, color="#333333", lw=0.9)
    ax.set_title("Panteon/Legacy long-run timeline, scaled to 2022-2026 window")
    ax.set_ylabel("PnL, % scaled to 38,375-bar full window")
    ax.set_xlabel("Run version date/time")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_markdown_report(
    df: pd.DataFrame,
    excluded: list[dict[str, Any]],
    report_path: Path,
    image_paths: dict[str, Path],
    long_df: pd.DataFrame | None = None,
) -> None:
    valid = df.dropna(subset=["panteon_pnl_pct"]).copy()
    best = valid.loc[valid["panteon_pnl_pct"].idxmax()]
    latest = valid.iloc[-1]
    first = valid.iloc[0]
    median_pnl = valid["panteon_pnl_pct"].median()
    median_dd = valid["panteon_max_dd_pct"].median()
    beat_rate = 100.0 * valid["beats_best_component"].fillna(False).astype(bool).mean()
    top = valid.sort_values("panteon_pnl_pct", ascending=False).head(10)
    legacy_scaled = pd.DataFrame()
    if long_df is not None and not long_df.empty:
        legacy_scaled = long_df[long_df["period_class"] == "2022-2025 legacy scaled"].copy()

    lines: list[str] = [
        "# Panteon Flash: timeline полных 5-летних прогонов",
        "",
        f"- Сформировано: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "- Фильтр включения: `executed_years` содержит 2022-2026, `bars_processed >= 38000`, период 2022-01-01 -> 2026-05-18.",
        f"- Включено полных 5-летних прогонов: {len(valid)}.",
        f"- Исключено неполных/годовых/H1 прогонов: {len(excluded)}.",
        "",
        "## Главная динамика",
        "",
        f"- Первый полный Flash-прогон: {first['run_datetime']} — {first['panteon_pnl_pct']:.2f}% PnL, MaxDD {first['panteon_max_dd_pct']:.2f}%.",
        f"- Лучший полный прогон: {best['run_datetime']} — {best['panteon_pnl_pct']:.2f}% PnL, MaxDD {best['panteon_max_dd_pct']:.2f}%, `{best['version_label']}`.",
        f"- Последний полный прогон: {latest['run_datetime']} — {latest['panteon_pnl_pct']:.2f}% PnL, MaxDD {latest['panteon_max_dd_pct']:.2f}%, `{latest['version_label']}`.",
        f"- Прирост от первого к последнему: {latest['panteon_pnl_pct'] - first['panteon_pnl_pct']:.2f} п.п.",
        f"- Медиана по всем полным версиям: {median_pnl:.2f}% PnL, {median_dd:.2f}% MaxDD.",
        f"- Доля версий, где Flash обошёл лучший standalone-компонент: {beat_rate:.1f}%.",
        "",
        "## Графики",
        "",
        f"![PnL timeline]({image_paths['pnl'].as_posix()})",
        "",
        f"![Risk timeline]({image_paths['dd'].as_posix()})",
        "",
        f"![Alpha timeline]({image_paths['alpha'].as_posix()})",
        "",
        "## Топ-10 полных 5-летних прогонов",
        "",
        "| # | Дата версии | PnL % | MaxDD % | Alpha п.п. | Closed | Best component | Версия |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        lines.append(
            "| "
            f"{rank} | {row['run_datetime']} | {row['panteon_pnl_pct']:.2f} | "
            f"{_safe_float(row['panteon_max_dd_pct']):.2f} | {_safe_float(row['panteon_alpha_pct']):.2f} | "
            f"{int(row['closed_trades']) if not pd.isna(row['closed_trades']) else ''} | "
            f"{row.get('best_component') or ''} | `{row['version_label']}` |"
        )
    lines.extend(["", "## Legacy scaled comparison", ""])
    if not legacy_scaled.empty and "legacy_scaled" in image_paths:
        best_legacy = legacy_scaled.loc[legacy_scaled["panteon_pnl_pct_scaled_to_full_window"].idxmax()]
        latest_legacy = legacy_scaled.iloc[-1]
        lines.extend(
            [
                f"- Added long legacy runs: {len(legacy_scaled)}.",
                "- Scaling rule: `scaled = ((1 + raw_pct / 100) ** (38375 / bars_processed) - 1) * 100`.",
                f"- Best scaled legacy run: {best_legacy['run_datetime']} — raw {best_legacy['panteon_pnl_pct_raw']:.2f}%, scaled {best_legacy['panteon_pnl_pct_scaled_to_full_window']:.2f}%, `{best_legacy['version_label']}`.",
                f"- Latest scaled legacy run: {latest_legacy['run_datetime']} — raw {latest_legacy['panteon_pnl_pct_raw']:.2f}%, scaled {latest_legacy['panteon_pnl_pct_scaled_to_full_window']:.2f}%, `{latest_legacy['version_label']}`.",
                "",
                f"![Legacy scaled timeline]({image_paths['legacy_scaled'].as_posix()})",
                "",
                "| # | Version date | Raw PnL % | Scaled PnL % | Bars | MaxDD raw % | Version |",
                "|---:|---|---:|---:|---:|---:|---|",
            ]
        )
        top_legacy = legacy_scaled.sort_values("panteon_pnl_pct_scaled_to_full_window", ascending=False).head(8)
        for rank, (_, row) in enumerate(top_legacy.iterrows(), start=1):
            lines.append(
                "| "
                f"{rank} | {row['run_datetime']} | {row['panteon_pnl_pct_raw']:.2f} | "
                f"{row['panteon_pnl_pct_scaled_to_full_window']:.2f} | {int(row['bars_processed'])} | "
                f"{_safe_float(row['panteon_max_dd_pct_raw']):.2f} | `{row['version_label']}` |"
            )
    else:
        lines.append("- No long legacy runs were found that can be scaled to the full 2022-2026 window.")
    lines.extend(
        [
            "",
            "## Интерпретация",
            "",
            "- Ранние версии до stable Flash давали отрицательный или около нуля результат: selection ещё не защищал от плохих клеток и переизбыточной активности.",
            "- Основной скачок появился после shadow-confirmation/cap/attribution-итераций: Flash перестал выбирать большую часть шумовых сигналов.",
            "- Поздние deny-итерации давали рост, но broad-deny был хрупким: часть улучшений переобучалась и не переносилась между 2025/2026.",
            "- Лучшая текущая линия — targeted LCB/terminal-deny + selected-subset positive boosts: она даёт максимум PnL при снижении MaxDD относительно предыдущего лучшего полного прогона.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(results_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts, excluded = _full_run_artifacts(results_dir)
    rows = [_row_from_artifacts(item) for item in artifacts]
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No full 2022-2026 run_summary.json files found.")

    numeric_cols = [
        "panteon_pnl_pct",
        "panteon_pnl_usd",
        "panteon_max_dd_pct",
        "panteon_realized_max_dd_pct",
        "selected_signals",
        "filled_signals",
        "closed_trades",
        "best_component_pnl_pct",
        "best_component_pnl_usd",
        "panteon_alpha_pct",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["run_dt"] = pd.to_datetime(df["run_datetime"])
    df = df.sort_values(["run_dt", "version_label"]).reset_index(drop=True)
    df["run_order"] = range(1, len(df) + 1)
    df["best_so_far_pnl_pct"] = df["panteon_pnl_pct"].cummax()

    csv_path = output_dir / "five_year_runs.csv"
    df.drop(columns=["run_dt"]).to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8-sig")

    excluded_path = output_dir / "excluded_runs.csv"
    pd.DataFrame(excluded).to_csv(excluded_path, index=False, encoding="utf-8-sig")

    long_df = pd.DataFrame(_long_run_rows(results_dir))
    if not long_df.empty:
        long_df["run_dt"] = pd.to_datetime(long_df["run_datetime"])
        for col in [
            "bars_processed",
            "scale_factor_to_full_window",
            "panteon_pnl_pct_raw",
            "panteon_pnl_pct_scaled_to_full_window",
            "panteon_max_dd_pct_raw",
        ]:
            long_df[col] = pd.to_numeric(long_df[col], errors="coerce")
        long_df = long_df.sort_values(["run_dt", "version_label"]).reset_index(drop=True)
        long_df.drop(columns=["run_dt"]).to_csv(
            output_dir / "long_legacy_and_full_runs_scaled.csv",
            index=False,
            quoting=csv.QUOTE_MINIMAL,
            encoding="utf-8-sig",
        )

    image_paths = {
        "pnl": output_dir / "five_year_pnl_timeline.png",
        "dd": output_dir / "five_year_maxdd_timeline.png",
        "alpha": output_dir / "five_year_alpha_timeline.png",
        "legacy_scaled": output_dir / "legacy_scaled_to_full_window_timeline.png",
    }
    _save_pnl_timeline(df, image_paths["pnl"])
    _save_dd_timeline(df, image_paths["dd"])
    _save_alpha_timeline(df, image_paths["alpha"])
    if not long_df.empty:
        _save_scaled_legacy_timeline(long_df, image_paths["legacy_scaled"])
    _write_markdown_report(df, excluded, output_dir / "FIVE_YEAR_RUNS_TIMELINE_REPORT.md", image_paths, long_df)

    summary = {
        "included_full_runs": int(len(df)),
        "included_long_runs_scaled": int(len(long_df)),
        "excluded_runs": int(len(excluded)),
        "csv": str(csv_path),
        "long_scaled_csv": str(output_dir / "long_legacy_and_full_runs_scaled.csv"),
        "excluded_csv": str(excluded_path),
        "report": str(output_dir / "FIVE_YEAR_RUNS_TIMELINE_REPORT.md"),
        "charts": {key: str(path) for key, path in image_paths.items()},
        "best_run": df.loc[df["panteon_pnl_pct"].idxmax()].drop(labels=["run_dt"]).to_dict(),
        "latest_run": df.iloc[-1].drop(labels=["run_dt"]).to_dict(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


def main() -> None:
    build(
        results_dir=Path("Results"),
        output_dir=Path("Reports") / "PanteonFlashFiveYearTimeline_20260525",
    )


if __name__ == "__main__":
    main()
