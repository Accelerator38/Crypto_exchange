"""Hard promotion gates for Panteon Flash/Ultima profit candidates."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PERIOD_2026_H1 = "2026_h1"
PERIOD_2025 = "2025"
PERIOD_FULL = "full_2022_2026"
REQUIRED_PERIODS = (PERIOD_2026_H1, PERIOD_2025, PERIOD_FULL)


@dataclass(frozen=True)
class ProfitabilityGates:
    min_2026_h1_usd: float = 112.72
    min_2025_usd: float = 221.80
    min_full_usd_exclusive: float = 1063.10
    max_full_drawdown_pct: float = 4.12
    currency_tolerance_usd: float = 0.005
    drawdown_tolerance_pct: float = 0.005

    def as_dict(self) -> dict[str, float]:
        return {
            "min_2026_h1_usd": float(self.min_2026_h1_usd),
            "min_2025_usd": float(self.min_2025_usd),
            "min_full_usd_exclusive": float(self.min_full_usd_exclusive),
            "max_full_drawdown_pct": float(self.max_full_drawdown_pct),
            "currency_tolerance_usd": float(self.currency_tolerance_usd),
            "drawdown_tolerance_pct": float(self.drawdown_tolerance_pct),
        }


DEFAULT_PROFITABILITY_GATES = ProfitabilityGates()


def load_flash_run_metrics(path: str | Path, *, name: str = "", period: str = "") -> dict[str, Any]:
    """Load the minimal PnL/DD metrics needed by the hard gates."""

    output_dir = _resolve_run_dir(path)
    status = _load_json(output_dir / "status.json")
    flash_attribution = _load_json(output_dir / "flash_attribution_summary.json")

    live = _mapping(status.get("live_session"))
    real_trades = _mapping(status.get("real_trades"))
    flash_summary = _mapping(flash_attribution.get("summary"))

    initial_capital = _float(status.get("initial_capital"), 1000.0)
    pnl_usd = _first_number(
        live.get("clean_panteon_total_pnl_usd"),
        live.get("panteon_owned_total_pnl_usd"),
        live.get("panteon_equity_pnl_usd"),
        status.get("pnl_usd"),
    )
    pnl_pct = _first_number(
        live.get("clean_panteon_pnl_pct"),
        live.get("panteon_owned_pnl_pct"),
        live.get("panteon_pnl_pct"),
        (pnl_usd / initial_capital * 100.0) if initial_capital > 0.0 else 0.0,
    )
    max_drawdown_pct = _first_number(
        live.get("clean_panteon_max_drawdown_pct"),
        status.get("clean_panteon_max_drawdown_pct"),
        live.get("panteon_max_drawdown_pct"),
        status.get("panteon_max_drawdown_pct"),
        status.get("max_drawdown_pct"),
    )
    closed_trades = int(_first_number(
        live.get("clean_real_closed_trades"),
        live.get("real_closed_trades"),
        real_trades.get("clean_closed"),
        real_trades.get("closed"),
        flash_summary.get("closed_trades"),
    ))

    return {
        "name": str(name or ""),
        "period": str(period or ""),
        "output_dir": str(output_dir),
        "pnl_usd": pnl_usd,
        "pnl_pct": pnl_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "closed_trades": closed_trades,
        "selected_signals": int(_first_number(flash_summary.get("selected_signals"))),
        "filled_signals": int(_first_number(flash_summary.get("filled_signals"))),
        "bars_processed": int(_first_number(_load_json(output_dir / "run_summary.json").get("bars_processed"))),
    }


def evaluate_profitability_gates(
    period_metrics: Mapping[str, Mapping[str, Any]],
    *,
    gates: ProfitabilityGates = DEFAULT_PROFITABILITY_GATES,
) -> dict[str, Any]:
    """Evaluate one candidate against the fixed promotion contract."""

    checks = [
        _min_usd_check(
            period_metrics,
            period=PERIOD_2026_H1,
            metric="pnl_usd",
            threshold=gates.min_2026_h1_usd,
            tolerance=gates.currency_tolerance_usd,
            label="2026 H1 PnL",
        ),
        _min_usd_check(
            period_metrics,
            period=PERIOD_2025,
            metric="pnl_usd",
            threshold=gates.min_2025_usd,
            tolerance=gates.currency_tolerance_usd,
            label="2025 PnL",
        ),
        _exclusive_min_usd_check(
            period_metrics,
            period=PERIOD_FULL,
            metric="pnl_usd",
            threshold=gates.min_full_usd_exclusive,
            label="full 2022-2026 PnL",
        ),
        _max_pct_check(
            period_metrics,
            period=PERIOD_FULL,
            metric="max_drawdown_pct",
            threshold=gates.max_full_drawdown_pct,
            tolerance=gates.drawdown_tolerance_pct,
            label="full 2022-2026 max DD",
        ),
    ]
    failures = [
        str(check["gate"])
        for check in checks
        if check.get("status") != "passed"
    ]
    return {
        "promotion_eligible": not failures,
        "checks": checks,
        "failures": failures,
    }


def build_profitability_gate_report(
    candidates: Sequence[Mapping[str, Any]],
    *,
    gates: ProfitabilityGates = DEFAULT_PROFITABILITY_GATES,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    eligible = 0
    for candidate in candidates:
        name = str(candidate.get("name") or "")
        raw_periods = _mapping(candidate.get("periods"))
        periods = {
            str(period): dict(_mapping(metrics))
            for period, metrics in raw_periods.items()
        }
        gate = evaluate_profitability_gates(periods, gates=gates)
        row = {
            "name": name,
            "periods": periods,
            "gate": gate,
            "promotion_eligible": bool(gate.get("promotion_eligible")),
        }
        rows.append(row)
        eligible += int(row["promotion_eligible"])

    return {
        "schema": "panteon_flash_profitability_gate_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gates": gates.as_dict(),
        "summary": {
            "candidates": len(rows),
            "promotion_eligible": eligible,
            "promotion_rejected": len(rows) - eligible,
        },
        "candidates": rows,
    }


def write_profitability_gate_report(
    *,
    output_json: str | Path,
    candidates: Sequence[Mapping[str, Any]],
    gates: ProfitabilityGates = DEFAULT_PROFITABILITY_GATES,
    output_md: str | Path | None = None,
) -> tuple[Path, Path | None]:
    report = build_profitability_gate_report(candidates, gates=gates)
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    md_path: Path | None = None
    if output_md is not None:
        md_path = Path(output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, md_path


def _min_usd_check(
    period_metrics: Mapping[str, Mapping[str, Any]],
    *,
    period: str,
    metric: str,
    threshold: float,
    tolerance: float,
    label: str,
) -> dict[str, Any]:
    value = _period_metric(period_metrics, period, metric)
    status = "missing"
    if value is not None:
        status = "passed" if value + float(tolerance) >= float(threshold) else "failed"
    return _check_payload(
        gate=f"{period}.{metric}>=${threshold:.2f}",
        label=label,
        period=period,
        metric=metric,
        value=value,
        threshold=threshold,
        status=status,
        comparator=">=",
    )


def _exclusive_min_usd_check(
    period_metrics: Mapping[str, Mapping[str, Any]],
    *,
    period: str,
    metric: str,
    threshold: float,
    label: str,
) -> dict[str, Any]:
    value = _period_metric(period_metrics, period, metric)
    status = "missing"
    if value is not None:
        status = "passed" if value > float(threshold) else "failed"
    return _check_payload(
        gate=f"{period}.{metric}>${threshold:.2f}",
        label=label,
        period=period,
        metric=metric,
        value=value,
        threshold=threshold,
        status=status,
        comparator=">",
    )


def _max_pct_check(
    period_metrics: Mapping[str, Mapping[str, Any]],
    *,
    period: str,
    metric: str,
    threshold: float,
    tolerance: float,
    label: str,
) -> dict[str, Any]:
    value = _period_metric(period_metrics, period, metric)
    status = "missing"
    if value is not None:
        status = "passed" if value <= float(threshold) + float(tolerance) else "failed"
    return _check_payload(
        gate=f"{period}.{metric}<={threshold:.2f}%",
        label=label,
        period=period,
        metric=metric,
        value=value,
        threshold=threshold,
        status=status,
        comparator="<=",
    )


def _check_payload(
    *,
    gate: str,
    label: str,
    period: str,
    metric: str,
    value: float | None,
    threshold: float,
    status: str,
    comparator: str,
) -> dict[str, Any]:
    return {
        "gate": gate,
        "label": label,
        "period": period,
        "metric": metric,
        "value": value,
        "threshold": float(threshold),
        "comparator": comparator,
        "status": status,
    }


def _period_metric(
    period_metrics: Mapping[str, Mapping[str, Any]],
    period: str,
    metric: str,
) -> float | None:
    row = period_metrics.get(period)
    if not isinstance(row, Mapping):
        return None
    raw = row.get(metric)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _resolve_run_dir(path: str | Path) -> Path:
    source = Path(path)
    if source.is_file():
        return source.parent
    if (source / "status.json").exists():
        return source
    candidates = sorted(
        source.rglob("status.json") if source.exists() else (),
        key=lambda item: item.stat().st_mtime,
    )
    if candidates:
        return candidates[-1].parent
    return source


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _first_number(*values: object) -> float:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _format_money(value: object) -> str:
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def _format_pct(value: object) -> str:
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "-"


def _markdown_report(report: Mapping[str, Any]) -> str:
    gates = _mapping(report.get("gates"))
    lines = [
        "# Panteon Flash Profitability Gate Report",
        "",
        "## Gates",
        "",
        f"- 2026 H1 PnL >= ${float(gates.get('min_2026_h1_usd', 0.0)):.2f}",
        f"- 2025 PnL >= ${float(gates.get('min_2025_usd', 0.0)):.2f}",
        f"- full 2022-2026 PnL > ${float(gates.get('min_full_usd_exclusive', 0.0)):.2f}",
        f"- full 2022-2026 max DD <= {float(gates.get('max_full_drawdown_pct', 0.0)):.2f}%",
        "",
        "## Candidates",
        "",
        "| Candidate | Promotion | 2026 H1 | 2025 | Full | Full DD | Failures |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in report.get("candidates", []) or []:
        if not isinstance(candidate, Mapping):
            continue
        periods = _mapping(candidate.get("periods"))
        h1 = _mapping(periods.get(PERIOD_2026_H1))
        y2025 = _mapping(periods.get(PERIOD_2025))
        full = _mapping(periods.get(PERIOD_FULL))
        gate = _mapping(candidate.get("gate"))
        failures = ", ".join(str(item) for item in gate.get("failures", []) or [])
        lines.append(
            "| {name} | {promotion} | {h1} | {y2025} | {full} | {dd} | {failures} |".format(
                name=str(candidate.get("name") or ""),
                promotion="PASS" if candidate.get("promotion_eligible") else "FAIL",
                h1=_format_money(h1.get("pnl_usd")),
                y2025=_format_money(y2025.get("pnl_usd")),
                full=_format_money(full.get("pnl_usd")),
                dd=_format_pct(full.get("max_drawdown_pct")),
                failures=failures or "-",
            )
        )
    return "\n".join(lines) + "\n"


def _parse_candidate_spec(raw: str) -> tuple[str, str, Path]:
    parts = [part.strip() for part in str(raw or "").split("|", 2)]
    if len(parts) != 3 or not all(parts):
        raise ValueError("--candidate must use 'name|period|path'")
    name, period, path = parts
    if period not in REQUIRED_PERIODS:
        raise ValueError(f"period must be one of: {', '.join(REQUIRED_PERIODS)}")
    return name, period, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate run as 'name|period|path'. Repeat for each period.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    args = parser.parse_args(argv)

    grouped: dict[str, dict[str, Any]] = {}
    for raw in args.candidate:
        name, period, path = _parse_candidate_spec(raw)
        candidate = grouped.setdefault(name, {"name": name, "periods": {}})
        candidate["periods"][period] = load_flash_run_metrics(
            path,
            name=name,
            period=period,
        )

    json_path, md_path = write_profitability_gate_report(
        output_json=args.output_json,
        output_md=args.output_md,
        candidates=tuple(grouped.values()),
    )
    print(json_path)
    if md_path is not None:
        print(md_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
