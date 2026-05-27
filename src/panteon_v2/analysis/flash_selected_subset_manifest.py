"""Build a selected-subset overlay for Panteon Flash.

The manifest is deliberately additive: it boosts/re-sizes proven positive cells and
protects sparse winners from broad LCB demotion, but it does not deny signals.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def build_flash_selected_subset_manifest(
    *,
    confirmation_attribution_paths: Sequence[Path | str],
    min_closed_trades: int = 4,
    score_boost: float = 0.25,
    risk_positive_mult: float = 1.15,
    risk_weak_mult: float = 0.75,
    sparse_min_pnl_usd: float = 25.0,
    context_min_closed_trades: int | None = None,
    context_score_boost: float = 0.15,
    context_risk_positive_mult: float = 1.0,
) -> dict:
    """Return an additive selected-subset manifest from attribution summaries.

    Score boosts require a cell to be positive and sufficiently sampled in every
    supplied confirmation period. Sparse winners are only protected from LCB
    demotion; they are not boosted.
    """

    if min_closed_trades < 1:
        raise ValueError("min_closed_trades must be >= 1")
    if score_boost < 0:
        raise ValueError("score_boost must be >= 0")
    if context_score_boost < 0:
        raise ValueError("context_score_boost must be >= 0")
    if risk_positive_mult <= 0 or risk_weak_mult <= 0:
        raise ValueError("risk multipliers must be > 0")
    if context_risk_positive_mult <= 0:
        raise ValueError("context_risk_positive_mult must be > 0")

    inputs = [Path(path) for path in confirmation_attribution_paths]
    periods = [_load_attribution_rows(path) for path in inputs]
    by_period = [_rows_by_signal_key(rows) for rows in periods]
    all_keys = sorted({key for period in by_period for key in period})
    context_periods = [_load_context_attribution_rows(path) for path in inputs]
    context_by_period = [_rows_by_context_key(rows) for rows in context_periods]
    all_context_keys = sorted({key for period in context_by_period for key in period})

    score_boosts: list[dict] = []
    do_not_demote: set[str] = set()
    risk_mults: dict[str, dict] = {}
    context_score_boosts: list[dict] = []
    context_risk_mults: list[dict] = []

    for signal_key in all_keys:
        period_rows = [period.get(signal_key) for period in by_period]
        observed_rows = [row for row in period_rows if row is not None]
        if not observed_rows:
            continue

        confirmed_positive = (
            len(period_rows) > 0
            and all(
                row is not None
                and _int(row.get("closed_trades")) >= min_closed_trades
                and _float(row.get("realized_pnl_usd")) > 0.0
                for row in period_rows
            )
        )
        weak_period = any(
            _int(row.get("closed_trades")) >= min_closed_trades
            and _float(row.get("realized_pnl_usd")) < 0.0
            for row in observed_rows
        )
        sparse_winner = any(
            _int(row.get("closed_trades")) > 0
            and _float(row.get("realized_pnl_usd")) >= sparse_min_pnl_usd
            for row in observed_rows
        )

        if confirmed_positive:
            score_boosts.append(
                {
                    "signal_key": signal_key,
                    "boost": float(score_boost),
                    "reason": "positive_in_all_confirmation_periods",
                    "periods": [_period_payload(row) for row in observed_rows],
                }
            )
            do_not_demote.add(signal_key)
            risk_mults[signal_key] = {
                "signal_key": signal_key,
                "risk_mult": float(risk_positive_mult),
                "reason": "positive_in_all_confirmation_periods",
            }
        elif sparse_winner:
            do_not_demote.add(signal_key)

        if weak_period and signal_key not in risk_mults:
            risk_mults[signal_key] = {
                "signal_key": signal_key,
                "risk_mult": float(risk_weak_mult),
                "reason": "negative_selected_subset_period",
            }

    context_min_closed = (
        int(context_min_closed_trades)
        if context_min_closed_trades is not None
        else int(min_closed_trades)
    )
    for context_key in all_context_keys:
        period_rows = [period.get(context_key) for period in context_by_period]
        observed_rows = [row for row in period_rows if row is not None]
        if not observed_rows:
            continue
        confirmed_positive = (
            len(period_rows) > 0
            and all(
                row is not None
                and _int(row.get("closed_trades")) >= context_min_closed
                and _float(row.get("realized_pnl_usd")) > 0.0
                for row in period_rows
            )
        )
        if not confirmed_positive:
            continue
        context_score_boosts.append(
            {
                "context_key": context_key,
                "boost": float(context_score_boost),
                "reason": "actor_symbol_action_regime_positive_in_all_confirmation_periods",
                "periods": [_period_payload(row) for row in observed_rows],
            }
        )
        context_risk_mults.append(
            {
                "context_key": context_key,
                "risk_mult": float(context_risk_positive_mult),
                "reason": "actor_symbol_action_regime_positive_in_all_confirmation_periods",
            }
        )

    return {
        "schema": "panteon_flash_selected_subset_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(path) for path in inputs],
        "parameters": {
            "min_closed_trades": int(min_closed_trades),
            "score_boost": float(score_boost),
            "risk_positive_mult": float(risk_positive_mult),
            "risk_weak_mult": float(risk_weak_mult),
            "sparse_min_pnl_usd": float(sparse_min_pnl_usd),
            "context_min_closed_trades": int(context_min_closed),
            "context_score_boost": float(context_score_boost),
            "context_risk_positive_mult": float(context_risk_positive_mult),
        },
        "score_boosts": sorted(score_boosts, key=lambda item: item["signal_key"]),
        "context_score_boosts": sorted(
            context_score_boosts,
            key=lambda item: item["context_key"],
        ),
        "do_not_demote_signal_keys": sorted(do_not_demote),
        "risk_mult_overrides": sorted(
            risk_mults.values(),
            key=lambda item: item["signal_key"],
        ),
        "context_risk_mult_overrides": sorted(
            context_risk_mults,
            key=lambda item: item["context_key"],
        ),
    }


def write_flash_selected_subset_manifest(
    *,
    output_path: Path | str,
    confirmation_attribution_paths: Sequence[Path | str],
    min_closed_trades: int = 4,
    score_boost: float = 0.25,
    risk_positive_mult: float = 1.15,
    risk_weak_mult: float = 0.75,
    sparse_min_pnl_usd: float = 25.0,
    context_min_closed_trades: int | None = None,
    context_score_boost: float = 0.15,
    context_risk_positive_mult: float = 1.0,
) -> Path:
    manifest = build_flash_selected_subset_manifest(
        confirmation_attribution_paths=confirmation_attribution_paths,
        min_closed_trades=min_closed_trades,
        score_boost=score_boost,
        risk_positive_mult=risk_positive_mult,
        risk_weak_mult=risk_weak_mult,
        sparse_min_pnl_usd=sparse_min_pnl_usd,
        context_min_closed_trades=context_min_closed_trades,
        context_score_boost=context_score_boost,
        context_risk_positive_mult=context_risk_positive_mult,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _load_attribution_rows(path: Path) -> list[dict]:
    source = path
    if source.is_dir():
        source = source / "flash_attribution_summary.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = data.get("rows", []) if isinstance(data, dict) else []
    return [row for row in rows if isinstance(row, dict)]


def _load_context_attribution_rows(path: Path) -> list[dict]:
    source = path
    if source.is_dir():
        source = source / "flash_attribution_summary.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    rows = data.get("context_rows")
    if not isinstance(rows, list):
        rows = [
            row
            for row in data.get("rows", [])
            if isinstance(row, dict)
            and (
                row.get("context_key")
                or row.get("regime")
                or row.get("entry_regime")
            )
        ]
    return [row for row in rows if isinstance(row, dict)]


def _rows_by_signal_key(rows: Iterable[Mapping[str, object]]) -> dict[str, dict]:
    by_key: dict[str, dict] = {}
    for row in rows:
        signal_key = _signal_key_from_row(row)
        if not signal_key:
            continue
        by_key[signal_key] = dict(row)
    return by_key


def _rows_by_context_key(rows: Iterable[Mapping[str, object]]) -> dict[str, dict]:
    by_key: dict[str, dict] = {}
    for row in rows:
        context_key = _context_key_from_row(row)
        if not context_key:
            continue
        by_key[context_key] = dict(row)
    return by_key


def _signal_key_from_row(row: Mapping[str, object]) -> str:
    explicit = str(row.get("signal_key") or "").strip()
    if explicit:
        return explicit
    actor_key = str(row.get("actor_key") or "").strip()
    symbol = str(row.get("symbol") or "").strip().upper()
    action = str(row.get("action") or "").strip().upper()
    if not actor_key or not symbol or not action:
        return ""
    return f"{actor_key}|{symbol}|{action}"


def _context_key_from_row(row: Mapping[str, object]) -> str:
    explicit = str(row.get("context_key") or "").strip()
    if explicit:
        parts = [part.strip() for part in explicit.split("|")]
        if len(parts) == 4 and all(parts):
            return (
                f"{parts[0]}|{parts[1].upper()}|{parts[2].upper()}|"
                f"{_normalize_regime_label(parts[3])}"
            )
        return explicit
    signal_key = _signal_key_from_row(row)
    regime = _normalize_regime_label(
        row.get("regime") or row.get("entry_regime") or ""
    )
    if not signal_key or not regime:
        return ""
    return f"{signal_key}|{regime}"


def _period_payload(row: Mapping[str, object]) -> dict:
    payload = {
        "closed_trades": _int(row.get("closed_trades")),
        "realized_pnl_usd": _float(row.get("realized_pnl_usd")),
        "selected_signals": _int(row.get("selected_signals")),
    }
    regime = _normalize_regime_label(
        row.get("regime") or row.get("entry_regime") or ""
    )
    if not regime:
        regime = _regime_from_context_key(row.get("context_key"))
    if regime:
        payload["regime"] = regime
    return payload


def _normalize_regime_label(raw: object) -> str:
    value = str(raw or "").strip().lower()
    aliases = {
        "bull": "bullish",
        "bear": "bearish",
        "flat": "neutral",
        "range": "neutral",
        "sideways": "neutral",
        "panic": "crash",
        "flash_crash": "crash",
    }
    return aliases.get(value, value)


def _regime_from_context_key(raw: object) -> str:
    parts = [part.strip() for part in str(raw or "").split("|")]
    if len(parts) != 4 or not all(parts):
        return ""
    return _normalize_regime_label(parts[3])


def _int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-closed-trades", type=int, default=4)
    parser.add_argument("--score-boost", type=float, default=0.25)
    parser.add_argument("--risk-positive-mult", type=float, default=1.15)
    parser.add_argument("--risk-weak-mult", type=float, default=0.75)
    parser.add_argument("--sparse-min-pnl-usd", type=float, default=25.0)
    parser.add_argument("--context-min-closed-trades", type=int, default=None)
    parser.add_argument("--context-score-boost", type=float, default=0.15)
    parser.add_argument("--context-risk-positive-mult", type=float, default=1.0)
    args = parser.parse_args(argv)

    path = write_flash_selected_subset_manifest(
        output_path=args.output,
        confirmation_attribution_paths=tuple(args.input),
        min_closed_trades=args.min_closed_trades,
        score_boost=args.score_boost,
        risk_positive_mult=args.risk_positive_mult,
        risk_weak_mult=args.risk_weak_mult,
        sparse_min_pnl_usd=args.sparse_min_pnl_usd,
        context_min_closed_trades=args.context_min_closed_trades,
        context_score_boost=args.context_score_boost,
        context_risk_positive_mult=args.context_risk_positive_mult,
    )
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
