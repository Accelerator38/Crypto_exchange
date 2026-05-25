"""Validation-aware manifest for Flash genetics contra rules."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def build_flash_contra_validation_manifest(
    *,
    contra_signal_keys: Sequence[str],
    split_results: Sequence[Mapping[str, Any]],
    min_validation_pnl_delta_pct: float = 0.0,
    max_validation_dd_worsening_pct: float = 0.0,
    require_validation_split: bool = True,
    require_oos_split: bool = False,
) -> dict[str, Any]:
    """Allow contra keys only if split-level validation beats baseline.

    This gate is intentionally portfolio-level. Static signal-key LCB can look
    good locally while changing later position/rate-limit sequencing, so the
    manifest is empty unless candidate runs beat their matched baselines.
    """

    keys = tuple(dict.fromkeys(str(key).strip() for key in contra_signal_keys if str(key).strip()))
    validation_delta = float(min_validation_pnl_delta_pct or 0.0)
    max_dd_worsening = float(max_validation_dd_worsening_pct or 0.0)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    validation_count = 0
    oos_count = 0

    for raw_split in split_results:
        row = _split_row(raw_split)
        rows.append(row)
        role = row["role"]
        if role == "validation":
            validation_count += 1
        elif role == "oos":
            oos_count += 1

        if role not in {"validation", "oos"}:
            continue
        name = row["name"]
        pnl_delta = row["pnl_delta_pct"]
        if pnl_delta == validation_delta:
            failures.append(f"{role}_tie:{name}")
        elif pnl_delta < validation_delta:
            failures.append(f"{role}_pnl_not_strictly_better:{name}")
        dd_delta = row["max_dd_delta_pct"]
        if dd_delta > max_dd_worsening:
            failures.append(f"{role}_max_dd_worse:{name}")

    if require_validation_split and validation_count <= 0:
        failures.append("validation_split_missing")
    if require_oos_split and oos_count <= 0:
        failures.append("oos_split_missing")
    if not keys:
        failures.append("contra_signal_keys_missing")

    eligible = not failures
    rejected_reason = failures[0] if failures else ""
    return {
        "summary": {
            "eligible": eligible,
            "contra_signal_keys": len(keys),
            "split_count": len(rows),
            "validation_split_count": validation_count,
            "oos_split_count": oos_count,
            "min_validation_pnl_delta_pct": validation_delta,
            "max_validation_dd_worsening_pct": max_dd_worsening,
        },
        "allowed_contra_signal_keys": list(keys) if eligible else [],
        "rejected_contra_signal_keys": [
            {"signal_key": key, "reason": rejected_reason}
            for key in keys
            if not eligible
        ],
        "promotion_failures": failures,
        "split_results": rows,
    }


def write_flash_contra_validation_manifest(
    output_dir: str | Path,
    *,
    contra_signal_keys: Sequence[str],
    split_results: Sequence[Mapping[str, Any]],
    min_validation_pnl_delta_pct: float = 0.0,
    max_validation_dd_worsening_pct: float = 0.0,
    require_validation_split: bool = True,
    require_oos_split: bool = False,
) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_flash_contra_validation_manifest(
        contra_signal_keys=contra_signal_keys,
        split_results=split_results,
        min_validation_pnl_delta_pct=min_validation_pnl_delta_pct,
        max_validation_dd_worsening_pct=max_validation_dd_worsening_pct,
        require_validation_split=require_validation_split,
        require_oos_split=require_oos_split,
    )
    json_path = out / "flash_genetics_contra_validation_manifest.json"
    md_path = out / "flash_genetics_contra_validation_manifest.md"
    json_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    md_path.write_text(_markdown_manifest(manifest), encoding="utf-8")
    return json_path, md_path


def load_flash_run_metrics(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir)
    status = _load_json(path / "status.json")
    summary = _load_json(path / "run_summary.json")
    live = _mapping(status.get("live_session"))
    trades = _mapping(status.get("real_trades"))
    attribution = _mapping(summary.get("flash_attribution_summary"))
    return {
        "run_dir": str(path),
        "panteon_pnl_pct": _float(live.get("panteon_pnl_pct")),
        "panteon_realized_pnl_usd": _float(live.get("panteon_realized_pnl_usd")),
        "panteon_max_drawdown_pct": _float(live.get("panteon_max_drawdown_pct")),
        "closed_trades": _int(trades.get("closed")),
        "winning_trades": _int(trades.get("successful")),
        "losing_trades": _int(trades.get("unsuccessful")),
        "selected_decisions": _int(attribution.get("selected_decisions")),
        "filled_signals": _int(attribution.get("filled_signals")),
        "rate_limited_open": _int(
            _mapping(attribution.get("filter_detail_counts")).get("rate_limited_open")
        ),
    }


def _split_row(raw_split: Mapping[str, Any]) -> dict[str, Any]:
    name = str(raw_split.get("name") or raw_split.get("split") or "").strip() or "split"
    role = str(raw_split.get("role") or "").strip().lower() or "validation"
    baseline = _mapping(raw_split.get("baseline"))
    candidate = _mapping(raw_split.get("candidate"))
    baseline_pnl = _float(baseline.get("panteon_pnl_pct"))
    candidate_pnl = _float(candidate.get("panteon_pnl_pct"))
    baseline_dd = _float(baseline.get("panteon_max_drawdown_pct"))
    candidate_dd = _float(candidate.get("panteon_max_drawdown_pct"))
    return {
        "name": name,
        "role": role,
        "baseline_pnl_pct": baseline_pnl,
        "candidate_pnl_pct": candidate_pnl,
        "pnl_delta_pct": candidate_pnl - baseline_pnl,
        "baseline_max_dd_pct": baseline_dd,
        "candidate_max_dd_pct": candidate_dd,
        "max_dd_delta_pct": candidate_dd - baseline_dd,
        "baseline_closed_trades": _int(baseline.get("closed_trades")),
        "candidate_closed_trades": _int(candidate.get("closed_trades")),
        "baseline_run_dir": str(baseline.get("run_dir") or ""),
        "candidate_run_dir": str(candidate.get("run_dir") or ""),
    }


def _markdown_manifest(manifest: Mapping[str, Any]) -> str:
    summary = _mapping(manifest.get("summary"))
    lines = [
        "# Flash Genetics Contra Validation Manifest",
        "",
        f"- Eligible: `{bool(summary.get('eligible'))}`",
        f"- Contra signal keys: `{summary.get('contra_signal_keys', 0)}`",
        f"- Validation splits: `{summary.get('validation_split_count', 0)}`",
        f"- OOS splits: `{summary.get('oos_split_count', 0)}`",
        "",
        "## Allowed Contra Keys",
    ]
    allowed = list(manifest.get("allowed_contra_signal_keys") or ())
    lines.extend(f"- `{key}`" for key in allowed) if allowed else lines.append("- none")
    lines.extend(["", "## Promotion Failures"])
    failures = list(manifest.get("promotion_failures") or ())
    lines.extend(f"- `{failure}`" for failure in failures) if failures else lines.append("- none")
    lines.extend(["", "## Split Results", ""])
    lines.append("| Split | Role | Baseline PnL % | Candidate PnL % | Delta % | Baseline DD % | Candidate DD % |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in manifest.get("split_results") or ():
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {name} | {role} | {bp:.4f} | {cp:.4f} | {delta:.4f} | {bdd:.4f} | {cdd:.4f} |".format(
                name=row.get("name", ""),
                role=row.get("role", ""),
                bp=_float(row.get("baseline_pnl_pct")),
                cp=_float(row.get("candidate_pnl_pct")),
                delta=_float(row.get("pnl_delta_pct")),
                bdd=_float(row.get("baseline_max_dd_pct")),
                cdd=_float(row.get("candidate_max_dd_pct")),
            )
        )
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        parsed = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _parse_split_arg(raw: str) -> dict[str, Any]:
    parts = str(raw or "").split("=", 1)
    if len(parts) != 2:
        raise ValueError("--split must be NAME:ROLE=BASELINE_DIR|CANDIDATE_DIR")
    name_role, dirs = parts
    left = name_role.split(":", 1)
    name = left[0].strip()
    role = left[1].strip() if len(left) > 1 else "validation"
    pair = dirs.split("|", 1)
    if len(pair) != 2:
        raise ValueError("--split must include BASELINE_DIR|CANDIDATE_DIR")
    return {
        "name": name,
        "role": role,
        "baseline": load_flash_run_metrics(pair[0].strip()),
        "candidate": load_flash_run_metrics(pair[1].strip()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--contra-signal-key", action="append", default=[])
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--min-validation-pnl-delta-pct", type=float, default=0.0)
    parser.add_argument("--max-validation-dd-worsening-pct", type=float, default=0.0)
    parser.add_argument("--require-oos-split", action="store_true")
    args = parser.parse_args(argv)
    splits = [_parse_split_arg(raw) for raw in args.split]
    json_path, md_path = write_flash_contra_validation_manifest(
        args.output_dir,
        contra_signal_keys=args.contra_signal_key,
        split_results=splits,
        min_validation_pnl_delta_pct=args.min_validation_pnl_delta_pct,
        max_validation_dd_worsening_pct=args.max_validation_dd_worsening_pct,
        require_oos_split=args.require_oos_split,
    )
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
