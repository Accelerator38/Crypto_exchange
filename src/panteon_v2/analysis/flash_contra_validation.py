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
    require_oos_split: bool = True,
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
            "require_oos_split": bool(require_oos_split),
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


def build_flash_contra_risk_sizing_manifest(
    *,
    contra_signal_keys: Sequence[str],
    split_results: Sequence[Mapping[str, Any]],
    risk_mult: float = 0.50,
    min_drawdown_improvement_pct: float = 1.0,
    max_pnl_degradation_pct: float = 0.25,
    require_validation_split: bool = True,
    require_oos_split: bool = True,
) -> dict[str, Any]:
    """Gate contra keys as risk-sizing reducers instead of binary allowlists.

    A contra overlay is eligible for sizing only when it improves max drawdown by
    at least ``min_drawdown_improvement_pct`` while limiting PnL degradation to
    ``max_pnl_degradation_pct`` on every validation/OOS split.
    """

    keys = tuple(dict.fromkeys(str(key).strip() for key in contra_signal_keys if str(key).strip()))
    dd_improvement_threshold = float(min_drawdown_improvement_pct or 0.0)
    pnl_decay_limit = float(max_pnl_degradation_pct or 0.0)
    risk_multiplier = max(0.0, min(1.0, float(risk_mult)))
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    validation_count = 0
    oos_count = 0

    for raw_split in split_results:
        row = _split_row(raw_split)
        row["max_dd_improvement_pct"] = -row["max_dd_delta_pct"]
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
        dd_improvement = row["max_dd_improvement_pct"]
        if pnl_delta < -pnl_decay_limit:
            failures.append(f"{role}_pnl_degradation_too_large:{name}")
        if dd_improvement < dd_improvement_threshold:
            failures.append(f"{role}_drawdown_improvement_too_small:{name}")

    if require_validation_split and validation_count <= 0:
        failures.append("validation_split_missing")
    if require_oos_split and oos_count <= 0:
        failures.append("oos_split_missing")
    if not keys:
        failures.append("contra_signal_keys_missing")
    if risk_multiplier <= 0.0:
        failures.append("risk_mult_not_positive")

    eligible = not failures
    rejected_reason = failures[0] if failures else ""
    return {
        "summary": {
            "eligible": eligible,
            "mode": "risk_sizing",
            "contra_signal_keys": len(keys),
            "split_count": len(rows),
            "validation_split_count": validation_count,
            "oos_split_count": oos_count,
            "require_oos_split": bool(require_oos_split),
            "risk_mult": risk_multiplier,
            "min_drawdown_improvement_pct": dd_improvement_threshold,
            "max_pnl_degradation_pct": pnl_decay_limit,
        },
        "risk_sizing_contra_signal_keys": [
            {"signal_key": key, "risk_mult": risk_multiplier}
            for key in keys
        ] if eligible else [],
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
    require_oos_split: bool = True,
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


def write_flash_contra_risk_sizing_manifest(
    output_dir: str | Path,
    *,
    contra_signal_keys: Sequence[str],
    split_results: Sequence[Mapping[str, Any]],
    risk_mult: float = 0.50,
    min_drawdown_improvement_pct: float = 1.0,
    max_pnl_degradation_pct: float = 0.25,
    require_validation_split: bool = True,
    require_oos_split: bool = True,
) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_flash_contra_risk_sizing_manifest(
        contra_signal_keys=contra_signal_keys,
        split_results=split_results,
        risk_mult=risk_mult,
        min_drawdown_improvement_pct=min_drawdown_improvement_pct,
        max_pnl_degradation_pct=max_pnl_degradation_pct,
        require_validation_split=require_validation_split,
        require_oos_split=require_oos_split,
    )
    json_path = out / "flash_genetics_contra_risk_sizing_manifest.json"
    md_path = out / "flash_genetics_contra_risk_sizing_manifest.md"
    json_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    md_path.write_text(_markdown_risk_sizing_manifest(manifest), encoding="utf-8")
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
        f"- Requires OOS: `{bool(summary.get('require_oos_split'))}`",
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


def _markdown_risk_sizing_manifest(manifest: Mapping[str, Any]) -> str:
    summary = _mapping(manifest.get("summary"))
    lines = [
        "# Flash Genetics Contra Risk Sizing Manifest",
        "",
        f"- Eligible: `{bool(summary.get('eligible'))}`",
        f"- Contra signal keys: `{summary.get('contra_signal_keys', 0)}`",
        f"- Risk multiplier: `{_float(summary.get('risk_mult')):.4f}`",
        f"- Min DD improvement %: `{_float(summary.get('min_drawdown_improvement_pct')):.4f}`",
        f"- Max PnL degradation %: `{_float(summary.get('max_pnl_degradation_pct')):.4f}`",
        f"- Validation splits: `{summary.get('validation_split_count', 0)}`",
        f"- OOS splits: `{summary.get('oos_split_count', 0)}`",
        f"- Requires OOS: `{bool(summary.get('require_oos_split'))}`",
        "",
        "## Risk Sizing Keys",
    ]
    sized = list(manifest.get("risk_sizing_contra_signal_keys") or ())
    if sized:
        for item in sized:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                f"- `{item.get('signal_key', '')}` risk_mult=`{_float(item.get('risk_mult')):.4f}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Promotion Failures"])
    failures = list(manifest.get("promotion_failures") or ())
    lines.extend(f"- `{failure}`" for failure in failures) if failures else lines.append("- none")
    lines.extend(["", "## Split Results", ""])
    lines.append("| Split | Role | Baseline PnL % | Candidate PnL % | Delta % | Baseline DD % | Candidate DD % | DD Improvement % |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in manifest.get("split_results") or ():
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {name} | {role} | {bp:.4f} | {cp:.4f} | {delta:.4f} | {bdd:.4f} | {cdd:.4f} | {ddi:.4f} |".format(
                name=row.get("name", ""),
                role=row.get("role", ""),
                bp=_float(row.get("baseline_pnl_pct")),
                cp=_float(row.get("candidate_pnl_pct")),
                delta=_float(row.get("pnl_delta_pct")),
                bdd=_float(row.get("baseline_max_dd_pct")),
                cdd=_float(row.get("candidate_max_dd_pct")),
                ddi=_float(row.get("max_dd_improvement_pct")),
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
    parser.add_argument("--risk-sizing-mode", action="store_true")
    parser.add_argument("--risk-mult", type=float, default=0.50)
    parser.add_argument("--min-drawdown-improvement-pct", type=float, default=1.0)
    parser.add_argument("--max-pnl-degradation-pct", type=float, default=0.25)
    parser.add_argument("--require-oos-split", action="store_true")
    parser.add_argument(
        "--allow-validation-only-manifest",
        action="store_true",
        help=(
            "Allow an intermediate validation-only manifest for OOS experiments. "
            "Final promotion manifests require OOS by default."
        ),
    )
    args = parser.parse_args(argv)
    splits = [_parse_split_arg(raw) for raw in args.split]
    require_oos = bool(args.require_oos_split) or not bool(args.allow_validation_only_manifest)
    if args.risk_sizing_mode:
        json_path, md_path = write_flash_contra_risk_sizing_manifest(
            args.output_dir,
            contra_signal_keys=args.contra_signal_key,
            split_results=splits,
            risk_mult=args.risk_mult,
            min_drawdown_improvement_pct=args.min_drawdown_improvement_pct,
            max_pnl_degradation_pct=args.max_pnl_degradation_pct,
            require_oos_split=require_oos,
        )
    else:
        json_path, md_path = write_flash_contra_validation_manifest(
            args.output_dir,
            contra_signal_keys=args.contra_signal_key,
            split_results=splits,
            min_validation_pnl_delta_pct=args.min_validation_pnl_delta_pct,
            max_validation_dd_worsening_pct=args.max_validation_dd_worsening_pct,
            require_oos_split=require_oos,
        )
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
