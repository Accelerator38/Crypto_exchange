"""Build a context-only selected-subset overlay on top of a base manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from panteon_v2.analysis.flash_selected_subset_manifest import (
    build_flash_selected_subset_manifest,
)


def build_flash_context_selected_subset_manifest(
    *,
    base_manifest_path: str | Path,
    confirmation_attribution_paths: Sequence[str | Path],
    context_min_closed_trades: int = 1,
    context_score_boost: float = 0.15,
    context_risk_positive_mult: float = 1.0,
) -> dict[str, Any]:
    """Return base selected-subset manifest plus generated context-only entries."""

    if context_min_closed_trades < 1:
        raise ValueError("context_min_closed_trades must be >= 1")
    if context_score_boost < 0:
        raise ValueError("context_score_boost must be >= 0")
    if context_risk_positive_mult <= 0:
        raise ValueError("context_risk_positive_mult must be > 0")

    base_path = Path(base_manifest_path)
    base = _load_manifest(base_path)
    inputs = tuple(Path(path) for path in confirmation_attribution_paths)
    context_source = build_flash_selected_subset_manifest(
        confirmation_attribution_paths=inputs,
        min_closed_trades=4,
        risk_weak_mult=1.0,
        context_min_closed_trades=context_min_closed_trades,
        context_score_boost=context_score_boost,
        context_risk_positive_mult=context_risk_positive_mult,
    )

    manifest = dict(base)
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["context_selected_subset_overlay"] = {
        "base_manifest": str(base_path),
        "inputs": [str(path) for path in inputs],
        "parameters": {
            "context_min_closed_trades": int(context_min_closed_trades),
            "context_score_boost": float(context_score_boost),
            "context_risk_positive_mult": float(context_risk_positive_mult),
        },
        "candidate_context_score_boosts": len(
            context_source.get("context_score_boosts", [])
        ),
        "candidate_context_risk_mult_overrides": len(
            context_source.get("context_risk_mult_overrides", [])
        ),
    }
    manifest["context_score_boosts"] = _merge_context_items(
        base.get("context_score_boosts", []),
        context_source.get("context_score_boosts", []),
    )
    manifest["context_risk_mult_overrides"] = _merge_context_items(
        base.get("context_risk_mult_overrides", []),
        context_source.get("context_risk_mult_overrides", []),
    )
    return manifest


def write_flash_context_selected_subset_manifest(
    *,
    output_path: str | Path,
    base_manifest_path: str | Path,
    confirmation_attribution_paths: Sequence[str | Path],
    context_min_closed_trades: int = 1,
    context_score_boost: float = 0.15,
    context_risk_positive_mult: float = 1.0,
) -> Path:
    manifest = build_flash_context_selected_subset_manifest(
        base_manifest_path=base_manifest_path,
        confirmation_attribution_paths=confirmation_attribution_paths,
        context_min_closed_trades=context_min_closed_trades,
        context_score_boost=context_score_boost,
        context_risk_positive_mult=context_risk_positive_mult,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("base manifest must be a JSON object")
    return data


def _merge_context_items(
    base_items: object,
    generated_items: object,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in _dict_rows(base_items):
        key = _context_key(item)
        if not key or key in seen:
            continue
        rows.append(dict(item))
        seen.add(key)
    additions: list[dict[str, Any]] = []
    for item in _dict_rows(generated_items):
        key = _context_key(item)
        if not key or key in seen:
            continue
        additions.append(dict(item))
        seen.add(key)
    return rows + sorted(additions, key=_context_key)


def _dict_rows(raw: object) -> list[Mapping[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _context_key(item: Mapping[str, Any]) -> str:
    return str(item.get("context_key") or item.get("signal_key") or "").strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context-min-closed-trades", type=int, default=1)
    parser.add_argument("--context-score-boost", type=float, default=0.15)
    parser.add_argument("--context-risk-positive-mult", type=float, default=1.0)
    args = parser.parse_args(argv)

    path = write_flash_context_selected_subset_manifest(
        output_path=args.output,
        base_manifest_path=args.base_manifest,
        confirmation_attribution_paths=tuple(args.input),
        context_min_closed_trades=args.context_min_closed_trades,
        context_score_boost=args.context_score_boost,
        context_risk_positive_mult=args.context_risk_positive_mult,
    )
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
