from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GENETICS_DIR = ROOT / "Genetics_DL_Agents"
if str(GENETICS_DIR) not in sys.path:
    sys.path.insert(0, str(GENETICS_DIR))

import crypto_genetics as cg  # noqa: E402


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _parse_labeled_path(raw_spec: str) -> tuple[str, Path]:
    label, sep, raw_path = str(raw_spec).partition("=")
    if sep:
        clean_label = label.strip()
        path = _resolve_path(raw_path.strip())
    else:
        path = _resolve_path(raw_spec)
        clean_label = path.parent.name or path.stem
    if not clean_label:
        clean_label = path.parent.name or path.stem
    return clean_label, path


def _parse_regime_open_bias_specs(raw_specs: Optional[List[str]]) -> Dict[str, float]:
    if not raw_specs:
        return {}
    parsed: Dict[str, float] = {}
    for raw_spec in raw_specs:
        key, sep, value = str(raw_spec).partition("=")
        regime = key.strip().lower()
        if not sep or not regime:
            raise ValueError(f"Invalid --regime-open-bias value: {raw_spec!r}; expected REGIME=BIAS")
        parsed[regime] = float(value)
    return parsed


def _row_values(row: Any) -> Dict[str, float]:
    raw = row.get("prices") if isinstance(row, dict) and isinstance(row.get("prices"), dict) else row
    out: Dict[str, float] = {}
    for key, value in dict(raw or {}).items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0.0 and np.isfinite(numeric):
            out[str(key).upper()] = numeric
    return out


def _load_bridge_cache(path: Path, bars: int) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    price_rows = [_row_values(row) for row in (payload.get("price_hist") or [])]
    volume_rows = [_row_values(row) for row in (payload.get("volume_hist") or [])]
    if bars > 0:
        price_rows = price_rows[-bars:]
        volume_rows = volume_rows[-bars:]
    return price_rows, volume_rows, payload


def _make_agent(genome_path: Path, regime_open_bias: Optional[Dict[str, float]] = None) -> Any:
    genome = np.load(genome_path).astype(np.float32).ravel()
    if genome.size != int(cg.GENOME_SIZE):
        raise ValueError(f"{genome_path} genome size {genome.size} != expected {cg.GENOME_SIZE}")
    agent = cg.GeneticsAgent(genome=genome)
    if regime_open_bias:
        configure = getattr(agent, "configure_regime_adaptive_output_bias", None)
        if not callable(configure):
            raise ValueError("GeneticsAgent does not support regime adaptive output bias")
        configure(regime_open_bias, enabled=True)
    return agent


def _empty_virtual_position() -> Dict[str, float]:
    return {
        "spot_qty": 0.0,
        "spot_entry": 0.0,
        "fut_qty": 0.0,
        "fut_entry": 0.0,
    }


def _apply_virtual_position_state(
    agent: Any,
    prices: Dict[str, float],
    actions: Dict[str, int],
    positions: Dict[str, Dict[str, float]],
) -> Dict[str, int]:
    """Apply live-style exchange state updates after raw GeneticsAgent actions."""
    update = getattr(agent, "update_from_exchange", None)
    if not callable(update):
        return {}

    counts: Dict[str, int] = {}
    for symbol, raw_action in dict(actions or {}).items():
        try:
            action = int(raw_action)
            price = float(prices[symbol])
        except (KeyError, TypeError, ValueError):
            continue
        if price <= 0.0 or not np.isfinite(price):
            continue

        state = positions.setdefault(str(symbol), _empty_virtual_position())
        changed = False
        event = ""
        if action == 1:  # buy_spot
            state["spot_qty"] = 1.0
            state["spot_entry"] = price if state["spot_entry"] <= 0.0 else state["spot_entry"]
            changed = True
            event = "spot_buy"
        elif action == 2:  # sell_spot
            state["spot_qty"] = 0.0
            state["spot_entry"] = 0.0
            changed = True
            event = "spot_sell"
        elif action == 3:  # fut_long
            state["fut_qty"] = 1.0
            state["fut_entry"] = price if state["fut_entry"] <= 0.0 else state["fut_entry"]
            changed = True
            event = "fut_long"
        elif action == 4:  # fut_short
            state["fut_qty"] = -1.0
            state["fut_entry"] = price if state["fut_entry"] <= 0.0 else state["fut_entry"]
            changed = True
            event = "fut_short"
        elif action == 5:  # close_fut
            state["fut_qty"] = 0.0
            state["fut_entry"] = 0.0
            changed = True
            event = "fut_close"

        if not changed:
            continue
        update(
            str(symbol),
            float(state["spot_qty"]),
            float(state["spot_entry"]),
            float(state["fut_qty"]),
            float(state["fut_entry"]),
        )
        counts[event] = int(counts.get(event, 0) + 1)
    return counts


def _evaluate_activation(
    *,
    label: str,
    genome_path: Path,
    price_rows: list[dict[str, float]],
    volume_rows: list[dict[str, float]],
    regime_open_bias: Optional[Dict[str, float]] = None,
    portfolio_value: float = 100.0,
    virtual_fill_position_state: bool = False,
) -> Dict[str, Any]:
    agent = _make_agent(genome_path, regime_open_bias)
    non_hold_bars = 0
    action_counts: Dict[str, int] = {}
    symbol_counts: Dict[str, int] = {}
    virtual_fill_counts: Dict[str, int] = {}
    virtual_positions: Dict[str, Dict[str, float]] = {}
    first_non_hold: list[dict[str, Any]] = []
    t0 = time.time()

    for idx, prices in enumerate(price_rows, start=1):
        volumes = volume_rows[idx - 1] if idx - 1 < len(volume_rows) else {}
        actions = agent.act(prices, volumes, month=6, portfolio_value=portfolio_value)
        active = {
            str(symbol): int(action)
            for symbol, action in dict(actions or {}).items()
            if int(action) != 0
        }
        if not active:
            continue
        non_hold_bars += 1
        for symbol, action in active.items():
            action_counts[str(action)] = int(action_counts.get(str(action), 0) + 1)
            symbol_counts[symbol] = int(symbol_counts.get(symbol, 0) + 1)
        if virtual_fill_position_state:
            for event, count in _apply_virtual_position_state(agent, prices, active, virtual_positions).items():
                virtual_fill_counts[event] = int(virtual_fill_counts.get(event, 0) + count)
        if len(first_non_hold) < 10:
            first_non_hold.append({"bar_offset": idx, "actions": dict(active)})

    history_lengths = {
        str(symbol): int(len(history))
        for symbol, history in getattr(agent, "ph", {}).items()
    }
    rows = len(price_rows)
    return {
        "label": label,
        "path": str(genome_path),
        "regime_adaptive_output_bias": {
            "enabled": bool(regime_open_bias),
            "regime_open_bias": dict(regime_open_bias or {}),
            "last_trace": dict(getattr(agent, "last_regime_adaptive_output_bias", {}) or {}),
        },
        "rows": int(rows),
        "non_hold_bars": int(non_hold_bars),
        "non_hold_bar_rate": float(non_hold_bars / rows) if rows else 0.0,
        "action_counts": action_counts,
        "symbol_counts": symbol_counts,
        "virtual_fill_position_state": bool(virtual_fill_position_state),
        "virtual_fill_counts": virtual_fill_counts,
        "virtual_open_positions": {
            symbol: state
            for symbol, state in sorted(virtual_positions.items())
            if state.get("spot_qty", 0.0) > 0.0 or state.get("fut_qty", 0.0) != 0.0
        },
        "first_non_hold": first_non_hold,
        "history_min": int(min(history_lengths.values())) if history_lengths else 0,
        "history_max": int(max(history_lengths.values())) if history_lengths else 0,
        "elapsed_sec": float(time.time() - t0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate whether saved genetics genomes produce live-style non-HOLD actions on bridge cache."
    )
    parser.add_argument("--exchange", default="", help="Exchange label for the report.")
    parser.add_argument("--bridge-cache", required=True, help="Path to mexc/bitget bridge cache JSON.")
    parser.add_argument("--bars", type=int, default=1440, help="Number of latest bridge-cache bars to replay.")
    parser.add_argument(
        "--genome",
        action="append",
        default=None,
        help="Genome as LABEL=path or path. Can be repeated.",
    )
    parser.add_argument(
        "--regime-adaptive-output-bias-genome",
        action="append",
        default=None,
        help="Genome evaluated with --regime-open-bias as LABEL=path or path. Can be repeated.",
    )
    parser.add_argument(
        "--regime-open-bias",
        action="append",
        default=None,
        help="Adaptive open-output bias as REGIME=BIAS. Can be repeated.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "Results" / "neiro_genetics" / "live_activation_eval.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--virtual-fill-position-state",
        action="store_true",
        help=(
            "After each non-HOLD action, inject a simple virtual exchange position "
            "state into GeneticsAgent.update_from_exchange before the next bar."
        ),
    )
    args = parser.parse_args()

    bridge_cache = _resolve_path(args.bridge_cache)
    price_rows, volume_rows, cache_payload = _load_bridge_cache(bridge_cache, max(0, int(args.bars)))
    if not price_rows:
        raise RuntimeError(f"No price rows in bridge cache: {bridge_cache}")

    regime_open_bias = _parse_regime_open_bias_specs(args.regime_open_bias)
    genomes = [_parse_labeled_path(spec) for spec in (args.genome or [])]
    adaptive_genomes = [
        _parse_labeled_path(spec)
        for spec in (args.regime_adaptive_output_bias_genome or [])
    ]
    if adaptive_genomes and not regime_open_bias:
        raise ValueError("--regime-adaptive-output-bias-genome requires --regime-open-bias")

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exchange": str(args.exchange or "").upper(),
        "bridge_cache": str(bridge_cache),
        "cache_bar": cache_payload.get("bar"),
        "cache_symbols": list(cache_payload.get("symbols") or []),
        "bars_requested": int(args.bars),
        "bars_used": int(len(price_rows)),
        "volume_rows_used": int(len(volume_rows)),
        "genomes": [],
    }

    for label, path in genomes:
        print(f"[live-activation] genome={label} path={path}")
        report["genomes"].append(
            _evaluate_activation(
                label=label,
                genome_path=path,
                price_rows=price_rows,
                volume_rows=volume_rows,
                virtual_fill_position_state=bool(args.virtual_fill_position_state),
            )
        )

    for label, path in adaptive_genomes:
        print(f"[live-activation] adaptive_genome={label} path={path}")
        report["genomes"].append(
            _evaluate_activation(
                label=f"{label}::regime_adaptive_output_bias",
                genome_path=path,
                price_rows=price_rows,
                volume_rows=volume_rows,
                regime_open_bias=regime_open_bias,
                virtual_fill_position_state=bool(args.virtual_fill_position_state),
            )
        )

    out = _resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[live-activation] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
