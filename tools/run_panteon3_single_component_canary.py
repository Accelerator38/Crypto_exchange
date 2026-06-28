from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for path in (SRC, TOOLS):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from panteon_v2.app import startup  # noqa: E402
from panteon_v2.selection import FlashAllocatorConfig  # noqa: E402
from run_panteon3_live_canary_check import build_canary_summary  # noqa: E402


DEFAULT_EXCHANGES = ("MEXC", "BITGET")
DEFAULT_ACTOR = "LiveOIBreakout"
DEFAULT_FALLBACK_ACTOR = "LiveVolCompress"


def _base_actor_label(raw: str) -> str:
    text = str(raw or "").strip()
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    changed = True
    while changed:
        changed = False
        for prefix in ("Solo_", "V_"):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                changed = True
    return text


def single_component_actor_aliases(actor_label: str) -> tuple[str, ...]:
    base = _base_actor_label(actor_label)
    if not base:
        raise ValueError("actor label is required")
    aliases = [
        base,
        f"agent:{base}",
        f"Solo_{base}",
        f"ensemble:Solo_{base}",
    ]
    raw = str(actor_label or "").strip()
    if raw and raw not in aliases:
        aliases.insert(0, raw)
    return tuple(dict.fromkeys(aliases))


def single_component_flash_config(
    base: FlashAllocatorConfig,
    actor_label: str,
    *,
    diagnostic: bool = True,
    exploration_risk_mult: float = 0.03,
) -> FlashAllocatorConfig:
    aliases = single_component_actor_aliases(actor_label)
    kwargs: dict[str, Any] = {
        "live_real_actor_whitelist": aliases,
        "range_low_vol_real_actor_allowlist": aliases,
        "promotion_derived_router_enabled": True,
        "promotion_derived_actor_labels": (_base_actor_label(actor_label),),
        "promotion_derived_min_closed_trades": 0,
        "promotion_derived_min_expectancy": -999.0 if diagnostic else base.promotion_derived_min_expectancy,
        "controlled_exploration_enabled": True if diagnostic else base.controlled_exploration_enabled,
        "controlled_exploration_allowed_reasons": (
            ("no_evidence",)
            if diagnostic
            else base.controlled_exploration_allowed_reasons
        ),
        "controlled_exploration_risk_mult": (
            float(exploration_risk_mult)
            if diagnostic
            else base.controlled_exploration_risk_mult
        ),
        "controlled_exploration_min_shadow_score": (
            0.0 if diagnostic else base.controlled_exploration_min_shadow_score
        ),
        "controlled_exploration_min_shadow_closed": (
            0 if diagnostic else base.controlled_exploration_min_shadow_closed
        ),
        "causal_actor_router_enabled": False,
        "max_signals_per_actor": 1,
    }
    if diagnostic:
        kwargs.update(
            {
                "min_closed_trades_to_trade": 0,
                "min_pnl_pct_to_trade": -999.0,
                "min_score_to_trade": 0.0,
                "global_health_gate_enabled": False,
                "regime_edge_gate_enabled": False,
                "real_loss_gate_enabled": False,
                "fee_aware_admission_enabled": False,
                "shadow_confirmation_enabled": False,
                "shadow_symbol_confirmation_enabled": False,
                "shadow_actor_fallback_confirmation_enabled": False,
                "shadow_base_fallback_confirmation_enabled": False,
                "shadow_signal_handoff_enabled": False,
                "shadow_quality_confirmation_enabled": False,
            }
        )
    return replace(base, **kwargs)


def _registry_lookup_labels(actor_label: str) -> tuple[str, ...]:
    base = _base_actor_label(actor_label)
    return tuple(dict.fromkeys((actor_label, base, f"V_{base}", f"Solo_{base}")))


def configure_single_component_pipeline(pipeline: object, actor_label: str) -> str:
    registry = getattr(pipeline, "registry", None)
    if registry is None:
        raise RuntimeError("pipeline registry is missing")

    selected = ""
    for label in _registry_lookup_labels(actor_label):
        getter = getattr(registry, "get", None)
        if callable(getter) and getter(label) is not None:
            selected = label
            break
    if not selected:
        labels = getattr(registry, "all_labels", lambda: [])()
        raise RuntimeError(
            f"single-component canary actor {actor_label!r} is not registered; "
            f"available={list(labels)}"
        )

    for label in list(getattr(registry, "all_labels", lambda: [])()):
        if label != selected:
            unregister = getattr(registry, "unregister", None)
            if callable(unregister):
                unregister(label)

    agent = getattr(registry, "get")(selected)
    setattr(agent, "shadow_only", False)
    setattr(agent, "paper_trading_eligible", True)
    setattr(agent, "live_trading_eligible", True)
    setattr(agent, "single_component_canary_enabled", True)

    setattr(pipeline, "profiles", ())
    setattr(pipeline, "manual_quarantine_labels", ())
    setattr(pipeline, "quarantine_override_labels", (selected,))
    qm = getattr(pipeline, "qm", None)
    release = getattr(qm, "force_release_override", None)
    if callable(release):
        try:
            release(selected, reason="single_component_canary", bar=0)
        except TypeError:
            release(selected)
    setattr(pipeline, "single_component_canary_actor", selected)
    return selected


@contextmanager
def _temporary_env(overrides: Mapping[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_exchange(
    exchange: str,
    *,
    actor_label: str,
    results_root: str | Path,
    initial_capital: float | None,
    max_bars: int,
    max_idle_polls: int,
    sleep_between_polls_sec: float,
    warmup_bars: int | None,
    diagnostic: bool,
    exploration_risk_mult: float,
) -> dict[str, Any]:
    exchange_key = str(exchange or "").strip().upper()
    if exchange_key not in DEFAULT_EXCHANGES:
        raise ValueError(f"unsupported exchange: {exchange}")

    root = Path(results_root)
    (root / "state").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    exchange_lower = exchange_key.lower()
    base_config = startup._resolve_flash_allocator_config(exchange_key)
    flash_config = single_component_flash_config(
        base_config,
        actor_label,
        diagnostic=diagnostic,
        exploration_risk_mult=exploration_risk_mult,
    )
    selected_holder = {"label": _base_actor_label(actor_label)}

    def _configure(pipeline: object) -> None:
        selected_holder["label"] = configure_single_component_pipeline(
            pipeline,
            actor_label,
        )

    env = {
        "CRYPTO_EXCHANGE": exchange_key,
        f"{exchange_key}_TRADING_MODE": "paper_live_feed",
        "PANTEON_V2_PAPER_CANARY_CLOSE_ON_SHUTDOWN": "1",
        "PANTEON_ALLOW_LIVE_FEED_FALLBACK": "0",
        "PANTEON_V2_FUTURES_SIGNAL_FIXES_ENABLED": "1",
        f"{exchange_key}_PANTEON_V2_FUTURES_SIGNAL_FIXES_ENABLED": "1",
    }

    started_at = datetime.now(timezone.utc)
    with _temporary_env(env):
        rc = startup.start_production(
            exchange=exchange_key,
            mode="paper_live_feed",
            initial_capital=initial_capital,
            max_bars=max_bars,
            max_idle_polls=max_idle_polls,
            sleep_between_polls_sec=sleep_between_polls_sec,
            use_v1_bridge=True,
            results_root=str(root),
            warmup_bars=warmup_bars,
            snapshot_path=str(root / "state" / f"{exchange_lower}_single_component_snapshot.json"),
            jsonl_event_log=str(root / "logs" / f"{exchange_lower}_single_component_events.jsonl"),
            include_genetics=False,
            flash_enabled_override=True,
            flash_allocator_config_override=flash_config,
            configure_pipeline=_configure,
        )

    return {
        "exchange": exchange_key,
        "actor_label": selected_holder["label"],
        "requested_actor_label": actor_label,
        "return_code": int(rc or 0),
        "started_at": started_at.isoformat(),
        "results_root": str(root),
        "diagnostic": bool(diagnostic),
        "initial_capital": initial_capital,
        "exploration_risk_mult": float(exploration_risk_mult),
    }


def _run_exchange_with_fallback(
    exchange: str,
    *,
    actor_label: str,
    fallback_actor_label: str,
    results_root: Path,
    initial_capital: float | None,
    max_bars: int,
    max_idle_polls: int,
    sleep_between_polls_sec: float,
    warmup_bars: int | None,
    diagnostic: bool,
    exploration_risk_mult: float,
) -> dict[str, Any]:
    try:
        return run_exchange(
            exchange,
            actor_label=actor_label,
            results_root=results_root,
            initial_capital=initial_capital,
            max_bars=max_bars,
            max_idle_polls=max_idle_polls,
            sleep_between_polls_sec=sleep_between_polls_sec,
            warmup_bars=warmup_bars,
            diagnostic=diagnostic,
            exploration_risk_mult=exploration_risk_mult,
        )
    except RuntimeError as exc:
        if not fallback_actor_label or fallback_actor_label == actor_label:
            raise
        if "is not registered" not in str(exc):
            raise
        return run_exchange(
            exchange,
            actor_label=fallback_actor_label,
            results_root=results_root,
            initial_capital=initial_capital,
            max_bars=max_bars,
            max_idle_polls=max_idle_polls,
            sleep_between_polls_sec=sleep_between_polls_sec,
            warmup_bars=warmup_bars,
            diagnostic=diagnostic,
            exploration_risk_mult=exploration_risk_mult,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a paper/live-feed canary for one allowlisted Panteon component.",
    )
    parser.add_argument("--exchange", action="append", dest="exchanges")
    parser.add_argument("--actor", default=DEFAULT_ACTOR)
    parser.add_argument("--fallback-actor", default=DEFAULT_FALLBACK_ACTOR)
    parser.add_argument(
        "--results-root",
        default=str(ROOT / "Results" / "Panteon3SingleComponentCanary"),
    )
    parser.add_argument(
        "--reports-dir",
        default=str(ROOT / "Reports" / "Panteon3Canary" / "single_component"),
    )
    parser.add_argument("--initial-capital", type=float, default=None)
    parser.add_argument("--exploration-risk-mult", type=float, default=0.03)
    parser.add_argument("--max-bars", type=int, default=3)
    parser.add_argument("--max-idle-polls", type=int, default=18)
    parser.add_argument("--sleep-between-polls-sec", type=float, default=5.0)
    parser.add_argument("--warmup-bars", type=int, default=1440)
    parser.add_argument(
        "--strict-gates",
        action="store_true",
        help="Keep normal admission gates; default is diagnostic single-component gating.",
    )
    parser.add_argument("--lookback-minutes", type=float, default=360.0)
    args = parser.parse_args(argv)

    exchanges = tuple(args.exchanges or DEFAULT_EXCHANGES)
    results: list[dict[str, Any]] = []
    results_root = Path(args.results_root)
    for exchange in exchanges:
        result = _run_exchange_with_fallback(
            exchange,
            actor_label=str(args.actor),
            fallback_actor_label=str(args.fallback_actor or ""),
            results_root=results_root,
            initial_capital=args.initial_capital,
            max_bars=max(0, int(args.max_bars)),
            max_idle_polls=max(0, int(args.max_idle_polls)),
            sleep_between_polls_sec=max(0.0, float(args.sleep_between_polls_sec)),
            warmup_bars=max(0, int(args.warmup_bars)),
            diagnostic=not bool(args.strict_gates),
            exploration_risk_mult=float(args.exploration_risk_mult),
        )
        results.append(result)

    summary = build_canary_summary(
        results_root=results_root,
        reports_dir=args.reports_dir,
        exchanges=exchanges,
        lookback_minutes=float(args.lookback_minutes),
    )
    payload = {
        "runner": "single_component_canary",
        "requested_actor": str(args.actor),
        "fallback_actor": str(args.fallback_actor or ""),
        "runs": results,
        "canary_summary": summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if summary.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
