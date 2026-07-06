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
RUNTIME = SRC / "panteon_runtime"
TOOLS = ROOT / "tools"
for path in (SRC, RUNTIME, TOOLS):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from panteon_v2.app import startup  # noqa: E402
from panteon_v2.domain.types import Action  # noqa: E402
from panteon_v2.selection import FlashAllocatorConfig  # noqa: E402
from run_panteon3_live_canary_check import build_canary_summary  # noqa: E402


DEFAULT_EXCHANGES = ("MEXC", "BITGET")
DEFAULT_ACTOR = "LiveOIBreakout"
DEFAULT_FALLBACK_ACTOR = "LiveVolCompress"
DEFAULT_CANARY_SYMBOLS = ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK")
DEFAULT_EXPLORATION_RISK_MULT = 1.0
DEFAULT_PAPER_PROBE_ACTION = Action.FUT_SHORT_FULL
DEFAULT_EXECUTION_SMOKE_MAX_BARS = 3
DEFAULT_PAPER_MIN_NOTIONAL_FLOOR_USD = 5.0
MIN_DIAGNOSTIC_WARMUP_BARS_BY_ACTOR = {
    "LiveOIBreakout": 2 * 60 + 21,
}
ACTOR_OVERRIDE_CASTS = {
    "CHECK_INT": int,
    "MOM_MIN": float,
    "VOL_MULT": float,
}


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
    terminal_denied_context_signal_keys: Sequence[str] | None = None,
    bypass_terminal_denies: bool = False,
) -> FlashAllocatorConfig:
    aliases = single_component_actor_aliases(actor_label)
    kwargs: dict[str, Any] = {
        "live_real_actor_whitelist": aliases,
        "range_low_vol_real_actor_allowlist": aliases,
        "promotion_derived_router_enabled": True,
        "promotion_derived_actor_labels": (_base_actor_label(actor_label),),
        "promotion_derived_min_closed_trades": 0,
        "promotion_derived_min_expectancy": -999.0 if diagnostic else base.promotion_derived_min_expectancy,
        "promotion_derived_dynamic_best_enabled": (
            True if diagnostic else base.promotion_derived_dynamic_best_enabled
        ),
        "promotion_derived_risk_mult": (
            float(exploration_risk_mult)
            if diagnostic
            else base.promotion_derived_risk_mult
        ),
        "promotion_derived_min_notional_sizing_enabled": (
            True if diagnostic else base.promotion_derived_min_notional_sizing_enabled
        ),
        "promotion_derived_min_notional_max_risk_mult": (
            1.0 if diagnostic else base.promotion_derived_min_notional_max_risk_mult
        ),
        "promotion_derived_default_min_notional_usd": (
            DEFAULT_PAPER_MIN_NOTIONAL_FLOOR_USD
            if diagnostic
            else base.promotion_derived_default_min_notional_usd
        ),
        "controlled_exploration_enabled": True if diagnostic else base.controlled_exploration_enabled,
        "controlled_exploration_allowed_reasons": (
            ("no_evidence", "score_below_threshold")
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
        "controlled_exploration_min_notional_sizing_enabled": (
            True if diagnostic else base.controlled_exploration_min_notional_sizing_enabled
        ),
        "controlled_exploration_min_notional_max_risk_mult": (
            1.0 if diagnostic else base.controlled_exploration_min_notional_max_risk_mult
        ),
        "controlled_exploration_default_min_notional_usd": (
            DEFAULT_PAPER_MIN_NOTIONAL_FLOOR_USD
            if diagnostic
            else base.controlled_exploration_default_min_notional_usd
        ),
        "causal_actor_router_enabled": False,
        "max_signals_per_actor": 8 if diagnostic else base.max_signals_per_actor,
    }
    if bypass_terminal_denies:
        kwargs["terminal_denied_signal_keys"] = ()
        kwargs["terminal_denied_context_signal_keys"] = ()
    elif terminal_denied_context_signal_keys:
        kwargs["terminal_denied_context_signal_keys"] = tuple(
            dict.fromkeys(
                [
                    *tuple(base.terminal_denied_context_signal_keys),
                    *_parse_context_signal_keys(terminal_denied_context_signal_keys),
                ]
            )
        )
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


def _base_symbol(raw: str) -> str:
    text = str(raw or "").strip().upper()
    if not text:
        return ""
    text = text.replace("-", "/")
    if "/" in text:
        text = text.split("/", 1)[0]
    if text.endswith("USDT") and len(text) > 4:
        text = text[:-4]
    return text.strip()


def _parse_symbols(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        if not raw.strip():
            return ()
        parts: Sequence[str] = tuple(raw.split(","))
    else:
        parts = raw
    symbols = tuple(dict.fromkeys(_base_symbol(part) for part in parts if _base_symbol(part)))
    return symbols


def _parse_context_signal_keys(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        parts: Sequence[str] = raw.replace(";", ",").split(",")
    else:
        collected: list[str] = []
        for item in raw:
            collected.extend(str(item or "").replace(";", ",").split(","))
        parts = collected
    return tuple(dict.fromkeys(part.strip() for part in parts if part.strip()))


def _load_candidate_policy(path: str | Path | None) -> dict[str, Any]:
    text = str(path or "").strip()
    if not text:
        return {}
    payload = json.loads(Path(text).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SystemExit("--candidate-policy must contain a JSON object")
    return dict(payload)


def _effective_terminal_denied_context_signal_keys(
    cli_keys: Sequence[str] | None,
    results: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    keys: list[str] = list(_parse_context_signal_keys(cli_keys))
    for result in results:
        keys.extend(
            _parse_context_signal_keys(
                result.get("runtime_terminal_denied_context_signal_keys")  # type: ignore[arg-type]
            )
        )
    return tuple(dict.fromkeys(keys))


def _configure_paper_min_notional_floor(
    pipeline: object,
    symbols: Sequence[str],
    *,
    min_notional_usd: float = DEFAULT_PAPER_MIN_NOTIONAL_FLOOR_USD,
) -> int:
    executor = getattr(pipeline, "executor", None)
    exchange = getattr(executor, "_exchange", None)
    setter = getattr(exchange, "set_min_notional", None)
    if not callable(setter):
        return 0
    count = 0
    for symbol in _parse_symbols(symbols):
        for key in (symbol, f"{symbol}/USDT"):
            setter(key, float(min_notional_usd))
            count += 1
    return count


def _clean_actor_overrides(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if not raw:
        return {}
    overrides: dict[str, Any] = {}
    for key, value in raw.items():
        clean_key = str(key or "").strip().upper()
        if value is None:
            continue
        caster = ACTOR_OVERRIDE_CASTS.get(clean_key)
        if caster is None:
            raise ValueError(f"unsupported actor override: {key}")
        overrides[clean_key] = caster(value)
    return overrides


def _actor_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return _clean_actor_overrides(
        {
            "CHECK_INT": args.actor_check_int,
            "MOM_MIN": args.actor_mom_min,
            "VOL_MULT": args.actor_vol_mult,
        }
    )


def _apply_actor_overrides(agent: object, overrides: Mapping[str, Any]) -> None:
    if not overrides:
        return
    for target in _actor_override_targets(agent):
        for key, value in overrides.items():
            setattr(target, key, value)


def _actor_override_targets(agent: object) -> tuple[object, ...]:
    targets: list[object] = []
    seen: set[int] = set()
    stack = [agent]
    while stack:
        target = stack.pop()
        ident = id(target)
        if ident in seen:
            continue
        seen.add(ident)
        targets.append(target)
        for attr in ("v1_agent", "base_agent", "inner"):
            child = getattr(target, attr, None)
            if child is not None:
                stack.append(child)
    return tuple(targets)


def _is_open_action(action: object) -> bool:
    if isinstance(action, Action):
        return bool(action.is_open)
    try:
        return bool(Action(int(action)).is_open)
    except Exception:
        return False


def _is_hold_action(action: object) -> bool:
    if isinstance(action, Action):
        return bool(action.is_hold)
    try:
        return Action(int(action)) == Action.HOLD
    except Exception:
        return False


def _market_regime_label(market: object, symbol: str) -> str:
    getter = getattr(market, "regime_for_symbol", None)
    regime = None
    if callable(getter):
        try:
            regime = getter(symbol)
        except Exception:
            regime = None
    if regime is None:
        regime = getattr(market, "regime", None)
    label = getattr(regime, "label", None) or getattr(regime, "name", None) or regime
    return str(label or "").strip().lower()


def _is_directional_probe_symbol(market: object, symbol: str) -> bool:
    return _market_regime_label(market, symbol) in {"bullish", "bearish", "crash"}


def _probe_symbol_price(prices: Mapping[object, object], symbol: str) -> float:
    try:
        return float(prices.get(symbol, 0.0) or 0.0)
    except Exception:
        return 0.0


class PaperCanaryProbeAgent:
    """Paper-only diagnostic wrapper that emits one probe open when the actor is idle."""

    def __init__(
        self,
        inner: object,
        *,
        label: str,
        probe_action: Action = DEFAULT_PAPER_PROBE_ACTION,
    ) -> None:
        self.inner = inner
        self.label = label
        self.probe_action = probe_action
        self.prefers_full_market_snapshot = bool(
            getattr(inner, "prefers_full_market_snapshot", False)
        )
        self.last_signal_diagnostics: dict[str, dict[str, Any]] = {}
        for attr in (
            "shadow_only",
            "paper_trading_eligible",
            "live_trading_eligible",
            "single_component_canary_enabled",
        ):
            if hasattr(inner, attr):
                setattr(self, attr, getattr(inner, attr))

    def act(self, market: object) -> dict[str, Action]:
        raw = self.inner.act(market) or {}
        actions = {
            str(symbol): (
                action if isinstance(action, Action) else Action(int(action))
            )
            for symbol, action in raw.items()
        }
        base_diag = dict(getattr(self.inner, "last_signal_diagnostics", {}) or {})
        open_symbols = [
            symbol for symbol, action in actions.items() if _is_open_action(action)
        ]
        if open_symbols and any(
            _is_directional_probe_symbol(market, symbol) for symbol in open_symbols
        ):
            self.last_signal_diagnostics = base_diag
            return actions

        prices = getattr(market, "prices", {}) or {}
        if not isinstance(prices, Mapping) or not prices:
            self.last_signal_diagnostics = base_diag
            return actions

        ordered_symbols = sorted(str(symbol) for symbol in prices if str(symbol or "").strip())
        if not ordered_symbols:
            self.last_signal_diagnostics = base_diag
            return actions

        probe_symbol = max(
            ordered_symbols,
            key=lambda symbol: (
                _is_directional_probe_symbol(market, symbol),
                _probe_symbol_price(prices, symbol),
            ),
        )
        out = {symbol: Action.HOLD for symbol in ordered_symbols}
        out.update({symbol: action for symbol, action in actions.items() if symbol in out})
        if not _is_hold_action(out.get(probe_symbol, Action.HOLD)):
            self.last_signal_diagnostics = base_diag
            return out

        out[probe_symbol] = self.probe_action
        symbol_diag = dict(base_diag.get(probe_symbol, {}) or {})
        self.last_signal_diagnostics = {
            **base_diag,
            probe_symbol: {
                **symbol_diag,
                "reason": "paper_canary_probe_open",
                "base_reason": str(symbol_diag.get("reason") or "hold"),
                "paper_canary_probe": True,
                "probe_action": self.probe_action.name,
            },
        }
        return out


def _diagnostic_warmup_bars(actor_label: str, requested: int | None) -> int | None:
    if requested is None:
        return None
    minimum = MIN_DIAGNOSTIC_WARMUP_BARS_BY_ACTOR.get(_base_actor_label(actor_label))
    if minimum is None:
        return max(0, int(requested))
    return max(int(requested), int(minimum))


def configure_single_component_pipeline(
    pipeline: object,
    actor_label: str,
    *,
    paper_probe_on_idle: bool = False,
    actor_overrides: Mapping[str, Any] | None = None,
) -> str:
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
    clean_actor_overrides = _clean_actor_overrides(actor_overrides)
    _apply_actor_overrides(agent, clean_actor_overrides)
    setattr(agent, "shadow_only", False)
    setattr(agent, "paper_trading_eligible", True)
    setattr(agent, "live_trading_eligible", True)
    setattr(agent, "single_component_canary_enabled", True)
    if paper_probe_on_idle:
        agent = PaperCanaryProbeAgent(agent, label=selected)
        setattr(agent, "shadow_only", False)
        setattr(agent, "paper_trading_eligible", True)
        setattr(agent, "live_trading_eligible", True)
        setattr(agent, "single_component_canary_enabled", True)
        register = getattr(registry, "register", None)
        if callable(register):
            register(agent, replace=True)

    setattr(pipeline, "profiles", ())
    setattr(pipeline, "manual_quarantine_labels", ())
    setattr(pipeline, "quarantine_override_labels", (selected,))
    setattr(pipeline, "single_component_canary_actor_overrides", clean_actor_overrides)
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
    symbols: Sequence[str] | None = None,
    paper_probe_on_idle: bool = False,
    actor_overrides: Mapping[str, Any] | None = None,
    terminal_denied_context_signal_keys: Sequence[str] | None = None,
    bypass_terminal_denies: bool = False,
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
        terminal_denied_context_signal_keys=terminal_denied_context_signal_keys,
        bypass_terminal_denies=bool(bypass_terminal_denies),
    )
    live_execution_config = startup._resolve_live_execution_config(exchange_key)
    if diagnostic:
        live_execution_config = replace(live_execution_config, max_new_opens_per_bar=8)
    selected_holder = {"label": _base_actor_label(actor_label)}
    clean_actor_overrides = _clean_actor_overrides(actor_overrides)
    runtime_flash_config_holder: dict[str, Any] = {
        "available": False,
        "terminal_denied_signal_keys": (),
        "terminal_denied_context_signal_keys": (),
    }

    def _configure(pipeline: object) -> None:
        selected_holder["label"] = configure_single_component_pipeline(
            pipeline,
            actor_label,
            paper_probe_on_idle=bool(paper_probe_on_idle),
            actor_overrides=clean_actor_overrides,
        )

    env = {
        "CRYPTO_EXCHANGE": exchange_key,
        f"{exchange_key}_TRADING_MODE": "paper_live_feed",
        "PANTEON_V2_PAPER_CANARY_CLOSE_ON_SHUTDOWN": "1",
        "PANTEON_ALLOW_LIVE_FEED_FALLBACK": "0",
        "PANTEON_V2_FUTURES_SIGNAL_FIXES_ENABLED": "1",
        f"{exchange_key}_PANTEON_V2_FUTURES_SIGNAL_FIXES_ENABLED": "1",
    }
    symbol_keys = _parse_symbols(symbols)
    if symbol_keys:
        env[f"{exchange_key}_SYMBOLS"] = ",".join(symbol_keys)
    min_notional_holder = {"configured": 0}

    started_at = datetime.now(timezone.utc)
    with _temporary_env(env):
        def _configure_with_runtime(pipeline: object) -> None:
            _configure(pipeline)
            flash_allocator = getattr(pipeline, "flash_allocator", None)
            if flash_allocator is not None:
                setattr(flash_allocator, "_config", flash_config)
                runtime_config = getattr(flash_allocator, "_config", None)
                runtime_flash_config_holder["available"] = True
                runtime_flash_config_holder[
                    "terminal_denied_signal_keys"
                ] = tuple(
                    getattr(
                        runtime_config,
                        "terminal_denied_signal_keys",
                        (),
                    )
                    or ()
                )
                runtime_flash_config_holder[
                    "terminal_denied_context_signal_keys"
                ] = tuple(
                    getattr(
                        runtime_config,
                        "terminal_denied_context_signal_keys",
                        (),
                    )
                    or ()
                )
            min_notional_holder["configured"] = _configure_paper_min_notional_floor(
                pipeline,
                symbol_keys,
            )

        rc = startup.start_production(
            exchange=exchange_key,
            mode="paper_live_feed",
            initial_capital=initial_capital,
            max_bars=max_bars,
            max_idle_polls=max_idle_polls,
            sleep_between_polls_sec=sleep_between_polls_sec,
            use_v1_bridge=True,
            results_root=str(root),
            warmup_bars=_diagnostic_warmup_bars(actor_label, warmup_bars)
            if diagnostic
            else warmup_bars,
            snapshot_path=str(root / "state" / f"{exchange_lower}_single_component_snapshot.json"),
            jsonl_event_log=str(root / "logs" / f"{exchange_lower}_single_component_events.jsonl"),
            include_genetics=False,
            live_execution_config_override=live_execution_config,
            flash_enabled_override=True,
            flash_allocator_config_override=flash_config,
            configure_pipeline=_configure_with_runtime,
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
        "symbols": list(symbol_keys),
        "paper_probe_on_idle": bool(paper_probe_on_idle),
        "paper_min_notional_floor_usd": DEFAULT_PAPER_MIN_NOTIONAL_FLOOR_USD,
        "paper_min_notional_symbols_configured": int(min_notional_holder["configured"]),
        "actor_overrides": dict(clean_actor_overrides),
        "calibration_only": bool(clean_actor_overrides),
        "execution_smoke_bypass_terminal_denies": bool(bypass_terminal_denies),
        "terminal_denied_context_signal_keys": list(
            _parse_context_signal_keys(terminal_denied_context_signal_keys)
        ),
        "runtime_flash_config_available": bool(
            runtime_flash_config_holder["available"]
        ),
        "runtime_terminal_denied_context_signal_keys": list(
            runtime_flash_config_holder["terminal_denied_context_signal_keys"]
        ),
        "runtime_terminal_denied_signal_keys": list(
            runtime_flash_config_holder["terminal_denied_signal_keys"]
        ),
        "effective_warmup_bars": (
            _diagnostic_warmup_bars(actor_label, warmup_bars)
            if diagnostic
            else warmup_bars
        ),
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
    symbols: Sequence[str] | None,
    paper_probe_on_idle: bool,
    actor_overrides: Mapping[str, Any] | None,
    terminal_denied_context_signal_keys: Sequence[str] | None,
    bypass_terminal_denies: bool,
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
            symbols=symbols,
            paper_probe_on_idle=paper_probe_on_idle,
            actor_overrides=actor_overrides,
            terminal_denied_context_signal_keys=terminal_denied_context_signal_keys,
            bypass_terminal_denies=bool(bypass_terminal_denies),
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
            symbols=symbols,
            paper_probe_on_idle=paper_probe_on_idle,
            actor_overrides=actor_overrides,
            terminal_denied_context_signal_keys=terminal_denied_context_signal_keys,
            bypass_terminal_denies=bool(bypass_terminal_denies),
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
    parser.add_argument("--exploration-risk-mult", type=float, default=DEFAULT_EXPLORATION_RISK_MULT)
    parser.add_argument("--max-bars", type=int, default=3)
    parser.add_argument("--max-idle-polls", type=int, default=18)
    parser.add_argument("--sleep-between-polls-sec", type=float, default=5.0)
    parser.add_argument("--warmup-bars", type=int, default=1440)
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_CANARY_SYMBOLS),
        help="Comma-separated exchange symbols for isolated canary runs; empty keeps runtime settings.",
    )
    parser.add_argument(
        "--strict-gates",
        action="store_true",
        help="Keep normal admission gates; default is diagnostic single-component gating.",
    )
    parser.add_argument(
        "--require-positive-expectancy",
        action="store_true",
        help="Keep positive expectancy as a hard canary gate even for diagnostic runs.",
    )
    parser.add_argument(
        "--execution-smoke",
        action="store_true",
        help="Treat this bounded diagnostic run as an execution-smoke check, not a performance gate.",
    )
    parser.add_argument(
        "--actor-check-int",
        type=int,
        default=None,
        help="Calibration-only override for the selected actor CHECK_INT.",
    )
    parser.add_argument(
        "--actor-mom-min",
        type=float,
        default=None,
        help="Calibration-only override for the selected actor MOM_MIN.",
    )
    parser.add_argument(
        "--actor-vol-mult",
        type=float,
        default=None,
        help="Calibration-only override for the selected actor VOL_MULT.",
    )
    parser.add_argument(
        "--terminal-deny-context-signal-key",
        action="append",
        default=None,
        help=(
            "Paper/prelive policy override in actor_key|symbol|action|regime format. "
            "Repeatable; comma-separated values are also accepted."
        ),
    )
    parser.add_argument("--candidate-policy", default="")
    parser.add_argument("--lookback-minutes", type=float, default=360.0)
    args = parser.parse_args(argv)

    exchanges = tuple(args.exchanges or DEFAULT_EXCHANGES)
    results: list[dict[str, Any]] = []
    results_root = Path(args.results_root)
    default_symbols_arg = ",".join(DEFAULT_CANARY_SYMBOLS)
    candidate_policy = _load_candidate_policy(args.candidate_policy)
    symbols = _parse_symbols(args.symbols)
    actor_overrides = _actor_overrides_from_args(args)
    terminal_denied_context_signal_keys = _parse_context_signal_keys(
        args.terminal_deny_context_signal_key
    )
    if candidate_policy:
        if str(candidate_policy.get("exchange") or "").upper() != "BITGET":
            raise SystemExit("--candidate-policy is not a BITGET policy")
        if any(str(exchange or "").upper() != "BITGET" for exchange in exchanges):
            raise SystemExit("--candidate-policy can only be used with --exchange BITGET")
        if str(args.actor) != str(candidate_policy.get("actor") or ""):
            raise SystemExit("--actor must match candidate policy actor")
        policy_symbols = _parse_symbols(candidate_policy.get("symbols"))  # type: ignore[arg-type]
        if policy_symbols and (str(args.symbols or "") == default_symbols_arg or not symbols):
            symbols = policy_symbols
        terminal_denied_context_signal_keys = tuple(dict.fromkeys((
            *terminal_denied_context_signal_keys,
            *_parse_context_signal_keys(
                candidate_policy.get("terminal_deny_context_signal_keys")  # type: ignore[arg-type]
            ),
        )))
    max_bars = max(0, int(args.max_bars))
    execution_smoke = bool(
        args.execution_smoke
        or (
            not bool(args.strict_gates)
            and max_bars <= int(DEFAULT_EXECUTION_SMOKE_MAX_BARS)
        )
    )
    require_positive_expectancy = bool(
        args.strict_gates
        or args.require_positive_expectancy
        or not execution_smoke
    )
    bypass_terminal_denies = bool(execution_smoke)
    for exchange in exchanges:
        result = _run_exchange_with_fallback(
            exchange,
            actor_label=str(args.actor),
            fallback_actor_label=str(args.fallback_actor or ""),
            results_root=results_root,
            initial_capital=args.initial_capital,
            max_bars=max_bars,
            max_idle_polls=max(0, int(args.max_idle_polls)),
            sleep_between_polls_sec=max(0.0, float(args.sleep_between_polls_sec)),
            warmup_bars=max(0, int(args.warmup_bars)),
            diagnostic=not bool(args.strict_gates),
            exploration_risk_mult=float(args.exploration_risk_mult),
            symbols=symbols,
            paper_probe_on_idle=execution_smoke,
            actor_overrides=actor_overrides,
            terminal_denied_context_signal_keys=terminal_denied_context_signal_keys,
            bypass_terminal_denies=bypass_terminal_denies,
        )
        results.append(result)

    effective_terminal_denied_context_signal_keys = (
        ()
        if bypass_terminal_denies
        else _effective_terminal_denied_context_signal_keys(
            terminal_denied_context_signal_keys,
            results,
        )
    )
    summary = build_canary_summary(
        results_root=results_root,
        reports_dir=args.reports_dir,
        exchanges=exchanges,
        lookback_minutes=float(args.lookback_minutes),
        require_positive_expectancy=require_positive_expectancy,
        calibration_only=bool(actor_overrides),
        actor_overrides=actor_overrides,
        terminal_denied_context_signal_keys=effective_terminal_denied_context_signal_keys,
        candidate_policy=candidate_policy,
        execution_smoke=execution_smoke,
    )
    payload = {
        "runner": "single_component_canary",
        "requested_actor": str(args.actor),
        "fallback_actor": str(args.fallback_actor or ""),
        "execution_smoke": execution_smoke,
        "execution_smoke_bypass_terminal_denies": bypass_terminal_denies,
        "calibration_only": bool(actor_overrides),
        "actor_overrides": actor_overrides,
        "candidate_policy": candidate_policy,
        "runs": results,
        "canary_summary": summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if summary.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
