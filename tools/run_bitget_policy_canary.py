from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
TOOLS = ROOT / "tools"
for path in (SRC, RUNTIME, TOOLS):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from panteon_v2.app import startup  # noqa: E402
from panteon_v2.selection import FlashAllocatorConfig  # noqa: E402
from run_panteon3_live_canary_check import build_canary_summary  # noqa: E402
from run_panteon3_single_component_canary import (  # noqa: E402
    DEFAULT_CANARY_SYMBOLS,
    DEFAULT_PAPER_MIN_NOTIONAL_FLOOR_USD,
    _configure_paper_min_notional_floor,
    _parse_context_signal_keys,
    _parse_symbols,
    _temporary_env,
    configure_single_component_pipeline,
    single_component_flash_config,
)


DEFAULT_PROFILE_ID = "controlled_exploration_loss_budget_deny_richard_btc"
DEFAULT_MANIFEST = ROOT / "configs" / "bitget_hypotheses_20260706.json"
DEFAULT_RESULTS_ROOT = ROOT / "Results" / "BitgetPolicyCanary"
DEFAULT_REPORTS_DIR = ROOT / "Reports" / "Panteon3Canary" / "bitget_policy"
DEFAULT_INITIAL_CAPITAL = 1000.0
DEFAULT_RISK_CAPITAL_FRACTION = 0.10
CONTROLLED_EXPLORATION_REASONS = (
    "expected_edge_below_cost",
    "range_low_vol_actor_not_allowed",
    "shadow_unconfirmed",
    "insufficient_closed_trades",
)
MATRIX_METRIC_KEYS = (
    "variant",
    "base_variant",
    "filled_signals",
    "closed_trades",
    "realized_pnl_usd",
    "expectancy_usd",
    "gross_profit",
    "gross_loss",
    "max_drawdown_usd",
    "profit_factor",
    "pnl_per_drawdown",
    "cost_attribution_present",
)
DEFAULT_BLOCKED_REGIMES = ("range_low_vol",)
SINGLE_COMPONENT_VARIANT_PREFIX = "single_component__"
ACTIVATION_GAP_REPORT_FILENAME = "latest_bitget_policy_activation_gap.json"


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(payload)


def load_hypothesis(manifest_path: str | Path, profile_id: str) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    operational_status = str(
        manifest.get("operational_status") or "active"
    ).strip()
    if operational_status != "active":
        raise ValueError(
            "hypothesis manifest is not operational: "
            f"{operational_status or 'unknown'}"
        )
    for item in manifest.get("hypotheses") or ():
        if not isinstance(item, Mapping):
            continue
        if str(item.get("id") or "") == str(profile_id):
            return dict(item)
    raise KeyError(f"Bitget hypothesis profile not found: {profile_id}")


def _option_values(args: Sequence[object], flag: str) -> tuple[str, ...]:
    values: list[str] = []
    items = [str(item) for item in args]
    prefix = f"{flag}="
    idx = 0
    while idx < len(items):
        item = items[idx]
        if item == flag:
            if idx + 1 >= len(items):
                raise ValueError(f"{flag} requires a value")
            values.append(items[idx + 1])
            idx += 2
            continue
        if item.startswith(prefix):
            values.append(item[len(prefix) :])
        idx += 1
    return tuple(values)


def _option_float(
    args: Sequence[object],
    flag: str,
    default: float,
) -> float:
    values = _option_values(args, flag)
    if not values:
        return float(default)
    return float(values[-1])


def _candidate_runner_args(hypothesis: Mapping[str, Any]) -> tuple[str, ...]:
    raw = hypothesis.get("candidate_runner_args") or ()
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(item) for item in raw)


def _policy_terminal_denies(
    hypothesis: Mapping[str, Any],
    extra_keys: Sequence[str] | None = None,
) -> tuple[str, ...]:
    args = _candidate_runner_args(hypothesis)
    keys: list[str] = []
    keys.extend(_option_values(args, "--flash-terminal-deny-context-signal-key"))
    keys.extend(_parse_context_signal_keys(hypothesis.get("terminal_deny_context_signal_keys")))  # type: ignore[arg-type]
    keys.extend(_parse_context_signal_keys(extra_keys))
    return _parse_context_signal_keys(keys)


def _candidate_variant_from_hypothesis(hypothesis: Mapping[str, Any]) -> str:
    variant = str(hypothesis.get("candidate_variant") or "").strip()
    if variant:
        return variant
    label = str(hypothesis.get("single_component_candidate_label") or "").strip()
    if label:
        return f"{SINGLE_COMPONENT_VARIANT_PREFIX}{label}"
    return ""


def _single_component_label_from_hypothesis(hypothesis: Mapping[str, Any]) -> str:
    label = str(hypothesis.get("single_component_candidate_label") or "").strip()
    if label:
        return label
    variant = _candidate_variant_from_hypothesis(hypothesis)
    if variant.startswith(SINGLE_COMPONENT_VARIANT_PREFIX):
        return variant[len(SINGLE_COMPONENT_VARIANT_PREFIX) :].strip()
    return ""


def _single_component_label_from_policy(policy: Mapping[str, Any]) -> str:
    variant = str(policy.get("candidate_variant") or "").strip()
    if variant.startswith(SINGLE_COMPONENT_VARIANT_PREFIX):
        label = variant[len(SINGLE_COMPONENT_VARIANT_PREFIX) :].strip()
        if label:
            return label
    return ""


def _disable_live_genetics(config: FlashAllocatorConfig) -> FlashAllocatorConfig:
    return replace(
        config,
        flash_genetics_core_primary_enabled=False,
        genetics_probation_bypass_min_closed_enabled=False,
        genetics_probation_bypass_trend_gate_enabled=False,
        genetics_probation_bypass_terminal_deny_enabled=False,
        genetics_probation_bypass_regime_edge_enabled=False,
        genetics_probation_bypass_pnl_enabled=False,
    )


def build_policy_flash_config(
    base: FlashAllocatorConfig,
    hypothesis: Mapping[str, Any],
    *,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    risk_capital_fraction: float = DEFAULT_RISK_CAPITAL_FRACTION,
    extra_terminal_denied_context_signal_keys: Sequence[str] | None = None,
) -> FlashAllocatorConfig:
    variant = _candidate_variant_from_hypothesis(hypothesis)
    terminal_denies = _policy_terminal_denies(
        hypothesis,
        extra_keys=extra_terminal_denied_context_signal_keys,
    )
    if variant.startswith(SINGLE_COMPONENT_VARIANT_PREFIX):
        label = _single_component_label_from_hypothesis(hypothesis)
        if not label:
            raise ValueError("single-component policy canary requires an actor label")
        clean_base = replace(
            base,
            terminal_denied_signal_keys=(),
            terminal_denied_context_signal_keys=(),
        )
        return _disable_live_genetics(
            single_component_flash_config(
                clean_base,
                label,
                diagnostic=True,
                exploration_risk_mult=0.03,
                terminal_denied_context_signal_keys=terminal_denies,
                bypass_terminal_denies=False,
            )
        )
    if variant != "controlled_exploration":
        raise ValueError(
            "policy canary currently supports candidate_variant=controlled_exploration "
            "or single_component__<Actor>"
        )
    args = _candidate_runner_args(hypothesis)
    allowed_reasons = _option_values(
        args,
        "--flash-controlled-exploration-allowed-reason",
    ) or CONTROLLED_EXPLORATION_REASONS
    range_low_vol_allowed_directions = _option_values(
        args,
        "--flash-controlled-exploration-range-low-vol-allowed-direction",
    )
    if (
        not range_low_vol_allowed_directions
        and "range_low_vol_actor_not_allowed" in set(allowed_reasons)
    ):
        range_low_vol_allowed_directions = ("short",)
    return _disable_live_genetics(replace(
        base,
        min_closed_trades_to_trade=3,
        min_pnl_pct_to_trade=0.0,
        min_score_to_trade=0.0,
        controlled_exploration_enabled=True,
        controlled_exploration_allowed_reasons=tuple(allowed_reasons),
        controlled_exploration_allow_range_low_vol_actor_not_allowed=(
            "range_low_vol_actor_not_allowed" in set(allowed_reasons)
        ),
        controlled_exploration_range_low_vol_allowed_directions=tuple(
            range_low_vol_allowed_directions
        ),
        controlled_exploration_risk_mult=_option_float(
            args,
            "--flash-controlled-exploration-risk-mult",
            0.05,
        ),
        controlled_exploration_min_shadow_score=_option_float(
            args,
            "--flash-controlled-exploration-min-shadow-score",
            0.0,
        ),
        controlled_exploration_min_shadow_closed=int(_option_float(
            args,
            "--flash-controlled-exploration-min-shadow-closed",
            0.0,
        )),
        controlled_exploration_max_daily_trades=int(_option_float(
            args,
            "--flash-controlled-exploration-max-daily-trades",
            6.0,
        )),
        controlled_exploration_max_open_positions=int(_option_float(
            args,
            "--flash-controlled-exploration-max-open-positions",
            2.0,
        )),
        controlled_exploration_min_rolling_expectancy=_option_float(
            args,
            "--flash-controlled-exploration-min-rolling-expectancy",
            0.0,
        ),
        controlled_exploration_min_notional_sizing_enabled=True,
        controlled_exploration_account_equity_usd=float(initial_capital),
        controlled_exploration_capital_fraction=float(risk_capital_fraction),
        controlled_exploration_min_notional_max_risk_mult=_option_float(
            args,
            "--flash-controlled-exploration-min-notional-max-risk-mult",
            0.10,
        ),
        controlled_exploration_default_min_notional_usd=_option_float(
            args,
            "--flash-controlled-exploration-default-min-notional-usd",
            DEFAULT_PAPER_MIN_NOTIONAL_FLOOR_USD,
        ),
        controlled_exploration_loss_budget_usd=_option_float(
            args,
            "--flash-controlled-exploration-loss-budget-usd",
            0.0,
        ),
        controlled_exploration_loss_budget_adverse_move_pct=_option_float(
            args,
            "--flash-controlled-exploration-loss-budget-adverse-move-pct",
            0.0,
        ),
        controlled_exploration_stop_loss_pct=_option_float(
            args,
            "--flash-controlled-exploration-stop-loss-pct",
            0.0,
        ),
        controlled_exploration_session_loss_budget_usd=_option_float(
            args,
            "--flash-controlled-exploration-session-loss-budget-usd",
            0.0,
        ),
        causal_actor_router_enabled=False,
        causal_actor_router_exploration_enabled=False,
        promotion_derived_router_enabled=False,
        promotion_derived_actor_labels=(),
        promotion_derived_dynamic_best_enabled=False,
        live_real_actor_whitelist=(),
        terminal_denied_signal_keys=(),
        terminal_denied_context_signal_keys=terminal_denies,
    ))


def find_latest_matrix_summary(profile_id: str) -> Path | None:
    reports_root = ROOT / "Reports" / "BitgetHypothesisSweep"
    if not reports_root.exists():
        return None
    candidates = [
        path
        for path in reports_root.rglob("panteon3_pre_live_matrix_summary.json")
        if path.parent.name == profile_id
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_matrix_evidence(
    profile_id: str,
    matrix_summary: str | Path | None = None,
    *,
    require_pass: bool = True,
) -> dict[str, Any]:
    path = Path(matrix_summary) if matrix_summary else find_latest_matrix_summary(profile_id)
    if path is None or not path.exists():
        if require_pass:
            raise FileNotFoundError(f"matrix summary not found for profile {profile_id}")
        return {}
    payload = _load_json(path)
    verdict = payload.get("promotion_verdict") or {}
    if require_pass and not bool(getattr(verdict, "get", lambda _k, _d=None: _d)("passed", False)):
        raise ValueError(
            f"matrix promotion gate is not PASS for profile {profile_id}: "
            f"{getattr(verdict, 'get', lambda _k, _d=None: _d)('fail_reasons', [])}"
        )
    candidate = payload.get("candidate") or {}
    gates = payload.get("promotion_gates") or {}
    candidate_slices = _summarize_candidate_slices(payload.get("candidate_slices"))
    return {
        "path": str(path),
        "promotion_verdict": verdict if isinstance(verdict, Mapping) else {},
        "promotion_gates": gates if isinstance(gates, Mapping) else {},
        "candidate_metrics": {
            key: candidate.get(key)
            for key in MATRIX_METRIC_KEYS
            if isinstance(candidate, Mapping) and key in candidate
        },
        "candidate_slice_summary": candidate_slices,
    }


def _summarize_candidate_slices(raw_slices: object) -> dict[str, Any]:
    if not isinstance(raw_slices, Mapping):
        return {
            "eligible_count": 0,
            "eligible_slices": [],
            "eligible_regimes_by_symbol": {},
        }
    eligible: list[dict[str, Any]] = []
    regimes_by_symbol: dict[str, set[str]] = {}
    for raw_key, raw_row in raw_slices.items():
        if not isinstance(raw_row, Mapping):
            continue
        if not bool(raw_row.get("promotion_eligible")):
            continue
        symbol = str(raw_row.get("symbol") or "").strip().upper()
        regime = _clean_regime(raw_row.get("regime"))
        direction = str(raw_row.get("direction") or "").strip().upper()
        actor_label = str(raw_row.get("actor_label") or "").strip()
        row = {
            "key": str(raw_row.get("key") or raw_key),
            "actor_label": actor_label,
            "symbol": symbol,
            "regime": regime,
            "direction": direction,
            "filled_signals": raw_row.get("filled_signals"),
            "closed_trades": raw_row.get("closed_trades"),
            "expectancy_usd": raw_row.get("expectancy_usd"),
            "lcb_usd": raw_row.get("lcb_usd"),
            "realized_pnl_usd": raw_row.get("realized_pnl_usd"),
            "max_drawdown_usd": raw_row.get("max_drawdown_usd"),
        }
        eligible.append(row)
        if symbol and regime:
            regimes_by_symbol.setdefault(symbol, set()).add(regime)
    eligible.sort(key=lambda item: str(item.get("key") or ""))
    return {
        "eligible_count": len(eligible),
        "eligible_slices": eligible,
        "eligible_regimes_by_symbol": {
            symbol: sorted(regimes)
            for symbol, regimes in sorted(regimes_by_symbol.items())
        },
    }


def _attempt_snapshots(monitor_payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(monitor_payload, Mapping):
        return []
    snapshots: list[dict[str, Any]] = []
    for attempt in monitor_payload.get("attempts") or ():
        if not isinstance(attempt, Mapping):
            continue
        snapshot = attempt.get("snapshot")
        if isinstance(snapshot, Mapping):
            snapshots.append(dict(snapshot))
    return snapshots


def _regime_counts_by_symbol(snapshots: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for snapshot in snapshots:
        raw = snapshot.get("regimes_by_symbol")
        if not isinstance(raw, Mapping):
            continue
        for raw_symbol, raw_regime in raw.items():
            symbols = _parse_symbols([str(raw_symbol or "")])
            if not symbols:
                continue
            symbol = symbols[0]
            regime = _clean_regime(raw_regime) or "unknown"
            per_symbol = counts.setdefault(symbol, {})
            per_symbol[regime] = per_symbol.get(regime, 0) + 1
    return {
        symbol: dict(sorted(regimes.items()))
        for symbol, regimes in sorted(counts.items())
    }


def build_activation_gap_report(
    *,
    profile_id: str,
    hypothesis: Mapping[str, Any],
    symbols: Sequence[str],
    matrix_evidence: Mapping[str, Any],
    monitor_payload: Mapping[str, Any] | None,
    blocked_regimes: Sequence[str] = DEFAULT_BLOCKED_REGIMES,
) -> dict[str, Any]:
    snapshots = _attempt_snapshots(monitor_payload)
    final_snapshot = (
        dict(monitor_payload.get("final_snapshot") or {})
        if isinstance(monitor_payload, Mapping)
        else {}
    )
    requested_symbols = _parse_symbols(symbols)
    blocked = sorted({_clean_regime(item) for item in blocked_regimes if _clean_regime(item)})
    observed_counts = _regime_counts_by_symbol(snapshots)
    final_regimes = {
        symbol: _clean_regime(regime)
        for symbol, regime in (final_snapshot.get("regimes_by_symbol") or {}).items()
        if str(symbol or "").strip()
    }
    slice_summary = (
        matrix_evidence.get("candidate_slice_summary")
        if isinstance(matrix_evidence, Mapping)
        else {}
    ) or {}
    eligible_by_symbol = (
        slice_summary.get("eligible_regimes_by_symbol")
        if isinstance(slice_summary, Mapping)
        else {}
    ) or {}
    eligible_overlap: dict[str, dict[str, Any]] = {}
    for symbol in requested_symbols:
        final_regime = final_regimes.get(symbol, "")
        allowed = tuple(str(item) for item in eligible_by_symbol.get(symbol, ()) or ())
        eligible_overlap[symbol] = {
            "final_regime": final_regime,
            "matrix_eligible_regimes": list(allowed),
            "final_regime_is_matrix_eligible": bool(final_regime and final_regime in allowed),
            "final_regime_is_blocked": bool(final_regime and final_regime in blocked),
        }

    final_actionable = list(final_snapshot.get("actionable_symbols") or [])
    final_blocked = list(final_snapshot.get("blocked_symbols") or [])
    policy_blocked = list(final_snapshot.get("policy_blocked_symbols") or [])
    all_observed_blocked = bool(
        requested_symbols
        and final_regimes
        and all(final_regimes.get(symbol) in blocked for symbol in requested_symbols)
    )
    has_matrix_overlap = any(
        row["final_regime_is_matrix_eligible"]
        for row in eligible_overlap.values()
    )
    if not snapshots:
        reason = "no_precheck_attempts"
    elif not bool(slice_summary.get("eligible_count", 0)):
        reason = "matrix_has_no_eligible_slices"
    elif all_observed_blocked:
        reason = "observed_regimes_all_blocked"
    elif not has_matrix_overlap:
        reason = "matrix_regime_mismatch"
    else:
        reason = "no_actionable_symbols_after_policy"

    return {
        "runner": "bitget_policy_activation_gap",
        "profile_id": profile_id,
        "actor": str(hypothesis.get("actor") or ""),
        "candidate_variant": _candidate_variant_from_hypothesis(hypothesis),
        "symbols": list(requested_symbols),
        "attempt_count": len(snapshots),
        "blocked_regimes": blocked,
        "final_actionable": bool(final_snapshot.get("actionable")),
        "final_actionable_symbols": final_actionable,
        "final_blocked_symbols": final_blocked,
        "final_policy_blocked_symbols": policy_blocked,
        "observed_regime_counts_by_symbol": observed_counts,
        "final_regimes_by_symbol": final_regimes,
        "matrix_eligible_slice_count": int(slice_summary.get("eligible_count") or 0),
        "matrix_eligible_regimes_by_symbol": dict(eligible_by_symbol),
        "final_regime_matrix_overlap_by_symbol": eligible_overlap,
        "activation_gap_reason": reason,
        "continue_same_test": reason not in {
            "observed_regimes_all_blocked",
            "matrix_regime_mismatch",
            "matrix_has_no_eligible_slices",
        },
        "recommended_next_steps": _activation_gap_next_steps(reason),
    }


def _activation_gap_next_steps(reason: str) -> list[str]:
    if reason == "observed_regimes_all_blocked":
        return [
            "Do not rerun the same neutral-only monitor while observed regimes remain blocked.",
            "Test a separate range_low_vol candidate only if it has positive costed matrix evidence.",
            "Otherwise wait for an eligible regime transition before strict paper canary.",
        ]
    if reason == "matrix_regime_mismatch":
        return [
            "Build a candidate policy from slices whose regimes overlap the observed feed.",
            "Keep current terminal-deny policy; do not bypass blocked regimes.",
        ]
    if reason == "matrix_has_no_eligible_slices":
        return [
            "Rebuild the Bitget matrix; current artifact has no promotion-eligible slices.",
        ]
    if reason == "no_precheck_attempts":
        return [
            "Run actionable-regime precheck before strict canary.",
        ]
    return [
        "Inspect candidate diagnostics and terminal-deny context for the final blocked symbols.",
    ]


def write_activation_gap_report(
    reports_dir: str | Path,
    report: Mapping[str, Any],
) -> Path:
    root = Path(reports_dir)
    root.mkdir(parents=True, exist_ok=True)
    latest = root / ACTIVATION_GAP_REPORT_FILENAME
    latest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = root / f"bitget_policy_activation_gap_{stamp}.json"
    snapshot.write_text(latest.read_text(encoding="utf-8"), encoding="utf-8")
    return latest


def build_candidate_policy_metadata(
    *,
    profile_id: str,
    hypothesis: Mapping[str, Any],
    symbols: Sequence[str],
    matrix_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "type": "bitget_policy_level_paper_canary",
        "exchange": "BITGET",
        "profile_id": profile_id,
        "actor": str(hypothesis.get("actor") or ""),
        "candidate_variant": _candidate_variant_from_hypothesis(hypothesis),
        "symbols": list(symbols),
        "candidate_runner_args": list(_candidate_runner_args(hypothesis)),
        "terminal_deny_context_signal_keys": list(_policy_terminal_denies(hypothesis)),
        "matrix_evidence": dict(matrix_evidence),
        "strict_runtime": {
            "mode": "paper_live_feed",
            "execution_smoke": False,
            "calibration_only": False,
            "include_genetics": False,
            "allow_live_feed_fallback": False,
            "paper_policy_resets_live_whitelist": True,
        },
    }


def run_exchange(
    *,
    results_root: str | Path,
    initial_capital: float,
    risk_capital_fraction: float,
    max_bars: int,
    max_idle_polls: int,
    sleep_between_polls_sec: float,
    warmup_bars: int | None,
    symbols: Sequence[str],
    flash_config: FlashAllocatorConfig,
    candidate_policy: Mapping[str, Any],
) -> dict[str, Any]:
    exchange_key = "BITGET"
    root = Path(results_root)
    (root / "state").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    symbol_keys = _parse_symbols(symbols)
    risk_config = startup._resolve_risk_config(
        exchange_key,
        float(risk_capital_fraction),
    )
    runtime_flash_config_holder: dict[str, Any] = {
        "available": False,
        "terminal_denied_context_signal_keys": (),
        "live_real_actor_whitelist": (),
    }
    min_notional_holder = {"configured": 0}
    single_component_holder = {"label": ""}

    def _configure_with_runtime(pipeline: object) -> None:
        single_component_label = _single_component_label_from_policy(candidate_policy)
        if single_component_label:
            single_component_holder["label"] = configure_single_component_pipeline(
                pipeline,
                single_component_label,
                paper_probe_on_idle=False,
                actor_overrides={},
            )
        flash_allocator = getattr(pipeline, "flash_allocator", None)
        if flash_allocator is not None:
            setattr(flash_allocator, "_config", flash_config)
            runtime_config = getattr(flash_allocator, "_config", None)
            runtime_flash_config_holder["available"] = True
            runtime_flash_config_holder["terminal_denied_context_signal_keys"] = tuple(
                getattr(runtime_config, "terminal_denied_context_signal_keys", ()) or ()
            )
            runtime_flash_config_holder["live_real_actor_whitelist"] = tuple(
                getattr(runtime_config, "live_real_actor_whitelist", ()) or ()
            )
        min_notional_holder["configured"] = _configure_paper_min_notional_floor(
            pipeline,
            symbol_keys,
        )
        setattr(pipeline, "bitget_policy_canary", dict(candidate_policy))

    env = {
        "CRYPTO_EXCHANGE": exchange_key,
        "BITGET_TRADING_MODE": "paper_live_feed",
        "PANTEON_V2_PAPER_CANARY_CLOSE_ON_SHUTDOWN": "1",
        "PANTEON_ALLOW_LIVE_FEED_FALLBACK": "0",
        "PANTEON_V2_FUTURES_SIGNAL_FIXES_ENABLED": "1",
        "BITGET_PANTEON_V2_FUTURES_SIGNAL_FIXES_ENABLED": "1",
    }
    if symbol_keys:
        env["BITGET_SYMBOLS"] = ",".join(symbol_keys)

    started_at = datetime.now(timezone.utc)
    with _temporary_env(env):
        rc = startup.start_production(
            exchange=exchange_key,
            mode="paper_live_feed",
            initial_capital=float(initial_capital),
            max_bars=max(0, int(max_bars)),
            max_idle_polls=max(0, int(max_idle_polls)),
            sleep_between_polls_sec=max(0.0, float(sleep_between_polls_sec)),
            use_v1_bridge=True,
            results_root=str(root),
            warmup_bars=None if warmup_bars is None else max(0, int(warmup_bars)),
            snapshot_path=str(root / "state" / "bitget_policy_canary_snapshot.json"),
            jsonl_event_log=str(root / "logs" / "bitget_policy_canary_events.jsonl"),
            include_genetics=False,
            risk_config_override=risk_config,
            flash_enabled_override=True,
            flash_allocator_config_override=flash_config,
            configure_pipeline=_configure_with_runtime,
        )

    return {
        "exchange": exchange_key,
        "return_code": int(rc or 0),
        "started_at": started_at.isoformat(),
        "results_root": str(root),
        "mode": "paper_live_feed",
        "initial_capital": float(initial_capital),
        "risk_capital_fraction": float(risk_capital_fraction),
        "risk_max_min_notional_upscale": float(
            getattr(risk_config, "max_min_notional_upscale", 0.0) or 0.0
        ),
        "symbols": list(symbol_keys),
        "max_bars": max(0, int(max_bars)),
        "max_idle_polls": max(0, int(max_idle_polls)),
        "sleep_between_polls_sec": max(0.0, float(sleep_between_polls_sec)),
        "warmup_bars": None if warmup_bars is None else max(0, int(warmup_bars)),
        "execution_smoke": False,
        "calibration_only": False,
        "include_genetics": False,
        "allow_live_feed_fallback": False,
        "paper_min_notional_floor_usd": DEFAULT_PAPER_MIN_NOTIONAL_FLOOR_USD,
        "paper_min_notional_symbols_configured": int(min_notional_holder["configured"]),
        "runtime_flash_config_available": bool(runtime_flash_config_holder["available"]),
        "runtime_terminal_denied_context_signal_keys": list(
            runtime_flash_config_holder["terminal_denied_context_signal_keys"]
        ),
        "runtime_live_real_actor_whitelist": list(
            runtime_flash_config_holder["live_real_actor_whitelist"]
        ),
        "runtime_single_component_actor": single_component_holder["label"],
    }


def _latest_status_path(results_root: str | Path, exchange: str = "BITGET") -> Path | None:
    exchange_root = Path(results_root) / str(exchange or "").strip().upper()
    if not exchange_root.exists():
        return None
    statuses = [path for path in exchange_root.glob("**/status.json") if path.is_file()]
    if not statuses:
        return None
    return max(statuses, key=lambda path: path.stat().st_mtime)


def _clean_regime(value: object) -> str:
    text = str(value or "").strip().lower()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def _policy_actor_aliases(actor_label: object) -> tuple[str, ...]:
    clean = str(actor_label or "").strip()
    if not clean:
        return ()
    if clean.startswith(("agent:", "ensemble:")):
        base = clean.split(":", 1)[1]
    else:
        base = clean
    aliases = [clean]
    if base:
        aliases.extend([
            base,
            f"agent:{base}",
            f"Solo_{base}",
            f"ensemble:Solo_{base}",
            f"agent:Solo_{base}",
        ])
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def _policy_symbol_aliases(symbol: object) -> tuple[str, ...]:
    clean = str(symbol or "").strip().upper()
    if not clean:
        return ()
    if clean == "*":
        return ("*",)
    aliases = [clean]
    if "/" in clean:
        base, quote = clean.split("/", 1)
        aliases.extend([base, f"{base}{quote}"])
    elif clean.endswith("USDT") and len(clean) > 4:
        base = clean[:-4]
        aliases.extend([base, f"{base}/USDT"])
    else:
        aliases.append(f"{clean}/USDT")
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def _terminal_deny_blocks_regime(
    *,
    keys: Sequence[str],
    actor_label: object,
    symbol: object,
    regime: object,
) -> bool:
    actor_aliases = set(_policy_actor_aliases(actor_label))
    symbol_aliases = set(_policy_symbol_aliases(symbol))
    clean_regime = _clean_regime(regime)
    if not actor_aliases or not symbol_aliases or not clean_regime:
        return False
    for raw_key in _parse_context_signal_keys(keys):
        parts = [part.strip() for part in str(raw_key or "").split("|")]
        if len(parts) != 4 or not all(parts):
            continue
        key_actor, key_symbol, key_action, key_regime = parts
        if key_actor not in actor_aliases:
            continue
        if key_symbol != "*" and key_symbol.upper() not in symbol_aliases:
            continue
        if key_action != "*":
            continue
        if key_regime != "*" and _clean_regime(key_regime) != clean_regime:
            continue
        return True
    return False


def build_actionable_regime_snapshot(
    status: Mapping[str, Any],
    *,
    symbols: Sequence[str],
    blocked_regimes: Sequence[str] = DEFAULT_BLOCKED_REGIMES,
    min_actionable_symbols: int = 1,
    actor_label: object = "",
    terminal_denied_context_signal_keys: Sequence[str] = (),
) -> dict[str, Any]:
    blocked = {
        _clean_regime(item)
        for item in blocked_regimes
        if _clean_regime(item)
    } or set(DEFAULT_BLOCKED_REGIMES)
    requested_symbols = _parse_symbols(symbols)
    raw_by_symbol = status.get("regimes_by_symbol")
    regimes_by_symbol: dict[str, str] = {}
    if isinstance(raw_by_symbol, Mapping):
        for raw_symbol, raw_regime in raw_by_symbol.items():
            symbol = _parse_symbols([str(raw_symbol or "")])
            if not symbol:
                continue
            if requested_symbols and symbol[0] not in requested_symbols:
                continue
            regimes_by_symbol[symbol[0]] = _clean_regime(raw_regime)
    if not regimes_by_symbol and requested_symbols:
        fallback = _clean_regime(status.get("regime") or status.get("market"))
        regimes_by_symbol = {symbol: fallback for symbol in requested_symbols}

    terminal_denies = _parse_context_signal_keys(terminal_denied_context_signal_keys)
    policy_blocked_symbols = sorted(
        symbol
        for symbol, regime in regimes_by_symbol.items()
        if _terminal_deny_blocks_regime(
            keys=terminal_denies,
            actor_label=actor_label,
            symbol=symbol,
            regime=regime,
        )
    )
    policy_blocked_set = set(policy_blocked_symbols)
    actionable_symbols = sorted(
        symbol
        for symbol, regime in regimes_by_symbol.items()
        if regime and regime not in blocked and symbol not in policy_blocked_set
    )
    blocked_symbols = sorted(
        symbol
        for symbol, regime in regimes_by_symbol.items()
        if not regime or regime in blocked or symbol in policy_blocked_set
    )
    return {
        "timestamp_utc": str(status.get("timestamp_utc") or status.get("timestamp") or ""),
        "status_path": str(status.get("_status_path") or ""),
        "mode": str(status.get("mode") or ""),
        "run_state": str(status.get("run_state") or ""),
        "feed_status": str(status.get("feed_status") or ""),
        "global_regime": _clean_regime(status.get("regime") or status.get("market")),
        "blocked_regimes": sorted(blocked),
        "min_actionable_symbols": max(1, int(min_actionable_symbols)),
        "regimes_by_symbol": regimes_by_symbol,
        "actionable_symbols": actionable_symbols,
        "blocked_symbols": blocked_symbols,
        "policy_blocked_symbols": policy_blocked_symbols,
        "policy_terminal_deny_keys_evaluated": len(terminal_denies),
        "actionable": len(actionable_symbols) >= max(1, int(min_actionable_symbols)),
    }


def load_actionable_regime_snapshot(
    results_root: str | Path,
    *,
    symbols: Sequence[str],
    blocked_regimes: Sequence[str] = DEFAULT_BLOCKED_REGIMES,
    min_actionable_symbols: int = 1,
    actor_label: object = "",
    terminal_denied_context_signal_keys: Sequence[str] = (),
) -> dict[str, Any]:
    status_path = _latest_status_path(results_root, "BITGET")
    if status_path is None:
        return {
            "actionable": False,
            "fail_reason": "status_missing",
            "regimes_by_symbol": {},
            "actionable_symbols": [],
            "blocked_symbols": [],
        }
    status = _load_json(status_path)
    status["_status_path"] = str(status_path)
    return build_actionable_regime_snapshot(
        status,
        symbols=symbols,
        blocked_regimes=blocked_regimes,
        min_actionable_symbols=min_actionable_symbols,
        actor_label=actor_label,
        terminal_denied_context_signal_keys=terminal_denied_context_signal_keys,
    )


def _write_actionable_precheck_report(
    reports_dir: str | Path,
    payload: Mapping[str, Any],
) -> None:
    report_root = Path(reports_dir)
    report_root.mkdir(parents=True, exist_ok=True)
    latest = report_root / "latest_actionable_regime_precheck.json"
    latest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (report_root / f"actionable_regime_precheck_{stamp}.json").write_text(
        latest.read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def run_actionable_regime_precheck(
    *,
    results_root: str | Path,
    reports_dir: str | Path,
    initial_capital: float,
    risk_capital_fraction: float,
    warmup_bars: int | None,
    max_bars: int,
    max_idle_polls: int,
    sleep_between_polls_sec: float,
    symbols: Sequence[str],
    flash_config: FlashAllocatorConfig,
    candidate_policy: Mapping[str, Any],
    blocked_regimes: Sequence[str],
    min_actionable_symbols: int,
    attempt_id: str = "",
) -> dict[str, Any]:
    precheck_root = Path(results_root) / "_regime_precheck"
    clean_attempt_id = str(attempt_id or "").strip()
    if clean_attempt_id:
        precheck_root = precheck_root / clean_attempt_id
    precheck_policy = {
        **dict(candidate_policy),
        "stage": "actionable_regime_precheck",
        "precheck_attempt_id": clean_attempt_id,
    }
    run = run_exchange(
        results_root=precheck_root,
        initial_capital=float(initial_capital),
        risk_capital_fraction=float(risk_capital_fraction),
        max_bars=max(1, int(max_bars)),
        max_idle_polls=max(0, int(max_idle_polls)),
        sleep_between_polls_sec=max(0.0, float(sleep_between_polls_sec)),
        warmup_bars=None if warmup_bars is None else max(0, int(warmup_bars)),
        symbols=symbols,
        flash_config=flash_config,
        candidate_policy=precheck_policy,
    )
    snapshot = load_actionable_regime_snapshot(
        precheck_root,
        symbols=symbols,
        blocked_regimes=blocked_regimes,
        min_actionable_symbols=min_actionable_symbols,
        actor_label=candidate_policy.get("actor", ""),
        terminal_denied_context_signal_keys=flash_config.terminal_denied_context_signal_keys,
    )
    payload = {
        "runner": "bitget_policy_canary_actionable_regime_precheck",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "attempt_id": clean_attempt_id,
        "results_root": str(precheck_root),
        "run": run,
        "snapshot": snapshot,
    }
    _write_actionable_precheck_report(reports_dir, payload)
    return payload


def run_actionable_regime_monitor(
    *,
    results_root: str | Path,
    reports_dir: str | Path,
    initial_capital: float,
    risk_capital_fraction: float,
    warmup_bars: int | None,
    max_bars: int,
    max_idle_polls: int,
    sleep_between_polls_sec: float,
    symbols: Sequence[str],
    flash_config: FlashAllocatorConfig,
    candidate_policy: Mapping[str, Any],
    blocked_regimes: Sequence[str],
    min_actionable_symbols: int,
    attempts: int = 1,
    interval_sec: float = 0.0,
) -> dict[str, Any]:
    max_attempts = max(1, int(attempts))
    delay = max(0.0, float(interval_sec))
    attempt_payloads: list[dict[str, Any]] = []
    final_payload: dict[str, Any] | None = None
    for attempt_index in range(1, max_attempts + 1):
        attempt_payload = run_actionable_regime_precheck(
            results_root=results_root,
            reports_dir=reports_dir,
            initial_capital=initial_capital,
            risk_capital_fraction=risk_capital_fraction,
            warmup_bars=warmup_bars,
            max_bars=max_bars,
            max_idle_polls=max_idle_polls,
            sleep_between_polls_sec=sleep_between_polls_sec,
            symbols=symbols,
            flash_config=flash_config,
            candidate_policy=candidate_policy,
            blocked_regimes=blocked_regimes,
            min_actionable_symbols=min_actionable_symbols,
            attempt_id=f"attempt_{attempt_index:03d}",
        )
        attempt_payloads.append(attempt_payload)
        final_payload = attempt_payload
        snapshot = attempt_payload.get("snapshot") or {}
        if bool(snapshot.get("actionable")):
            break
        if attempt_index < max_attempts and delay > 0.0:
            time.sleep(delay)
    final_snapshot = (final_payload or {}).get("snapshot") or {}
    monitor_payload = {
        "runner": "bitget_policy_canary_actionable_regime_monitor",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_root": str(results_root),
        "attempts_requested": max_attempts,
        "attempts_completed": len(attempt_payloads),
        "interval_sec": delay,
        "actionable": bool(final_snapshot.get("actionable")),
        "final_snapshot": final_snapshot,
        "attempts": attempt_payloads,
    }
    report_root = Path(reports_dir)
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "latest_actionable_regime_monitor.json").write_text(
        json.dumps(monitor_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return monitor_payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run strict Bitget paper/live-feed canary for a policy-level matrix profile.",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--profile-id", default=DEFAULT_PROFILE_ID)
    parser.add_argument("--matrix-summary", default="")
    parser.add_argument("--allow-missing-matrix", action="store_true")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument(
        "--run-id",
        default="",
        help="Stable run folder id; default is current UTC timestamp.",
    )
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument(
        "--risk-capital-fraction",
        type=float,
        default=DEFAULT_RISK_CAPITAL_FRACTION,
    )
    parser.add_argument("--max-bars", type=int, default=65)
    parser.add_argument("--max-idle-polls", type=int, default=120)
    parser.add_argument("--sleep-between-polls-sec", type=float, default=2.0)
    parser.add_argument("--warmup-bars", type=int, default=240)
    parser.add_argument(
        "--require-actionable-regime",
        action="store_true",
        help="Run a paper-only regime precheck and skip strict canary while symbols stay in blocked regimes.",
    )
    parser.add_argument(
        "--precheck-max-bars",
        type=int,
        default=1,
        help="Number of paper-live bars for actionable-regime precheck.",
    )
    parser.add_argument(
        "--precheck-max-idle-polls",
        type=int,
        default=24,
        help="Idle polls for actionable-regime precheck.",
    )
    parser.add_argument(
        "--precheck-sleep-between-polls-sec",
        type=float,
        default=1.0,
        help="Sleep between precheck polls.",
    )
    parser.add_argument(
        "--actionable-monitor-attempts",
        type=int,
        default=1,
        help="Number of actionable-regime precheck attempts before skipping strict canary.",
    )
    parser.add_argument(
        "--actionable-monitor-interval-sec",
        type=float,
        default=60.0,
        help="Delay between actionable-regime precheck attempts.",
    )
    parser.add_argument(
        "--min-actionable-symbols",
        type=int,
        default=1,
        help="Minimum symbols outside blocked regimes required to run strict canary.",
    )
    parser.add_argument(
        "--blocked-regime",
        action="append",
        default=None,
        help="Regime that blocks strict canary precheck; repeatable, default range_low_vol.",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated Bitget symbols; default uses profile symbols.",
    )
    parser.add_argument(
        "--terminal-deny-context-signal-key",
        action="append",
        default=None,
        help="Extra policy terminal deny context key; repeatable or comma-separated.",
    )
    parser.add_argument("--lookback-minutes", type=float, default=360.0)
    args = parser.parse_args(argv)

    hypothesis = load_hypothesis(args.manifest, args.profile_id)
    profile_symbols = _parse_symbols(hypothesis.get("symbols"))  # type: ignore[arg-type]
    symbols = _parse_symbols(args.symbols) or profile_symbols or DEFAULT_CANARY_SYMBOLS
    run_id = str(args.run_id or "").strip() or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    effective_results_root = Path(args.results_root) / str(args.profile_id) / run_id
    matrix_evidence = load_matrix_evidence(
        args.profile_id,
        args.matrix_summary or None,
        require_pass=not bool(args.allow_missing_matrix),
    )
    base_config = startup._resolve_flash_allocator_config("BITGET")
    flash_config = build_policy_flash_config(
        base_config,
        hypothesis,
        initial_capital=float(args.initial_capital),
        risk_capital_fraction=float(args.risk_capital_fraction),
        extra_terminal_denied_context_signal_keys=args.terminal_deny_context_signal_key,
    )
    candidate_policy = build_candidate_policy_metadata(
        profile_id=args.profile_id,
        hypothesis=hypothesis,
        symbols=symbols,
        matrix_evidence=matrix_evidence,
    )
    candidate_policy["run_id"] = run_id
    candidate_policy["results_root"] = str(effective_results_root)
    extra_keys = _parse_context_signal_keys(args.terminal_deny_context_signal_key)
    if extra_keys:
        candidate_policy["terminal_deny_context_signal_keys"] = list(
            dict.fromkeys([
                *candidate_policy["terminal_deny_context_signal_keys"],
                *extra_keys,
            ])
        )
    precheck_payload: dict[str, Any] | None = None
    monitor_payload: dict[str, Any] | None = None
    blocked_regimes = tuple(args.blocked_regime or DEFAULT_BLOCKED_REGIMES)
    if bool(args.require_actionable_regime):
        monitor_payload = run_actionable_regime_monitor(
            results_root=effective_results_root,
            reports_dir=args.reports_dir,
            initial_capital=float(args.initial_capital),
            risk_capital_fraction=float(args.risk_capital_fraction),
            warmup_bars=max(0, int(args.warmup_bars)),
            max_bars=max(1, int(args.precheck_max_bars)),
            max_idle_polls=max(0, int(args.precheck_max_idle_polls)),
            sleep_between_polls_sec=max(
                0.0,
                float(args.precheck_sleep_between_polls_sec),
            ),
            symbols=symbols,
            flash_config=flash_config,
            candidate_policy=candidate_policy,
            blocked_regimes=blocked_regimes,
            min_actionable_symbols=max(1, int(args.min_actionable_symbols)),
            attempts=max(1, int(args.actionable_monitor_attempts)),
            interval_sec=max(0.0, float(args.actionable_monitor_interval_sec)),
        )
        attempts = monitor_payload.get("attempts") or []
        if attempts:
            precheck_payload = attempts[-1]
        snapshot = monitor_payload.get("final_snapshot") or {}
        candidate_policy["actionable_regime_precheck"] = snapshot
        if not bool(snapshot.get("actionable")):
            activation_gap_report = build_activation_gap_report(
                profile_id=args.profile_id,
                hypothesis=hypothesis,
                symbols=symbols,
                matrix_evidence=matrix_evidence,
                monitor_payload=monitor_payload,
                blocked_regimes=blocked_regimes,
            )
            activation_gap_path = write_activation_gap_report(
                args.reports_dir,
                activation_gap_report,
            )
            activation_gap_report["path"] = str(activation_gap_path)
            payload = {
                "runner": "bitget_policy_canary",
                "profile_id": args.profile_id,
                "execution_smoke": False,
                "calibration_only": False,
                "canary_skipped": True,
                "skip_reason": "no_actionable_regime",
                "activation_gap_report": activation_gap_report,
                "candidate_policy": candidate_policy,
                "actionable_regime_precheck": precheck_payload,
                "actionable_regime_monitor": monitor_payload,
                "canary_summary": None,
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
    result = run_exchange(
        results_root=effective_results_root,
        initial_capital=float(args.initial_capital),
        risk_capital_fraction=float(args.risk_capital_fraction),
        max_bars=max(0, int(args.max_bars)),
        max_idle_polls=max(0, int(args.max_idle_polls)),
        sleep_between_polls_sec=max(0.0, float(args.sleep_between_polls_sec)),
        warmup_bars=max(0, int(args.warmup_bars)),
        symbols=symbols,
        flash_config=flash_config,
        candidate_policy=candidate_policy,
    )
    terminal_denies = tuple(
        result["runtime_terminal_denied_context_signal_keys"]
        or flash_config.terminal_denied_context_signal_keys
    )
    summary = build_canary_summary(
        results_root=effective_results_root,
        reports_dir=args.reports_dir,
        exchanges=("BITGET",),
        lookback_minutes=float(args.lookback_minutes),
        require_positive_expectancy=True,
        calibration_only=False,
        actor_overrides={},
        terminal_denied_context_signal_keys=terminal_denies,
        candidate_policy=candidate_policy,
        execution_smoke=False,
    )
    payload = {
        "runner": "bitget_policy_canary",
        "profile_id": args.profile_id,
        "execution_smoke": False,
        "calibration_only": False,
        "runs": [result],
        "candidate_policy": candidate_policy,
        "actionable_regime_precheck": precheck_payload,
        "actionable_regime_monitor": monitor_payload,
        "canary_summary": summary,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
