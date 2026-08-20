from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
RUNTIME_ROOT = SRC_ROOT / "panteon_runtime"
for path in (SRC_ROOT, RUNTIME_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from exia.scoring import deterministic_seed, moving_block_lcb  # noqa: E402
from timeframe_profiles import (  # noqa: E402
    FOUR_HOUR_TIMEFRAME_MINUTES,
    apply_timeframe_profile,
    get_timeframe_profile,
)


DEFAULT_SPEC = ROOT / "configs" / "exia_timebase_audit_v1.json"
DEFAULT_OUTPUT = ROOT / "Reports" / "Exia" / "timebase_audit_v1"
MINUTE_MS = 60_000
SPOT_LONG_ACTIONS = frozenset((1, 2))
FUTURES_LONG_ACTIONS = frozenset((4, 5))
LONG_ACTIONS = SPOT_LONG_ACTIONS | FUTURES_LONG_ACTIONS
SHORT_ACTIONS = frozenset((6, 7))
CLOSE_ACTIONS = frozenset((3, 8))
TIMEFRAME_NAMES = {
    1: "1m",
    5: "5m",
    15: "15m",
    30: "30m",
    60: "1h",
    120: "2h",
    180: "3h",
    240: "4h",
    300: "5h",
    360: "6h",
}
TIMEBASE_MODES = ("physical_time", "native_bar", "fixed_profile")


@dataclass(frozen=True)
class AuditBar:
    timestamp: int
    opens: dict[str, float]
    closes: dict[str, float]
    volumes: dict[str, float]


class AuditBarSequence:
    """Compact complete panel that materializes one legacy bar at a time."""

    def __init__(self, frame: pd.DataFrame) -> None:
        ordered = frame
        self.symbols = tuple(sorted(ordered["symbol"].unique()))
        width = len(self.symbols)
        if not width or len(ordered) % width:
            raise ValueError("bar panel is not rectangular")
        rows = len(ordered) // width
        codes = pd.Categorical(ordered["symbol"], categories=self.symbols).codes.reshape(
            rows, width
        )
        expected_codes = np.arange(width, dtype=codes.dtype)
        if not np.all(codes == expected_codes):
            raise ValueError("bar panel symbol order is inconsistent")
        timestamp_matrix = ordered["timestamp"].to_numpy(dtype=np.int64, copy=False).reshape(
            rows, width
        )
        if not np.all(timestamp_matrix == timestamp_matrix[:, :1]):
            raise ValueError("bar panel timestamps are inconsistent")
        self.timestamps = timestamp_matrix[:, 0].copy()
        self.opens = ordered["open"].to_numpy(dtype=np.float64, copy=False).reshape(rows, width)
        self.closes = ordered["close"].to_numpy(dtype=np.float64, copy=False).reshape(rows, width)
        self.volumes = ordered["volume"].to_numpy(dtype=np.float64, copy=False).reshape(rows, width)

    def __len__(self) -> int:
        return int(len(self.timestamps))

    def __getitem__(self, index: int) -> AuditBar:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return AuditBar(
            timestamp=int(self.timestamps[index]),
            opens=dict(zip(self.symbols, self.opens[index], strict=True)),
            closes=dict(zip(self.symbols, self.closes[index], strict=True)),
            volumes=dict(zip(self.symbols, self.volumes[index], strict=True)),
        )

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _checkpoint_stem(
    output_dir: Path,
    *,
    panel_id: str,
    timeframe: int,
    mode: str,
    agent_id: str,
) -> Path:
    safe_agent = re.sub(r"[^A-Za-z0-9_.-]+", "_", agent_id)
    safe_panel = re.sub(r"[^A-Za-z0-9_.-]+", "_", panel_id)
    return output_dir / "checkpoints" / safe_panel / f"{timeframe}m" / mode / safe_agent


def _load_checkpoint(stem: Path, *, fingerprint: str) -> dict[str, Any] | None:
    metadata_path = stem.with_suffix(".json")
    if not metadata_path.exists():
        return None
    payload = load_json(metadata_path)
    if payload.get("fingerprint") != fingerprint:
        raise ValueError(f"stale audit checkpoint fingerprint: {metadata_path}")
    trade_payload = payload.get("trades")
    if trade_payload is None:
        trades = pd.DataFrame()
    else:
        trades_path = ROOT / str(trade_payload["path"])
        if not trades_path.is_file() or sha256_file(trades_path) != trade_payload["sha256"]:
            raise ValueError(f"audit checkpoint trade SHA mismatch: {trades_path}")
        trades = pd.read_parquet(trades_path)
    return {
        "trades": trades,
        "parameter": payload["parameter"],
        "metrics": payload["metrics"],
    }


def _save_checkpoint(
    stem: Path,
    *,
    fingerprint: str,
    trades: pd.DataFrame,
    parameter: dict[str, Any],
    metrics: list[dict[str, Any]],
) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    trade_payload: dict[str, str] | None = None
    if not trades.empty:
        trades_path = stem.with_suffix(".parquet")
        temporary = trades_path.with_name(f"{trades_path.name}.{os.getpid()}.partial")
        trades.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, trades_path)
        trade_payload = {
            "path": str(trades_path.relative_to(ROOT)),
            "sha256": sha256_file(trades_path),
        }
    _atomic_json(
        stem.with_suffix(".json"),
        {
            "schema_version": "exia.timebase_audit_checkpoint.v1",
            "fingerprint": fingerprint,
            "parameter": parameter,
            "metrics": metrics,
            "trades": trade_payload,
            "orders_enabled": False,
            "promotion_authority": False,
        },
    )


def validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != "exia.timebase_audit.v1":
        raise ValueError("unexpected timebase audit schema")
    safety = spec.get("safety", {})
    for flag in ("paper_allowed", "live_allowed", "orders_enabled", "promotion_authority"):
        if safety.get(flag) is not False:
            raise ValueError(f"safety flag must be false: {flag}")
    modes = spec.get("modes")
    if (
        not isinstance(modes, list)
        or not modes
        or len(modes) != len(set(modes))
        or not set(modes) <= set(TIMEBASE_MODES)
    ):
        raise ValueError("timebase modes must be preregistered, unique and supported")
    if "fixed_profile" in modes:
        profile_id = str(spec.get("runtime_profile_id", ""))
        get_timeframe_profile(profile_id)
    if not spec.get("agents") or not spec.get("panels"):
        raise ValueError("timebase audit requires agents and panels")
    for agent in spec["agents"]:
        agent_modes = agent.get("modes", spec["modes"])
        if not agent_modes or not set(agent_modes) <= set(spec["modes"]):
            raise ValueError(f"invalid component modes: {agent.get('agent_id')}")
        if agent.get("engine", "act") not in {"act", "simple_research"}:
            raise ValueError(f"unknown component engine: {agent.get('engine')}")
    for panel in spec["panels"]:
        source_minutes = int(panel["source_timeframe_minutes"])
        for timeframe in panel["timeframes_minutes"]:
            timeframe = int(timeframe)
            if timeframe < source_minutes or timeframe % source_minutes:
                raise ValueError("target timeframe must be an integer source multiple")
            if "fixed_profile" in modes and timeframe != FOUR_HOUR_TIMEFRAME_MINUTES:
                raise ValueError("fixed_profile is sealed to the 4h timeframe")


def _manifest_files(manifest_path: Path, expected_dataset_sha: str) -> list[Path]:
    manifest = load_json(manifest_path)
    if manifest.get("dataset_sha256") != expected_dataset_sha:
        raise ValueError(f"dataset SHA mismatch: {manifest_path}")
    if manifest.get("validation", {}).get("passed") is not True:
        raise ValueError(f"source dataset validation failed: {manifest_path}")
    paths: list[Path] = []
    for item in manifest.get("files", []):
        path = ROOT / str(item["path"]).replace("\\", "/")
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"source file SHA mismatch: {path}")
        paths.append(path)
    if not paths:
        raise ValueError(f"manifest has no source files: {manifest_path}")
    return paths


def load_source_manifest_panel(
    manifest_path: Path,
    dataset_sha256: str,
) -> pd.DataFrame:
    manifest = load_json(manifest_path)
    files = _manifest_files(manifest_path, dataset_sha256)
    required = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
    frames = [
        pd.read_parquet(path, columns=required)
        if path.suffix.lower() == ".parquet"
        else pd.read_csv(path, usecols=required)
        for path in files
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="raise").astype("int64")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
    normalized_symbols = frame["symbol"].astype(str).str.upper()
    frame["symbol"] = pd.Categorical(
        normalized_symbols, categories=sorted(normalized_symbols.unique())
    )
    if frame.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("source panel contains duplicate timestamp/symbol rows")
    if manifest.get("sort_order") == ["timestamp", "symbol"]:
        timestamps = frame["timestamp"].to_numpy(dtype=np.int64, copy=False)
        if len(timestamps) > 1 and np.any(timestamps[1:] < timestamps[:-1]):
            raise ValueError("source manifest declares an invalid timestamp sort order")
        return frame.reset_index(drop=True)
    return frame.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)


def load_source_panel(panel: dict[str, Any]) -> pd.DataFrame:
    manifest_path = ROOT / str(panel["manifest"])
    return load_source_manifest_panel(manifest_path, str(panel["dataset_sha256"]))


def aggregate_ohlcv(
    source: pd.DataFrame,
    *,
    source_minutes: int,
    target_minutes: int,
) -> pd.DataFrame:
    """Build complete epoch-aligned OHLCV bars; never downsample by row selection."""
    source_minutes = int(source_minutes)
    target_minutes = int(target_minutes)
    if target_minutes < source_minutes or target_minutes % source_minutes:
        raise ValueError("target timeframe is not compatible with source timeframe")
    expected_rows = target_minutes // source_minutes
    if target_minutes == source_minutes:
        symbols = tuple(sorted(source["symbol"].unique()))
        complete = source.groupby("timestamp", sort=False)["symbol"].nunique()
        complete_timestamps = complete.index[complete == len(symbols)]
        if len(complete_timestamps) == len(complete):
            return source
        grouped = source.loc[source["timestamp"].isin(complete_timestamps)].copy()
        if grouped.empty:
            raise ValueError("source panel contains no complete bars")
        return grouped.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)
    target_ms = target_minutes * MINUTE_MS
    frame = source.copy()
    frame["bucket"] = (frame["timestamp"] // target_ms) * target_ms
    grouped = (
        frame.groupby(["bucket", "symbol"], sort=True, observed=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            source_rows=("timestamp", "size"),
        )
        .reset_index()
        .rename(columns={"bucket": "timestamp"})
    )
    grouped = grouped.loc[grouped["source_rows"] == expected_rows].copy()
    symbols = tuple(sorted(source["symbol"].unique()))
    complete = grouped.groupby("timestamp", sort=False)["symbol"].nunique()
    complete_timestamps = complete.index[complete == len(symbols)]
    grouped = grouped.loc[grouped["timestamp"].isin(complete_timestamps)].copy()
    if grouped.empty:
        raise ValueError("OHLCV aggregation produced no complete bars")
    if not (grouped["high"] >= grouped[["open", "close", "low"]].max(axis=1)).all():
        raise ValueError("aggregated high is inconsistent")
    if not (grouped["low"] <= grouped[["open", "close", "high"]].min(axis=1)).all():
        raise ValueError("aggregated low is inconsistent")
    return grouped.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)


def frame_to_bars(frame: pd.DataFrame) -> AuditBarSequence:
    return AuditBarSequence(frame)


def scaled_parameters(
    agent_class: type,
    temporal_fields: Iterable[str],
    *,
    timeframe_minutes: int,
    mode: str,
    explicit_overrides: dict[str, Any] | None = None,
    parameter_base_minutes: int = 1,
) -> tuple[dict[str, Any], list[str]]:
    overrides = dict(explicit_overrides or {})
    resolution_limited: list[str] = []
    for field in temporal_fields:
        if not hasattr(agent_class, field):
            raise ValueError(f"unknown temporal field {agent_class.__name__}.{field}")
        original = getattr(agent_class, field)
        if not isinstance(original, int) or isinstance(original, bool) or original <= 0:
            raise ValueError(f"temporal field must be a positive integer: {field}")
        if mode == "physical_time":
            scaled = max(
                1,
                int(
                    math.ceil(
                        original * int(parameter_base_minutes) / int(timeframe_minutes)
                    )
                ),
            )
            overrides[field] = scaled
            if original * int(parameter_base_minutes) < int(timeframe_minutes):
                resolution_limited.append(field)
        elif mode == "native_bar":
            overrides[field] = original
        elif mode == "fixed_profile":
            continue
        else:
            raise ValueError(f"unknown mode: {mode}")
    return overrides, resolution_limited


def build_agent(
    agent_spec: dict[str, Any],
    *,
    timeframe_minutes: int,
    mode: str,
    runtime_profile_id: str | None = None,
) -> tuple[Any, dict[str, Any], list[str]]:
    module = importlib.import_module(str(agent_spec.get("module", "panteon_agents")))
    base_class = getattr(module, str(agent_spec["class_name"]))
    overrides, resolution_limited = scaled_parameters(
        base_class,
        agent_spec["temporal_fields"],
        timeframe_minutes=timeframe_minutes,
        mode=mode,
        explicit_overrides=agent_spec.get("overrides", {}),
        parameter_base_minutes=int(agent_spec.get("parameter_base_minutes", 1)),
    )
    audit_class = type(
        f"{base_class.__name__}_{mode}_{timeframe_minutes}m",
        (base_class,),
        overrides,
    )
    agent = audit_class(**dict(agent_spec.get("constructor_kwargs", {})))
    if agent_spec.get("shadow_bootstrap_mode"):
        setter = getattr(agent, "set_shadow_bootstrap_mode", None)
        if callable(setter):
            setter(True)
        else:
            setattr(agent, "_shadow_bootstrap_mode", True)
    if mode == "fixed_profile":
        if timeframe_minutes != FOUR_HOUR_TIMEFRAME_MINUTES:
            raise ValueError("fixed profile can only be applied to 4h bars")
        profile_report = apply_timeframe_profile(agent, str(runtime_profile_id or ""))
        overrides = {
            field: getattr(agent, field) for field in agent_spec["temporal_fields"]
        }
        overrides["__runtime_profile__"] = profile_report
        resolution_limited = []
    return agent, overrides, resolution_limited


def scaled_strategy_definition(
    component_spec: dict[str, Any],
    *,
    timeframe_minutes: int,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    strategy = {
        "kind": str(component_spec["strategy_kind"]),
        "params": dict(component_spec.get("params", {})),
    }
    effective = dict(strategy["params"])
    limited: list[str] = []
    base_minutes = int(component_spec.get("parameter_base_minutes", 60))
    for field in component_spec.get("temporal_fields", []):
        original = effective.get(field)
        if not isinstance(original, int) or isinstance(original, bool) or original <= 0:
            raise ValueError(f"simple strategy temporal field must be positive: {field}")
        if mode == "physical_time":
            effective[field] = max(
                1,
                int(math.ceil(original * base_minutes / int(timeframe_minutes))),
            )
            if original * base_minutes < int(timeframe_minutes):
                limited.append(str(field))
        elif mode not in {"native_bar", "fixed_profile"}:
            raise ValueError(f"unknown mode: {mode}")
    strategy["params"] = effective
    return strategy, effective, limited


def _sync_agent_entry(
    agent: Any,
    symbol: str,
    fill_price: float,
    bar_index: int,
    _visited: set[int] | None = None,
) -> None:
    visited = _visited if _visited is not None else set()
    identity = id(agent)
    if identity in visited:
        return
    visited.add(identity)
    old_entry: float | None = None
    for field in ("entry_px", "ep"):
        values = getattr(agent, field, None)
        if isinstance(values, dict):
            try:
                candidate = float(values.get(symbol, 0.0) or 0.0)
            except (TypeError, ValueError):
                candidate = 0.0
            if candidate > 0:
                old_entry = candidate
            values[symbol] = float(fill_price)
    if old_entry and old_entry > 0:
        for field in ("sl", "tp", "stop"):
            values = getattr(agent, field, None)
            if not isinstance(values, dict) or symbol not in values:
                continue
            try:
                level = float(values[symbol])
            except (TypeError, ValueError):
                continue
            if level > 0:
                values[symbol] = float(fill_price) * level / old_entry
    for field in ("et", "last_entry"):
        values = getattr(agent, field, None)
        if isinstance(values, dict):
            values[symbol] = int(bar_index)
    open_positions = getattr(agent, "_open_pos", None)
    if isinstance(open_positions, dict) and isinstance(open_positions.get(symbol), dict):
        open_positions[symbol]["entry_price"] = float(fill_price)
        open_positions[symbol]["entry_bar"] = int(bar_index)
    children: list[Any] = []
    iterator = getattr(agent, "iter_subagents", None)
    if callable(iterator):
        try:
            children.extend(item[2] for item in iterator())
        except Exception:
            pass
    for field in ("_inner", "_b1", "_b2", "_fa", "_ms"):
        child = getattr(agent, field, None)
        if child is not None:
            children.append(child)
    for child in children:
        _sync_agent_entry(child, symbol, fill_price, bar_index, visited)


def simulate_simple_strategy(
    frame: pd.DataFrame,
    *,
    component_spec: dict[str, Any],
    agent_id: str,
    panel_id: str,
    timeframe_minutes: int,
    mode: str,
    base_cost_bps: float,
    stress_cost_bps: float,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, Any], list[str]]:
    from simple_research.simulator import CostModel, simulate_targets
    from simple_research.strategies import build_signal

    strategy, parameters, limited = scaled_strategy_definition(
        component_spec,
        timeframe_minutes=timeframe_minutes,
        mode=mode,
    )
    signal = build_signal(frame, strategy)
    result = simulate_targets(
        frame,
        signal,
        start_timestamp=int(frame["timestamp"].min()),
        end_timestamp=int(frame["timestamp"].max()) + timeframe_minutes * MINUTE_MS,
        costs=CostModel(
            fee_bps_per_fill=4.0,
            slippage_bps_per_fill=max(stress_cost_bps / 2.0 - 4.0, 0.0),
        ),
        trial_count=1,
    )
    ledger = result.ledger.copy()
    counters = {
        "entry_signals": int(len(ledger)),
        "close_signals": int(len(ledger)),
        "unsupported_actions": 0,
        "conflicting_actions": 0,
        "agent_errors": 0,
    }
    if ledger.empty:
        return ledger, counters, parameters, limited
    ledger["panel_id"] = panel_id
    ledger["agent_id"] = agent_id
    ledger["mode"] = mode
    ledger["timeframe_minutes"] = int(timeframe_minutes)
    ledger["direction"] = ledger["direction"].map({1: "LONG", -1: "SHORT"})
    ledger["signal_timestamp"] = (
        ledger["entry_timestamp"].astype("int64") - timeframe_minutes * MINUTE_MS
    )
    ledger["entry_action"] = ledger["direction"].map({"LONG": 5, "SHORT": 7})
    ledger["source_action_lane"] = "simple_research_target"
    ledger["holding_hours"] = (
        ledger["exit_timestamp"].astype("int64")
        - ledger["entry_timestamp"].astype("int64")
    ) / 3_600_000.0
    ledger["base_net_bps"] = ledger["gross_bps"].astype(float) - base_cost_bps
    ledger["stress_net_bps"] = ledger["gross_bps"].astype(float) - stress_cost_bps
    return ledger, counters, parameters, limited


def _close_trade(
    position: dict[str, Any],
    *,
    exit_timestamp: int,
    exit_price: float,
    exit_reason: str,
    base_cost_bps: float,
    stress_cost_bps: float,
) -> dict[str, Any]:
    entry = float(position["entry_price"])
    if position["direction"] == "LONG":
        gross = (float(exit_price) / entry - 1.0) * 10_000.0
    else:
        gross = (1.0 - float(exit_price) / entry) * 10_000.0
    return {
        **position,
        "exit_timestamp": int(exit_timestamp),
        "exit_price": float(exit_price),
        "exit_reason": str(exit_reason),
        "holding_hours": (int(exit_timestamp) - int(position["entry_timestamp"])) / 3_600_000.0,
        "gross_bps": float(gross),
        "base_net_bps": float(gross - base_cost_bps),
        "stress_net_bps": float(gross - stress_cost_bps),
        "fills": 2,
    }


def simulate_agent(
    bars: Any,
    *,
    agent: Any,
    agent_id: str,
    panel_id: str,
    timeframe_minutes: int,
    mode: str,
    base_cost_bps: float,
    stress_cost_bps: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    positions: dict[str, dict[str, Any]] = {}
    pending: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    counters = {
        "entry_signals": 0,
        "close_signals": 0,
        "unsupported_actions": 0,
        "conflicting_actions": 0,
        "agent_errors": 0,
    }
    compact = isinstance(bars, AuditBarSequence)
    symbol_indexes = (
        {symbol: index for index, symbol in enumerate(bars.symbols)}
        if compact
        else {}
    )
    for bar_index in range(len(bars)):
        if compact:
            timestamp = int(bars.timestamps[bar_index])
            closes = dict(zip(bars.symbols, bars.closes[bar_index], strict=True))
            volumes = dict(zip(bars.symbols, bars.volumes[bar_index], strict=True))
        else:
            bar = bars[bar_index]
            timestamp = int(bar.timestamp)
            closes = dict(bar.closes)
            volumes = dict(bar.volumes)
        for symbol, order in pending.items():
            if compact:
                column = symbol_indexes.get(symbol)
                if column is None:
                    continue
                fill_price = float(bars.opens[bar_index, column])
            else:
                if symbol not in bar.opens:
                    continue
                fill_price = float(bar.opens[symbol])
            action = int(order["action"])
            if action in LONG_ACTIONS or action in SHORT_ACTIONS:
                direction = "LONG" if action in LONG_ACTIONS else "SHORT"
                if symbol in positions:
                    counters["conflicting_actions"] += 1
                    continue
                positions[symbol] = {
                    "panel_id": panel_id,
                    "agent_id": agent_id,
                    "mode": mode,
                    "timeframe_minutes": int(timeframe_minutes),
                    "symbol": symbol,
                    "direction": direction,
                    "signal_timestamp": int(order["signal_timestamp"]),
                    "entry_timestamp": timestamp,
                    "entry_price": fill_price,
                    "entry_action": action,
                    "source_action_lane": (
                        "spot_long_normalized"
                        if action in SPOT_LONG_ACTIONS
                        else "futures_native"
                    ),
                }
                _sync_agent_entry(agent, symbol, fill_price, bar_index)
            elif action in CLOSE_ACTIONS:
                position = positions.pop(symbol, None)
                if position is not None:
                    trades.append(
                        _close_trade(
                            position,
                            exit_timestamp=timestamp,
                            exit_price=fill_price,
                            exit_reason="agent_close_next_open",
                            base_cost_bps=base_cost_bps,
                            stress_cost_bps=stress_cost_bps,
                        )
                    )
        pending = {}
        dt = datetime.fromtimestamp(timestamp / 1000.0, timezone.utc)
        try:
            replay_clock = getattr(agent, "set_replay_timestamp_ms", None)
            if callable(replay_clock):
                replay_clock(timestamp)
            actions = agent.act(
                closes,
                volumes,
                month=dt.month,
                portfolio_value=10_000.0,
                bar_index=bar_index,
            )
        except Exception:
            counters["agent_errors"] += 1
            continue
        if not isinstance(actions, dict):
            counters["agent_errors"] += 1
            continue
        signal_timestamp = int(timestamp + timeframe_minutes * MINUTE_MS)
        for symbol, raw_action in actions.items():
            try:
                action = int(raw_action)
            except (TypeError, ValueError):
                counters["unsupported_actions"] += 1
                continue
            symbol = str(symbol).upper()
            if action in LONG_ACTIONS or action in SHORT_ACTIONS:
                counters["entry_signals"] += 1
                pending[symbol] = {"action": action, "signal_timestamp": signal_timestamp}
            elif action in CLOSE_ACTIONS:
                counters["close_signals"] += 1
                pending[symbol] = {"action": action, "signal_timestamp": signal_timestamp}
            elif action not in (0,):
                counters["unsupported_actions"] += 1
    if bars:
        if compact:
            last_timestamp = int(bars.timestamps[-1])
            last_closes = dict(zip(bars.symbols, bars.closes[-1], strict=True))
        else:
            last = bars[-1]
            last_timestamp = int(last.timestamp)
            last_closes = last.closes
        exit_timestamp = int(last_timestamp + timeframe_minutes * MINUTE_MS)
        for symbol, position in list(positions.items()):
            trades.append(
                _close_trade(
                    position,
                    exit_timestamp=exit_timestamp,
                    exit_price=float(last_closes[symbol]),
                    exit_reason="end_of_panel",
                    base_cost_bps=base_cost_bps,
                    stress_cost_bps=stress_cost_bps,
                )
            )
    return pd.DataFrame(trades), counters


def _window_name(timestamp: int, windows: list[dict[str, Any]]) -> str:
    instant = pd.Timestamp(timestamp, unit="ms", tz="UTC")
    for window in windows:
        if pd.Timestamp(window["start"]) <= instant < pd.Timestamp(window["end"]):
            return str(window["name"])
    return "outside"


def _max_drawdown_bps(values: pd.Series) -> float | None:
    if values.empty:
        return None
    equity = values.astype("float64").cumsum().to_numpy()
    equity = np.concatenate(([0.0], equity))
    peaks = np.maximum.accumulate(equity)
    return float(np.max(peaks - equity))


def summarize_trades(
    trades: pd.DataFrame,
    *,
    seed_key: str,
    minimum_trades: int,
    bootstrap_samples: int,
    alpha: float,
    familywise_alpha: float,
) -> dict[str, Any]:
    closed = len(trades)
    if not closed:
        return {
            "closed_trades": 0,
            "fills": 0,
            "long_trades": 0,
            "short_trades": 0,
            "mean_gross_bps": None,
            "mean_base_net_bps": None,
            "mean_stress_net_bps": None,
            "stress_lcb_bps": None,
            "familywise_stress_lcb_bps": None,
            "positive_stress_rate": None,
            "max_drawdown_stress_bps": None,
            "status": "INSUFFICIENT",
        }
    values = trades["stress_net_bps"].astype("float64")
    timestamps = trades["entry_timestamp"].astype("int64")
    lcb = moving_block_lcb(
        values,
        timestamps,
        samples=bootstrap_samples,
        alpha=alpha,
        seed=deterministic_seed(seed_key),
    )
    familywise_lcb = moving_block_lcb(
        values,
        timestamps,
        samples=bootstrap_samples,
        alpha=familywise_alpha,
        seed=deterministic_seed(f"{seed_key}:familywise"),
    )
    mean_stress = float(values.mean())
    if closed < minimum_trades or lcb is None:
        status = "INSUFFICIENT"
    elif mean_stress > 0 and lcb > 0:
        status = "PASS"
    else:
        status = "FAIL"
    return {
        "closed_trades": int(closed),
        "fills": int(trades["fills"].sum()),
        "long_trades": int((trades["direction"] == "LONG").sum()),
        "short_trades": int((trades["direction"] == "SHORT").sum()),
        "mean_gross_bps": float(trades["gross_bps"].mean()),
        "mean_base_net_bps": float(trades["base_net_bps"].mean()),
        "mean_stress_net_bps": mean_stress,
        "stress_lcb_bps": lcb,
        "familywise_stress_lcb_bps": familywise_lcb,
        "positive_stress_rate": float((values > 0).mean()),
        "max_drawdown_stress_bps": _max_drawdown_bps(
            trades.sort_values("exit_timestamp", kind="stable")["stress_net_bps"]
        ),
        "status": status,
    }


def _longest_contiguous_pass(timeframes: list[int], passing: set[int]) -> int:
    longest = 0
    current = 0
    for timeframe in timeframes:
        if timeframe in passing:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def build_sensitivity_summary(
    metrics: pd.DataFrame,
    *,
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    gate = spec["stability_gate"]
    stability_panel_id = str(
        spec.get("stability_panel_id", spec["panels"][0]["panel_id"])
    )
    panel = next(
        item for item in spec["panels"] if item["panel_id"] == stability_panel_id
    )
    timeframes = [int(value) for value in panel["timeframes_minutes"]]
    rows: list[dict[str, Any]] = []
    common = metrics.loc[
        (metrics["panel_id"] == stability_panel_id) & (metrics["split"] == "overall")
    ]
    for (agent_id, mode), group in common.groupby(["agent_id", "mode"], sort=True):
        group = group.set_index("timeframe_minutes")
        passing: set[int] = set()
        means: list[float] = []
        sign_sequence: list[int] = []
        for timeframe in timeframes:
            if timeframe not in group.index:
                sign_sequence.append(0)
                continue
            row = group.loc[timeframe]
            value = row["mean_stress_net_bps"]
            if pd.notna(value):
                value = float(value)
                means.append(value)
                sign_sequence.append(1 if value > 0 else -1 if value < 0 else 0)
            else:
                sign_sequence.append(0)
            if (
                int(row["closed_trades"]) >= int(gate["minimum_trades_per_timeframe"])
                and float(row["mean_stress_net_bps"]) > 0
                and pd.notna(row["stress_lcb_bps"])
                and float(row["stress_lcb_bps"]) > 0
            ):
                passing.add(timeframe)
        sign_changes = sum(
            1
            for left, right in zip(sign_sequence, sign_sequence[1:])
            if left and right and left != right
        )
        split_rows = metrics.loc[
            (metrics["panel_id"] == stability_panel_id)
            & (metrics["agent_id"] == agent_id)
            & (metrics["mode"] == mode)
            & (metrics["split"] != "overall")
            & (metrics["timeframe_minutes"].isin(passing))
            & (metrics["closed_trades"] >= int(gate["supported_split_minimum_trades"]))
        ]
        negative_supported = split_rows.loc[
            (split_rows["mean_stress_net_bps"] <= 0)
            | split_rows["stress_lcb_bps"].isna()
            | (split_rows["stress_lcb_bps"] <= 0)
        ]
        contiguous = _longest_contiguous_pass(timeframes, passing)
        stable = (
            contiguous >= int(gate["minimum_adjacent_passing_timeframes"])
            and negative_supported.empty
        )
        best_tf: int | None = None
        worst_tf: int | None = None
        supported = group.loc[group["closed_trades"] >= int(spec["bootstrap"]["minimum_trades"])]
        if not supported.empty:
            best_tf = int(supported["mean_stress_net_bps"].idxmax())
            worst_tf = int(supported["mean_stress_net_bps"].idxmin())
        rows.append(
            {
                "agent_id": str(agent_id),
                "mode": str(mode),
                "component_type": str(group["component_type"].iloc[0]),
                "engine": str(group["engine"].iloc[0]),
                "passing_timeframes_minutes": sorted(passing),
                "longest_adjacent_pass_run": int(contiguous),
                "negative_supported_splits": int(len(negative_supported)),
                "sign_changes": int(sign_changes),
                "stress_mean_range_bps": None if not means else float(max(means) - min(means)),
                "best_timeframe_minutes": best_tf,
                "worst_timeframe_minutes": worst_tf,
                "stable_timebase_candidate": bool(stable),
            }
        )
    return rows


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)) or pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}"


def _cell(row: pd.Series | None) -> str:
    if row is None or int(row["closed_trades"]) == 0:
        return "n/a / n/a / 0"
    return (
        f"{_fmt(row['mean_stress_net_bps'])} / "
        f"{_fmt(row['stress_lcb_bps'])} / {int(row['closed_trades'])}"
    )


def _matrix_markdown(metrics: pd.DataFrame, panel_id: str, mode: str, timeframes: list[int]) -> list[str]:
    subset = metrics.loc[
        (metrics["panel_id"] == panel_id)
        & (metrics["mode"] == mode)
        & (metrics["split"] == "overall")
    ]
    lines = [
        "| Agent | " + " | ".join(TIMEFRAME_NAMES.get(tf, f"{tf}m") for tf in timeframes) + " |",
        "|---|" + "---:|" * len(timeframes),
    ]
    for agent_id in sorted(subset["agent_id"].unique()):
        agent_rows = subset.loc[subset["agent_id"] == agent_id].set_index("timeframe_minutes")
        cells = [_cell(agent_rows.loc[tf] if tf in agent_rows.index else None) for tf in timeframes]
        lines.append(f"| {agent_id} | " + " | ".join(cells) + " |")
    return lines


def render_report(
    *,
    report: dict[str, Any],
    metrics: pd.DataFrame,
    sensitivity: list[dict[str, Any]],
    spec: dict[str, Any],
) -> str:
    stable = [row for row in sensitivity if row["stable_timebase_candidate"]]
    lines = [
        "# Exia Panteon component timebase audit v1",
        "",
        f"Verdict: `{report['verdict']}`.",
        "",
        "This is a diagnostic component audit. It cannot authorize paper or live trading.",
        "",
        "## Contract",
        "",
        "- True epoch-aligned OHLCV aggregation; no timestamp-stride row sampling.",
        "- Signal on a completed bar, execution at the next bar open.",
        "- Legacy spot LONG actions `1/2/3` are normalized to equivalent linear futures LONG/open/close actions; the source action and lane remain in every trade row.",
        f"- Round-trip costs: base `{spec['costs_bps']['base_round_trip']} bps`, stress `{spec['costs_bps']['stress_round_trip']} bps`.",
        "- `physical_time` converts legacy minute-bar windows to the target bar duration.",
        "- `native_bar` keeps integer bar parameters fixed and therefore retargets the strategy horizon.",
        "- `fixed_profile` applies one sealed bar-based runtime profile to standalone and nested components.",
        "- Cell format below: `stress mean / UTC-day bootstrap LCB / closed trades`.",
        "",
    ]
    for panel in spec["panels"]:
        panel_id = str(panel["panel_id"])
        timeframes = [int(value) for value in panel["timeframes_minutes"]]
        lines.extend([f"## {panel_id}", ""])
        coverage = report["panels"][panel_id]
        lines.append(
            f"Source `{coverage['source_first_utc']}` to `{coverage['source_last_utc']}`; "
            f"dataset `{coverage['dataset_sha256']}`."
        )
        lines.append("")
        for mode in spec["modes"]:
            lines.extend([f"### {mode}", ""])
            lines.extend(_matrix_markdown(metrics, panel_id, mode, timeframes))
            lines.append("")
    lines.extend(["## Sensitivity decision", ""])
    lines.append(
        "| Agent | Mode | Passing timeframes | Adjacent run | Negative splits | Sign changes | Mean range | Stable |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    for row in sensitivity:
        passing = ",".join(TIMEFRAME_NAMES.get(tf, f"{tf}m") for tf in row["passing_timeframes_minutes"]) or "none"
        lines.append(
            f"| {row['agent_id']} | {row['mode']} | {passing} | "
            f"{row['longest_adjacent_pass_run']} | {row['negative_supported_splits']} | "
            f"{row['sign_changes']} | {_fmt(row['stress_mean_range_bps'])} | "
            f"{str(row['stable_timebase_candidate']).lower()} |"
        )
    lines.extend(["", "## Exclusions", ""])
    for item in spec["excluded_components"]:
        lines.append(f"- `{item['component']}`: {item['reason']}.")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Every panel is diagnostic sensitivity evidence, not promotion evidence.",
            "- Legacy agents consume close and volume snapshots, so true high/low bars are preserved in the tape but are not exposed to agents whose original API cannot consume them.",
            "- Spot-action normalization compares signal logic under one Bitget-futures cost model; it is not a claim of spot execution parity.",
            "- Physical-time windows below the selected bar resolution collapse to one bar and are reported as resolution-limited parameters.",
            "- Every configuration belongs to one disclosed trial family; isolated positive rows are not new candidates.",
            "- Dynamic Panteon/Flash selection remains blocked until independently stable components exist.",
            "",
            "## Result",
            "",
        ]
    )
    if stable:
        labels = ", ".join(f"{row['agent_id']} ({row['mode']})" for row in stable)
        lines.append(f"Diagnostic stability gate passed by: {labels}. These rows still have no promotion authority.")
    else:
        lines.append("No component passed the preregistered adjacent-timeframe stability gate.")
    lines.append("")
    lines.append("All paper/live/orders/promotion flags remain false.")
    lines.append("")
    return "\n".join(lines)


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def run(
    spec_path: Path,
    output_dir: Path,
    *,
    agent_shard_index: int = 0,
    agent_shard_count: int = 1,
    checkpoint_only: bool = False,
) -> dict[str, Any]:
    spec = load_json(spec_path)
    validate_spec(spec)
    if agent_shard_count < 1 or not 0 <= agent_shard_index < agent_shard_count:
        raise ValueError("invalid agent shard")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_fingerprint = hashlib.sha256(
        spec_path.read_bytes() + b"\0" + Path(__file__).read_bytes()
    ).hexdigest()
    base_cost = float(spec["costs_bps"]["base_round_trip"])
    stress_cost = float(spec["costs_bps"]["stress_round_trip"])
    minimum = int(spec["bootstrap"]["minimum_trades"])
    samples = int(spec["bootstrap"]["samples"])
    alpha = float(spec["bootstrap"]["alpha"])
    total_configs = sum(
        len(panel["timeframes_minutes"])
        * sum(len(agent.get("modes", spec["modes"])) for agent in spec["agents"])
        for panel in spec["panels"]
    )
    familywise_alpha = alpha / total_configs
    all_trades: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    panel_report: dict[str, Any] = {}
    parameter_report: list[dict[str, Any]] = []
    agent_shards = {
        str(agent["agent_id"]): index % agent_shard_count
        for index, agent in enumerate(spec["agents"])
    }

    for panel in spec["panels"]:
        panel_id = str(panel["panel_id"])
        derived_manifests = panel.get("derived_timeframe_manifests", {})
        source: pd.DataFrame | None = None
        if not derived_manifests:
            source = load_source_panel(panel)
        panel_manifest_path = ROOT / str(panel["manifest"])
        panel_manifest = load_json(panel_manifest_path)
        source_rows = int(
            panel_manifest.get("validation", {}).get("rows", len(source) if source is not None else 0)
        )
        first_timestamp = int(
            pd.Timestamp(panel["windows"][0]["start"]).timestamp() * 1000
            if source is None
            else source["timestamp"].min()
        )
        requested_end = panel_manifest.get("requested_end_date")
        last_timestamp = int(
            (
                pd.Timestamp(str(requested_end), tz="UTC")
                + pd.Timedelta(days=1)
                - pd.Timedelta(minutes=int(panel["source_timeframe_minutes"]))
            ).timestamp()
            * 1000
            if source is None and requested_end
            else source["timestamp"].max()
        )
        panel_report[panel_id] = {
            "dataset_sha256": panel["dataset_sha256"],
            "source_rows": source_rows,
            "source_first_utc": pd.Timestamp(first_timestamp, unit="ms", tz="UTC").isoformat(),
            "source_last_utc": pd.Timestamp(last_timestamp, unit="ms", tz="UTC").isoformat(),
            "timeframes": {},
        }
        for timeframe in panel["timeframes_minutes"]:
            timeframe = int(timeframe)
            derived = derived_manifests.get(str(timeframe))
            if derived is not None:
                aggregated = load_source_manifest_panel(
                    ROOT / str(derived["manifest"]), str(derived["dataset_sha256"])
                )
                if "source_rows" not in aggregated.columns:
                    aggregated["source_rows"] = (
                        timeframe // int(panel["source_timeframe_minutes"])
                    )
            else:
                if source is None:
                    source = load_source_panel(panel)
                aggregated = aggregate_ohlcv(
                    source,
                    source_minutes=int(panel["source_timeframe_minutes"]),
                    target_minutes=timeframe,
                )
            bars = frame_to_bars(aggregated)
            panel_report[panel_id]["timeframes"][str(timeframe)] = {
                "bars": len(bars),
                "rows": int(len(aggregated)),
                "first_utc": pd.Timestamp(bars[0].timestamp, unit="ms", tz="UTC").isoformat(),
                "last_utc": pd.Timestamp(bars[-1].timestamp, unit="ms", tz="UTC").isoformat(),
            }
            for mode in spec["modes"]:
                for agent_spec in spec["agents"]:
                    if mode not in agent_spec.get("modes", spec["modes"]):
                        continue
                    agent_id = str(agent_spec["agent_id"])
                    if agent_shards[agent_id] != agent_shard_index:
                        continue
                    engine = str(agent_spec.get("engine", "act"))
                    component_type = str(agent_spec.get("component_type", "agent"))
                    checkpoint_stem = _checkpoint_stem(
                        output_dir,
                        panel_id=panel_id,
                        timeframe=timeframe,
                        mode=str(mode),
                        agent_id=agent_id,
                    )
                    checkpoint = _load_checkpoint(
                        checkpoint_stem, fingerprint=checkpoint_fingerprint
                    )
                    if checkpoint is not None:
                        trades = checkpoint["trades"]
                        if not checkpoint_only and not trades.empty:
                            all_trades.append(trades)
                        parameter_report.append(checkpoint["parameter"])
                        metric_rows.extend(checkpoint["metrics"])
                        print(
                            f"checkpoint {panel_id}/{timeframe}m/{mode}/{agent_id}",
                            flush=True,
                        )
                        continue
                    if engine == "simple_research":
                        trades, counters, parameters, resolution_limited = (
                            simulate_simple_strategy(
                                aggregated,
                                component_spec=agent_spec,
                                agent_id=agent_id,
                                panel_id=panel_id,
                                timeframe_minutes=timeframe,
                                mode=str(mode),
                                base_cost_bps=base_cost,
                                stress_cost_bps=stress_cost,
                            )
                        )
                    else:
                        agent, parameters, resolution_limited = build_agent(
                            agent_spec,
                            timeframe_minutes=timeframe,
                            mode=str(mode),
                            runtime_profile_id=spec.get("runtime_profile_id"),
                        )
                        trades, counters = simulate_agent(
                            bars,
                            agent=agent,
                            agent_id=agent_id,
                            panel_id=panel_id,
                            timeframe_minutes=timeframe,
                            mode=str(mode),
                            base_cost_bps=base_cost,
                            stress_cost_bps=stress_cost,
                        )
                    if counters["agent_errors"]:
                        raise RuntimeError(
                            f"agent invocation failed: {panel_id}/{timeframe}/{mode}/{agent_id}"
                        )
                    if not trades.empty:
                        trades["split"] = trades["entry_timestamp"].map(
                            lambda value: _window_name(int(value), panel["windows"])
                        )
                        if not checkpoint_only:
                            all_trades.append(trades)
                    parameter_row = {
                        "panel_id": panel_id,
                        "agent_id": agent_id,
                        "component_type": component_type,
                        "engine": engine,
                        "mode": str(mode),
                        "timeframe_minutes": timeframe,
                        "effective_temporal_parameters": {
                            field: parameters[field] for field in agent_spec["temporal_fields"]
                        },
                        "runtime_profile": parameters.get("__runtime_profile__"),
                        "resolution_limited_fields": resolution_limited,
                        **counters,
                    }
                    parameter_report.append(parameter_row)
                    scopes = [("overall", trades)]
                    for window in panel["windows"]:
                        name = str(window["name"])
                        subset = trades.loc[trades["split"] == name] if not trades.empty else trades
                        scopes.append((name, subset))
                    config_metric_rows: list[dict[str, Any]] = []
                    for split, subset in scopes:
                        summary = summarize_trades(
                            subset,
                            seed_key=f"{spec['audit_id']}:{panel_id}:{agent_id}:{mode}:{timeframe}:{split}",
                            minimum_trades=minimum,
                            bootstrap_samples=samples,
                            alpha=alpha,
                            familywise_alpha=familywise_alpha,
                        )
                        config_metric_rows.append(
                            {
                                "panel_id": panel_id,
                                "agent_id": agent_id,
                                "component_type": component_type,
                                "engine": engine,
                                "mode": str(mode),
                                "timeframe_minutes": timeframe,
                                "split": split,
                                "bars": len(bars),
                                "resolution_limited_fields": len(resolution_limited),
                                **counters,
                                **summary,
                            }
                        )
                    metric_rows.extend(config_metric_rows)
                    _save_checkpoint(
                        checkpoint_stem,
                        fingerprint=checkpoint_fingerprint,
                        trades=trades,
                        parameter=parameter_row,
                        metrics=config_metric_rows,
                    )
                    print(
                        f"completed {panel_id}/{timeframe}m/{mode}/{agent_id}",
                        flush=True,
                    )

    if checkpoint_only:
        return {
            "schema_version": "exia.timebase_audit_checkpoint_run.v1",
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "agent_shard_index": agent_shard_index,
            "agent_shard_count": agent_shard_count,
            "completed_configurations": len(parameter_report),
            "orders_enabled": False,
            "promotion_authority": False,
        }

    trades_frame = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    metrics_frame = pd.DataFrame(metric_rows)
    sensitivity = build_sensitivity_summary(metrics_frame, spec=spec)
    stable = [row for row in sensitivity if row["stable_timebase_candidate"]]
    fixed_profile = spec.get("modes") == ["fixed_profile"]
    if fixed_profile:
        verdict = (
            "FIXED_PROFILE_STABLE_COMPONENTS_FOUND_DIAGNOSTIC_ONLY"
            if stable
            else "NO_FIXED_PROFILE_STABLE_COMPONENT"
        )
    else:
        verdict = (
            "TIMEBASE_STABLE_COMPONENTS_FOUND_DIAGNOSTIC_ONLY"
            if stable
            else "NO_TIMEBASE_STABLE_COMPONENT"
        )
    trades_path = output_dir / "trades.parquet"
    metrics_path = output_dir / "metrics.parquet"
    trades_frame.to_parquet(trades_path, index=False)
    metrics_frame.to_parquet(metrics_path, index=False)
    report = {
        "schema_version": "exia.timebase_audit_report.v1",
        "audit_id": spec["audit_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "trial_family_size": total_configs,
        "familywise_alpha": familywise_alpha,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "runtime_profile_id": spec.get("runtime_profile_id"),
        "selection_policy": spec.get("selection_policy"),
        "panels": panel_report,
        "sensitivity": sensitivity,
        "stable_timebase_candidates": stable,
        "parameters": parameter_report,
        "excluded_components": spec["excluded_components"],
        "equivalent_components": spec.get("equivalent_components", []),
        "action_normalization": spec["action_normalization"],
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
        "artifacts": {
            "trades": {"path": str(trades_path.relative_to(ROOT)), "sha256": sha256_file(trades_path)},
            "metrics": {"path": str(metrics_path.relative_to(ROOT)), "sha256": sha256_file(metrics_path)},
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md = render_report(report=report, metrics=metrics_frame, sensitivity=sensitivity, spec=spec)
    report_md_path = output_dir / "report.md"
    report_md_path.write_text(report_md, encoding="utf-8")
    manifest = {
        "schema_version": "exia.timebase_audit_manifest.v1",
        "audit_id": spec["audit_id"],
        "generated_at": report["generated_at"],
        "source_revision": _git_value("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git_value("status", "--porcelain")),
        "command": f".venv\\Scripts\\python.exe tools\\run_exia_timebase_audit_v1.py --spec {spec_path.relative_to(ROOT)} --output {output_dir.relative_to(ROOT)}",
        "spec": {"path": str(spec_path.relative_to(ROOT)), "sha256": sha256_file(spec_path)},
        "runner": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256_file(Path(__file__).resolve())},
        "report": {"path": str(report_path.relative_to(ROOT)), "sha256": sha256_file(report_path)},
        "report_md": {"path": str(report_md_path.relative_to(ROOT)), "sha256": sha256_file(report_md_path)},
        "trades": report["artifacts"]["trades"],
        "metrics": report["artifacts"]["metrics"],
        "orders_enabled": False,
        "promotion_authority": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the preregistered Exia legacy-component timebase audit")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--agent-shard-index", type=int, default=0)
    parser.add_argument("--agent-shard-count", type=int, default=1)
    parser.add_argument("--checkpoint-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec_path = args.spec if args.spec.is_absolute() else ROOT / args.spec
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    report = run(
        spec_path.resolve(),
        output_dir.resolve(),
        agent_shard_index=args.agent_shard_index,
        agent_shard_count=args.agent_shard_count,
        checkpoint_only=args.checkpoint_only,
    )
    if args.checkpoint_only:
        print(json.dumps(report, indent=2))
        return 0
    print(json.dumps({
        "audit_id": report["audit_id"],
        "verdict": report["verdict"],
        "stable_timebase_candidates": len(report["stable_timebase_candidates"]),
        "orders_enabled": report["orders_enabled"],
        "promotion_authority": report["promotion_authority"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
