from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from panteon_v2.analysis.retrodate_market_runner import (  # noqa: E402
    write_component_benchmark_report,
)
from panteon_v2.analysis.soft_allocator import (  # noqa: E402
    ShadowPnLEvent,
    simulate_perfect_monthly_panteon,
    simulate_soft_allocator_policies,
    write_perfect_panteon_report,
    write_shadow_pnl_events,
    write_soft_allocator_report,
)
from panteon_v2.app.agent_bootstrap import register_all_v1_agents  # noqa: E402
from panteon_v2.selection import AgentRegistry  # noqa: E402
from tools.build_panteon_regime_breakdown_report import build as build_regime_report  # noqa: E402
from tools.run_legend_full_experiment import (  # noqa: E402
    LATEST_OPTIONAL_AGENT_LABELS,
    build_config,
)


SENTINEL_NO_SHADOW_AGENT = "__NO_SHADOW_AGENT__"
SENTINEL_NO_SHADOW_PLAYER = "__NO_SHADOW_PLAYER__"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run latest Legend/Panteon shadow actors in parallel shards and merge reports."
    )
    parser.add_argument("--profile", default="soft_regime_top1_24_confirmed_only")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "Results" / "PanteonLegend_parallel_shadow_latest",
    )
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 4))
    parser.add_argument("--max-bars", type=int, default=None)
    parser.add_argument("--include-optional-agents", action="store_true")
    parser.add_argument("--optional-agent-labels", default="")
    parser.add_argument("--keep-shards", action="store_true")
    args = parser.parse_args(argv)

    optional_labels = _csv_labels(args.optional_agent_labels) or LATEST_OPTIONAL_AGENT_LABELS
    base_config = build_config(
        args.profile,
        Path(args.results_root),
        include_optional_agents=bool(args.include_optional_agents),
        optional_agent_labels=optional_labels,
    )
    if args.max_bars is not None:
        base_config = replace(base_config, max_bars=int(args.max_bars))

    agent_labels = _registered_agent_labels(base_config)
    player_labels = _player_labels(base_config, agent_labels)
    shards = _build_shards(
        agent_labels,
        player_labels,
        workers=max(1, int(args.workers)),
    )
    if not shards:
        raise RuntimeError("no shards to run")

    run_root = Path(args.results_root).resolve() / f"{args.profile}_parallel_shadow_full2022_2026"
    shard_root = run_root / "SHARDS" / datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    shard_root.mkdir(parents=True, exist_ok=True)

    print(
        f"START parallel profile={args.profile} workers={args.workers} "
        f"agents={len(agent_labels)} players={len(player_labels)} shards={len(shards)}",
        flush=True,
    )
    futures = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        for shard in shards:
            futures.append(
                pool.submit(
                    _run_shard,
                    profile=args.profile,
                    results_root=str(shard_root / shard["name"]),
                    include_optional_agents=bool(args.include_optional_agents),
                    optional_agent_labels=optional_labels,
                    max_bars=args.max_bars,
                    shadow_agent_include_labels=tuple(shard["agents"]),
                    shadow_player_include_labels=tuple(shard["players"]),
                )
            )
        shard_results = []
        for future in as_completed(futures):
            result = future.result()
            shard_results.append(result)
            print(
                f"DONE shard={result['name']} seconds={result['seconds']:.1f} "
                f"output={result['output_dir']}",
                flush=True,
            )

    merged = _merge_shards(
        base_config=base_config,
        profile=args.profile,
        run_root=run_root,
        shard_results=sorted(shard_results, key=lambda item: item["name"]),
    )
    if not args.keep_shards:
        # Keep run summaries in merged metadata, but remove bulky shard artifacts.
        shutil.rmtree(shard_root, ignore_errors=True)
    print(f"output_dir={merged['output_dir']}", flush=True)
    print(f"analysis_report={merged['analysis_report']}", flush=True)
    print(f"run_summary={merged['run_summary']}", flush=True)
    return 0


def _run_shard(
    *,
    profile: str,
    results_root: str,
    include_optional_agents: bool,
    optional_agent_labels: Sequence[str],
    max_bars: int | None,
    shadow_agent_include_labels: Sequence[str],
    shadow_player_include_labels: Sequence[str],
) -> dict[str, Any]:
    start = datetime.now(timezone.utc)
    config = build_config(
        profile,
        Path(results_root),
        include_optional_agents=include_optional_agents,
        optional_agent_labels=optional_agent_labels,
    )
    config = replace(
        config,
        max_bars=max_bars,
        shadow_agent_include_labels=tuple(shadow_agent_include_labels),
        shadow_player_include_labels=tuple(shadow_player_include_labels),
    )
    from panteon_v2.analysis.retrodate_market_runner import run_retrodate_market_benchmark

    summary = run_retrodate_market_benchmark(config)
    seconds = (datetime.now(timezone.utc) - start).total_seconds()
    return {
        "name": Path(results_root).name,
        "seconds": seconds,
        "output_dir": str(summary.output_dir),
        "run_summary": str(summary.summary_path),
        "agents": list(shadow_agent_include_labels),
        "players": list(shadow_player_include_labels),
    }


def _merge_shards(
    *,
    base_config: Any,
    profile: str,
    run_root: Path,
    shard_results: Sequence[dict[str, Any]],
) -> dict[str, str]:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = run_root / "RETRODATE_MARKET" / f"{ts}_parallel_shadow_v2"
    output_dir.mkdir(parents=True, exist_ok=False)

    summaries = [_load_json(Path(item["run_summary"])) for item in shard_results]
    if not summaries:
        raise RuntimeError("no shard summaries")
    representative = summaries[0]
    initial_capital = float(representative.get("initial_capital") or base_config.initial_capital)

    agent_events = _read_events_from_shards(shard_results, "shadow_agent_pnl_events.jsonl")
    player_events = _read_events_from_shards(shard_results, "shadow_player_pnl_events.jsonl")
    write_shadow_pnl_events(
        output_dir,
        sorted(player_events, key=lambda item: (item.bar, item.label, item.timestamp)),
    )
    write_shadow_pnl_events(
        output_dir,
        sorted(agent_events, key=lambda item: (item.bar, item.label, item.timestamp)),
        filename="shadow_agent_pnl_events.jsonl",
    )
    _write_merged_trading_log(output_dir / "trading.log", shard_results)
    _write_merged_leaderboards(output_dir, shard_results)

    selected_policy = base_config.soft_allocator_execution_policy
    policy_report = simulate_soft_allocator_policies(
        player_events,
        initial_capital=initial_capital,
        policies=(selected_policy,) if selected_policy is not None else None,
    )
    full_soft_report = simulate_soft_allocator_policies(
        player_events,
        initial_capital=initial_capital,
    )
    perfect_report = simulate_perfect_monthly_panteon(
        player_events,
        initial_capital=initial_capital,
    )
    write_soft_allocator_report(output_dir, full_soft_report)
    write_perfect_panteon_report(output_dir, perfect_report)
    panteon_result = (
        policy_report.policy_results[0]
        if policy_report.policy_results
        else full_soft_report.policy_results[0]
    )
    write_component_benchmark_report(
        output_dir,
        panteon_pnl_usd=panteon_result.pnl_usd,
        initial_capital=initial_capital,
        shadow_agent_pnl_events=agent_events,
        shadow_player_pnl_events=player_events,
    )

    analysis_report = output_dir / "analysis_report.md"
    analysis_report.write_text(
        "\n".join(
            [
                "# Parallel shadow Panteon report",
                "",
                "Mode: offline actor-shard merge. Panteon result is soft allocator replay over merged shadow player events.",
                "",
                f"Panteon owned PnL: {panteon_result.pnl_pct:.6f}%",
                f"Panteon realized PnL USD: {panteon_result.pnl_usd:.6f}",
                f"Panteon max drawdown: {panteon_result.max_drawdown_pct:.6f}%",
                f"Panteon realized max drawdown: {panteon_result.max_drawdown_pct:.6f}%",
                f"Policy: {panteon_result.policy_name}",
                f"Weighted closed trades: {panteon_result.weighted_closed_trades:.6f}",
                f"Average cash weight: {panteon_result.average_cash_weight_pct:.6f}%",
                f"Average leader count: {panteon_result.average_leader_count:.6f}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    run_summary = dict(representative)
    run_summary.update(
        {
            "output_dir": str(output_dir),
            "analysis_report": str(analysis_report),
            "parallel_shadow_merge_enabled": True,
            "parallel_shadow_profile": profile,
            "parallel_shadow_shards": list(shard_results),
            "shadow_agent_include_labels": [],
            "shadow_player_include_labels": [],
        }
    )
    run_summary_path = output_dir / "run_summary.json"
    run_summary_path.write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    build_regime_report(output_dir)
    return {
        "output_dir": str(output_dir),
        "analysis_report": str(analysis_report),
        "run_summary": str(run_summary_path),
    }


def _registered_agent_labels(config: Any) -> tuple[str, ...]:
    registry = AgentRegistry()
    labels = register_all_v1_agents(
        registry,
        include_optional=bool(config.include_optional_agents),
        optional_agent_labels=(
            tuple(config.optional_agent_labels)
            if tuple(config.optional_agent_labels or ())
            else None
        ),
        skip_on_error=True,
    )
    return tuple(dict.fromkeys(str(label) for label in labels if str(label).strip()))


def _player_labels(config: Any, agent_labels: Sequence[str]) -> tuple[str, ...]:
    labels: list[str] = []
    _extend(labels, (str(getattr(profile, "label", "")) for profile in config.player_profiles))
    _extend(labels, (label for label, _agents in config.fixed_agent_player_sets))
    _extend(labels, (_first_label(item) for item in config.regime_switch_player_sets))
    _extend(labels, (_first_label(item) for item in config.rotating_agent_player_sets))
    _extend(labels, (f"Solo_{label}" for label in agent_labels))
    return tuple(labels)


def _build_shards(
    agent_labels: Sequence[str],
    player_labels: Sequence[str],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    workers = max(1, int(workers))
    agent_workers = 1 if workers <= 3 else max(1, min(3, workers // 3))
    player_workers = max(1, workers - agent_workers)
    shards: list[dict[str, Any]] = []
    for index, chunk in enumerate(_chunks(agent_labels, agent_workers), start=1):
        shards.append(
            {
                "name": f"agent_shard_{index:02d}",
                "agents": tuple(chunk),
                "players": (SENTINEL_NO_SHADOW_PLAYER,),
            }
        )
    for index, chunk in enumerate(_chunks(player_labels, player_workers), start=1):
        shards.append(
            {
                "name": f"player_shard_{index:02d}",
                "agents": (SENTINEL_NO_SHADOW_AGENT,),
                "players": tuple(chunk),
            }
        )
    return shards


def _chunks(values: Sequence[str], count: int) -> list[tuple[str, ...]]:
    clean = tuple(str(value).strip() for value in values if str(value).strip())
    count = max(1, min(int(count), len(clean) or 1))
    return [tuple(clean[index::count]) for index in range(count) if clean[index::count]]


def _read_events_from_shards(
    shard_results: Sequence[dict[str, Any]],
    filename: str,
) -> list[ShadowPnLEvent]:
    rows: list[ShadowPnLEvent] = []
    for item in shard_results:
        path = Path(item["output_dir"]) / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                label = str(payload.get("label") or "").strip()
                bar = _safe_int(payload.get("bar"))
                if not label or bar <= 0:
                    continue
                rows.append(
                    ShadowPnLEvent(
                        bar=bar,
                        label=label,
                        timestamp=str(payload.get("timestamp") or ""),
                        regime=str(payload.get("regime") or "all"),
                        pnl_usd=_safe_float(payload.get("pnl_usd")),
                        closed_trades=max(0, _safe_int(payload.get("closed_trades"))),
                        wins=max(0, _safe_int(payload.get("wins"))),
                    )
                )
    rows.sort(key=lambda item: (item.bar, item.label, item.timestamp))
    return rows


def _write_merged_trading_log(path: Path, shard_results: Sequence[dict[str, Any]]) -> None:
    by_bar: dict[int, dict[str, Any]] = {}
    for item in shard_results:
        log_path = Path(item["output_dir"]) / "trading.log"
        if not log_path.exists():
            continue
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parsed = _parse_trading_log_line(line)
            bar = _safe_int(parsed.get("bar"))
            if bar <= 0:
                continue
            bucket = by_bar.setdefault(bar, dict(parsed))
            if bucket is not parsed:
                bucket["shadow_signals"] = _safe_int(bucket.get("shadow_signals")) + _safe_int(
                    parsed.get("shadow_signals")
                )
                bucket["shadow_filled"] = _safe_int(bucket.get("shadow_filled")) + _safe_int(
                    parsed.get("shadow_filled")
                )
    lines = []
    for bar in sorted(by_bar):
        row = by_bar[bar]
        lines.append(
            f"{row.get('_prefix', '')}  bar={bar}  market={row.get('market', '')}  "
            f"regime={row.get('regime', '')}  leader=PanteonSoft:parallel_shadow_merge  "
            "selected_leader=ParallelShadowMerge  executed_leader=PanteonSoft:parallel_shadow_merge  "
            f"balance={row.get('balance', '$1000.00')}  positions={row.get('positions', 0)}  "
            "raw_signals=0  signals=0  filled=0  rejected=0  blocked=0  "
            f"shadow_signals={_safe_int(row.get('shadow_signals'))}  "
            f"shadow_filled={_safe_int(row.get('shadow_filled'))}  "
            f"market_confidence={row.get('market_confidence', '')}  "
            "fallback_used=False  fallback_skipped=True  "
            "fallback_reason=parallel shadow merge: execution log is not live-like"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_merged_leaderboards(output_dir: Path, shard_results: Sequence[dict[str, Any]]) -> None:
    for filename, section in (
        ("leaderboard_agents.json", "agents"),
        ("leaderboard_shadow_only_agents.json", "agents"),
        ("leaderboard_players.json", "players"),
        ("leaderboard_shadow_only_players.json", "players"),
    ):
        merged: dict[str, Any] = {}
        for item in shard_results:
            path = Path(item["output_dir"]) / filename
            payload = _load_json(path)
            rows = payload.get(section) if isinstance(payload, dict) else {}
            if isinstance(rows, dict):
                merged.update(rows)
        (output_dir / filename).write_text(
            json.dumps({section: merged}, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )


def _parse_trading_log_line(line: str) -> dict[str, Any]:
    row: dict[str, Any] = {}
    parts = line.split("  ")
    row["_prefix"] = parts[0].strip()
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.strip().split("=", 1)
        row[key.strip()] = value.strip()
    return row


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}


def _csv_labels(raw: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        values = raw.replace(";", ",").split(",")
    else:
        values = [str(value) for value in raw]
    return tuple(str(value).strip() for value in values if str(value).strip())


def _extend(target: list[str], values: Sequence[str] | Any) -> None:
    for value in values:
        label = str(value or "").strip()
        if label and label not in target:
            target.append(label)


def _first_label(value: object) -> str:
    try:
        parts = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return ""
    return str(parts[0] or "").strip() if parts else ""


def _safe_int(value: object) -> int:
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: object) -> float:
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0
    return number if number == number else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
