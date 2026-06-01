from __future__ import annotations

from pathlib import Path
from typing import Optional

from panteon_v2.app.startup import start_production

from .config import build_legend_profile, legend_player_profiles
from .runtime import (
    configure_legend_pipeline,
    legend_live_execution_config,
    legend_risk_config,
    legend_strategist_config,
)


def start_legend_production(
    *,
    exchange: str,
    mode: str = "live_futures",
    initial_capital: Optional[float] = None,
    snapshot_path: Optional[str] = None,
    jsonl_event_log: Optional[str] = None,
    results_root: str = "Results",
    sleep_between_polls_sec: float = 5.0,
) -> int:
    profile = build_legend_profile()
    return start_production(
        exchange=exchange,
        mode=mode,
        initial_capital=initial_capital,
        seed_quarantine=(),
        profiles=legend_player_profiles(),
        snapshot_path=snapshot_path,
        jsonl_event_log=jsonl_event_log,
        results_root=results_root,
        sleep_between_polls_sec=sleep_between_polls_sec,
        risk_config_override=legend_risk_config(profile),
        strategist_config_override=legend_strategist_config(profile),
        live_execution_config_override=legend_live_execution_config(profile),
        flash_enabled_override=False,
        configure_pipeline=lambda pipeline: configure_legend_pipeline(pipeline, profile),
    )


def default_state_path(project_root: str | Path, exchange: str) -> Path:
    return Path(project_root) / "panteon_legend_state" / f"{exchange.lower()}_snapshot.json"
