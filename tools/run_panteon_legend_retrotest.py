from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from panteon_legend.config import LEGEND_YEARS  # noqa: E402
from panteon_legend.retrotest import build_legend_retrodate_config, run_legend_retrodate_benchmark  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "Results" / "PanteonLegend_20260531",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Retrodate")
    parser.add_argument(
        "--years",
        default=",".join(str(year) for year in LEGEND_YEARS),
    )
    parser.add_argument("--max-bars", type=int, default=None)
    parser.add_argument("--include-optional-agents", action="store_true")
    parser.add_argument("--optional-agent-labels", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    years = tuple(int(part.strip()) for part in str(args.years).split(",") if part.strip())
    if args.dry_run:
        config = build_legend_retrodate_config(
            results_root=args.results_root,
            data_dir=args.data_dir,
            years=years,
            max_bars=args.max_bars,
            include_optional_agents=args.include_optional_agents,
            optional_agent_labels=_csv_labels(args.optional_agent_labels),
        )
        print(json.dumps(_config_preview(config), indent=2, sort_keys=True))
        return 0

    summary = run_legend_retrodate_benchmark(
        results_root=args.results_root,
        data_dir=args.data_dir,
        years=years,
        max_bars=args.max_bars,
        include_optional_agents=args.include_optional_agents,
        optional_agent_labels=_csv_labels(args.optional_agent_labels),
    )
    print(f"output_dir={summary.output_dir}", flush=True)
    print(f"analysis_report={summary.report_path}", flush=True)
    print(f"run_summary={summary.summary_path}", flush=True)
    return 0


def _config_preview(config) -> dict:
    return {
        "results_root": str(config.results_root),
        "data_dir": str(config.data_dir),
        "years": list(config.years),
        "max_bars": config.max_bars,
        "flash_enabled": config.flash_enabled,
        "hard_policy_enabled": config.hard_policy_enabled,
        "include_optional_agents": config.include_optional_agents,
        "optional_agent_labels": list(config.optional_agent_labels),
        "risk_capital_fraction": config.risk_capital_fraction,
        "risk_max_leverage": config.risk_max_leverage,
        "apply_risk_leverage_to_notional": config.apply_risk_leverage_to_notional,
        "max_new_opens_per_bar": config.max_new_opens_per_bar,
        "risk_max_open_positions": config.risk_max_open_positions,
        "fixed_agent_players_enabled": config.fixed_agent_players_enabled,
        "fixed_agent_player_sets": [
            {"label": label, "agents": list(agents)}
            for label, agents in config.fixed_agent_player_sets
        ],
        "player_profile_labels": [
            str(getattr(profile, "label", ""))
            for profile in config.player_profiles
        ],
        "regime_switch_player_sets": [
            str(item[0])
            for item in config.regime_switch_player_sets
        ],
        "rotating_agent_player_sets": [
            str(item[0])
            for item in config.rotating_agent_player_sets
        ],
        "solo_agent_candidate_limit": config.solo_agent_candidate_limit,
    }


def _csv_labels(raw: str) -> tuple[str, ...]:
    return tuple(
        part.strip()
        for part in str(raw or "").replace(";", ",").split(",")
        if part.strip()
    )


if __name__ == "__main__":
    raise SystemExit(main())
