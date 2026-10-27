from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exia.genetic_primus.benchmarks_v4 import (  # noqa: E402
    NoTradePolicy,
    momentum_policy,
    relative_momentum_policy,
    summarize_benchmark,
    v7_registered_proposal_policies,
)
from exia.genetic_primus.contracts_v4 import load_v4_config  # noqa: E402
from exia.genetic_primus.data import load_panel  # noqa: E402
from exia.genetic_primus.features import build_feature_frame  # noqa: E402
from exia.genetic_primus.promotion_v4 import _outer_block_lcb  # noqa: E402
from exia.genetic_primus.rolling import fixed_baseline_scaler  # noqa: E402
from exia.genetic_primus.simulator import simulate_policy  # noqa: E402


MATRIX = ROOT / "configs/genetic_primus_v7_proposal_matrix.json"


def _timestamp(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def _compound(values: pd.Series) -> float:
    return float(np.prod(1.0 + values.to_numpy(float)) - 1.0)


def run(matrix_path: Path = MATRIX) -> dict[str, object]:
    raw = matrix_path.read_bytes()
    matrix = json.loads(raw)
    if matrix.get("schema_version") != "exia.genetic_primus.v7_proposal_matrix/1":
        raise ValueError("unexpected V7 proposal matrix schema")
    if matrix.get("status") != "REGISTERED_BEFORE_EVALUATION":
        raise ValueError("V7 proposal matrix was not registered before evaluation")
    safety = matrix.get("safety", {})
    expected_safety = {
        "credentials_allowed",
        "paper_allowed",
        "live_allowed",
        "orders_enabled",
        "promotion_authority",
    }
    if set(safety) != expected_safety or any(safety.values()):
        raise PermissionError("V7 proposal evaluation must remain fail-closed")
    output_dir = ROOT / str(matrix["result_path"])
    if output_dir.exists():
        raise FileExistsError(f"V7 proposal result path already exists: {output_dir}")

    registered = [(item["policy_id"], item["rule"]) for item in matrix["proposals"]]
    proposals = v7_registered_proposal_policies()
    implemented = [(policy.policy_id, policy.rule) for policy in proposals]
    if implemented != registered:
        raise ValueError("implemented V7 proposals do not match the registered matrix")

    config_path = ROOT / str(matrix["dataset_config_path"])
    config = load_v4_config(config_path, root=ROOT)
    if config.dataset["dataset_sha256"] != matrix["dataset_sha256"]:
        raise ValueError("V7 proposal matrix dataset SHA mismatch")
    features = build_feature_frame(load_panel(config, root=ROOT))
    scaler = fixed_baseline_scaler()
    policies = (
        NoTradePolicy(),
        momentum_policy(),
        relative_momentum_policy(),
        *proposals,
    )

    start = _timestamp(matrix["development_start"])
    stop = _timestamp(matrix["development_end"])
    step = int(pd.Timedelta(days=int(matrix["origin_cadence_days"])).total_seconds() * 1000)
    horizon = int(
        pd.Timedelta(days=int(matrix["deployment_horizon_days"])).total_seconds() * 1000
    )
    origins: list[tuple[int, int]] = []
    cursor = start
    while cursor + horizon <= stop:
        origins.append((cursor, cursor + horizon))
        cursor += step

    rows: list[dict[str, object]] = []
    portfolio_rows: list[pd.DataFrame] = []
    costs = {
        "base": float(matrix["round_trip_cost_bps"]),
        "stress": float(matrix["stress_round_trip_cost_bps"]),
    }
    for origin_index, (origin_start, origin_end) in enumerate(origins):
        for policy in policies:
            for cost_scenario, cost_bps in costs.items():
                simulation = simulate_policy(
                    features,
                    policy=policy,
                    scaler=scaler,
                    start_timestamp=origin_start,
                    end_timestamp=origin_end,
                    round_trip_cost_bps=cost_bps,
                )
                rows.append(
                    {
                        "origin_index": origin_index,
                        "origin_timestamp": origin_start,
                        "deployment_end": origin_end,
                        "policy_id": policy.policy_id,
                        "cost_scenario": cost_scenario,
                        "round_trip_cost_bps": cost_bps,
                        **summarize_benchmark(simulation),
                    }
                )
                local_portfolio = simulation.portfolio.loc[:, ["timestamp", "net_return"]].copy()
                local_portfolio["origin_index"] = origin_index
                local_portfolio["policy_id"] = policy.policy_id
                local_portfolio["cost_scenario"] = cost_scenario
                portfolio_rows.append(local_portfolio)

    outcomes = pd.DataFrame(rows)
    portfolio = pd.concat(portfolio_rows, ignore_index=True)
    rule = matrix["selection_rule"]
    base_no_trade = outcomes.loc[
        outcomes.policy_id.eq("NoTrade") & outcomes.cost_scenario.eq("base")
    ].set_index("origin_index")
    stress_no_trade = outcomes.loc[
        outcomes.policy_id.eq("NoTrade") & outcomes.cost_scenario.eq("stress")
    ].set_index("origin_index")
    summaries: dict[str, dict[str, object]] = {}
    for policy in policies:
        policy_summary: dict[str, object] = {}
        for scenario in costs:
            local = outcomes.loc[
                outcomes.policy_id.eq(policy.policy_id)
                & outcomes.cost_scenario.eq(scenario)
            ].sort_values("origin_index")
            path = portfolio.loc[
                portfolio.policy_id.eq(policy.policy_id)
                & portfolio.cost_scenario.eq(scenario)
            ].sort_values(["origin_index", "timestamp"])
            equity = (1.0 + path.net_return).cumprod()
            drawdown = float(-(equity / equity.cummax() - 1.0).min())
            policy_summary[scenario] = {
                "compound_return": _compound(local.net_return),
                "compound_bps": _compound(local.net_return) * 10_000.0,
                "mean_origin_bps": float(local.net_bps.mean()),
                "positive_origins": int((local.net_return > 0.0).sum()),
                "zero_origins": int(local.net_return.eq(0.0).sum()),
                "trades": int(local.trades.sum()),
                "turnover_fills": int(local.turnover_fills.sum()),
                "trades_per_day": float(local.trades.sum() / (len(local) * 3.0)),
                "max_drawdown": drawdown,
            }
        base = outcomes.loc[
            outcomes.policy_id.eq(policy.policy_id) & outcomes.cost_scenario.eq("base")
        ].set_index("origin_index")
        stress = outcomes.loc[
            outcomes.policy_id.eq(policy.policy_id) & outcomes.cost_scenario.eq("stress")
        ].set_index("origin_index")
        normal_delta = (base.net_bps - base_no_trade.net_bps).to_numpy(float)
        stress_delta = (stress.net_bps - stress_no_trade.net_bps).to_numpy(float)
        policy_summary["paired_vs_no_trade"] = {
            "mean_delta_bps": float(normal_delta.mean()),
            "selection_lcb_bps": _outer_block_lcb(
                normal_delta,
                samples=int(rule["outer_bootstrap_samples"]),
                block_origins=int(rule["outer_block_origins"]),
                seed=int(rule["selection_seed"]),
            ),
            "audit_lcb_bps": _outer_block_lcb(
                normal_delta,
                samples=int(rule["outer_bootstrap_samples"]),
                block_origins=int(rule["outer_block_origins"]),
                seed=int(rule["audit_seed"]),
            ),
            "stress_audit_lcb_bps": _outer_block_lcb(
                stress_delta,
                samples=int(rule["outer_bootstrap_samples"]),
                block_origins=int(rule["outer_block_origins"]),
                seed=int(rule["audit_seed"]) + 1,
            ),
        }
        block_returns: list[float] = []
        for block_start, block_end in matrix["calendar_blocks"]:
            lo, hi = _timestamp(block_start), _timestamp(block_end)
            local = base.loc[(base.origin_timestamp >= lo) & (base.origin_timestamp < hi)]
            block_returns.append(_compound(local.net_return))
        policy_summary["calendar_block_returns"] = block_returns
        policy_summary["positive_compound_blocks"] = int(
            sum(value > 0.0 for value in block_returns)
        )
        if policy.policy_id.startswith("V7_"):
            gates = {
                "positive_compound_blocks": policy_summary["positive_compound_blocks"]
                >= int(rule["required_positive_compound_blocks"]),
                "positive_full_period_return": policy_summary["base"]["compound_return"] > 0.0,
                "positive_paired_audit_lcb": policy_summary["paired_vs_no_trade"][
                    "audit_lcb_bps"
                ]
                > 0.0,
                "positive_stress_return": policy_summary["stress"]["compound_return"] > 0.0,
                "maximum_drawdown": policy_summary["base"]["max_drawdown"]
                <= float(rule["maximum_drawdown"]),
                "maximum_trades_per_day": policy_summary["base"]["trades_per_day"]
                <= float(rule["maximum_trades_per_day"]),
            }
            policy_summary["selection_gates"] = gates
            policy_summary["selected_for_v7_overlay"] = bool(all(gates.values()))
        summaries[policy.policy_id] = policy_summary

    output_dir.mkdir(parents=True)
    outcomes_path = output_dir / "origin_outcomes.parquet"
    portfolio_path = output_dir / "portfolio.parquet"
    outcomes.to_parquet(outcomes_path, index=False)
    portfolio.to_parquet(portfolio_path, index=False)
    artifact_sha256 = {
        "origin_outcomes": hashlib.sha256(outcomes_path.read_bytes()).hexdigest(),
        "portfolio": hashlib.sha256(portfolio_path.read_bytes()).hexdigest(),
    }
    result = {
        "schema_version": "exia.genetic_primus.v7_proposal_matrix_result/1",
        "status": "REVEALED_DEVELOPMENT_ONLY",
        "matrix_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_sha256": matrix["dataset_sha256"],
        "origins": len(origins),
        "policy_count": len(policies),
        "summaries": summaries,
        "selected_policy_ids": [
            key for key, value in summaries.items() if value.get("selected_for_v7_overlay")
        ],
        "artifact_sha256": artifact_sha256,
        **{key: False for key in sorted(safety)},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    result = run()
    print(
        json.dumps(
            {
                "status": result["status"],
                "origins": result["origins"],
                "policy_count": result["policy_count"],
                "selected_policy_ids": result["selected_policy_ids"],
                "orders_enabled": result["orders_enabled"],
            }
        )
    )


if __name__ == "__main__":
    main()
