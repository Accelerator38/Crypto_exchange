from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
TOOLS = ROOT / "tools"
for path in (SRC, RUNTIME, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_exia_timebase_audit_v1 import (  # noqa: E402
    load_source_manifest_panel,
    summarize_trades,
)
from simple_research.strategies import build_market_regime  # noqa: E402


DEFAULT_SPEC = ROOT / "configs" / "exia_4h_regime_discovery_v1.json"
DEFAULT_AUDIT = ROOT / "Reports" / "Exia" / "four_hour_regime_discovery_v1_raw"
DEFAULT_OUTPUT = ROOT / "Reports" / "Exia" / "four_hour_regime_discovery_v1"
REGIME_ORDER = ("all", "bullish", "bearish", "neutral")
STATUS_ORDER = {
    "NO_ACTIVATION": 0,
    "LOW_INCIDENCE": 1,
    "NEGATIVE": 2,
    "MIXED": 3,
    "POSITIVE_UNCERTAIN": 4,
    "ROBUST_POSITIVE": 5,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _regime_map(spec: dict[str, Any]) -> tuple[dict[str, pd.Series], list[dict[str, Any]]]:
    params = dict(spec["regime_attribution"])
    maps: dict[str, pd.Series] = {}
    coverage: list[dict[str, Any]] = []
    value_map = {1: "bullish", -1: "bearish", 0: "neutral"}
    for panel in spec["panels"]:
        derived = panel["derived_timeframe_manifests"]["240"]
        frame = load_source_manifest_panel(
            ROOT / str(derived["manifest"]), str(derived["dataset_sha256"])
        )
        numeric = build_market_regime(frame, params)
        labels = numeric.map(value_map).astype("string")
        maps[str(panel["panel_id"])] = labels
        counts = labels.value_counts(dropna=False)
        for regime in ("bullish", "bearish", "neutral"):
            coverage.append(
                {
                    "panel_id": panel["panel_id"],
                    "regime": regime,
                    "bars": int(counts.get(regime, 0)),
                    "share": float(counts.get(regime, 0) / len(labels)),
                }
            )
    return maps, coverage


def _attribute_regimes(trades: pd.DataFrame, maps: dict[str, pd.Series]) -> pd.DataFrame:
    attributed = trades.copy()
    attributed["entry_regime"] = pd.Series(pd.NA, index=attributed.index, dtype="string")
    for panel_id, labels in maps.items():
        mask = attributed["panel_id"].astype(str).eq(panel_id)
        attributed.loc[mask, "entry_regime"] = (
            attributed.loc[mask, "entry_timestamp"].map(labels).astype("string")
        )
    if attributed["entry_regime"].isna().any():
        missing = int(attributed["entry_regime"].isna().sum())
        raise ValueError(f"market regime is missing for {missing} trades")
    unexpected = set(attributed["entry_regime"].astype(str)) - {"bullish", "bearish", "neutral"}
    if unexpected:
        raise ValueError(f"unexpected market regimes: {sorted(unexpected)}")
    return attributed


def _basic_summary(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "closed_trades": 0,
            "fills": 0,
            "mean_gross_bps": None,
            "mean_base_net_bps": None,
            "mean_stress_net_bps": None,
            "median_stress_net_bps": None,
            "positive_stress_rate": None,
        }
    stress = trades["stress_net_bps"].astype("float64")
    return {
        "closed_trades": int(len(trades)),
        "fills": int(trades["fills"].sum()),
        "mean_gross_bps": float(trades["gross_bps"].mean()),
        "mean_base_net_bps": float(trades["base_net_bps"].mean()),
        "mean_stress_net_bps": float(stress.mean()),
        "median_stress_net_bps": float(stress.median()),
        "positive_stress_rate": float((stress > 0).mean()),
    }


def _scope_trades(trades: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "historical_all":
        return trades.loc[trades["panel_id"].eq("full8_4h_20220101_20260714")]
    if scope == "prospective_holdout":
        return trades.loc[trades["panel_id"].eq("untouched_4h_20260715_20260817")]
    raise ValueError(f"unknown scope: {scope}")


def _full_metrics(
    trades: pd.DataFrame,
    *,
    spec: dict[str, Any],
    component_meta: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    samples = int(spec["bootstrap"]["samples"])
    alpha = float(spec["bootstrap"]["alpha"])
    minimum = int(spec["stability_gate"]["minimum_trades_per_timeframe"])
    familywise_alpha = alpha / (len(component_meta) * len(REGIME_ORDER))
    for scope in ("historical_all", "prospective_holdout"):
        scoped = _scope_trades(trades, scope)
        for agent_id, meta in component_meta.items():
            agent_trades = scoped.loc[scoped["agent_id"].astype(str).eq(agent_id)]
            for regime in REGIME_ORDER:
                subset = (
                    agent_trades
                    if regime == "all"
                    else agent_trades.loc[agent_trades["entry_regime"].eq(regime)]
                )
                summary = summarize_trades(
                    subset,
                    seed_key=f"{spec['audit_id']}:{scope}:{agent_id}:{regime}",
                    minimum_trades=minimum,
                    bootstrap_samples=samples,
                    alpha=alpha,
                    familywise_alpha=familywise_alpha,
                )
                summary["median_stress_net_bps"] = (
                    None
                    if subset.empty
                    else float(subset["stress_net_bps"].astype("float64").median())
                )
                rows.append(
                    {
                        "scope": scope,
                        "agent_id": agent_id,
                        "base_agent_id": meta["base_agent_id"],
                        "variant_id": meta["variant_id"],
                        "component_type": meta["component_type"],
                        "engine": meta["engine"],
                        "regime": regime,
                        **summary,
                    }
                )
    return pd.DataFrame(rows)


def _split_metrics(
    trades: pd.DataFrame,
    *,
    component_meta: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    historical = trades.loc[trades["panel_id"].eq("full8_4h_20220101_20260714")]
    for split in ("development", "validation", "oos", "sanity"):
        split_trades = historical.loc[historical["split"].eq(split)]
        for agent_id, meta in component_meta.items():
            agent_trades = split_trades.loc[split_trades["agent_id"].astype(str).eq(agent_id)]
            for regime in REGIME_ORDER:
                subset = (
                    agent_trades
                    if regime == "all"
                    else agent_trades.loc[agent_trades["entry_regime"].eq(regime)]
                )
                rows.append(
                    {
                        "split": split,
                        "agent_id": agent_id,
                        "base_agent_id": meta["base_agent_id"],
                        "variant_id": meta["variant_id"],
                        "component_type": meta["component_type"],
                        "engine": meta["engine"],
                        "regime": regime,
                        **_basic_summary(subset),
                    }
                )
    return pd.DataFrame(rows)


def _ranking_status(row: pd.Series) -> str:
    trades = int(row["closed_trades"])
    if trades == 0:
        return "NO_ACTIVATION"
    if trades < 20:
        return "LOW_INCIDENCE"
    mean = float(row["mean_stress_net_bps"])
    median = float(row["median_stress_net_bps"])
    lcb = row["stress_lcb_bps"]
    familywise = row["familywise_stress_lcb_bps"]
    if mean > 0 and median > 0 and pd.notna(lcb) and lcb > 0 and pd.notna(familywise) and familywise > 0:
        return "ROBUST_POSITIVE"
    if mean > 0 and median > 0 and pd.notna(lcb) and lcb > 0:
        return "POSITIVE_UNCERTAIN"
    if mean > 0 or median > 0:
        return "MIXED"
    return "NEGATIVE"


def _build_rankings(metrics: pd.DataFrame, split_metrics: pd.DataFrame) -> pd.DataFrame:
    supported = (
        split_metrics.assign(supported=split_metrics["closed_trades"].ge(10))
        .groupby(["agent_id", "regime"], sort=True, observed=True)
        .agg(
            supported_windows=("supported", "sum"),
            worst_window_mean_bps=("mean_stress_net_bps", "min"),
            worst_window_median_bps=("median_stress_net_bps", "min"),
        )
        .reset_index()
    )
    ranked_source = metrics.copy()
    historical_mask = ranked_source["scope"].eq("historical_all")
    historical = ranked_source.loc[historical_mask].merge(
        supported, on=["agent_id", "regime"], how="left"
    )
    holdout = ranked_source.loc[~historical_mask].copy()
    holdout["supported_windows"] = 0
    holdout["worst_window_mean_bps"] = np.nan
    holdout["worst_window_median_bps"] = np.nan
    ranked_source = pd.concat([historical, holdout], ignore_index=True)
    ranked_source["ranking_status"] = ranked_source.apply(_ranking_status, axis=1)
    ranked_source["status_tier"] = ranked_source["ranking_status"].map(STATUS_ORDER).astype(int)
    numeric_sort = [
        "familywise_stress_lcb_bps",
        "stress_lcb_bps",
        "median_stress_net_bps",
        "mean_stress_net_bps",
    ]
    for column in numeric_sort:
        ranked_source[f"_sort_{column}"] = ranked_source[column].fillna(-np.inf)
    ranked_source["_sort_drawdown"] = -ranked_source["max_drawdown_stress_bps"].fillna(np.inf)
    ranked: list[pd.DataFrame] = []
    for scope in ("historical_all", "prospective_holdout"):
        for regime in REGIME_ORDER:
            group = ranked_source.loc[
                ranked_source["scope"].eq(scope) & ranked_source["regime"].eq(regime)
            ].sort_values(
                [
                    "status_tier",
                    "supported_windows",
                    "_sort_familywise_stress_lcb_bps",
                    "_sort_stress_lcb_bps",
                    "_sort_median_stress_net_bps",
                    "_sort_mean_stress_net_bps",
                    "_sort_drawdown",
                    "agent_id",
                ],
                ascending=[False, False, False, False, False, False, False, True],
                kind="stable",
            )
            group["rank"] = np.arange(1, len(group) + 1)
            ranked.append(group)
    result = pd.concat(ranked, ignore_index=True)
    return result.drop(columns=[column for column in result if column.startswith("_sort_")])


def _agent_summary(rankings: pd.DataFrame) -> pd.DataFrame:
    rankings = rankings.loc[rankings["scope"].eq("historical_all")].copy()
    value_columns = [
        "mean_stress_net_bps",
        "median_stress_net_bps",
        "stress_lcb_bps",
        "closed_trades",
        "rank",
        "ranking_status",
    ]
    records: list[dict[str, Any]] = []
    for agent_id, group in rankings.groupby("agent_id", sort=True):
        first = group.iloc[0]
        row: dict[str, Any] = {
            "agent_id": agent_id,
            "base_agent_id": first["base_agent_id"],
            "variant_id": first["variant_id"],
            "component_type": first["component_type"],
            "engine": first["engine"],
        }
        for regime in REGIME_ORDER:
            metric = group.loc[group["regime"].eq(regime)].iloc[0]
            for column in value_columns:
                row[f"{regime}_{column}"] = metric[column]
        market_only = group.loc[group["regime"].ne("all")].sort_values("rank", kind="stable")
        row["best_market_regime"] = market_only.iloc[0]["regime"]
        row["best_market_rank"] = int(market_only.iloc[0]["rank"])
        records.append(row)
    return pd.DataFrame(records).sort_values("all_rank", kind="stable").reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    labels = [column.replace("_", " ") for column in columns]
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _render_report(
    *,
    spec: dict[str, Any],
    rankings: pd.DataFrame,
    summary: pd.DataFrame,
    coverage: list[dict[str, Any]],
) -> str:
    historical_rankings = rankings.loc[rankings["scope"].eq("historical_all")]
    holdout_rankings = rankings.loc[rankings["scope"].eq("prospective_holdout")]
    lines = [
        "# Exia 4h all-agent regime discovery v1",
        "",
        "## Contract",
        "",
        f"- Components: {spec['experiment_counts']['total_components']}.",
        f"- Current components: {spec['experiment_counts']['current_components']}.",
        f"- Additional current-agent variations: {spec['experiment_counts']['additional_variants']}.",
        f"- New causal OHLCV/cross-sectional agents: {spec['experiment_counts']['new_agents']}.",
        "- Timeframe: completed 4h bars; signal at close, fill on the next bar open.",
        "- External regime NoTrade gate: disabled. Regime is an attribution label only.",
        "- Costs: 8 bps base and 16 bps stress round trip.",
        "- Mean and median are recorded overall and for bullish, bearish and neutral entry regimes.",
        "- This is diagnostic discovery and grants no paper/live/promotion authority.",
        "",
        "## Regime coverage",
        "",
    ]
    lines.extend(_markdown_table(pd.DataFrame(coverage), ["panel_id", "regime", "bars", "share"]))
    lines.extend(["", "## Top 15 by regime", ""])
    top_columns = [
        "rank",
        "agent_id",
        "variant_id",
        "ranking_status",
        "closed_trades",
        "mean_stress_net_bps",
        "median_stress_net_bps",
        "stress_lcb_bps",
        "familywise_stress_lcb_bps",
        "supported_windows",
    ]
    for regime in REGIME_ORDER:
        lines.extend([f"### {regime}", ""])
        top = historical_rankings.loc[historical_rankings["regime"].eq(regime)].head(15)
        lines.extend(_markdown_table(top, top_columns))
        lines.append("")
    lines.extend(["## Untouched prospective holdout: top 10", ""])
    for regime in REGIME_ORDER:
        lines.extend([f"### {regime}", ""])
        top = holdout_rankings.loc[holdout_rankings["regime"].eq(regime)].head(10)
        lines.extend(_markdown_table(top, top_columns))
        lines.append("")
    lines.extend(
        [
            "## Every component by market regime",
            "",
            "The table below records stress-costed mean and median for every component. Full LCB, drawdown, fills, split metrics and ranks are in the parquet/TSV artifacts.",
            "",
        ]
    )
    summary_columns = [
        "all_rank",
        "agent_id",
        "variant_id",
        "all_closed_trades",
        "all_mean_stress_net_bps",
        "all_median_stress_net_bps",
        "bullish_mean_stress_net_bps",
        "bullish_median_stress_net_bps",
        "bearish_mean_stress_net_bps",
        "bearish_median_stress_net_bps",
        "neutral_mean_stress_net_bps",
        "neutral_median_stress_net_bps",
        "best_market_regime",
    ]
    lines.extend(_markdown_table(summary, summary_columns))
    lines.extend(
        [
            "",
            "## Data limitation",
            "",
            "A true L2 order-book agent is excluded because there is no continuous point-in-time L2 history for the common full8 2022-2026 window. It was not replaced with a misleading OHLCV proxy.",
            "",
            "Legacy Panteon player rows used the current runtime fallback because the optional crypto_exchange genetics dependency is unavailable. They are not evidence for a validated genetic ensemble.",
            "",
            "All paper/live/orders/promotion flags remain false.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_tsv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(frame.columns)
        for row in frame.itertuples(index=False, name=None):
            writer.writerow(row)


def analyze(spec_path: Path, audit_dir: Path, output_dir: Path) -> dict[str, Any]:
    spec = _load_json(spec_path)
    if set(spec.get("safety", {}).values()) != {False}:
        raise ValueError("discovery analysis cannot grant trading authority")
    if spec["selection_policy"]["external_regime_gate"] is not False:
        raise ValueError("external regime gate must remain disabled")
    raw_trades_path = audit_dir / "trades.parquet"
    if not raw_trades_path.is_file():
        raise FileNotFoundError(raw_trades_path)
    trades = pd.read_parquet(raw_trades_path)
    maps, coverage = _regime_map(spec)
    attributed = _attribute_regimes(trades, maps)
    component_meta = {
        str(row["agent_id"]): {
            "base_agent_id": str(row.get("base_agent_id", row["agent_id"])),
            "variant_id": str(row.get("variant_id", "BASE")),
            "component_type": str(row.get("component_type", "agent")),
            "engine": str(row.get("engine", "act")),
        }
        for row in spec["agents"]
    }
    metrics = _full_metrics(attributed, spec=spec, component_meta=component_meta)
    split_metrics = _split_metrics(attributed, component_meta=component_meta)
    rankings = _build_rankings(metrics, split_metrics)
    summary = _agent_summary(rankings)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "regime_trades": output_dir / "regime_trades.parquet",
        "agent_regime_metrics": output_dir / "agent_regime_metrics.parquet",
        "agent_regime_metrics_tsv": output_dir / "agent_regime_metrics.tsv",
        "agent_regime_split_metrics": output_dir / "agent_regime_split_metrics.parquet",
        "agent_regime_split_metrics_tsv": output_dir / "agent_regime_split_metrics.tsv",
        "rankings": output_dir / "rankings.parquet",
        "rankings_tsv": output_dir / "rankings.tsv",
        "agent_summary_tsv": output_dir / "agent_summary.tsv",
    }
    attributed.to_parquet(paths["regime_trades"], index=False)
    metrics.to_parquet(paths["agent_regime_metrics"], index=False)
    split_metrics.to_parquet(paths["agent_regime_split_metrics"], index=False)
    rankings.to_parquet(paths["rankings"], index=False)
    _write_tsv(paths["agent_regime_metrics_tsv"], metrics)
    _write_tsv(paths["agent_regime_split_metrics_tsv"], split_metrics)
    _write_tsv(paths["rankings_tsv"], rankings)
    _write_tsv(paths["agent_summary_tsv"], summary)

    historical_rankings = rankings.loc[rankings["scope"].eq("historical_all")]
    holdout_rankings = rankings.loc[rankings["scope"].eq("prospective_holdout")]
    robust = historical_rankings.loc[
        historical_rankings["ranking_status"].eq("ROBUST_POSITIVE")
    ]
    holdout_robust = holdout_rankings.loc[
        holdout_rankings["ranking_status"].eq("ROBUST_POSITIVE")
    ]
    report = {
        "schema_version": "exia.4h_regime_discovery_report.v1",
        "experiment_id": spec["audit_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "component_count": len(component_meta),
        "trade_count": int(len(attributed)),
        "regime_label_coverage": 1.0,
        "external_regime_gate": False,
        "mean_and_median_recorded_overall_and_by_regime": True,
        "robust_positive_by_regime": {
            regime: int((robust["regime"] == regime).sum()) for regime in REGIME_ORDER
        },
        "holdout_robust_positive_by_regime": {
            regime: int((holdout_robust["regime"] == regime).sum()) for regime in REGIME_ORDER
        },
        "top_by_regime": {
            regime: historical_rankings.loc[
                historical_rankings["regime"].eq(regime), "agent_id"
            ].head(10).tolist()
            for regime in REGIME_ORDER
        },
        "holdout_top_by_regime": {
            regime: holdout_rankings.loc[
                holdout_rankings["regime"].eq(regime), "agent_id"
            ].head(10).tolist()
            for regime in REGIME_ORDER
        },
        "coverage": coverage,
        "excluded_components": spec.get("excluded_components", []),
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
        "artifacts": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }
    report_json = output_dir / "report.json"
    report_md = output_dir / "report.md"
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_md.write_text(
        _render_report(spec=spec, rankings=rankings, summary=summary, coverage=coverage),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "exia.4h_regime_discovery_manifest.v1",
        "generated_at": report["generated_at"],
        "spec": {"path": str(spec_path.relative_to(ROOT)), "sha256": _sha256(spec_path)},
        "raw_audit_report": {
            "path": str((audit_dir / "report.json").relative_to(ROOT)),
            "sha256": _sha256(audit_dir / "report.json"),
        },
        "report": {"path": str(report_json.relative_to(ROOT)), "sha256": _sha256(report_json)},
        "report_md": {"path": str(report_md.relative_to(ROOT)), "sha256": _sha256(report_md)},
        "orders_enabled": False,
        "promotion_authority": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Exia 4h all-agent results by market regime")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = args.spec if args.spec.is_absolute() else ROOT / args.spec
    audit = args.audit_dir if args.audit_dir.is_absolute() else ROOT / args.audit_dir
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = analyze(spec.resolve(), audit.resolve(), output.resolve())
    print(json.dumps({
        "components": report["component_count"],
        "trades": report["trade_count"],
        "robust_positive_by_regime": report["robust_positive_by_regime"],
        "orders_enabled": report["orders_enabled"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
