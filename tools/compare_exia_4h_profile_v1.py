from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(frame: pd.DataFrame, minimum_trades: int = 20) -> dict[str, Any]:
    sufficient = frame.loc[
        (frame["closed_trades"] >= minimum_trades) & frame["mean_stress_net_bps"].notna()
    ].copy()
    weighted = sufficient.loc[sufficient["closed_trades"] > 0]
    return {
        "configurations": int(len(frame)),
        "sample_sufficient": int(len(sufficient)),
        "closed_trades": int(frame["closed_trades"].sum()),
        "pooled_stress_net_bps": (
            None
            if weighted.empty
            else float(
                np.average(
                    weighted["mean_stress_net_bps"], weights=weighted["closed_trades"]
                )
            )
        ),
        "mean_of_component_means_bps": (
            None if sufficient.empty else float(sufficient["mean_stress_net_bps"].mean())
        ),
        "median_component_mean_bps": (
            None if sufficient.empty else float(sufficient["mean_stress_net_bps"].median())
        ),
        "positive_mean": int((sufficient["mean_stress_net_bps"] > 0).sum()),
        "positive_lcb": int((sufficient["stress_lcb_bps"] > 0).sum()),
        "positive_familywise_lcb": int(
            (sufficient["familywise_stress_lcb_bps"] > 0).sum()
        ),
    }


def _overall(
    metrics: pd.DataFrame,
    *,
    panel_id: str | None = None,
    mode: str | None = None,
    timeframe_minutes: int = 240,
) -> pd.DataFrame:
    selected = metrics.loc[
        (metrics["split"].astype(str) == "overall")
        & (metrics["timeframe_minutes"] == timeframe_minutes)
    ].copy()
    if panel_id is not None:
        selected = selected.loc[selected["panel_id"].astype(str) == panel_id].copy()
    if mode is not None:
        selected = selected.loc[selected["mode"].astype(str) == mode].copy()
    selected["agent_id"] = selected["agent_id"].astype(str)
    return selected


def build_comparison(
    baseline_dir: Path,
    adapted_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    baseline_metrics_path = baseline_dir / "metrics.parquet"
    adapted_metrics_path = adapted_dir / "metrics.parquet"
    adapted_report_path = adapted_dir / "report.json"
    baseline_metrics = pd.read_parquet(baseline_metrics_path)
    adapted_metrics = pd.read_parquet(adapted_metrics_path)
    adapted_report = json.loads(adapted_report_path.read_text(encoding="utf-8"))
    if adapted_report.get("orders_enabled") is not False:
        raise ValueError("adapted audit is not read-only")

    main_panel = "full8_4h_20220101_20260714"
    holdout_panel = "untouched_4h_20260715_20260817"
    baseline = _overall(baseline_metrics, mode="native_bar")
    adapted = _overall(adapted_metrics, panel_id=main_panel, mode="fixed_profile")
    holdout = _overall(adapted_metrics, panel_id=holdout_panel, mode="fixed_profile")
    if len(baseline) != 43 or len(adapted) != 43 or len(holdout) != 43:
        raise ValueError("expected exactly 43 comparable components in every panel")

    columns = [
        "agent_id",
        "component_type",
        "closed_trades",
        "mean_stress_net_bps",
        "stress_lcb_bps",
        "familywise_stress_lcb_bps",
        "status",
    ]
    comparison = baseline[columns].merge(
        adapted[columns], on="agent_id", suffixes=("_baseline", "_adapted")
    ).merge(
        holdout[columns], on="agent_id", how="left"
    )
    comparison = comparison.rename(
        columns={
            "closed_trades": "closed_trades_holdout",
            "mean_stress_net_bps": "mean_stress_net_bps_holdout",
            "stress_lcb_bps": "stress_lcb_bps_holdout",
            "familywise_stress_lcb_bps": "familywise_stress_lcb_bps_holdout",
            "status": "status_holdout",
            "component_type": "component_type_holdout",
        }
    )
    comparison["mean_delta_bps"] = (
        comparison["mean_stress_net_bps_adapted"]
        - comparison["mean_stress_net_bps_baseline"]
    )
    comparison["lcb_delta_bps"] = (
        comparison["stress_lcb_bps_adapted"] - comparison["stress_lcb_bps_baseline"]
    )
    comparison["mean_improved"] = comparison["mean_delta_bps"] > 0
    comparison["lcb_improved"] = comparison["lcb_delta_bps"] > 0
    comparison["activation_fixed"] = (
        (comparison["closed_trades_baseline"] < 20)
        & (comparison["closed_trades_adapted"] >= 20)
    )
    comparison = comparison.sort_values(
        ["component_type_adapted", "agent_id"], kind="stable"
    ).reset_index(drop=True)

    main_windows = adapted_metrics.loc[
        (adapted_metrics["panel_id"].astype(str) == main_panel)
        & (adapted_metrics["split"].astype(str) != "overall")
    ]
    all_windows_positive = 0
    for _, group in main_windows.groupby("agent_id"):
        supported = group.loc[group["closed_trades"] >= 10]
        if len(supported) == 4 and (supported["stress_lcb_bps"] > 0).all():
            all_windows_positive += 1

    by_type: dict[str, dict[str, Any]] = {}
    for component_type in sorted(set(adapted["component_type"].astype(str))):
        by_type[component_type] = {
            "baseline": _summary(
                baseline.loc[baseline["component_type"].astype(str) == component_type]
            ),
            "adapted": _summary(
                adapted.loc[adapted["component_type"].astype(str) == component_type]
            ),
        }

    payload = {
        "schema_version": "exia.4h_profile_comparison.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile_id": adapted_report.get("runtime_profile_id"),
        "timeframe_minutes": 240,
        "verdict": adapted_report["verdict"],
        "baseline": _summary(baseline),
        "adapted": _summary(adapted),
        "prospective_holdout": _summary(holdout),
        "improvement_counts": {
            "components": int(len(comparison)),
            "mean_improved": int(comparison["mean_improved"].sum()),
            "lcb_improved": int(comparison["lcb_improved"].sum()),
            "activation_fixed": int(comparison["activation_fixed"].sum()),
            "all_main_windows_positive_lcb": int(all_windows_positive),
        },
        "by_component_type": by_type,
        "best_adapted": adapted.sort_values("stress_lcb_bps", ascending=False)
        .head(10)[
            [
                "component_type",
                "agent_id",
                "closed_trades",
                "mean_stress_net_bps",
                "stress_lcb_bps",
                "familywise_stress_lcb_bps",
                "status",
            ]
        ]
        .to_dict("records"),
        "source": {
            "baseline_metrics": str(baseline_metrics_path.relative_to(ROOT)),
            "baseline_metrics_sha256": sha256_file(baseline_metrics_path),
            "adapted_metrics": str(adapted_metrics_path.relative_to(ROOT)),
            "adapted_metrics_sha256": sha256_file(adapted_metrics_path),
            "adapted_report": str(adapted_report_path.relative_to(ROOT)),
            "adapted_report_sha256": sha256_file(adapted_report_path),
        },
        "orders_enabled": False,
        "promotion_authority": False,
    }
    return payload, comparison


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}"


def render_markdown(payload: dict[str, Any], comparison: pd.DataFrame) -> str:
    baseline = payload["baseline"]
    adapted = payload["adapted"]
    holdout = payload["prospective_holdout"]
    counts = payload["improvement_counts"]
    lines = [
        "# Exia 4h profile v1: baseline vs adapted",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "The profile is a research baseline only. It grants no paper/live or promotion authority.",
        "",
        "## Aggregate comparison",
        "",
        "| Metric | Native 4h baseline | Adapted 4h | Prospective holdout |",
        "|---|---:|---:|---:|",
        f"| Sample-sufficient configurations | {baseline['sample_sufficient']} | {adapted['sample_sufficient']} | {holdout['sample_sufficient']} |",
        f"| Closed trades | {baseline['closed_trades']} | {adapted['closed_trades']} | {holdout['closed_trades']} |",
        f"| Pooled stress net, bps/trade | {_fmt(baseline['pooled_stress_net_bps'])} | {_fmt(adapted['pooled_stress_net_bps'])} | {_fmt(holdout['pooled_stress_net_bps'])} |",
        f"| Median component mean, bps/trade | {_fmt(baseline['median_component_mean_bps'])} | {_fmt(adapted['median_component_mean_bps'])} | {_fmt(holdout['median_component_mean_bps'])} |",
        f"| Positive mean | {baseline['positive_mean']} | {adapted['positive_mean']} | {holdout['positive_mean']} |",
        f"| Positive LCB | {baseline['positive_lcb']} | {adapted['positive_lcb']} | {holdout['positive_lcb']} |",
        "",
        "## Interpretation",
        "",
        f"- Mean improved for {counts['mean_improved']} of {counts['components']} components; LCB improved for {counts['lcb_improved']}.",
        f"- The explicit profile repaired insufficient activation for {counts['activation_fixed']} components.",
        f"- Components with positive LCB in all development/validation/OOS/sanity windows: {counts['all_main_windows_positive_lcb']}.",
        "- The old rare-signal outliers lost most of their apparent mean after check intervals and holding periods were expressed in actual 4h bars.",
        "- The prospective holdout has no positive LCB, so the profile must remain research-only.",
        "",
        "## Best adapted full-history configurations",
        "",
        "| Component | Type | Trades | Stress mean | LCB | Family-wise LCB |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["best_adapted"]:
        lines.append(
            f"| {row['agent_id']} | {row['component_type']} | {row['closed_trades']} | "
            f"{_fmt(row['mean_stress_net_bps'])} | {_fmt(row['stress_lcb_bps'])} | "
            f"{_fmt(row['familywise_stress_lcb_bps'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-component evidence",
            "",
            "The complete typed comparison is stored in `component_comparison.parquet`.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare native and adapted Exia 4h audits")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--adapted-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload, comparison = build_comparison(
        args.baseline_dir.resolve(), args.adapted_dir.resolve()
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "component_comparison.parquet"
    comparison.to_parquet(comparison_path, index=False)
    payload["component_comparison"] = {
        "path": str(comparison_path.relative_to(ROOT)),
        "sha256": sha256_file(comparison_path),
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "comparison.md").write_text(
        render_markdown(payload, comparison), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "mean_improved": payload["improvement_counts"]["mean_improved"],
                "lcb_improved": payload["improvement_counts"]["lcb_improved"],
                "holdout_positive_lcb": payload["prospective_holdout"]["positive_lcb"],
                "orders_enabled": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
