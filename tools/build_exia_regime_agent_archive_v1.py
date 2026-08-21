from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "configs" / "research_archive" / "exia_regime_agent_archive_v1.json"
DEFAULT_OUTPUT = ROOT / "docs" / "research_archive" / "exia_regime_agents_v1"
DEFAULT_WORKBOOK_DATA = ROOT / "outputs" / "exia_regime_agent_archive_v1" / "workbook_data.json"
REGIMES = ("bullish", "bearish", "neutral")
SCOPES = ("historical_all", "prospective_holdout")
WINDOWS = ("development", "validation", "oos", "sanity")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_symbol(path: Path, symbol: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == symbol
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one top-level source symbol: {path}/{symbol}")
    node = matches[0]
    block = ast.get_source_segment(text, node)
    if not block:
        raise ValueError(f"cannot extract source block: {path}/{symbol}")
    normalized = block.replace("\r\n", "\n").encode("utf-8")
    return {
        "path": str(path.relative_to(ROOT)),
        "symbol": symbol,
        "line_start": int(node.lineno),
        "line_end": int(node.end_lineno or node.lineno),
        "implementation_sha256": hashlib.sha256(normalized).hexdigest(),
    }


def validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != "exia.regime_agent_archive.v1":
        raise ValueError("unexpected regime-agent archive schema")
    if set(spec.get("safety", {}).values()) != {False}:
        raise ValueError("archive cannot grant trading authority")
    policy = spec.get("selection_policy", {})
    if policy.get("not_a_promotion_shortlist") is not True:
        raise ValueError("archive must explicitly reject promotion authority")
    if policy.get("positive_mean_without_positive_lcb_is_not_an_edge") is not True:
        raise ValueError("positive mean cannot be treated as an edge")
    agents = spec.get("agents")
    if not isinstance(agents, list) or len(agents) != 10:
        raise ValueError("archive must contain exactly ten selected profiles")
    identifiers = [str(row.get("agent_id")) for row in agents]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("archive agent ids must be unique")
    counts = {regime: 0 for regime in REGIMES}
    for row in agents:
        regime = str(row.get("regime"))
        if regime not in counts:
            raise ValueError(f"unsupported archive regime: {regime}")
        counts[regime] += 1
    if counts != {"bullish": 3, "bearish": 3, "neutral": 4}:
        raise ValueError(f"unexpected regime allocation: {counts}")


def _one(frame: pd.DataFrame, **filters: str) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in filters.items():
        mask &= frame[column].eq(value)
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"expected one metric row for {filters}, got {len(selected)}")
    return selected.iloc[0]


def _metric_value(row: pd.Series, field: str) -> float | int | None:
    value = row.get(field)
    if pd.isna(value):
        return None
    if field in {"closed_trades", "fills", "long_trades", "short_trades"}:
        return int(value)
    return float(value)


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty TSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


def _render_readme(catalog: dict[str, Any]) -> str:
    lines = [
        "# Exia regime agent archive v1",
        "",
        "Статус: исследовательский архив. Ни один профиль не прошёл полный promotion gate.",
        "",
        "Архив сохраняет наиболее информативные 4h-профили по режимам рынка, их точные параметры, реализацию и costed evidence. Это не whitelist и не рекомендация для paper/live.",
        "",
        "## Краткий каталог",
        "",
        "| Regime | Rank | Agent | Role | Hist trades | Mean | Median | LCB | Family LCB | Holdout trades | Holdout mean | Holdout median | Holdout LCB |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    history = {(r["agent_id"], r["scope"]): r for r in catalog["selected_metrics"]}
    for agent in catalog["agents"]:
        hist = history[(agent["agent_id"], "historical_all")]
        hold = history[(agent["agent_id"], "prospective_holdout")]
        lines.append(
            f"| {agent['regime']} | {agent['archive_rank']} | `{agent['agent_id']}` | "
            f"{agent['archive_role']} | {hist['closed_trades']} | {_fmt(hist['mean_stress_net_bps'])} | "
            f"{_fmt(hist['median_stress_net_bps'])} | {_fmt(hist['stress_lcb_bps'])} | "
            f"{_fmt(hist['familywise_stress_lcb_bps'])} | {hold['closed_trades']} | "
            f"{_fmt(hold['mean_stress_net_bps'])} | {_fmt(hold['median_stress_net_bps'])} | "
            f"{_fmt(hold['stress_lcb_bps'])} |"
        )
    for regime in REGIMES:
        lines.extend(["", f"## {regime.capitalize()}", ""])
        for agent in [row for row in catalog["agents"] if row["regime"] == regime]:
            source = agent["source"]
            spec = agent["source_spec"]
            params = spec.get("post_profile_overrides") or spec.get("params") or {}
            lines.extend(
                [
                    f"### {agent['archive_rank']}. `{agent['agent_id']}`",
                    "",
                    agent["description_ru"],
                    "",
                    f"- Роль: `{agent['archive_role']}`.",
                    f"- Механика: {agent['mechanism']}.",
                    f"- Почему сохранён: {agent['why_preserved_ru']}",
                    f"- Ограничения: {agent['limitations_ru']}",
                    f"- Фиксированные параметры: `{json.dumps(params, ensure_ascii=False, sort_keys=True)}`.",
                    f"- Исходник: `{source['path']}::{source['symbol']}`; implementation SHA `{source['implementation_sha256']}`.",
                    "",
                ]
            )
    lines.extend(
        [
            "## Правила повторного использования",
            "",
            "1. Не включать архивный профиль в live whitelist или ensemble по факту нахождения в архиве.",
            "2. Не подбирать параметры на уже раскрытых validation/OOS/sanity/holdout окнах.",
            "3. Новая попытка требует materially different hypothesis и нового prospective evidence.",
            "4. Обязательны realistic costs, positive mean+median+LCB, direction/regime checks и drawdown gate.",
            "5. `LiveVolCompress` остаётся terminal negative reference; neutral directional compression family закрыта.",
            "",
            "## Файлы",
            "",
            "- `catalog.json` - полный машинный каталог.",
            "- `selected_metrics.tsv` - history и holdout, отдельные ячейки Excel.",
            "- `window_metrics.tsv` - development/validation/OOS/sanity.",
            "- `source_specs.json` - точные параметры и SHA блоков реализации.",
            "- `agent_catalog.xlsx` - человекочитаемый Excel-каталог.",
            "- `manifest.json` - evidence и artifact hashes.",
            "",
        ]
    )
    return "\n".join(lines)


def build(spec_path: Path, output_dir: Path, workbook_data_path: Path) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    output_dir = output_dir.resolve()
    workbook_data_path = workbook_data_path.resolve()
    spec = load_json(spec_path)
    validate_spec(spec)
    source = spec["source_experiment"]
    experiment_config_path = ROOT / str(source["config"])
    metrics_path = ROOT / str(source["metrics"])
    split_path = ROOT / str(source["split_metrics"])
    report_path = ROOT / str(source["report"])
    for path in (experiment_config_path, metrics_path, split_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    experiment = load_json(experiment_config_path)
    source_specs = {str(row["agent_id"]): row for row in experiment["agents"]}
    metrics = pd.read_parquet(metrics_path)
    split_metrics = pd.read_parquet(split_path)

    agents: list[dict[str, Any]] = []
    selected_metrics: list[dict[str, Any]] = []
    window_metrics: list[dict[str, Any]] = []
    source_spec_rows: list[dict[str, Any]] = []
    for selection in spec["agents"]:
        agent_id = str(selection["agent_id"])
        regime = str(selection["regime"])
        if agent_id not in source_specs:
            raise ValueError(f"agent spec missing from source experiment: {agent_id}")
        source_path = ROOT / str(selection["source"]["path"])
        source_info = _source_symbol(source_path, str(selection["source"]["symbol"]))
        source_spec = source_specs[agent_id]
        agent = {**selection, "source": source_info, "source_spec": source_spec}
        agents.append(agent)
        source_spec_rows.append(
            {
                "agent_id": agent_id,
                "regime": regime,
                "source": source_info,
                "source_spec": source_spec,
            }
        )
        for scope in SCOPES:
            metric = _one(metrics, agent_id=agent_id, regime=regime, scope=scope)
            row = {
                "regime": regime,
                "archive_rank": int(selection["archive_rank"]),
                "agent_id": agent_id,
                "archive_role": str(selection["archive_role"]),
                "scope": scope,
            }
            for field in (
                "closed_trades",
                "fills",
                "long_trades",
                "short_trades",
                "mean_gross_bps",
                "mean_base_net_bps",
                "mean_stress_net_bps",
                "median_stress_net_bps",
                "stress_lcb_bps",
                "familywise_stress_lcb_bps",
                "positive_stress_rate",
                "max_drawdown_stress_bps",
            ):
                row[field] = _metric_value(metric, field)
            row["status"] = str(metric["status"])
            selected_metrics.append(row)
        for window in WINDOWS:
            metric = _one(split_metrics, agent_id=agent_id, regime=regime, split=window)
            window_metrics.append(
                {
                    "regime": regime,
                    "archive_rank": int(selection["archive_rank"]),
                    "agent_id": agent_id,
                    "archive_role": str(selection["archive_role"]),
                    "window": window,
                    "closed_trades": int(metric["closed_trades"]),
                    "fills": int(metric["fills"]),
                    "mean_gross_bps": float(metric["mean_gross_bps"]),
                    "mean_base_net_bps": float(metric["mean_base_net_bps"]),
                    "mean_stress_net_bps": float(metric["mean_stress_net_bps"]),
                    "median_stress_net_bps": float(metric["median_stress_net_bps"]),
                    "positive_stress_rate": float(metric["positive_stress_rate"]),
                }
            )

    catalog = {
        "schema_version": "exia.regime_agent_archive_catalog.v1",
        "archive_id": str(spec["archive_id"]),
        "created_on": str(spec["created_on"]),
        "status": "RESEARCH_ARCHIVE_NOT_PROMOTABLE",
        "selection_policy": spec["selection_policy"],
        "source_experiment": {
            **source,
            "config_sha256": sha256_file(experiment_config_path),
            "metrics_sha256": sha256_file(metrics_path),
            "split_metrics_sha256": sha256_file(split_path),
            "report_sha256": sha256_file(report_path),
        },
        "agents": agents,
        "selected_metrics": selected_metrics,
        "window_metrics": window_metrics,
        "safety": dict(spec["safety"]),
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    workbook_data_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path = output_dir / "catalog.json"
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_tsv(output_dir / "selected_metrics.tsv", selected_metrics)
    _write_tsv(output_dir / "window_metrics.tsv", window_metrics)
    (output_dir / "source_specs.json").write_text(
        json.dumps(source_spec_rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(_render_readme(catalog), encoding="utf-8")
    workbook_data_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifact_names = ["catalog.json", "selected_metrics.tsv", "window_metrics.tsv", "source_specs.json", "README.md"]
    if (output_dir / "agent_catalog.xlsx").is_file():
        artifact_names.append("agent_catalog.xlsx")
    manifest = {
        "schema_version": "exia.regime_agent_archive_manifest.v1",
        "archive_id": str(spec["archive_id"]),
        "spec": {"path": str(spec_path.relative_to(ROOT)), "sha256": sha256_file(spec_path)},
        "source_experiment": catalog["source_experiment"],
        "artifacts": {
            name: {"path": str((output_dir / name).relative_to(ROOT)), "sha256": sha256_file(output_dir / name)}
            for name in artifact_names
        },
        "orders_enabled": False,
        "promotion_authority": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Exia regime-agent research archive")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workbook-data", type=Path, default=DEFAULT_WORKBOOK_DATA)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec_path = args.spec if args.spec.is_absolute() else ROOT / args.spec
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    workbook_data = args.workbook_data if args.workbook_data.is_absolute() else ROOT / args.workbook_data
    catalog = build(spec_path, output_dir, workbook_data)
    print(
        json.dumps(
            {
                "archive_id": catalog["archive_id"],
                "agent_count": len(catalog["agents"]),
                "status": catalog["status"],
                "orders_enabled": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
