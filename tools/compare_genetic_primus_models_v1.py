from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from openpyxl import load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Font, PatternFill
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for path in (SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from exia.genetic_primus.contracts import load_primus_config, sha256_file  # noqa: E402
from exia.genetic_primus.data import load_panel  # noqa: E402
from exia.genetic_primus.features import RobustScaler, build_feature_frame  # noqa: E402
from exia.genetic_primus.policy import GeneticPrimusPolicy  # noqa: E402
from exia.genetic_primus.simulator import simulate_policy  # noqa: E402
from exia.scoring import moving_block_lcb  # noqa: E402
from simple_research.simulator import CostModel, simulate_targets  # noqa: E402
from simple_research.strategies import build_signal  # noqa: E402
from run_exia_4h_candidate_screen_v1 import (  # noqa: E402
    _load_panel as load_standard_panel,
    _window_map,
    load_json,
    validate_family,
)


DEFAULT_FAMILY = ROOT / "configs" / "exia_4h_candidate_family_v1.json"
DEFAULT_PRIMUS_CONFIG = ROOT / "configs" / "genetic_primus_v3.json"
DEFAULT_OUTPUT = (
    ROOT / "Reports" / "Exia" / "Genetic_Primus" / "model_comparison_v1_20260821"
)
BAR_MS = 240 * 60 * 1000
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_ALPHA = 0.05
STRESS_COST_BPS = 16.0

PRIMUS_RUNS = {
    "Genetic_Primus_V1": ROOT
    / "Reports/Exia/Genetic_Primus/development_v1_20260821",
    "Genetic_Primus_V2": ROOT
    / "Reports/Exia/Genetic_Primus/development_v2_20260821",
    "Genetic_Primus_V3": ROOT
    / "Reports/Exia/Genetic_Primus/development_v3_100g_20260821",
}

LEGACY_RUNS = {
    "player_one": ROOT / "Reports/Exia/player_one_v1/trades.parquet",
    "player_two": ROOT / "Reports/Exia/player_two_v1/trades.parquet",
}


def _seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _max_share(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    counts = values.value_counts()
    return float(counts.max() / counts.sum())


def summarize_ledger(
    ledger: pd.DataFrame,
    *,
    model_id: str,
    split: str,
    max_drawdown: float | None,
) -> dict[str, Any]:
    if ledger.empty:
        return {
            "trades": 0,
            "fills": 0,
            "mean_bps": None,
            "median_bps": None,
            "lcb_bps": None,
            "positive_rate": None,
            "total_net_bps": 0.0,
            "max_drawdown": max_drawdown,
            "max_direction_share": 0.0,
            "max_symbol_share": 0.0,
        }
    values = ledger["net_bps"].astype("float64")
    timestamps = ledger["signal_timestamp"].astype("int64")
    return {
        "trades": int(len(ledger)),
        "fills": int(ledger["fills"].sum()),
        "mean_bps": float(values.mean()),
        "median_bps": float(values.median()),
        "lcb_bps": moving_block_lcb(
            values,
            timestamps,
            samples=BOOTSTRAP_SAMPLES,
            alpha=BOOTSTRAP_ALPHA,
            seed=_seed(f"comparison:{model_id}:{split}"),
        ),
        "positive_rate": float((values > 0.0).mean()),
        "total_net_bps": float(values.sum()),
        "max_drawdown": max_drawdown,
        "max_direction_share": _max_share(ledger["direction"]),
        "max_symbol_share": _max_share(ledger["symbol"]),
    }


def _common_row(
    *,
    model_id: str,
    model_family: str,
    split: str,
    ledger: pd.DataFrame,
    max_drawdown: float | None,
    protocol_status: str,
    native_lcb_bps: float | None,
    comparison_scope: str,
    notes: str,
) -> dict[str, Any]:
    metrics = summarize_ledger(
        ledger,
        model_id=model_id,
        split=split,
        max_drawdown=max_drawdown,
    )
    enough = metrics["trades"] >= 20
    positive = (
        enough
        and metrics["mean_bps"] is not None
        and metrics["mean_bps"] > 0.0
        and metrics["lcb_bps"] is not None
        and metrics["lcb_bps"] > 0.0
    )
    return {
        "model_id": model_id,
        "model_family": model_family,
        "split": split,
        **metrics,
        "native_lcb_bps": native_lcb_bps,
        "comparison_positive_gate": bool(positive),
        "protocol_status": protocol_status,
        "comparison_scope": comparison_scope,
        "notes": notes,
    }


def _load_primus_development_rows() -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    ledgers: list[pd.DataFrame] = []
    for model_id, run_dir in PRIMUS_RUNS.items():
        metrics = json.loads(
            (run_dir / "strict_development_metrics.json").read_text(encoding="utf-8")
        )
        freeze = json.loads((run_dir / "freeze_manifest.json").read_text(encoding="utf-8"))
        ledger = pd.read_parquet(run_dir / "trades.parquet").copy()
        ledger["model_id"] = model_id
        ledger["split"] = "development"
        ledgers.append(ledger)
        rows.append(
            _common_row(
                model_id=model_id,
                model_family="neurogenetic",
                split="development",
                ledger=ledger,
                max_drawdown=float(metrics["max_drawdown"]),
                protocol_status=str(freeze["status"]),
                native_lcb_bps=_safe_float(metrics.get("lcb_bps")),
                comparison_scope="primary_same_simulator",
                notes=(
                    "V1/V2/V3 were trained only on development; common LCB is "
                    "recomputed with 5000 moving-block samples."
                ),
            )
        )
    return rows, ledgers


def _load_scaler(path: Path) -> RobustScaler:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RobustScaler(
        median=np.asarray(payload["median"], dtype=np.float64),
        scale=np.asarray(payload["scale"], dtype=np.float64),
    )


def _evaluate_v3_validation(config_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    run_dir = PRIMUS_RUNS["Genetic_Primus_V3"]
    config = load_primus_config(config_path, root=ROOT)
    validation_metrics = json.loads(
        (run_dir / "validation_metrics.json").read_text(encoding="utf-8")
    )
    validation_manifest = json.loads(
        (run_dir / "validation_manifest.json").read_text(encoding="utf-8")
    )
    with np.load(run_dir / "champion.npz", allow_pickle=False) as payload:
        genome = np.asarray(payload["genome"], dtype=np.float32).reshape(-1)
    features = build_feature_frame(load_panel(config, root=ROOT))
    window = config.windows["validation"]
    simulation = simulate_policy(
        features,
        policy=GeneticPrimusPolicy(genome),
        scaler=_load_scaler(run_dir / "scaler.json"),
        start_timestamp=int(pd.Timestamp(window["start"]).timestamp() * 1000),
        end_timestamp=int(pd.Timestamp(window["end"]).timestamp() * 1000),
        round_trip_cost_bps=float(config.costs["stress_round_trip"]),
    )
    ledger = simulation.ledger.copy()
    ledger["model_id"] = "Genetic_Primus_V3"
    ledger["split"] = "validation"
    row = _common_row(
        model_id="Genetic_Primus_V3",
        model_family="neurogenetic",
        split="validation",
        ledger=ledger,
        max_drawdown=float(validation_metrics["max_drawdown"]),
        protocol_status=str(validation_manifest["status"]),
        native_lcb_bps=_safe_float(validation_metrics.get("lcb_bps")),
        comparison_scope="primary_same_simulator",
        notes="One-time validation already opened by the frozen V3 protocol; OOS remains sealed.",
    )
    return row, ledger


def _prepare_standard_ledger(result: Any, model_id: str, split: str) -> pd.DataFrame:
    ledger = result.ledger.copy()
    if ledger.empty:
        ledger = pd.DataFrame(
            columns=[
                "symbol",
                "direction",
                "signal_timestamp",
                "entry_timestamp",
                "net_bps",
                "fills",
            ]
        )
    else:
        ledger["signal_timestamp"] = ledger["entry_timestamp"].astype("int64") - BAR_MS
    ledger["model_id"] = model_id
    ledger["split"] = split
    return ledger


def _evaluate_standard_models(
    family_path: Path,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], dict[str, Any]]:
    family = load_json(family_path)
    validate_family(family)
    if float(family["costs_bps"]["stress_round_trip"]) != STRESS_COST_BPS:
        raise ValueError("comparison is sealed to the 16 bps stress cost")
    panel = load_standard_panel(family)
    windows = _window_map(family)
    model_specs: list[tuple[str, pd.Series, str]] = []
    for candidate in family["candidates"]:
        model_specs.append(
            (
                str(candidate["candidate_id"]),
                build_signal(
                    panel,
                    {"kind": candidate["strategy_kind"], "params": candidate["params"]},
                ),
                "standard_market_quorum",
            )
        )
    model_specs.extend(
        [
            ("CONTROL_ALWAYS_LONG", pd.Series(1, index=panel.index, dtype="int8"), "control"),
            ("CONTROL_ALWAYS_SHORT", pd.Series(-1, index=panel.index, dtype="int8"), "control"),
            ("CONTROL_NO_TRADE", pd.Series(0, index=panel.index, dtype="int8"), "control"),
        ]
    )
    rows: list[dict[str, Any]] = []
    ledgers: list[pd.DataFrame] = []
    for model_id, signal, family_name in model_specs:
        for split in ("development", "validation"):
            start, end = windows[split]
            isolated_signal = signal.where(
                (panel["timestamp"] >= start) & (panel["timestamp"] < end), 0
            )
            result = simulate_targets(
                panel,
                isolated_signal,
                start_timestamp=start,
                end_timestamp=end,
                costs=CostModel(fee_bps_per_fill=4.0, slippage_bps_per_fill=4.0),
                trial_count=1,
            )
            ledger = _prepare_standard_ledger(result, model_id, split)
            ledgers.append(ledger)
            protocol = (
                "DEVELOPMENT_REJECTED_OR_CONTROL"
                if split == "development"
                else "POST_REJECTION_VALIDATION_DIAGNOSTIC"
            )
            rows.append(
                _common_row(
                    model_id=model_id,
                    model_family=family_name,
                    split=split,
                    ledger=ledger,
                    max_drawdown=float(result.metrics["max_drawdown"]),
                    protocol_status=protocol,
                    native_lcb_bps=None,
                    comparison_scope="primary_common_execution",
                    notes=(
                        "Validation is diagnostic only because every market-quorum rule "
                        "failed its preregistered development gate."
                    ),
                )
            )
    identity = {
        "family_path": str(family_path.relative_to(ROOT)),
        "family_sha256": sha256_file(family_path),
        "dataset_sha256": family["source"]["dataset_sha256"],
    }
    return rows, ledgers, identity


def _load_legacy_rows() -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    ledgers: list[pd.DataFrame] = []
    for model_id, path in LEGACY_RUNS.items():
        all_trades = pd.read_parquet(path)
        for split in ("development", "validation"):
            ledger = all_trades.loc[all_trades["split"] == split].copy()
            ledger["net_bps"] = ledger["stress_net_bps"].astype(float)
            ledger["model_id"] = model_id
            ledgers.append(ledger)
            rows.append(
                _common_row(
                    model_id=model_id,
                    model_family="legacy_ensemble_reference",
                    split=split,
                    ledger=ledger,
                    max_drawdown=None,
                    protocol_status="RESEARCH_FAIL_NOT_PROMOTABLE",
                    native_lcb_bps=None,
                    comparison_scope="reference_different_routing",
                    notes=(
                        "Same full8/4h/16 bps data windows, but different routing and "
                        "position ownership; excluded from the primary rank."
                    ),
                )
            )
    return rows, ledgers


def _rank(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["lcb_rank_within_split"] = np.nan
    primary = ranked["comparison_scope"].str.startswith("primary")
    for split, indices in ranked.loc[primary].groupby("split").groups.items():
        values = ranked.loc[indices, "lcb_bps"]
        ranked.loc[indices, "lcb_rank_within_split"] = values.rank(
            method="min", ascending=False, na_option="bottom"
        )
    return ranked.sort_values(
        ["split", "comparison_scope", "lcb_rank_within_split", "model_id"],
        kind="stable",
    ).reset_index(drop=True)


def _write_sheet(writer: pd.ExcelWriter, name: str, frame: pd.DataFrame) -> None:
    frame.to_excel(writer, sheet_name=name, index=False)
    sheet = writer.book[name]
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
    for column in sheet.columns:
        values = [str(cell.value) if cell.value is not None else "" for cell in column[:250]]
        sheet.column_dimensions[column[0].column_letter].width = min(
            52, max(10, max(len(value) for value in values) + 2)
        )
    for metric in ("mean_bps", "median_bps", "lcb_bps"):
        if metric in frame.columns and len(frame):
            column_number = list(frame.columns).index(metric) + 1
            letter = sheet.cell(row=1, column=column_number).column_letter
            sheet.conditional_formatting.add(
                f"{letter}2:{letter}{len(frame) + 1}",
                ColorScaleRule(
                    start_type="min",
                    start_color="F8696B",
                    mid_type="num",
                    mid_value=0,
                    mid_color="FFEB84",
                    end_type="max",
                    end_color="63BE7B",
                ),
            )


def write_workbook(path: Path, *, leaderboard: pd.DataFrame, methodology: pd.DataFrame) -> None:
    primus = leaderboard.loc[leaderboard["model_family"] == "neurogenetic"]
    standards = leaderboard.loc[
        leaderboard["model_family"].isin(["standard_market_quorum", "control"])
    ]
    legacy = leaderboard.loc[leaderboard["model_family"] == "legacy_ensemble_reference"]
    sheets = {
        "Common Leaderboard": leaderboard,
        "Primus Versions": primus,
        "Standard Models": standards,
        "Legacy References": legacy,
        "Methodology": methodology,
    }
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            _write_sheet(writer, name, frame)
    workbook = load_workbook(path, read_only=False)
    for name, frame in sheets.items():
        sheet = workbook[name]
        if sheet.max_column != len(frame.columns) or sheet.max_column <= 1:
            raise RuntimeError(f"invalid separate-cell layout in {name}")
        if sheet.merged_cells.ranges:
            raise RuntimeError(f"merged cells are forbidden in {name}")
    workbook.close()


def plot_development(leaderboard: pd.DataFrame, path: Path) -> None:
    subset = leaderboard.loc[
        (leaderboard["split"] == "development")
        & ~leaderboard["model_family"].eq("legacy_ensemble_reference")
        & leaderboard["lcb_bps"].notna()
    ].sort_values("lcb_bps")
    figure, axis = plt.subplots(figsize=(13, 9), constrained_layout=True)
    y = np.arange(len(subset))
    axis.barh(y, subset["mean_bps"], color="#4C78A8", alpha=0.75, label="Mean")
    axis.scatter(subset["median_bps"], y, color="#F2CF5B", edgecolor="#333333", label="Median")
    axis.scatter(subset["lcb_bps"], y, color="#E45756", marker="D", label="LCB 95%")
    axis.axvline(0.0, color="#222222", linewidth=1)
    axis.set_yticks(y, subset["model_id"])
    axis.set_xlabel("Stress-costed bps per closed trade")
    axis.set_title("Genetic_Primus V1-V3 vs standard 4h models: development")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="lower right")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_split_heatmap(leaderboard: pd.DataFrame, path: Path) -> None:
    subset = leaderboard.loc[
        leaderboard["comparison_scope"].str.startswith("primary")
        & leaderboard["model_family"].ne("control")
    ].copy()
    matrix = subset.pivot(index="model_id", columns="split", values="lcb_bps")
    matrix = matrix.reindex(columns=["development", "validation"])
    values = matrix.to_numpy(dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    limit = max(1.0, float(np.quantile(finite, 0.95))) if len(finite) else 1.0
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    image = axis.imshow(
        values,
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
    )
    axis.set_xticks(range(len(matrix.columns)), matrix.columns)
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_title("Common moving-block LCB by split")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = "sealed" if not np.isfinite(value) else f"{value:.1f}"
            axis.text(column, row, label, ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, shrink=0.8, label="stress-costed bps/trade")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _fmt(value: Any) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{float(value):.2f}"


def _render_report(leaderboard: pd.DataFrame, identity: dict[str, Any]) -> str:
    development = leaderboard.loc[
        (leaderboard["split"] == "development")
        & leaderboard["comparison_scope"].str.startswith("primary")
    ].sort_values("lcb_bps", ascending=False, na_position="last")
    validation = leaderboard.loc[
        (leaderboard["split"] == "validation")
        & leaderboard["comparison_scope"].str.startswith("primary")
    ].sort_values("lcb_bps", ascending=False, na_position="last")
    primus = development.loc[development["model_family"] == "neurogenetic"]
    v3_validation = validation.loc[validation["model_id"] == "Genetic_Primus_V3"].iloc[0]
    best_standard = development.loc[
        development["model_family"] == "standard_market_quorum"
    ].iloc[0]
    best_primus = primus.iloc[0]
    lines = [
        "# Genetic_Primus V1-V3 vs standard 4h models",
        "",
        "Статус: retrospective research comparison; paper/live authority отсутствует.",
        "",
        "## Единый контракт",
        "",
        "- Bitget full8, 4h, решения по закрытой свече, исполнение на следующем open.",
        "- Stress cost: 16 bps round trip.",
        "- Development: 2022-01-01..2023-12-31; validation: 2024-01-01..2024-12-31.",
        f"- Common LCB: moving-block bootstrap, {BOOTSTRAP_SAMPLES} samples, alpha={BOOTSTRAP_ALPHA}.",
        f"- Dataset SHA: `{identity['dataset_sha256']}`.",
        "- V1/V2 validation остаётся закрытой, потому что версии не прошли собственный development gate.",
        "- OOS и sanity не открывались для этого сравнения.",
        "",
        "## Development: три версии Primus",
        "",
        "| Model | Trades | Mean | Median | Common LCB | Drawdown | Protocol status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in primus.itertuples(index=False):
        lines.append(
            f"| {row.model_id} | {row.trades} | {_fmt(row.mean_bps)} | "
            f"{_fmt(row.median_bps)} | {_fmt(row.lcb_bps)} | "
            f"{_fmt(100.0 * row.max_drawdown)}% | {row.protocol_status} |"
        )
    lines.extend(
        [
            "",
            "## Development: общий рейтинг",
            "",
            "| Rank | Model | Family | Trades | Mean | Median | Common LCB | DD |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in development.itertuples(index=False):
        rank = "n/a" if pd.isna(row.lcb_rank_within_split) else str(int(row.lcb_rank_within_split))
        dd = "n/a" if pd.isna(row.max_drawdown) else f"{100.0 * row.max_drawdown:.2f}%"
        lines.append(
            f"| {rank} | {row.model_id} | {row.model_family} | {row.trades} | "
            f"{_fmt(row.mean_bps)} | {_fmt(row.median_bps)} | {_fmt(row.lcb_bps)} | {dd} |"
        )
    lines.extend(
        [
            "",
            "## Validation diagnostic",
            "",
            "| Model | Trades | Mean | Median | Common LCB | DD | Status |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in validation.itertuples(index=False):
        dd = "n/a" if pd.isna(row.max_drawdown) else f"{100.0 * row.max_drawdown:.2f}%"
        lines.append(
            f"| {row.model_id} | {row.trades} | {_fmt(row.mean_bps)} | "
            f"{_fmt(row.median_bps)} | {_fmt(row.lcb_bps)} | {dd} | {row.protocol_status} |"
        )
    lines.extend(
        [
            "",
            "## Вывод",
            "",
            f"- На development V3 сильнее лучшего стандартного правила `{best_standard.model_id}`: "
            f"LCB {_fmt(best_primus.lcb_bps)} bps у лучшего Primus "
            f"против {_fmt(best_standard.lcb_bps)} bps у standard baseline.",
            f"- Но V3 не переносится на validation: mean {_fmt(v3_validation.mean_bps)}, "
            f"median {_fmt(v3_validation.median_bps)}, LCB {_fmt(v3_validation.lcb_bps)} bps, "
            f"drawdown {100.0 * v3_validation.max_drawdown:.2f}%.",
            "- Следовательно, рост V1 -> V3 показывает улучшение оптимизации development, а не подтверждённое улучшение обобщения.",
            "- Ни одна строка этого отчёта не открывает OOS, paper или live.",
            "",
            "`player_one` и `player_two` находятся в XLSX как справочные результаты: их routing contract отличается, поэтому они не входят в основной rank.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def _normalize_comparison_ledgers(ledgers: list[pd.DataFrame]) -> pd.DataFrame:
    columns = [
        "model_id",
        "split",
        "symbol",
        "direction",
        "market_state",
        "signal_timestamp",
        "entry_timestamp",
        "exit_timestamp",
        "entry_price",
        "exit_price",
        "holding_bars",
        "gross_bps",
        "cost_bps",
        "net_bps",
        "fills",
        "exit_reason",
    ]
    normalized: list[pd.DataFrame] = []
    for ledger in ledgers:
        local = ledger.copy()
        for column in columns:
            if column not in local:
                local[column] = None
        local["direction"] = local["direction"].replace({1: "LONG", -1: "SHORT"}).astype("string")
        local["model_id"] = local["model_id"].astype("string")
        local["split"] = local["split"].astype("string")
        local["symbol"] = local["symbol"].astype("string")
        local["market_state"] = local["market_state"].astype("string")
        local["exit_reason"] = local["exit_reason"].astype("string")
        normalized.append(local.loc[:, columns])
    return pd.concat(normalized, ignore_index=True)


def run(*, family_path: Path, primus_config_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    primus_rows, primus_ledgers = _load_primus_development_rows()
    v3_validation_row, v3_validation_ledger = _evaluate_v3_validation(primus_config_path)
    standard_rows, standard_ledgers, identity = _evaluate_standard_models(family_path)
    legacy_rows, legacy_ledgers = _load_legacy_rows()
    leaderboard = _rank(
        pd.DataFrame(primus_rows + [v3_validation_row] + standard_rows + legacy_rows)
    )
    all_ledgers = _normalize_comparison_ledgers(
        primus_ledgers + [v3_validation_ledger] + standard_ledgers + legacy_ledgers
    )
    methodology = pd.DataFrame(
        [
            {"field": "dataset", "value": identity["dataset_sha256"]},
            {"field": "timeframe_minutes", "value": 240},
            {"field": "symbols", "value": 8},
            {"field": "round_trip_cost_bps", "value": STRESS_COST_BPS},
            {"field": "bootstrap_method", "value": "moving_block_lcb"},
            {"field": "bootstrap_samples", "value": BOOTSTRAP_SAMPLES},
            {"field": "bootstrap_alpha", "value": BOOTSTRAP_ALPHA},
            {"field": "development", "value": "2022-01-01..2024-01-01"},
            {"field": "validation", "value": "2024-01-01..2025-01-01"},
            {"field": "oos_opened", "value": False},
            {"field": "sanity_opened", "value": False},
            {"field": "paper_allowed", "value": False},
            {"field": "live_allowed", "value": False},
            {"field": "orders_enabled", "value": False},
            {"field": "promotion_authority", "value": False},
        ]
    )
    leaderboard_path = output_dir / "leaderboard.parquet"
    ledger_path = output_dir / "comparison_trades.parquet"
    workbook_path = output_dir / "model_comparison.xlsx"
    development_plot = output_dir / "development_comparison.png"
    split_plot = output_dir / "split_lcb_heatmap.png"
    report_path = output_dir / "report.md"
    leaderboard.to_parquet(leaderboard_path, index=False)
    all_ledgers.to_parquet(ledger_path, index=False)
    write_workbook(workbook_path, leaderboard=leaderboard, methodology=methodology)
    plot_development(leaderboard, development_plot)
    plot_split_heatmap(leaderboard, split_plot)
    report_path.write_text(_render_report(leaderboard, identity), encoding="utf-8")

    artifacts = {
        "leaderboard": _artifact(leaderboard_path),
        "trades": _artifact(ledger_path),
        "workbook": _artifact(workbook_path),
        "development_plot": _artifact(development_plot),
        "split_plot": _artifact(split_plot),
        "report": _artifact(report_path),
    }
    result = {
        "schema_version": "exia.genetic_primus_model_comparison.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        **identity,
        "primus_config": str(primus_config_path.relative_to(ROOT)),
        "primus_config_sha256": sha256_file(primus_config_path),
        "stress_round_trip_cost_bps": STRESS_COST_BPS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_alpha": BOOTSTRAP_ALPHA,
        "models": int(leaderboard["model_id"].nunique()),
        "rows": int(len(leaderboard)),
        "oos_opened": False,
        "sanity_opened": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result["manifest"] = _artifact(manifest_path)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Genetic_Primus V1-V3 with standard 4h models"
    )
    parser.add_argument("--family", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument("--primus-config", type=Path, default=DEFAULT_PRIMUS_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(
        family_path=args.family.resolve(),
        primus_config_path=args.primus_config.resolve(),
        output_dir=args.output.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
