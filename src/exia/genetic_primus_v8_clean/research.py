from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


EVENT_ID = "isolated_symbol_cascade_reversal"
HOLDOUT_SYMBOL = "LINK/USDT"
FEATURE_NAMES = (
    "log_liquidation_total",
    "log_cross_liquidation_total",
    "absolute_symbol_return",
    "absolute_market_return",
    "available_symbol_count",
    "direction_value",
)
SAFETY = {
    "credentials_allowed": False,
    "orders_enabled": False,
    "paper_trading_enabled": False,
    "live_trading_enabled": False,
    "promotion_authority": False,
}


@dataclass(frozen=True)
class ShadowTrainingResult:
    genome: np.ndarray
    median: np.ndarray
    scale: np.ndarray
    history: pd.DataFrame
    train_rows: int
    holdout_rows: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _assert_fail_closed(v8_contract: Mapping[str, Any], v7_manifest: Mapping[str, Any]) -> None:
    safety = v8_contract.get("safety")
    if not isinstance(safety, Mapping) or any(safety.get(key) is not False for key in SAFETY):
        raise PermissionError("V8 Clean research requires all safety authorities to remain false")
    if v8_contract.get("source_hypothesis", {}).get("parameter_tuning_allowed") is not False:
        raise PermissionError("V8 Clean frozen thresholds may not be tuned")
    if v7_manifest.get("status") != "V7_TERMINAL_REJECTED_NO_EVENT_CAPACITY":
        raise PermissionError("shadow research is pinned to the rejected V7 source experiment")
    if v7_manifest.get("evolution_allowed") is not False:
        raise PermissionError("unexpected V7 evolution authority")


def load_symbol_holdout_ledger(events_path: Path, v7_contract_path: Path) -> pd.DataFrame:
    required = {
        "event_id",
        "symbol",
        "direction_value",
        "timestamp",
        "exit_timestamp",
        "entry_price",
        "exit_price",
        "net_bps",
        "return_1",
        "market_return_1",
        "liquidation_total_usd",
        "cross_liquidation_total_usd",
        "available_symbol_count",
    }
    frame = pd.read_parquet(events_path)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"V7 event ledger missing columns: {missing}")
    frame = frame.loc[frame["event_id"].eq(EVENT_ID)].copy()
    contract = _read_json(v7_contract_path)
    fold_masks: list[pd.Series] = []
    fold_names: list[str] = []
    for fold in contract.get("walk_forward", []):
        start = int(pd.Timestamp(fold["evaluation_start"]).timestamp() * 1000)
        end = int(pd.Timestamp(fold["evaluation_end"]).timestamp() * 1000)
        mask = (
            frame["timestamp"].ge(start)
            & frame["timestamp"].lt(end)
            & frame["exit_timestamp"].lt(end)
        )
        fold_masks.append(mask)
        fold_names.append(str(fold["name"]))
    if not fold_masks:
        raise ValueError("V7 contract has no walk-forward folds")
    accepted = fold_masks[0].copy()
    for mask in fold_masks[1:]:
        accepted |= mask
    frame = frame.loc[accepted].copy()
    frame["fold"] = ""
    for name, mask in zip(fold_names, fold_masks, strict=True):
        frame.loc[mask.reindex(frame.index, fill_value=False), "fold"] = name
    if frame["fold"].eq("").any():
        raise ValueError("accepted V7 event without a fold")
    return frame.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)


def _raw_features(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.log1p(frame["liquidation_total_usd"].clip(lower=0.0).to_numpy(float)),
            np.log1p(frame["cross_liquidation_total_usd"].clip(lower=0.0).to_numpy(float)),
            frame["return_1"].abs().to_numpy(float),
            frame["market_return_1"].abs().to_numpy(float),
            frame["available_symbol_count"].to_numpy(float),
            frame["direction_value"].to_numpy(float),
        ]
    )


def _normal_lcb(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean
    return mean - 1.645 * float(np.std(values, ddof=1)) / math.sqrt(len(values))


def _fitness(genome: np.ndarray, x: np.ndarray, values: np.ndarray, folds: np.ndarray) -> float:
    selected = x @ genome[:-1] + genome[-1] >= 0.0
    selection_rate = float(np.mean(selected))
    if selection_rate < 0.15 or selection_rate > 0.80:
        return -20_000.0 - abs(selection_rate - 0.45) * 1_000.0
    fold_means: list[float] = []
    for fold in np.unique(folds):
        local = values[selected & (folds == fold)]
        if len(local) < 15:
            return -10_000.0 + float(len(local))
        fold_means.append(float(np.mean(local)))
    lcb = _normal_lcb(values[selected])
    return min(fold_means) + 0.5 * float(lcb) - abs(selection_rate - 0.45) * 2.0


def train_symbol_holdout_shadow(
    ledger: pd.DataFrame,
    *,
    seed: int = 80824,
    population_size: int = 72,
    generations: int = 48,
) -> ShadowTrainingResult:
    if population_size < 16 or generations < 2:
        raise ValueError("shadow genetic search is intentionally bounded but non-trivial")
    train = ledger.loc[ledger["symbol"].ne(HOLDOUT_SYMBOL)].copy()
    holdout = ledger.loc[ledger["symbol"].eq(HOLDOUT_SYMBOL)].copy()
    if len(holdout) < 50:
        raise ValueError("historical LINK holdout is unexpectedly small")
    raw = _raw_features(train)
    median = np.median(raw, axis=0)
    scale = np.median(np.abs(raw - median), axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    x = (raw - median) / scale
    values = train["net_bps"].to_numpy(float)
    folds = train["fold"].to_numpy(str)
    rng = np.random.default_rng(seed)
    population = rng.normal(0.0, 0.8, size=(population_size, x.shape[1] + 1))
    population[0] = 0.0
    population[0, -1] = 1.0
    history: list[dict[str, Any]] = []
    best_genome = population[0].copy()
    best_fitness = -math.inf
    elite_count = max(4, population_size // 8)
    for generation in range(generations):
        scores = np.asarray([_fitness(genome, x, values, folds) for genome in population])
        order = np.argsort(scores)[::-1]
        if float(scores[order[0]]) > best_fitness:
            best_fitness = float(scores[order[0]])
            best_genome = population[order[0]].copy()
        selected = x @ best_genome[:-1] + best_genome[-1] >= 0.0
        history.append(
            {
                "generation": generation,
                "best_fitness": best_fitness,
                "population_median_fitness": float(np.median(scores)),
                "selected_train_events": int(selected.sum()),
            }
        )
        elites = population[order[:elite_count]].copy()
        next_population = [elites[0].copy()]
        sigma = 0.35 * (1.0 - generation / generations) + 0.05
        while len(next_population) < population_size:
            left, right = rng.integers(0, elite_count, size=2)
            mix = rng.uniform(0.0, 1.0, size=best_genome.shape)
            child = mix * elites[left] + (1.0 - mix) * elites[right]
            child += rng.normal(0.0, sigma, size=child.shape)
            next_population.append(child)
        population = np.asarray(next_population)
    return ShadowTrainingResult(
        genome=best_genome,
        median=median,
        scale=scale,
        history=pd.DataFrame(history),
        train_rows=len(train),
        holdout_rows=len(holdout),
    )


def _summary(agent_id: str, values: np.ndarray, total_events: int) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "events": int(len(values)),
        "selection_rate": float(len(values) / total_events),
        "mean_net_bps": float(np.mean(values)) if len(values) else None,
        "median_net_bps": float(np.median(values)) if len(values) else None,
        "normal_95_lcb_bps": _normal_lcb(values),
        "positive_rate": float(np.mean(values > 0.0)) if len(values) else None,
        "total_net_bps": float(np.sum(values)),
    }


def _metric_text(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.2f}{suffix}"


def compare_on_link_holdout(
    ledger: pd.DataFrame,
    result: ShadowTrainingResult,
    *,
    cost_bps: float = 24.0,
) -> pd.DataFrame:
    holdout = ledger.loc[ledger["symbol"].eq(HOLDOUT_SYMBOL)].copy()
    raw = _raw_features(holdout)
    x = (raw - result.median) / result.scale
    selected = x @ result.genome[:-1] + result.genome[-1] >= 0.0
    price_return_bps = (holdout["exit_price"].to_numpy(float) / holdout["entry_price"].to_numpy(float) - 1.0) * 10_000.0
    event_direction = holdout["direction_value"].to_numpy(float)
    momentum_direction = np.sign(holdout["return_1"].to_numpy(float))
    frozen = holdout["net_bps"].to_numpy(float)
    values = {
        "V8_FROZEN_LINK_REVERSAL": frozen,
        "V8_SHADOW_GENETIC_GATE": frozen[selected],
        "ORDINARY_ALWAYS_LONG": price_return_bps - cost_bps,
        "ORDINARY_ALWAYS_SHORT": -price_return_bps - cost_bps,
        "ORDINARY_4H_MOMENTUM": momentum_direction * price_return_bps - cost_bps,
        "ORDINARY_4H_MEAN_REVERSION": -momentum_direction * price_return_bps - cost_bps,
    }
    rows = [_summary(agent_id, net, len(holdout)) for agent_id, net in values.items()]
    return pd.DataFrame(rows).sort_values("normal_95_lcb_bps", ascending=False, na_position="last")


def run_retrospective_shadow(
    *,
    root: Path,
    output_dir: Path,
    retrospective_acknowledged: bool,
) -> dict[str, Any]:
    if retrospective_acknowledged is not True:
        raise PermissionError("explicit retrospective acknowledgement is required")
    root = root.resolve()
    output_dir = output_dir.resolve()
    v8_contract_path = root / "configs/genetic_primus_v8_clean.json"
    v7_contract_path = root / "configs/genetic_primus_v7_contract.json"
    v7_manifest_path = root / "Reports/Exia/Genetic_Primus/v7_event_feasibility_20260824/manifest.json"
    events_path = root / "Reports/Exia/Genetic_Primus/v7_event_feasibility_20260824/events.parquet"
    v8_contract = _read_json(v8_contract_path)
    v7_manifest = _read_json(v7_manifest_path)
    _assert_fail_closed(v8_contract, v7_manifest)
    ledger = load_symbol_holdout_ledger(events_path, v7_contract_path)
    result = train_symbol_holdout_shadow(ledger)
    comparison = compare_on_link_holdout(ledger, result)
    indexed = comparison.set_index("agent_id")
    frozen = indexed.loc["V8_FROZEN_LINK_REVERSAL"]
    shadow = indexed.loc["V8_SHADOW_GENETIC_GATE"]
    ordinary = indexed.loc["ORDINARY_4H_MEAN_REVERSION"]
    diagnostics = {
        "shadow_lcb_uplift_bps": float(
            shadow["normal_95_lcb_bps"] - frozen["normal_95_lcb_bps"]
        ),
        "shadow_mean_uplift_bps": float(shadow["mean_net_bps"] - frozen["mean_net_bps"]),
        "frozen_equals_ordinary_mean_reversion": bool(
            abs(float(frozen["mean_net_bps"] - ordinary["mean_net_bps"])) < 1e-12
            and abs(float(frozen["normal_95_lcb_bps"] - ordinary["normal_95_lcb_bps"])) < 1e-12
        ),
    }
    diagnostics["research_conclusion"] = (
        "SHADOW_PASSES_RETROSPECTIVE_ROBUSTNESS_UPLIFT"
        if diagnostics["shadow_lcb_uplift_bps"] > 0.0
        else "SHADOW_REJECTED_NO_ROBUSTNESS_UPLIFT"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    result.history.to_parquet(output_dir / "training_history.parquet", index=False)
    comparison.to_parquet(output_dir / "link_holdout_comparison.parquet", index=False)
    model = {
        "schema": "genetic_primus.v8.clean.retrospective_shadow_model/1",
        "status": "ENGINEERING_ONLY_RETROSPECTIVE_SYMBOL_HOLDOUT",
        "event_id": EVENT_ID,
        "training_symbols": sorted(set(ledger["symbol"]) - {HOLDOUT_SYMBOL}),
        "holdout_symbol": HOLDOUT_SYMBOL,
        "feature_names": list(FEATURE_NAMES),
        "genome": result.genome.tolist(),
        "median": result.median.tolist(),
        "scale": result.scale.tolist(),
        "train_rows": result.train_rows,
        "holdout_rows": result.holdout_rows,
        "source_sha256": {
            "v8_contract": _sha256(v8_contract_path),
            "v7_contract": _sha256(v7_contract_path),
            "v7_manifest": _sha256(v7_manifest_path),
            "v7_events": _sha256(events_path),
        },
        "not_prospective_evidence": True,
        **SAFETY,
    }
    (output_dir / "shadow_model.json").write_text(
        json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "genetic_primus.v8.clean.retrospective_shadow_run/1",
        "status": model["status"],
        "model_sha256": _sha256(output_dir / "shadow_model.json"),
        "comparison": comparison.to_dict(orient="records"),
        "diagnostics": diagnostics,
        "warning": "Historical V7 symbol holdout is not independent prospective V8 evidence.",
        **SAFETY,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Genetic Primus V8 Clean retrospective shadow",
        "",
        "Status: `ENGINEERING_ONLY_RETROSPECTIVE_SYMBOL_HOLDOUT`.",
        "",
        "LINK is excluded from training, but the historical V7 window is not prospective V8 evidence.",
        "",
        "| Agent | Events | Selection | Mean net bps | Median | Normal 95% LCB | Positive |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.agent_id} | {row.events} | {row.selection_rate:.1%} | "
            f"{_metric_text(row.mean_net_bps)} | {_metric_text(row.median_net_bps)} | "
            f"{_metric_text(row.normal_95_lcb_bps)} | "
            f"{_metric_text(None if row.positive_rate is None else row.positive_rate * 100.0, '%')} |"
        )
    lines.extend(
        [
            "",
            f"Research conclusion: `{diagnostics['research_conclusion']}`.",
            f"Shadow LCB uplift versus frozen V8: `{diagnostics['shadow_lcb_uplift_bps']:.2f}` bps.",
            f"Frozen V8 equals ordinary 4H mean reversion on this event set: "
            f"`{diagnostics['frozen_equals_ordinary_mean_reversion']}`.",
            "",
            "No model in this report has paper, live, order, credential, or promotion authority.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest
