from __future__ import annotations

import argparse
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

from exia.genetic_primus.contracts import FEATURE_NAMES, STATIC_FEATURE_NAMES  # noqa: E402
from exia.genetic_primus.evolution_v4 import run_evolution_v4  # noqa: E402
from exia.genetic_primus.features import RobustScaler  # noqa: E402
from exia.genetic_primus.measurement import build_bootstrap_bank  # noqa: E402
from exia.genetic_primus.policy_v4 import V4NetworkShape  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toy_frame() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-01", "2025-01-21", freq="4h", inclusive="left", tz="UTC"
    )
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(("BTC/USDT", "LINK/USDT")):
        for index, timestamp in enumerate(timestamps):
            price = 100.0 + symbol_index * 10.0 + index * 0.05
            row: dict[str, object] = {
                "timestamp": int(timestamp.timestamp() * 1000),
                "symbol": symbol,
                "open": price,
                "high": price + 0.10,
                "low": price - 0.10,
                "close": price + 0.02,
                "market_state": "quiet_range",
                "feature_supported": True,
            }
            for feature_index, name in enumerate(STATIC_FEATURE_NAMES):
                row[name] = ((index % 11) - 5) * (feature_index + 1) * 1.0e-4
            rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["timestamp", "symbol"], kind="stable")
        .reset_index(drop=True)
    )


def run(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"checkpoint smoke output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = _toy_frame()
    start = int(frame["timestamp"].min())
    end = int(frame["timestamp"].max()) + 4 * 60 * 60 * 1000
    fold = {
        "name": "toy_full",
        "start": "2025-01-01T00:00:00Z",
        "end": "2025-01-21T00:00:00Z",
    }
    fitness = dict(json.loads((ROOT / "configs/genetic_primus_v4.json").read_text(encoding="utf-8"))["fitness"])
    bank = build_bootstrap_bank(
        frame,
        start_timestamp=start,
        cutoff_timestamp=end,
        folds=(fold,),
        bank_id="checkpoint-resume-smoke",
        seed=91,
        samples=int(fitness["bootstrap_samples"]),
        block_days=int(fitness["bootstrap_block_days"]),
    )
    shape = V4NetworkShape(inputs=len(FEATURE_NAMES), hidden=(), outputs=3)
    scaler = RobustScaler(
        median=np.zeros(len(STATIC_FEATURE_NAMES), dtype=np.float64),
        scale=np.ones(len(STATIC_FEATURE_NAMES), dtype=np.float64),
    )
    evolution = {
        "seed": 17,
        "workers": 1,
        "population_size": 2,
        "generations": 1,
        "elite_count": 1,
        "tournament_size": 2,
        "crossover_rate": 0.5,
        "mutation_rate": 0.1,
        "mutation_sigma": 0.05,
    }
    checkpoint = output_dir / "checkpoint.npz"
    fingerprint = "checkpoint-resume-smoke-v1"
    common = {
        "frame": frame,
        "scaler": scaler,
        "shape": shape,
        "start_timestamp": start,
        "end_timestamp": end,
        "round_trip_cost_bps": 16.0,
        "folds": (fold,),
        "fitness_config": fitness,
        "bootstrap_bank": bank,
        "evolution_config": evolution,
        "warm_start": np.zeros(shape.genome_size, dtype=np.float32),
        "checkpoint_path": checkpoint,
        "run_fingerprint": fingerprint,
        "finalist_count": 1,
    }
    first = run_evolution_v4(**common, resume=False)
    first_sha = _sha256(checkpoint)
    second = run_evolution_v4(**common, resume=True)
    second_sha = _sha256(checkpoint)
    mismatch_rejected = False
    try:
        run_evolution_v4(**{**common, "run_fingerprint": "wrong-fingerprint"}, resume=True)
    except ValueError as error:
        mismatch_rejected = "fingerprint mismatch" in str(error)
    final_sha = _sha256(checkpoint)
    with np.load(checkpoint, allow_pickle=True) as payload:
        checkpoint_complete = bool(payload["complete"].item())
        next_generation = int(payload["next_generation"].item())
    result = {
        "schema_version": "exia.genetic_primus.checkpoint_resume_smoke/1",
        "status": "OFFLINE_CHECKPOINT_RESUME_OK",
        "checkpoint_complete": checkpoint_complete,
        "next_generation": next_generation,
        "sha_unchanged_after_resume": first_sha == second_sha == final_sha,
        "champion_unchanged": bool(
            np.array_equal(first.champion.genome, second.champion.genome)
            and first.champion.score.fitness == second.champion.score.fitness
        ),
        "fingerprint_mismatch_rejected": mismatch_rejected,
        "checkpoint_sha256": final_sha,
        "credentials_allowed": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    if not all(
        (
            result["checkpoint_complete"],
            result["sha_unchanged_after_resume"],
            result["champion_unchanged"],
            result["fingerprint_mismatch_rejected"],
        )
    ):
        raise AssertionError(f"checkpoint resume smoke failed: {result}")
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny offline V4 checkpoint-resume smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
