from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GENETICS_DIR = ROOT / "Genetics_DL_Agents"
if str(GENETICS_DIR) not in sys.path:
    sys.path.insert(0, str(GENETICS_DIR))

import crypto_genetics as cg  # noqa: E402


def _open_output_bias_indices() -> np.ndarray:
    b4_start = cg.GENOME_SIZE - cg.N_ACTIONS
    actions = [code for code in cg._OPEN_ACTION_CODES if 0 <= code < cg.N_ACTIONS]
    return np.asarray([b4_start + code for code in actions], dtype=np.int64)


def _seed_feasible_best_population(
    population: np.ndarray,
    best_genome: np.ndarray,
    rng: np.random.Generator,
    *,
    start_slot: int,
    seed_frac: float,
    sigma: float,
    open_logit_bias: float,
) -> int:
    if population.ndim != 2 or population.shape[1] != cg.GENOME_SIZE:
        raise ValueError(f"population must have shape (N, {cg.GENOME_SIZE})")
    best = np.asarray(best_genome, dtype=np.float32).ravel()
    if best.shape[0] != cg.GENOME_SIZE:
        raise ValueError(f"best genome size {best.shape[0]} != {cg.GENOME_SIZE}")

    slot0 = max(0, int(start_slot))
    available = max(0, population.shape[0] - slot0)
    n_seed = min(available, max(0, int(round(population.shape[0] * float(seed_frac)))))
    if n_seed <= 0:
        return 0

    seeded = np.broadcast_to(best, (n_seed, cg.GENOME_SIZE)).astype(np.float32, copy=True)
    if sigma > 0:
        seeded += rng.normal(0.0, float(sigma), size=seeded.shape).astype(np.float32)

    bias_idx = _open_output_bias_indices()
    if bias_idx.size and open_logit_bias:
        seeded[:, bias_idx] -= np.float32(open_logit_bias)

    population[slot0:slot0 + n_seed] = np.clip(seeded, -5.0, 5.0)
    return n_seed


def _force_torch_cpu_evaluator() -> None:
    try:
        import torch
    except Exception:
        return

    def _detect_cpu_torch():
        print("  Smoke mode: PyTorch CPU evaluator")
        return "torch", torch.device("cpu")

    cg._detect_gpu = _detect_cpu_torch


def _force_cpu_numba_evaluator() -> None:
    def _detect_cpu_numba():
        print("  Smoke mode: CPU evaluator with Numba simulator")
        return None, None

    cg._detect_gpu = _detect_cpu_numba


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small isolated genetics training smoke test.")
    parser.add_argument("--population", type=int, default=18)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument(
        "--evaluator",
        choices=("cpu-numba", "torch-cpu"),
        default="cpu-numba",
        help="Evaluator backend for isolated training.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override genetics CPU worker count. Use 1 to debug worker exceptions inline.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "Results" / "neiro_genetics" / f"genetics_smoke_train_{time.strftime('%Y%m%d_%H%M%S')}"),
    )
    parser.add_argument(
        "--feasible-best-seed-frac",
        type=float,
        default=0.0,
        help="Fraction of population to seed as best-genome micro-mutants with damped open-action logits.",
    )
    parser.add_argument(
        "--feasible-best-seed-sigma",
        type=float,
        default=0.03,
        help="Gaussian sigma for feasible best-genome micro-mutants.",
    )
    parser.add_argument(
        "--feasible-open-logit-bias",
        type=float,
        default=1.5,
        help="Amount subtracted from final-layer open-action biases in feasible warm seeds.",
    )
    args = parser.parse_args()

    source_agents_dir = Path(cg.AGENTS_DIR)
    best_path = source_agents_dir / "best_genome.npy"
    if not best_path.exists():
        raise FileNotFoundError(best_path)
    best_genome = np.load(best_path).astype(np.float32).ravel()
    if best_genome.shape[0] != cg.GENOME_SIZE:
        raise ValueError(f"best genome size {best_genome.shape[0]} != {cg.GENOME_SIZE}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    old_cfg_start = cg._cx._CFG.get("start_date")
    old_cfg_end = cg._cx._CFG.get("end_date")
    old_agents_dir = cg.AGENTS_DIR

    try:
        cg._cx._CFG["start_date"] = args.start_date
        cg._cx._CFG["end_date"] = args.end_date

        cg.POP_SIZE = int(args.population)
        cg.N_GENERATIONS = int(args.generations)
        cg.ELITE_SIZE = min(2, max(1, cg.POP_SIZE // 3))
        cg.ARCHIVE_SIZE = min(3, cg.POP_SIZE)
        cg.N_ISLANDS = 3
        cg.PHASE1_ENABLED = False
        cg.PHASE2_ENABLED = False
        cg.BC_ENABLED = False
        if args.workers is not None:
            cg.N_WORKERS = int(args.workers)
        elif args.evaluator == "cpu-numba":
            cg.N_WORKERS = 1
        cg.CONTINUE_TRAINING = False
        cg.LOAD_BEST_GENOME = False
        cg.LOAD_EXTRA_GENOMES = False
        cg.LOAD_REGIME_GENOMES = False
        cg.LOAD_ISLAND_GENOMES = False
        cg.SAVE_EVERY = 1
        cg.CHART_EVERY = max(1, cg.N_GENERATIONS)
        cg.TRAIN_EXECUTION_LAG_BARS = 1

        if args.evaluator == "torch-cpu":
            _force_torch_cpu_evaluator()
        else:
            _force_cpu_numba_evaluator()

        cg.AGENTS_DIR = str(out_dir)
        os.makedirs(cg.AGENTS_DIR, exist_ok=True)

        print(f"[smoke] loading data {args.start_date} -> {args.end_date}")
        precomp = cg._load_precomp()
        if not precomp:
            raise RuntimeError("No periods loaded")

        if args.evaluator == "cpu-numba" and getattr(cg, "_NUMBA_OK", False):
            print("[smoke] warming up Numba simulator")
            cg._warm_up_numba()

        trainer = cg.GeneticTrainer(precomp)
        trainer.pop[0] = best_genome
        trainer.best_g = best_genome.copy()
        trainer.best_fit = -np.inf
        feasible_seeded = _seed_feasible_best_population(
            trainer.pop,
            best_genome,
            trainer.rng,
            start_slot=1,
            seed_frac=args.feasible_best_seed_frac,
            sigma=args.feasible_best_seed_sigma,
            open_logit_bias=args.feasible_open_logit_bias,
        )
        if feasible_seeded:
            print(
                "[smoke] feasible-best seeds="
                f"{feasible_seeded} sigma={args.feasible_best_seed_sigma:.4f} "
                f"open_bias={args.feasible_open_logit_bias:.3f}"
            )

        print(
            f"[smoke] training pop={cg.POP_SIZE} gens={cg.N_GENERATIONS} "
            f"periods={len(precomp)} out={out_dir}"
        )
        t0 = time.time()
        trainer.run()
        elapsed = time.time() - t0

        summary = {
            "out_dir": str(out_dir),
            "source_best_genome": str(best_path),
            "population": cg.POP_SIZE,
            "generations": cg.N_GENERATIONS,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "periods": len(precomp),
            "execution_lag_bars": cg.TRAIN_EXECUTION_LAG_BARS,
            "feasible_best_seeded": feasible_seeded,
            "feasible_best_seed_frac": args.feasible_best_seed_frac,
            "feasible_best_seed_sigma": args.feasible_best_seed_sigma,
            "feasible_open_logit_bias": args.feasible_open_logit_bias,
            "elapsed_sec": elapsed,
            "best_fit": float(trainer.best_fit),
        }
        (out_dir / "smoke_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[smoke] summary {out_dir / 'smoke_summary.json'}")
    finally:
        cg.AGENTS_DIR = old_agents_dir
        if old_cfg_start is not None:
            cg._cx._CFG["start_date"] = old_cfg_start
        if old_cfg_end is not None:
            cg._cx._CFG["end_date"] = old_cfg_end

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
