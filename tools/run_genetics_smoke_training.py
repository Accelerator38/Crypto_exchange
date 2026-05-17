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
