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


def _genome_size_for_arch(arch: tuple[int, int, int, int, int]) -> int:
    n_in, h1, h2, h3, n_actions = [int(v) for v in arch]
    return (
        n_in * h1 + h1
        + h1 * h2 + h2
        + h2 * h3 + h3
        + h3 * n_actions + n_actions
    )


def _unpack_mlp_genome(genome: np.ndarray, arch: tuple[int, int, int, int, int]):
    n_in, h1, h2, h3, n_actions = [int(v) for v in arch]
    i = 0
    w1 = genome[i:i + n_in * h1].reshape(n_in, h1); i += n_in * h1
    b1 = genome[i:i + h1]; i += h1
    w2 = genome[i:i + h1 * h2].reshape(h1, h2); i += h1 * h2
    b2 = genome[i:i + h2]; i += h2
    w3 = genome[i:i + h2 * h3].reshape(h2, h3); i += h2 * h3
    b3 = genome[i:i + h3]; i += h3
    w4 = genome[i:i + h3 * n_actions].reshape(h3, n_actions); i += h3 * n_actions
    b4 = genome[i:i + n_actions]
    return w1, b1, w2, b2, w3, b3, w4, b4


def _pack_mlp_genome(layers: tuple[np.ndarray, ...]) -> np.ndarray:
    return np.concatenate([np.asarray(layer, dtype=np.float32).ravel() for layer in layers]).astype(np.float32)


def _transfer_mlp_genome(
    source: np.ndarray,
    source_arch: tuple[int, int, int, int, int],
    target_arch: tuple[int, int, int, int, int],
) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32).ravel()
    if source.shape[0] != _genome_size_for_arch(source_arch):
        raise ValueError(
            f"source genome size {source.shape[0]} does not match source_arch "
            f"{source_arch}: {_genome_size_for_arch(source_arch)}"
        )
    if target_arch[0] < source_arch[0]:
        raise ValueError("target input width must be >= source input width")
    if target_arch[4] != source_arch[4]:
        raise ValueError("target action count must equal source action count")
    if any(target_arch[i] < source_arch[i] for i in (1, 2, 3)):
        raise ValueError("target hidden widths must be >= source hidden widths")

    sw1, sb1, sw2, sb2, sw3, sb3, sw4, sb4 = _unpack_mlp_genome(source, source_arch)
    n_in, h1, h2, h3, n_actions = target_arch
    w1 = np.zeros((n_in, h1), dtype=np.float32)
    b1 = np.zeros(h1, dtype=np.float32)
    w2 = np.zeros((h1, h2), dtype=np.float32)
    b2 = np.zeros(h2, dtype=np.float32)
    w3 = np.zeros((h2, h3), dtype=np.float32)
    b3 = np.zeros(h3, dtype=np.float32)
    w4 = np.zeros((h3, n_actions), dtype=np.float32)
    b4 = np.zeros(n_actions, dtype=np.float32)

    old_n_in, old_h1, old_h2, old_h3, _ = source_arch
    w1[:old_n_in, :old_h1] = sw1
    b1[:old_h1] = sb1
    w2[:old_h1, :old_h2] = sw2
    b2[:old_h2] = sb2
    w3[:old_h2, :old_h3] = sw3
    b3[:old_h3] = sb3
    w4[:old_h3, :] = sw4
    b4[:] = sb4
    return _pack_mlp_genome((w1, b1, w2, b2, w3, b3, w4, b4))


def _current_arch() -> tuple[int, int, int, int, int]:
    return (cg.N_INPUT, cg.N_HIDDEN1, cg.N_HIDDEN2, cg.N_HIDDEN3, cg.N_ACTIONS)


def _source_arch_from_reports(source_agents_dir: Path) -> tuple[int, int, int, int, int]:
    report_path = source_agents_dir / "training_report.json"
    if report_path.exists():
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            arch = payload.get("arch", {})
            return (
                int(arch.get("in", 28)),
                int(arch.get("h1", 128)),
                int(arch.get("h2", 64)),
                int(arch.get("h3", 32)),
                int(arch.get("out", 9)),
            )
        except Exception:
            pass
    return (28, 128, 64, 32, 9)


def _write_warm_start_artifacts(
    out_dir: Path,
    genome: np.ndarray,
    metadata: dict,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(genome, dtype=np.float32).ravel()
    genome_path = out_dir / "warm_start_genome.npy"
    meta_path = out_dir / "warm_start_genome_meta.json"
    np.save(genome_path, arr)
    payload = {
        "role": "warm_start_baseline",
        "genome_size": int(arr.shape[0]),
        **dict(metadata),
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"genome": genome_path, "meta": meta_path}


def _open_output_bias_indices() -> np.ndarray:
    b4_start = cg.GENOME_SIZE - cg.N_ACTIONS
    actions = [code for code in cg._OPEN_ACTION_CODES if 0 <= code < cg.N_ACTIONS]
    return np.asarray([b4_start + code for code in actions], dtype=np.int64)


def _output_layer_start_index() -> int:
    return int(cg.GENOME_SIZE - (cg.N_HIDDEN3 * cg.N_ACTIONS + cg.N_ACTIONS))


def _seed_feasible_best_population(
    population: np.ndarray,
    best_genome: np.ndarray,
    rng: np.random.Generator,
    *,
    start_slot: int,
    seed_frac: float,
    sigma: float,
    open_logit_bias: float,
    mutation_scope: str = "all",
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
        if mutation_scope == "all":
            seeded += rng.normal(0.0, float(sigma), size=seeded.shape).astype(np.float32)
        elif mutation_scope == "output":
            out_start = _output_layer_start_index()
            seeded[:, out_start:] += rng.normal(
                0.0,
                float(sigma),
                size=seeded[:, out_start:].shape,
            ).astype(np.float32)
        else:
            raise ValueError(f"Unsupported feasible seed mutation scope: {mutation_scope}")

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


def _resolve_position_state_features_enabled(
    mode: str,
    source_agents_dir: Path,
    *,
    settings_default: bool,
) -> bool:
    if mode == "settings":
        return bool(settings_default)
    if mode == "on":
        return True
    if mode == "off":
        return False
    if mode != "source-meta":
        raise ValueError(f"Unsupported position-state feature mode: {mode}")

    meta_path = source_agents_dir / "best_genome_meta.json"
    if not meta_path.exists():
        return bool(settings_default)
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if "position_state_features_enabled" in payload:
            return bool(payload["position_state_features_enabled"])
    except Exception:
        return bool(settings_default)
    return bool(settings_default)


def _filter_precomp_by_regime(precomp: list, regime_filter: str) -> list:
    regime = str(regime_filter or "all").lower()
    if regime == "all":
        return list(precomp)
    if regime not in {"bearish", "neutral", "bullish"}:
        raise ValueError(f"Unsupported regime filter: {regime_filter}")
    return [
        entry for entry in precomp
        if len(entry) > 6 and cg.map_regime_3(entry[6]) == regime
    ]


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
    parser.add_argument(
        "--feasible-mutation-scope",
        choices=("all", "output"),
        default="all",
        help="Scope for warm-start micro-mutants. output mutates only the final action layer.",
    )
    parser.add_argument(
        "--position-state-features",
        choices=("source-meta", "settings", "on", "off"),
        default="source-meta",
        help=(
            "Feature contract for isolated training. source-meta keeps warm-start "
            "best_genome compatible with its saved metadata; use on/off to force."
        ),
    )
    parser.add_argument(
        "--regime-filter",
        choices=("all", "bearish", "neutral", "bullish"),
        default="all",
        help="Train only on periods mapped to this market regime.",
    )
    parser.add_argument(
        "--source-genome-mismatch",
        choices=("fail", "transfer", "random"),
        default="fail",
        help=(
            "How to handle warm-start best_genome size mismatches. transfer embeds "
            "the saved MLP into the current wider architecture; random starts from "
            "a fresh current-architecture seed."
        ),
    )
    args = parser.parse_args()

    source_agents_dir = Path(cg.AGENTS_DIR)
    best_path = source_agents_dir / "best_genome.npy"
    if not best_path.exists():
        raise FileNotFoundError(best_path)
    best_genome = np.load(best_path).astype(np.float32).ravel()
    source_genome_size = int(best_genome.shape[0])
    target_genome_size = int(cg.GENOME_SIZE)
    source_arch = _source_arch_from_reports(source_agents_dir)
    target_arch = _current_arch()
    source_genome_transferred = False
    if source_genome_size != target_genome_size:
        if args.source_genome_mismatch == "transfer":
            best_genome = _transfer_mlp_genome(best_genome, source_arch, target_arch)
            source_genome_transferred = True
            print(
                "[smoke] transferred source genome "
                f"{source_arch}/{source_genome_size} -> {target_arch}/{target_genome_size}"
            )
        elif args.source_genome_mismatch == "random":
            best_genome = cg.init_population(1)[0].astype(np.float32).ravel()
            print(
                "[smoke] source genome size mismatch; using random current-architecture seed "
                f"{source_genome_size} -> {target_genome_size}"
            )
        else:
            raise ValueError(
                f"best genome size {source_genome_size} != {target_genome_size}; "
                "use --source-genome-mismatch transfer or random"
            )

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    warm_start_paths = _write_warm_start_artifacts(
        out_dir,
        best_genome,
        {
            "source_best_genome": str(best_path),
            "source_arch": {
                "in": int(source_arch[0]),
                "h1": int(source_arch[1]),
                "h2": int(source_arch[2]),
                "h3": int(source_arch[3]),
                "out": int(source_arch[4]),
            },
            "target_arch": {
                "in": int(target_arch[0]),
                "h1": int(target_arch[1]),
                "h2": int(target_arch[2]),
                "h3": int(target_arch[3]),
                "out": int(target_arch[4]),
            },
            "source_genome_size": source_genome_size,
            "target_genome_size": target_genome_size,
            "source_genome_mismatch": args.source_genome_mismatch,
            "source_genome_transferred": source_genome_transferred,
            "position_state_features_enabled": bool(
                _resolve_position_state_features_enabled(
                    args.position_state_features,
                    source_agents_dir,
                    settings_default=bool(cg.POSITION_STATE_FEATURES_ENABLED),
                )
            ),
        },
    )

    old_cfg_start = cg._cx._CFG.get("start_date")
    old_cfg_end = cg._cx._CFG.get("end_date")
    old_agents_dir = cg.AGENTS_DIR
    old_position_state_features_enabled = cg.POSITION_STATE_FEATURES_ENABLED
    source_position_state_features_enabled = _resolve_position_state_features_enabled(
        "source-meta",
        source_agents_dir,
        settings_default=bool(cg.POSITION_STATE_FEATURES_ENABLED),
    )
    position_state_features_enabled = _resolve_position_state_features_enabled(
        args.position_state_features,
        source_agents_dir,
        settings_default=bool(cg.POSITION_STATE_FEATURES_ENABLED),
    )

    try:
        cg._cx._CFG["start_date"] = args.start_date
        cg._cx._CFG["end_date"] = args.end_date
        cg.POSITION_STATE_FEATURES_ENABLED = bool(position_state_features_enabled)

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
        loaded_periods = len(precomp)
        precomp = _filter_precomp_by_regime(precomp, args.regime_filter)
        if not precomp:
            raise RuntimeError(f"No periods left after regime filter: {args.regime_filter}")
        if args.regime_filter != "all":
            print(
                f"[smoke] regime_filter={args.regime_filter} "
                f"periods={len(precomp)}/{loaded_periods}"
            )

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
            mutation_scope=args.feasible_mutation_scope,
        )
        if feasible_seeded:
            print(
                "[smoke] feasible-best seeds="
                f"{feasible_seeded} sigma={args.feasible_best_seed_sigma:.4f} "
                f"open_bias={args.feasible_open_logit_bias:.3f} "
                f"scope={args.feasible_mutation_scope}"
            )
        print(
            "[smoke] position_state_features="
            f"{bool(cg.POSITION_STATE_FEATURES_ENABLED)} "
            f"(mode={args.position_state_features}, source_meta={source_position_state_features_enabled})"
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
            "source_arch": {
                "in": int(source_arch[0]),
                "h1": int(source_arch[1]),
                "h2": int(source_arch[2]),
                "h3": int(source_arch[3]),
                "out": int(source_arch[4]),
            },
            "target_arch": {
                "in": int(target_arch[0]),
                "h1": int(target_arch[1]),
                "h2": int(target_arch[2]),
                "h3": int(target_arch[3]),
                "out": int(target_arch[4]),
            },
            "source_genome_size": source_genome_size,
            "target_genome_size": target_genome_size,
            "source_genome_mismatch": args.source_genome_mismatch,
            "source_genome_transferred": source_genome_transferred,
            "warm_start_genome": str(warm_start_paths["genome"]),
            "warm_start_genome_meta": str(warm_start_paths["meta"]),
            "population": cg.POP_SIZE,
            "generations": cg.N_GENERATIONS,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "regime_filter": args.regime_filter,
            "loaded_periods": loaded_periods,
            "periods": len(precomp),
            "execution_lag_bars": cg.TRAIN_EXECUTION_LAG_BARS,
            "feasible_best_seeded": feasible_seeded,
            "feasible_best_seed_frac": args.feasible_best_seed_frac,
            "feasible_best_seed_sigma": args.feasible_best_seed_sigma,
            "feasible_open_logit_bias": args.feasible_open_logit_bias,
            "feasible_mutation_scope": args.feasible_mutation_scope,
            "position_state_features_mode": args.position_state_features,
            "position_state_features_enabled": bool(cg.POSITION_STATE_FEATURES_ENABLED),
            "source_position_state_features_enabled": bool(source_position_state_features_enabled),
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
        cg.POSITION_STATE_FEATURES_ENABLED = old_position_state_features_enabled
        if old_cfg_start is not None:
            cg._cx._CFG["start_date"] = old_cfg_start
        if old_cfg_end is not None:
            cg._cx._CFG["end_date"] = old_cfg_end

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
