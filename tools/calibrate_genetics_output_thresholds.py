from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GENETICS_DIR = ROOT / "Genetics_DL_Agents"
NEIRO_GENETICS_DIR = ROOT / "Results" / "neiro_genetics"
if str(GENETICS_DIR) not in sys.path:
    sys.path.insert(0, str(GENETICS_DIR))

import crypto_genetics as cg  # noqa: E402


def _resolve_under_neiro_genetics(path: Path | str) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = ROOT / out
    resolved = out.resolve()
    root = NEIRO_GENETICS_DIR.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"genetics calibration artifacts must be under {root}") from exc
    return resolved


def _bias_token(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def open_output_bias_indices() -> np.ndarray:
    b4_start = int(cg.GENOME_SIZE - cg.N_ACTIONS)
    actions = [code for code in cg._OPEN_ACTION_CODES if 0 <= int(code) < cg.N_ACTIONS]
    return np.asarray([b4_start + int(code) for code in actions], dtype=np.int64)


def calibrate_open_action_bias(genome: np.ndarray, *, open_bias: float) -> np.ndarray:
    arr = np.asarray(genome, dtype=np.float32).ravel()
    if arr.shape[0] != cg.GENOME_SIZE:
        raise ValueError(f"genome size {arr.shape[0]} != expected {cg.GENOME_SIZE}")
    calibrated = arr.copy()
    idx = open_output_bias_indices()
    calibrated[idx] -= np.float32(open_bias)
    return calibrated.astype(np.float32, copy=False)


def write_calibrated_variants(
    *,
    source_genome: Path | str,
    out_dir: Path | str,
    open_biases: Iterable[float],
) -> dict:
    source = Path(source_genome)
    if not source.is_absolute():
        source = ROOT / source
    source = source.resolve()
    genome = np.load(source).astype(np.float32).ravel()
    if genome.shape[0] != cg.GENOME_SIZE:
        raise ValueError(f"{source} genome size {genome.shape[0]} != expected {cg.GENOME_SIZE}")

    out = _resolve_under_neiro_genetics(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    variants: list[dict] = []
    for raw_bias in open_biases:
        bias = float(raw_bias)
        calibrated = calibrate_open_action_bias(genome, open_bias=bias)
        token = _bias_token(bias)
        genome_path = out / f"open_bias_{token}.npy"
        meta_path = out / f"open_bias_{token}_meta.json"
        np.save(genome_path, calibrated)
        meta = {
            "role": "output_layer_open_bias_calibration",
            "source_genome": str(source),
            "open_bias": bias,
            "genome_size": int(calibrated.shape[0]),
            "changed_indices": [int(v) for v in open_output_bias_indices().tolist()],
            "promotion_allowed": False,
            "requires_validation": True,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        variants.append({
            "open_bias": bias,
            "path": str(genome_path),
            "meta_path": str(meta_path),
            "promotion_allowed": False,
            "requires_validation": True,
        })

    manifest_path = out / "output_calibration_manifest.json"
    manifest = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_genome": str(source),
        "promotion_allowed": False,
        "paper_trading_eligible": False,
        "live_trading_eligible": False,
        "requires_validation": True,
        "variants": variants,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": manifest_path, "variants": [Path(item["path"]) for item in variants]}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create output-layer open-action bias calibrated genetics genomes."
    )
    parser.add_argument("--source-genome", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--open-bias", type=float, action="append", required=True)
    args = parser.parse_args()

    result = write_calibrated_variants(
        source_genome=args.source_genome,
        out_dir=args.out_dir,
        open_biases=args.open_bias,
    )
    print(json.dumps({
        "manifest": str(result["manifest"]),
        "variants": [str(path) for path in result["variants"]],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
