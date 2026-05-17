from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GENETICS_DIR = ROOT / "Genetics_DL_Agents"
if str(GENETICS_DIR) not in sys.path:
    sys.path.insert(0, str(GENETICS_DIR))

import crypto_genetics as cg  # noqa: E402


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a validated genetic genome to the production slot.")
    parser.add_argument("--genome", required=True, help="Candidate .npy genome path.")
    parser.add_argument(
        "--agents-dir",
        default=str(GENETICS_DIR / "Agents" / "genetics"),
        help="Target genetics agents directory.",
    )
    parser.add_argument("--eval-report", default=None, help="Optional evaluation JSON used for promotion.")
    parser.add_argument("--reason", default="", help="Human-readable promotion reason.")
    args = parser.parse_args()

    genome_path = _resolve(args.genome)
    agents_dir = _resolve(args.agents_dir)
    eval_report = _resolve(args.eval_report) if args.eval_report else None

    if not genome_path.exists():
        raise FileNotFoundError(genome_path)
    genome = np.load(genome_path).astype(np.float32).ravel()
    if genome.shape[0] != cg.GENOME_SIZE:
        raise ValueError(f"{genome_path} genome size {genome.shape[0]} != expected {cg.GENOME_SIZE}")

    agents_dir.mkdir(parents=True, exist_ok=True)
    target = agents_dir / "best_genome.npy"
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = None
    if target.exists():
        backup = agents_dir / f"best_genome_bak_{ts}_pre_promotion.npy"
        shutil.copy2(target, backup)

    np.save(target, genome)

    position_state_features_enabled = bool(cg.POSITION_STATE_FEATURES_ENABLED)
    if eval_report is not None and eval_report.exists():
        try:
            report_payload = json.loads(eval_report.read_text(encoding="utf-8"))
            matched_genome = None
            for genome_report in report_payload.get("genomes", []):
                if Path(str(genome_report.get("path", ""))).resolve() == genome_path.resolve():
                    matched_genome = genome_report
                    break
            source = matched_genome if matched_genome is not None else report_payload
            position_state_features_enabled = bool(
                source.get(
                    "position_state_features_enabled",
                    source.get("default_position_state_features_enabled", position_state_features_enabled),
                )
            )
        except Exception:
            pass

    metadata = {
        "promoted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_genome": str(genome_path),
        "target_genome": str(target),
        "backup_genome": str(backup) if backup else None,
        "eval_report": str(eval_report) if eval_report else None,
        "reason": args.reason,
        "genome_size": int(cg.GENOME_SIZE),
        "position_state_features_enabled": position_state_features_enabled,
    }
    meta_path = agents_dir / "best_genome_meta.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
