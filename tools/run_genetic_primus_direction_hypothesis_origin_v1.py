"""One finite origin: fit, verify saved evidence, audit; safe to resume locally."""
from __future__ import annotations
import argparse
import json
import gc
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_genetic_primus_direction_hypotheses_v1 as runner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=int, choices=[0, 1, 10, 30], default=0)
    parser.add_argument("--origin", type=int, required=True)
    args = parser.parse_args()
    for minutes in ([1, 10, 30] if args.minutes == 0 else [args.minutes]):
        runner.fit_origin(minutes, args.origin)
        audit_path = runner.OUT / f"audit_tf_{minutes}m_origin_{args.origin:02d}.json"
        if audit_path.exists():
            saved = runner.read(audit_path)
            if saved["registration_sha256"] != runner.registration()["sha256"]:
                raise ValueError("completed audit belongs to another registration")
            if runner.identity({k: v for k, v in saved.items() if k != "sha256"}) != saved["sha256"]:
                raise ValueError("completed audit was modified")
            print(json.dumps({"audit_reused": True}))
        else:
            runner.audit(minutes, args.origin)
        gc.collect()
