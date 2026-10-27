"""Direct, small unittest entry point. Never pytest or collector."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import argparse
import importlib.util
import io
import unittest
import hashlib
import json

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("core", "rolling", "all"), default="core")
    parser.add_argument("--report-path")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("compact_tests", ROOT / "tests/test_genetic_primus_compact_forecast_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = ["ModelTests", "FeatureTests"] if args.scope == "core" else ["RollingTests"] if args.scope == "rolling" else ["ModelTests", "FeatureTests", "RollingTests"]
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(getattr(module, n)) for n in names)
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=0).run(suite)
    print(f"scope={args.scope}; checks={result.testsRun}; failures={len(result.failures)}; errors={len(result.errors)}; synthetic_only; network=0; collector=0")
    if not result.wasSuccessful():
        print(output.getvalue().encode("utf-8")[:720].decode("utf-8", "ignore"))
    if args.report_path:
        path = (ROOT / args.report_path).resolve()
        path.relative_to(ROOT / "Reports/Exia/Genetic_Primus")
        receipt = {"scope": args.scope, "checks": result.testsRun,
                   "failures": len(result.failures), "errors": len(result.errors),
                   "synthetic_only": True, "network": 0, "collector": 0,
                   "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__), ROOT / "tests/test_genetic_primus_compact_forecast_v1.py",
                                 ROOT / "src/exia/genetic_primus/compact_forecast_v1.py",
                                 ROOT / "src/exia/genetic_primus/compact_forecast_data_v1.py"]}}
        with path.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return int(not result.wasSuccessful())


if __name__ == "__main__":
    raise SystemExit(main())
