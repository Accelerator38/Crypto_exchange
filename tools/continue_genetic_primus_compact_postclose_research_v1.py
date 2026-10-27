"""Explicit serialization-only adapter for the already pinned research runner.

Does not edit the runner, models, thresholds, training banks, or sealed receipts.
Numpy scalar booleans in gate results require conversion to Python scalars before
canonical JSON hashing. Register this adapter before benchmark model fits.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys


def plain(value):
    # Only JSON-container recursion and numpy scalar normalization; no rounding.
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def main():
    path = Path(__file__).with_name("run_genetic_primus_compact_postclose_research_v1.py")
    spec = importlib.util.spec_from_file_location("pinned_postclose_research_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    if len(sys.argv) < 2 or sys.argv[1] not in ("calibrate", "summarize", "verify"):
        raise ValueError("serialization adapter only for named exporting steps")
    reg, _, _, _ = runner.load()
    adapter_pin = runner.pin(Path(__file__))
    registration_path = runner.OUT / "serialization_adapter_registration.json"
    if not registration_path.exists():
        if any(runner.OUT.glob("report_origin_*.json")):
            raise ValueError("adapter must be registered before benchmark fits")
        runner.write(registration_path.name, {"schema": "serialization_scalar_erratum/1",
                                              "registration_sha256": reg["sha256"], "adapter_pin": adapter_pin,
                                              "source": runner.relative(Path(__file__)),
                                              "reason": "numpy_bool_gate_values_not_JSON_serializable",
                                              "conversion": "numpy_scalar_item_no_numeric_or_ranking_change",
                                              "existing_pre_august_fits_preserved": 168,
                                              "before_benchmark_fits": True, "safety": runner.r.SAFETY})
    adapter_registration = runner.verify_body(runner.read(registration_path))
    if adapter_registration["adapter_pin"] != adapter_pin or adapter_registration["registration_sha256"] != reg["sha256"]:
        raise ValueError("serialization adapter substitution")
    original_write = runner.write

    def write_with_adapter(name, value):
        normalized = plain(value)
        normalized["serialization_adapter_registration_sha256"] = adapter_registration["sha256"]
        normalized["serialization_adapter_source_pin"] = adapter_pin
        return original_write(name, normalized)

    runner.write = write_with_adapter
    runner.main()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print((type(error).__name__ + ": " + str(error))[:600], file=sys.stderr)
        sys.exit(1)
