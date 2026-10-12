"""Small self-contained parity test; no pytest, historical archive or runtime."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import copy
import json
from pathlib import Path
import numpy as np
import offline_models as api


def read(name):
    p = Path(__file__).resolve().parent / name
    if p.stat().st_size > 250_000:
        raise ValueError("oversized fixture")
    v = json.loads(p.read_text(encoding="utf-8"))
    if v.get("sha256") != api.identity({k: x for k, x in v.items() if k != "sha256"}):
        raise ValueError("fixture SHA mismatch")
    return v


def main():
    vectors = read("test_vectors.json")
    passed = 0
    for name, v in vectors["models"].items():
        fit = read(name + ".json")
        if not fit.get("safety") or any(value is not False for value in fit["safety"].values()):
            raise ValueError("runtime authority must remain false")
        assert fit["model"]["sha256"] == v["model_sha256"]
        np.testing.assert_allclose(api.predict(fit["model"], v["features"]), v["prediction"], rtol=0, atol=1e-13)
        passed += 2
        broken = copy.deepcopy(fit["model"])
        broken["parameters"][0] += .1
        try:
            api.predict(broken, v["features"])
        except ValueError:
            passed += 1
        else:
            raise AssertionError("corrupted model accepted")
    for name, v in vectors["policies"].items():
        np.testing.assert_allclose(api.research_targets(name, v["features"], v["covariance"]), v["targets"], rtol=0, atol=1e-13)
        passed += 1
    t = np.arange(49, dtype=np.int64) * api.BAR_MS
    flat = api.latest_features(t, np.ones((49, 8)), asof_ms=int(t[-1]), history_anchor_ms=0, symbols=api.SYMBOLS)
    np.testing.assert_array_equal(flat, np.zeros((8, 6)))
    passed += 1
    print(json.dumps({"standalone_checks_passed": passed, "network": False, "runtime": False}))


if __name__ == "__main__":
    main()
