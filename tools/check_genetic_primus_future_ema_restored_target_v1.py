"""Synthetic imports/arithmetic from offline target using workspace Python -I -S -B."""

import hashlib
import importlib
import importlib.metadata as metadata
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "Reports/Exia/Genetic_Primus/future_ema_offline_bundle_v1_20261003"
TARGET = DESTINATION / "isolated_target"
ENVIRONMENT = ROOT / "configs/genetic_primus_future_ema_environment_v1.json"
ENVIRONMENT_SHA = "8f82114b946f09afcd4aa8b6e5dcd721fb7c2f303a09f1c2dac3a0fc51137360"


def main():
    if (Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve()
            or not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode):
        raise ValueError("use workspace Python with -I -S -B")
    if not TARGET.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("target outside workspace")
    if any("site-packages" in Path(path).parts for path in sys.path):
        raise ValueError("original site-packages visible before target injection")
    raw = ENVIRONMENT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ENVIRONMENT_SHA:
        raise ValueError("registered environment changed")
    expected = json.loads(raw)["snapshot"]
    if sys.version != expected["python"]["version"]:
        raise ValueError("shared Python build changed")
    sys.path.insert(0, str(TARGET.resolve()))
    modules = {"numpy": "numpy", "pandas": "pandas", "pyarrow": "pyarrow",
               "python-dateutil": "dateutil", "six": "six", "tzdata": "tzdata"}
    observed = {}
    for name, module_name in modules.items():
        dist = metadata.distribution(name)
        pin = expected["packages"][name]
        text = dist.read_text("METADATA")
        if dist.version != pin["version"] or hashlib.sha256(text.encode()).hexdigest() != pin["metadata_sha256"]:
            raise ValueError("restored version/metadata mismatch")
        module = importlib.import_module(module_name)
        if not Path(module.__file__).resolve().is_relative_to(TARGET.resolve()):
            raise ValueError("import escaped isolated target")
        observed[name] = dist.version
    for optional in ("bottleneck", "numexpr"):
        try:
            metadata.distribution(optional)
        except metadata.PackageNotFoundError:
            continue
        raise ValueError("unexpected optional numerical backend")
    np, pd, pa = (sys.modules[name] for name in ("numpy", "pandas", "pyarrow"))
    if float(np.dot([1.0, 2.0], [3.0, 4.0])) != 11.0:
        raise ValueError("numpy synthetic arithmetic failed")
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    values = frame.x.ewm(span=2, adjust=False).mean().tolist()
    if not all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(values, [1.0, 5.0 / 3, 23.0 / 9])):
        raise ValueError("pandas synthetic EWM failed")
    if pa.Table.from_pandas(frame, preserve_index=False).to_pandas().x.tolist() != frame.x.tolist():
        raise ValueError("arrow in-memory roundtrip failed")
    result = {"schema_version": "exia.genetic_primus.offline_restore_smoke/1",
              "status": "OFFLINE_PACKAGE_TARGET_VERIFIED", "versions": observed,
              "environment_sha256": ENVIRONMENT_SHA,
              "all_imports_from_isolated_target": True, "original_site_packages_visible": False,
              "shared_workspace_python": True, "independent_python_runtime_restored": False,
              "full_os_isolation": False, "new_model_fits": 0,
              "market_data_opened": False, "network_used": False, "promotion_authority": False}
    with (DESTINATION / "restore_smoke.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "imports": len(observed),
                      "network_used": False, "new_model_fits": 0}, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, ImportError) as exc:
        print(json.dumps({"status": "RESTORE_SMOKE_BLOCKED", "error_type": type(exc).__name__,
                          "detail": str(exc)[:160]}, separators=(",", ":")))
        raise SystemExit(1)
