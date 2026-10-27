"""Read-only normalized recording intake; optional NEW validated export, no replay."""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/recorded_admission_v1_20260911"
FROZEN_RECEIPT = "Reports/Exia/Genetic_Primus/portfolio_replay_v1_20260910/policy_tests.json"
FROZEN_SHA = "007d49f7eb878cdafb699ac599d93e328b6fe92c1726abf059f5c0ff968a4c56"
MAX_BYTES = 2 * 1024 * 1024


def strict_json(raw):
    text = raw.decode("utf-8")
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > 64:
                raise ValueError("excessive_json_nesting")
        elif char in "]}":
            depth -= 1
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_json_key:" + str(key)[:60])
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonstandard_json_constant:" + value)
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except RecursionError as exc:
        raise ValueError("excessive_json_nesting") from exc


def context():
    raw = (ROOT / FROZEN_RECEIPT).read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_SHA:
        raise ValueError("frozen_portfolio_receipt_changed")
    receipt = strict_json(raw)
    for name, expected in receipt["source_hashes"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("frozen_portfolio_source_changed:" + name)
    api = runpy.run_path(str(ROOT / "tools/run_genetic_primus_portfolio_replay_v1.py"))
    portfolio, release, identity = api["read_inputs"]()
    if identity["sha256"] != receipt["implementation_sha256"]:
        raise ValueError("frozen_portfolio_identity_changed")
    sys.path.insert(0, str(ROOT / "src/exia/genetic_primus"))
    return portfolio, release, identity


def reference_path(base, name):
    if not isinstance(name, str) or not name or ":" in name or "\\" in name:
        raise ValueError("reference_must_be_relative_local_posix_json")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".json":
        raise ValueError("reference_escape_or_unsupported_format")
    result = base.joinpath(*relative.parts).resolve()
    if not result.is_relative_to(base.resolve()):
        raise ValueError("reference_symlink_escape")
    return result


def read_bounded(path, limit):
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError("file_missing_or_byte_budget_exceeded")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("file_grew_beyond_byte_budget")
    return raw


def load_package(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT) or path.suffix != ".json":
        raise ValueError("workspace_local_manifest_json_required")
    raw = read_bounded(path, min(MAX_BYTES, 65536))
    manifest = strict_json(raw)
    refs = manifest.get("artifacts", {})
    if set(refs) != {"bundle", "provenance"}:
        raise ValueError("bundle_and_provenance_artifacts_required")
    used, values, hashes = len(raw), {}, {"manifest": hashlib.sha256(raw).hexdigest()}
    resolved = {path}
    for role in ("bundle", "provenance"):
        ref = refs[role]
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256", "bytes"}:
            raise ValueError("explicit_file_path_sha_and_size_required")
        if type(ref["bytes"]) is not int or ref["bytes"] <= 0 or ref["bytes"] > MAX_BYTES - used:
            raise ValueError("package_byte_budget_exceeded")
        target = reference_path(path.parent, ref["path"])
        if target in resolved:
            raise ValueError("artifact_paths_must_be_distinct")
        resolved.add(target)
        content = read_bounded(target, ref["bytes"])
        digest = hashlib.sha256(content).hexdigest()
        if len(content) != ref["bytes"] or digest != ref["sha256"]:
            raise ValueError("artifact_size_or_hash_mismatch:" + role)
        used += len(content)
        hashes[role] = digest
        values[role] = strict_json(content)
    return manifest, values["bundle"], values["provenance"], hashes


def save_new(name, value):
    target = (ROOT / name).resolve()
    if target.parent != OUTPUT.resolve() or target.suffix != ".json":
        raise ValueError("new_scoped_json_output_required")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)


def importer_sources():
    names = ("src/exia/genetic_primus/recorded_admission_v1.py", "tools/import_genetic_primus_recorded_v1.py")
    return {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="local normalized package manifest, no network URLs")
    parser.add_argument("--report", help="optional NEW scoped admission report")
    parser.add_argument("--export", help="optional NEW validated bundle; NEVER runs the portfolio")
    args = parser.parse_args()
    portfolio, release, identity = context()
    if not args.manifest:
        if args.report or args.export:
            parser.error("--manifest required for outputs")
        print(json.dumps({"status": "WAITING_FOR_RECORDINGS", "frozen_sources_verified": True,
                          "replay_run": False, "trading_authorized": False}))
        return 0
    destinations = []
    for name in (args.report, args.export):
        if name:
            path = (ROOT / name).resolve()
            if path.parent != OUTPUT.resolve() or path.suffix != ".json" or path.exists() or path in destinations:
                raise ValueError("distinct_new_scoped_outputs_required")
            destinations.append(path)
    manifest, bundle, provenance, hashes = load_package(ROOT / args.manifest)
    from recorded_admission_v1 import admitted_bundle
    from release_contract_v1 import object_sha
    exported, admission = admitted_bundle(manifest, bundle, provenance, portfolio, release, identity["sha256"])
    receipt = {"schema_version": "exia.genetic_primus.recorded_intake_receipt/1", "admission": admission,
               "input_file_hashes": hashes, "frozen_implementation": identity,
               "importer_sources": importer_sources(), "replay_run": False,
               "network_used": False, "training_run": False, "trading_authorized": False}
    receipt["receipt_body_sha256"] = object_sha(receipt)
    if args.report:
        save_new(args.report, receipt)
    if args.export:
        save_new(args.export, exported)
    print(json.dumps({"status": "ADMITTED_OFFLINE_ONLY", "evidence_kind": admission["evidence_kind"],
                      "events": admission["unique_events"], "decision_boundaries": admission["decision_boundaries"],
                      "funding_settlements": admission["funding_settlements"],
                      "replay_run": False, "qualification_eligible": False}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": type(exc).__name__ + ": " + str(exc)[:240]}))
        raise SystemExit(1)
