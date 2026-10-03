"""Package-only offline handoff: export, verify, bounded exclusive extraction.

Stdlib only. Never installs packages, executes archive contents or evaluates models.
"""

import argparse
import hashlib
import json
import re
import stat
import struct
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "Reports/Exia/Genetic_Primus/future_ema_offline_bundle_v1_20261003"
DESTINATION = ROOT / "Reports/Exia/Genetic_Primus/future_ema_handoff_v1_20261004"
OBJECTIVE = "future_ema_base_viability_60x3d_from_2026_10_05_v1"
BUNDLE_SHA = "fc74409c9cadc180dc18e6c7a0fb47befba4b06ff60529aaf6292b6be2c69cef"
MAX_BYTES = 128 * 1024 * 1024
MAX_FILES = 64
INDEX = "archive_index.json"
PACKAGES = ("numpy", "pandas", "pyarrow", "python-dateutil", "six", "tzdata")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(name):
    if not isinstance(name, str):
        raise ValueError("archive member name must be text")
    parts = PurePosixPath(name).parts
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
    if (not parts or len(name) > 240 or "\\" in name or ":" in name
            or name != "/".join(parts) or PurePosixPath(name).is_absolute()
            or any(p in (".", "..") or p.endswith((".", " "))
                   or p.split(".")[0].upper() in reserved for p in parts)):
        raise ValueError("unsafe archive member name")
    return name


def _unique(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON key")
        result[name] = value
    return result


def _nonfinite(_):
    raise ValueError("nonfinite JSON")


def _central_directory_guard(path):
    size = path.stat().st_size
    if not 22 <= size <= MAX_BYTES:
        raise ValueError("archive size outside bound")
    with path.open("rb") as stream:
        stream.seek(max(0, size - 65557))
        tail = stream.read(65557)
    position = tail.rfind(b"PK\x05\x06")
    if position < 0 or len(tail) - position < 22:
        raise ValueError("invalid ZIP end")
    _, disk, cd_disk, count_disk, count, cd_size, cd_offset, comment = struct.unpack(
        "<4s4H2LH", tail[position:position + 22])
    end = size - len(tail) + position
    if (disk or cd_disk or count != count_disk or not 1 <= count <= MAX_FILES
            or cd_size > 65536 or cd_offset + cd_size != end
            or end + 22 + comment != size):
        raise ValueError("unsupported ZIP layout or excessive central directory")


def verify_archive(path, expected_sha):
    path = Path(path)
    _central_directory_guard(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or file_sha(path) != expected_sha:
        raise ValueError("archive SHA mismatch")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_FILES or sum(i.file_size for i in infos) > MAX_BYTES:
            raise ValueError("uncompressed payload exceeds bound")
        names = set()
        folded = set()
        for info in infos:
            name = safe_name(info.filename)
            if (name in names or name.casefold() in folded or info.is_dir()
                    or stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1
                    or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                raise ValueError("unsafe or duplicate archive entry")
            names.add(name)
            folded.add(name.casefold())
        if INDEX not in names or archive.getinfo(INDEX).file_size > 32768:
            raise ValueError("missing/oversized archive index")
        index = json.loads(archive.read(INDEX), object_pairs_hook=_unique, parse_constant=_nonfinite)
        if (not isinstance(index, dict) or index.get("schema_version") != "exia.genetic_primus.package_handoff/1"
                or index.get("objective_id") != OBJECTIVE or index.get("promotion_authority") is not False
                or index.get("kind") != "package_bundle_not_model_release"):
            raise ValueError("handoff contract mismatch")
        members = index["members"]
        if not isinstance(members, dict) or names != set(members) | {INDEX}:
            raise ValueError("index/member set mismatch")
        total = 0
        for name, pin in members.items():
            info = archive.getinfo(name)
            if (not isinstance(pin, dict) or type(pin.get("size_bytes")) is not int
                    or info.file_size != pin["size_bytes"]
                    or not re.fullmatch(r"[0-9a-f]{64}", pin.get("sha256", ""))):
                raise ValueError("invalid member pin")
            hasher = hashlib.sha256()
            count = 0
            with archive.open(info) as stream:
                while chunk := stream.read(65536):
                    count += len(chunk)
                    if count > pin["size_bytes"]:
                        raise ValueError("member expanded beyond declared bound")
                    hasher.update(chunk)
            if count != pin["size_bytes"] or hasher.hexdigest() != pin["sha256"]:
                raise ValueError("member SHA mismatch")
            total += count
    return {"status": "ARCHIVE_VERIFIED", "archive_sha256": expected_sha,
            "members_verified": len(members), "payload_bytes": total,
            "promotion_authority": False, "contents_executed": False}


def create_archive(path, sources):
    """Sources map: archive name -> (local Path, precommitted content SHA)."""
    if len(sources) + 1 > MAX_FILES:
        raise ValueError("too many handoff members")
    members = {}
    for name, (source, expected_sha) in sources.items():
        safe_name(name)
        if name == INDEX or file_sha(source) != expected_sha:
            raise ValueError("source SHA differs before export")
        members[name] = {"sha256": expected_sha, "size_bytes": source.stat().st_size}
    if sum(p["size_bytes"] for p in members.values()) > MAX_BYTES - 65536:
        raise ValueError("export exceeds archive budget")
    index = {"schema_version": "exia.genetic_primus.package_handoff/1", "objective_id": OBJECTIVE,
             "kind": "package_bundle_not_model_release", "members": members,
             "independent_python_runtime_restored": False, "promotion_authority": False}
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        for name, (source, expected_sha) in sources.items():
            digest = hashlib.sha256()
            count = 0
            with source.open("rb") as original, archive.open(name, "w") as output:
                while chunk := original.read(65536):
                    count += len(chunk)
                    if count > members[name]["size_bytes"]:
                        raise ValueError("source grew during export; preserve incomplete archive")
                    output.write(chunk)
                    digest.update(chunk)
            if count != members[name]["size_bytes"] or digest.hexdigest() != expected_sha:
                raise ValueError("source changed during export; preserve incomplete archive")
        archive.writestr(INDEX, json.dumps(index, sort_keys=True, indent=2))
    return verify_archive(path, file_sha(path))


def extract_archive(path, expected_sha, destination):
    verification = verify_archive(path, expected_sha)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            target = (root / safe_name(info.filename)).resolve()
            if not target.is_relative_to(root):
                raise ValueError("extraction escaped destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.open(info) as original, target.open("xb") as output:
                while chunk := original.read(65536):
                    output.write(chunk)
                    digest.update(chunk)
            if info.filename != INDEX:
                # Verify extracted bytes again against archived index, not just CRC.
                index = json.loads(archive.read(INDEX), object_pairs_hook=_unique)
                if digest.hexdigest() != index["members"][info.filename]["sha256"]:
                    raise ValueError("extraction hash mismatch; preserve destination")
    return {**verification, "status": "ARCHIVE_EXTRACTED_AND_VERIFIED",
            "independent_machine_verified": False, "contents_executed": False}


def export_bundle():
    if not DESTINATION.resolve().is_relative_to(ROOT.resolve()) or file_sha(BUNDLE / "bundle_manifest.json") != BUNDLE_SHA:
        raise ValueError("unsafe destination or changed source bundle")
    manifest = json.loads((BUNDLE / "bundle_manifest.json").read_text())
    sources = {"metadata/bundle_manifest.json": (BUNDLE / "bundle_manifest.json", BUNDLE_SHA)}
    for name, digest in manifest["artifact_sha256"].items():
        sources["metadata/" + safe_name(name)] = (BUNDLE / name, digest)
    requirements = BUNDLE / "requirements_offline.txt"
    if hashlib.sha256(requirements.read_text().encode()).hexdigest() != manifest["requirements_lf_normalized_sha256"]:
        raise ValueError("requirements changed")
    sources["metadata/requirements_offline.txt"] = (requirements, file_sha(requirements))
    for name in PACKAGES:
        record_path = BUNDLE / (name + ".json")
        if file_sha(record_path) != manifest["artifact_sha256"][name + ".json"]:
            raise ValueError("package evidence changed")
        record = json.loads(record_path.read_text())
        if record["record_payload_verified"] is not True or record["installed_payload_matches"] is not True:
            raise ValueError("unverified package evidence")
        wheel = (ROOT / record["wheel_path"]).resolve()
        if not wheel.is_relative_to((BUNDLE / "wheels").resolve()):
            raise ValueError("wheel outside source bundle")
        sources["wheels/" + wheel.name] = (wheel, record["wheel_sha256"])
    environment = ROOT / "configs/genetic_primus_future_ema_environment_v1.json"
    sources["metadata/environment.json"] = (environment, manifest["environment_sha256"])
    cache_sources = ROOT / "configs/genetic_primus_future_ema_cache_sources_v1.json"
    if any(json.loads((BUNDLE / (n + ".json")).read_text())["sources_config_sha256"] != file_sha(cache_sources) for n in PACKAGES):
        raise ValueError("cache sources registration changed")
    sources["metadata/cache_sources.json"] = (cache_sources, file_sha(cache_sources))
    instructions = ROOT / "docs/GENETIC_PRIMUS_FUTURE_EMA_OFFLINE_BUNDLE_V1.md"
    sources["README.md"] = (instructions, file_sha(instructions))
    sources["verify_handoff.py"] = (Path(__file__), file_sha(Path(__file__)))
    archive = DESTINATION / "genetic_primus_future_ema_packages_v1.zip"
    result = create_archive(archive, sources)
    evidence = {**result, "archive_path": archive.relative_to(ROOT).as_posix(),
                "archive_size_bytes": archive.stat().st_size, "source_bundle_sha256": BUNDLE_SHA,
                "kind": "package_bundle_not_model_release", "independent_machine_verified": False,
                "power_loss_durability_verified": False, "network_used": False, "new_model_fits": 0}
    with (DESTINATION / "handoff.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(evidence, sort_keys=True, indent=2) + "\n")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "verify", "extract"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expected-sha")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != Path("E:/Crypto_exchange/.venv/Scripts/python.exe").resolve():
        raise ValueError("use approved workspace Python")
    if args.mode == "export":
        result = export_bundle()
    elif args.mode == "verify":
        if args.archive is None or args.expected_sha is None:
            raise ValueError("archive and independently saved expected SHA required")
        result = verify_archive(args.archive, args.expected_sha)
    else:
        if args.archive is None or args.expected_sha is None or not DESTINATION.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("archive/SHA and scoped destination required")
        result = extract_archive(args.archive, args.expected_sha, DESTINATION / "extraction_check")
        with (DESTINATION / "extraction_check.json").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "HANDOFF_STEP_BLOCKED", "error_type": type(exc).__name__,
                          "detail": str(exc)[:140], "promotion_authority": False}, separators=(",", ":")))
        raise SystemExit(1)
