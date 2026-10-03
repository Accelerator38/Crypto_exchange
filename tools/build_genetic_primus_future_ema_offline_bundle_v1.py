"""Validate one cached wheel at a time and copy ZIP payload to an exclusive bundle."""

import argparse
import base64
import csv
import hashlib
import importlib.metadata as metadata
import io
import json
import stat
import struct
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_genetic_primus_future_ema_wheel_cache_v1 import inspect_candidate, MAX_BODY_BYTES

ENVIRONMENT = ROOT / "configs/genetic_primus_future_ema_environment_v1.json"
ENVIRONMENT_SHA = "8f82114b946f09afcd4aa8b6e5dcd721fb7c2f303a09f1c2dac3a0fc51137360"
SOURCES = ROOT / "configs/genetic_primus_future_ema_cache_sources_v1.json"
DESTINATION = ROOT / "Reports/Exia/Genetic_Primus/future_ema_offline_bundle_v1_20261003"
MAX_UNCOMPRESSED = 512 * 1024 * 1024


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def wheel_bounds(path):
    size = path.stat().st_size
    if not 22 <= size <= MAX_BODY_BYTES:
        raise ValueError("source outside wheel size bound")
    with path.open("rb") as stream:
        stream.seek(max(0, size - 65557))
        tail = stream.read(65557)
        index = tail.rfind(b"PK\x05\x06")
        if index < 0 or len(tail) - index < 22:
            raise ValueError("missing ZIP end")
        _, disk, cd_disk, disk_count, count, cd_size, cd_offset, comment = struct.unpack(
            "<4s4H2LH", tail[index:index + 22])
        if disk or cd_disk or disk_count != count or count == 65535 or max(cd_size, cd_offset) == 0xffffffff:
            raise ValueError("unsupported multipart/ZIP64 wheel")
        end_position = size - len(tail) + index
        start, end = end_position - cd_size - cd_offset, end_position + 22 + comment
        if not 0 <= start < end <= size:
            raise ValueError("invalid ZIP boundaries")
        stream.seek(start)
        if stream.read(4) != b"PK\x03\x04":
            raise ValueError("invalid ZIP start")
    return start, end


def validate_payload(path, installed_root=None):
    """Check CRC/RECORD of all wheel files; optionally compare installed raw bytes.

    Does not execute archive members. The RECORD CSV itself is not self-hashed.
    """
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > 10000 or sum(m.file_size for m in infos) > MAX_UNCOMPRESSED:
            raise ValueError("archive exceeds validation budget")
        members = {}
        for item in infos:
            name = item.filename
            parts = PurePosixPath(name).parts
            if (not parts or PurePosixPath(name).is_absolute() or ".." in parts
                    or "\\" in name or ":" in name or item.flag_bits & 1
                    or stat.S_ISLNK(item.external_attr >> 16) or name in members):
                raise ValueError("unsafe or duplicate wheel member")
            members[name] = item
        records = [n for n in members if n.endswith(".dist-info/RECORD")]
        if len(records) != 1 or members[records[0]].file_size > 1024 * 1024:
            raise ValueError("missing or oversized wheel RECORD")
        record_name = records[0]
        rows = csv.reader(io.StringIO(archive.read(record_name).decode("utf-8")))
        seen, checked, total = set(), 0, 0
        for row in rows:
            if len(row) != 3:
                raise ValueError("invalid RECORD row")
            name, digest, length = row
            if name in seen or name not in members:
                raise ValueError("duplicate or unknown RECORD member")
            seen.add(name)
            if name == record_name:
                if digest or length:
                    raise ValueError("RECORD must not hash itself")
                continue
            if not digest.startswith("sha256=") or not length.isdecimal():
                raise ValueError("unhashed wheel payload")
            item = members[name]
            if int(length) != item.file_size:
                raise ValueError("RECORD size mismatch")
            hasher = hashlib.sha256()
            count = 0
            with archive.open(item) as stream:
                while chunk := stream.read(65536):
                    hasher.update(chunk)
                    count += len(chunk)
            encoded = base64.urlsafe_b64encode(hasher.digest()).rstrip(b"=").decode("ascii")
            if count != int(length) or encoded != digest.removeprefix("sha256="):
                raise ValueError("RECORD payload hash mismatch")
            if installed_root is not None:
                installed = (Path(installed_root) / name).resolve()
                if not installed.is_relative_to(Path(installed_root).resolve()) or file_sha(installed) != hasher.hexdigest():
                    raise ValueError(f"installed payload mismatch: {name}")
            checked += 1
            total += count
        if any(not m.is_dir() and n not in seen for n, m in members.items()):
            raise ValueError("payload absent from RECORD")
    return {"checked_files": checked, "uncompressed_bytes": total,
            "record_payload_verified": True, "installed_payload_matches": installed_root is not None}


def copy_payload(source, destination):
    start, end = wheel_bounds(source)
    with source.open("rb") as original, destination.open("xb") as output:
        original.seek(start)
        remaining = end - start
        while remaining:
            chunk = original.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("source truncated during copy")
            output.write(chunk)
            remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", choices=("numpy", "pandas", "pyarrow", "python-dateutil", "six", "tzdata"))
    args = parser.parse_args()
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("use workspace Python")
    raw = ENVIRONMENT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ENVIRONMENT_SHA:
        raise ValueError("environment registration changed")
    expected = json.loads(raw)["snapshot"]["packages"]
    sources_raw = SOURCES.read_bytes()
    config = json.loads(sources_raw)
    if config["environment_sha256"] != ENVIRONMENT_SHA:
        raise ValueError("source registration lineage mismatch")
    item = config["sources"][args.package]
    source = Path(item["path"]).resolve()
    if not source.is_relative_to(Path("C:/Users/anton/AppData/Local/pip/cache").resolve()):
        raise ValueError("source outside registered cache")
    source_sha = file_sha(source)
    candidate = inspect_candidate(source, expected, legacy=item["legacy"])
    if candidate is None or candidate["name"] != args.package:
        raise ValueError("cache candidate no longer matches")
    installed_root = Path(metadata.distribution(args.package).locate_file(""))
    if installed_root.resolve() != (ROOT / ".venv/Lib/site-packages").resolve():
        raise ValueError("installed baseline outside workspace venv")
    validation = validate_payload(source, installed_root)
    filename = f"{args.package.replace('-', '_')}-{candidate['version']}-{candidate['compatible_tags'][0]}.whl"
    destination = (DESTINATION / "wheels" / filename).resolve()
    if (not DESTINATION.resolve().is_relative_to(ROOT.resolve())
            or not destination.is_relative_to(DESTINATION.resolve()) or destination.exists()):
        raise ValueError("unsafe or existing bundle destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    copy_payload(source, destination)
    copied = validate_payload(destination)
    if file_sha(source) != source_sha or copied["checked_files"] != validation["checked_files"]:
        raise ValueError("source changed or copied payload differs; preserve files for review")
    evidence = {"schema_version": "exia.genetic_primus.offline_wheel/1",
                "name": args.package, "version": candidate["version"],
                "source_path": source.as_posix(), "source_sha256": source_sha,
                "sources_config_sha256": hashlib.sha256(sources_raw).hexdigest(),
                "environment_sha256": ENVIRONMENT_SHA,
                "wheel_path": destination.relative_to(ROOT).as_posix(),
                "wheel_sha256": file_sha(destination), "wheel_size_bytes": destination.stat().st_size,
                **validation, "network_used": False, "packages_installed": False,
                "independent_pypi_authenticity_verified": False}
    record_path = DESTINATION / (args.package + ".json")
    with record_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(evidence, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"name": args.package, "status": "WHEEL_VERIFIED_AND_COPIED",
                      "checked_files": validation["checked_files"], "wheel_sha256": evidence["wheel_sha256"]}, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "BUNDLE_STEP_BLOCKED", "error_type": type(exc).__name__,
                          "detail": str(exc)[:160], "packages_installed": False}, separators=(",", ":")))
        raise SystemExit(1)
