"""Bounded read-only pip HTTP-cache audit; no download, install or wheel execution."""

import argparse
import email.parser
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / "configs/genetic_primus_future_ema_environment_v1.json"
ENVIRONMENT_SHA = "8f82114b946f09afcd4aa8b6e5dcd721fb7c2f303a09f1c2dac3a0fc51137360"
MAX_BODY_BYTES = 128 * 1024 * 1024
MAX_METADATA_BYTES = 128 * 1024
COMPATIBLE_TAGS = {"cp312-cp312-win_amd64", "cp312-none-win_amd64", "py3-none-any"}


def normalized(name):
    return name.lower().replace("_", "-").replace(".", "-")


def inspect_candidate(path: Path, expected: dict, *, legacy=False) -> dict | None:
    if path.is_symlink() or not 4 <= path.stat().st_size <= MAX_BODY_BYTES:
        return None
    with path.open("rb") as stream:
        header = stream.read(256 if legacy else 4)
        if (not legacy and header != b"PK\x03\x04") or (legacy and (
                not header.startswith(b"cc=4,") or b"PK\x03\x04" not in header)):
            return None
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) > 10000:
            raise ValueError("too many archive members")
        members = [m for m in archive.infolist() if m.filename.endswith(".dist-info/METADATA")]
        if len(members) != 1 or members[0].file_size > MAX_METADATA_BYTES:
            return None
        with archive.open(members[0]) as stream:
            raw = stream.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES:
            return None
        meta = email.parser.BytesParser().parsebytes(raw)
        name = normalized(meta.get("Name", ""))
        version = meta.get("Version")
        pin = expected.get(name)
        if pin is None or pin["version"] != version:
            return None
        text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != pin["metadata_sha256"]:
            return None
        wheel_name = members[0].filename.removesuffix("METADATA") + "WHEEL"
        wheel = archive.getinfo(wheel_name)
        if wheel.file_size > 16384:
            return None
        with archive.open(wheel) as stream:
            wheel_raw = stream.read(16385)
        if len(wheel_raw) > 16384:
            return None
        headers = email.parser.BytesParser().parsebytes(wheel_raw)
        tags = headers.get_all("Tag", [])
        if not any(tag in COMPATIBLE_TAGS for tag in tags):
            return None
        return {"name": name, "version": version, "source": path.as_posix(),
                "compatible_tags": [tag for tag in tags if tag in COMPATIBLE_TAGS],
                "metadata_matches_installed": True, "payload_integrity_verified": False}


def audit_cache(directory: Path, expected: dict, *, max_bodies=160, max_directories=8000,
                offset=0, legacy=False) -> dict:
    if not 1 <= max_bodies <= 200 or not 1 <= max_directories <= 10000 or not 0 <= offset <= 10000:
        raise ValueError("scan budget outside safe bounds")
    candidates, errors, bodies, directories, capped, skipped = {}, 0, 0, 0, False, 0
    exists = directory.is_dir()
    for current, dirs, files in os.walk(directory, followlinks=False):
        directories += 1
        if directories > max_directories:
            capped = True
            break
        dirs[:] = sorted(d for d in dirs if not (Path(current) / d).is_symlink())
        for filename in sorted(files):
            if not legacy and not filename.endswith(".body"):
                continue
            if skipped < offset:
                skipped += 1
                continue
            if bodies >= max_bodies:
                capped = True
                break
            bodies += 1
            try:
                candidate = inspect_candidate(Path(current) / filename, expected, legacy=legacy)
                if candidate:
                    candidates.setdefault(candidate["name"], candidate)
            except (OSError, ValueError, KeyError, UnicodeError, zipfile.BadZipFile, RuntimeError):
                errors += 1
        if capped:
            break
    required = sorted(name for name, value in expected.items() if value is not None)
    return {"status": "CANDIDATES_FOUND_NOT_VERIFIED" if all(n in candidates for n in required)
            else "OFFLINE_BUNDLE_INCOMPLETE", "cache_exists": exists, "bodies_examined": bodies,
            "directories_examined": min(directories, max_directories), "scan_capped": capped,
            "offset": offset, "bodies_skipped": skipped,
            "cache_format": "legacy_cc4_prefixed_zip" if legacy else "http-v2-body",
            "candidate_errors": errors, "candidates": candidates,
            "missing": [n for n in required if n not in candidates],
            "network_used": False, "packages_installed": False, "payload_integrity_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--legacy", action="store_true",
                        help="inspect ZIP bodies embedded in cc=4 cache; never deserialize cache")
    parser.add_argument("--candidate-source", help="show provenance for one matched distribution only")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("use workspace Python")
    raw = ENVIRONMENT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ENVIRONMENT_SHA:
        raise ValueError("environment registration changed")
    expected = json.loads(raw)["snapshot"]["packages"]
    result = audit_cache(args.cache, expected, offset=args.offset, legacy=args.legacy)
    if args.candidate_source:
        candidate = result["candidates"].get(args.candidate_source)
        print(json.dumps({"candidate": candidate, "payload_integrity_verified": False}, separators=(",", ":")))
        return 0 if candidate else 1
    result["candidates"] = {name: item["version"] for name, item in result["candidates"].items()}
    print(json.dumps(result, separators=(",", ":")))
    return 0 if not result["missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
