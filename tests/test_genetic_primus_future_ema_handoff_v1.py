"""Tiny synthetic archive regression suite; no installs, models or network."""

import hashlib
import json
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.package_genetic_primus_future_ema_handoff_v1 import (
    INDEX, OBJECTIVE, create_archive, extract_archive, file_sha, safe_name, verify_archive,
)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self):
        source = self.root / "source.py"
        with source.open("xb") as stream:
            stream.write(b"raise RuntimeError('archive content must never execute')\n")
        return source

    def archive(self, *, member="payload.txt", bad_pin=False, symlink=False, bad_index=None):
        path = self.root / "crafted.zip"
        data = b"synthetic-only"
        index = {"schema_version": "exia.genetic_primus.package_handoff/1", "objective_id": OBJECTIVE,
                 "kind": "package_bundle_not_model_release", "promotion_authority": False,
                 "members": {member: {"sha256": "0" * 64 if bad_pin else hashlib.sha256(data).hexdigest(),
                                     "size_bytes": len(data)}}}
        with zipfile.ZipFile(path, "x") as archive:
            if symlink:
                info = zipfile.ZipInfo(member)
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, data)
            else:
                archive.writestr(member, data)
            archive.writestr(INDEX, bad_index if bad_index is not None else json.dumps(index))
        return path

    def test_roundtrip_does_not_execute_or_overwrite(self):
        source = self.source()
        archive = self.root / "valid.zip"
        result = create_archive(archive, {"tools/payload.py": (source, file_sha(source))})
        self.assertFalse(result["contents_executed"])
        self.assertEqual(result["members_verified"], 1)
        destination = self.root / "restore"
        result = extract_archive(archive, file_sha(archive), destination)
        self.assertEqual((destination / "tools/payload.py").read_bytes(), source.read_bytes())
        self.assertFalse(result["independent_machine_verified"])
        with self.assertRaises(FileExistsError):
            extract_archive(archive, file_sha(archive), destination)
        with self.assertRaises(FileExistsError):
            create_archive(archive, {"tools/payload.py": (source, file_sha(source))})

    def test_changed_source_rejected_before_archive_creation(self):
        source = self.source()
        path = self.root / "not-created.zip"
        with self.assertRaisesRegex(ValueError, "source SHA"):
            create_archive(path, {"payload.py": (source, "0" * 64)})
        self.assertFalse(path.exists())

    def test_wrong_archive_sha_rejected_before_extraction(self):
        path = self.archive()
        destination = self.root / "not-created"
        with self.assertRaisesRegex(ValueError, "archive SHA"):
            extract_archive(path, "0" * 64, destination)
        self.assertFalse(destination.exists())

    def test_member_hash_mismatch_rejected_before_extraction(self):
        path = self.archive(bad_pin=True)
        destination = self.root / "not-created"
        with self.assertRaisesRegex(ValueError, "member SHA"):
            extract_archive(path, file_sha(path), destination)
        self.assertFalse(destination.exists())

    def test_traversal_and_windows_names_rejected(self):
        for name in ("../outside", "/outside", "C:/outside", "x\\y", "CON.txt", "x/../y", "a.", "a//b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_name(name)
        path = self.archive(member="../outside")
        with self.assertRaises(ValueError):
            verify_archive(path, file_sha(path))
        self.assertFalse((self.root.parent / "outside").exists())

    def test_symlink_rejected(self):
        path = self.archive(symlink=True)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            verify_archive(path, file_sha(path))

    def test_malformed_index_rejected(self):
        path = self.archive(bad_index=b'{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            verify_archive(path, file_sha(path))

    def test_central_directory_budget_rejected(self):
        path = self.root / "too-many.zip"
        with zipfile.ZipFile(path, "x") as archive:
            for i in range(65):
                archive.writestr(str(i), b"x")
        with self.assertRaisesRegex(ValueError, "central directory"):
            verify_archive(path, file_sha(path))


if __name__ == "__main__":
    unittest.main()
