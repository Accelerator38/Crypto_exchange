"""Synthetic archive checks; does not import or install wheel payloads."""

import base64
import csv
import hashlib
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_genetic_primus_future_ema_offline_bundle_v1 import (
    copy_payload, validate_payload, wheel_bounds,
)
from tools.audit_genetic_primus_future_ema_wheel_cache_v1 import audit_cache, inspect_candidate


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def wheel(self, path, *, prefix=b"", suffix=b"", bad_digest=False, unsafe=False):
        metadata = b"Name: six\nVersion: 1.17.0\n\n"
        files = {"six.py": b"# synthetic, never imported\n",
                 "six-1.17.0.dist-info/METADATA": metadata,
                 "six-1.17.0.dist-info/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n\n"}
        if unsafe:
            files["../outside.py"] = b"not executable evidence"
        csv_text = io.StringIO(newline="")
        writer = csv.writer(csv_text)
        for name, value in files.items():
            digest = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode()
            writer.writerow([name, "sha256=" + ("wrong" if bad_digest else digest), len(value)])
        writer.writerow(["six-1.17.0.dist-info/RECORD", "", ""])
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, value in files.items():
                archive.writestr(name, value)
            archive.writestr("six-1.17.0.dist-info/RECORD", csv_text.getvalue())
        with path.open("xb") as stream:
            stream.write(prefix + data.getvalue() + suffix)
        return files, {"six": {"version": "1.17.0",
                               "metadata_sha256": hashlib.sha256(metadata).hexdigest()}}

    def test_legacy_copy_removes_only_cache_envelope(self):
        source, target = self.root / "cache", self.root / "six.whl"
        self.wheel(source, prefix=b"cc=4,synthetic-envelope:", suffix=b"synthetic-trailer")
        start, end = wheel_bounds(source)
        self.assertGreater(start, 0)
        self.assertLess(end, source.stat().st_size)
        self.assertEqual(validate_payload(source)["checked_files"], 3)
        copy_payload(source, target)
        self.assertEqual(wheel_bounds(target), (0, target.stat().st_size))
        self.assertTrue(validate_payload(target)["record_payload_verified"])
        with self.assertRaises(FileExistsError):
            copy_payload(source, target)

    def test_record_and_unsafe_paths_are_rejected(self):
        for name, kwargs in (("bad", {"bad_digest": True}), ("unsafe", {"unsafe": True})):
            path = self.root / name
            self.wheel(path, **kwargs)
            with self.assertRaises(ValueError):
                validate_payload(path)

    def test_installed_comparison_detects_changed_payload(self):
        path = self.root / "valid.whl"
        files, _ = self.wheel(path)
        installed = self.root / "installed"
        for name, value in files.items():
            destination = installed / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(value)
        self.assertTrue(validate_payload(path, installed)["installed_payload_matches"])
        with (installed / "six.py").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "installed payload mismatch"):
            validate_payload(path, installed)

    def test_metadata_and_compatibility_filters(self):
        path = self.root / "cache.body"
        _, expected = self.wheel(path)
        self.assertEqual(inspect_candidate(path, expected)["name"], "six")
        expected["six"]["version"] = "wrong"
        self.assertIsNone(inspect_candidate(path, expected))

    def test_scan_budget_and_offset_are_bounded(self):
        expected = {}
        for i in range(3):
            _, expected = self.wheel(self.root / f"{i}.body")
        report = audit_cache(self.root, expected, max_bodies=1, offset=1)
        self.assertEqual(report["bodies_examined"], 1)
        self.assertEqual(report["bodies_skipped"], 1)
        self.assertTrue(report["scan_capped"])
        self.assertFalse(report["packages_installed"])
        with self.assertRaises(ValueError):
            audit_cache(self.root, expected, max_bodies=201)

    def test_legacy_header_requires_expected_format(self):
        path = self.root / "legacy"
        _, expected = self.wheel(path, prefix=b"cc=4,synthetic-envelope:", suffix=b"tail")
        self.assertIsNotNone(inspect_candidate(path, expected, legacy=True))
        self.assertIsNone(inspect_candidate(path, expected, legacy=False))


if __name__ == "__main__":
    unittest.main()
