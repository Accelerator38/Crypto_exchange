"""Offline regressions for per-origin commitment and the final-origin cutoff."""

import json
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime as RealDateTime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from exia.genetic_primus import prospective_manifest_v2 as v2


ROOT = Path(__file__).parents[1]
BASE = ROOT / "configs/genetic_primus_prospective_3d_v1.json"
FIXED_NOW = RealDateTime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


class FixedDateTime(RealDateTime):
    @classmethod
    def now(cls, tz=None):
        return cls.fromtimestamp(FIXED_NOW.timestamp(), tz or timezone.utc)


def contract():
    result = deepcopy(json.loads(BASE.read_text(encoding="utf-8")))
    result["frozen_at_utc"] = "2026-09-27T00:00:00Z"
    result["outer"].update(start_utc="2026-09-28T00:00:00Z",
                           end_exclusive_utc="2026-10-28T00:00:00Z")
    result["training"]["initial_history_start_utc"] = "2026-09-27T00:00:00Z"
    result["data_commitment"].update(
        manifest_version=2,
        origin_snapshot_commit_rule=v2.RULE,
        origin_commit_deadline_hours=72,
    )
    return result


def frame(start: RealDateTime, bars: int, symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame([
        {"timestamp": int((start + timedelta(hours=4 * i)).timestamp() * 1000),
         "symbol": symbol, "open": 100.0, "high": 101.0,
         "low": 99.0, "close": 100.0, "volume": 1.0}
        for i in range(bars) for symbol in symbols
    ])


def clock_at(moment: RealDateTime):
    class Clock(RealDateTime):
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(moment.timestamp(), tz or timezone.utc)
    return Clock


class PerOriginManifestTest(unittest.TestCase):
    def test_last_origin_allows_commit_after_full_outer_end(self):
        old = json.loads(BASE.read_text(encoding="utf-8"))
        old["data_commitment"].update(
            manifest_version=2, origin_snapshot_commit_rule=v2.RULE,
            origin_commit_deadline_hours=72)
        closed, deadline = v2.origin_commit_window(old, 9, FIXED_NOW)
        self.assertEqual(closed, RealDateTime(2026, 10, 1, tzinfo=timezone.utc))
        self.assertEqual(deadline, RealDateTime(2026, 10, 4, tzinfo=timezone.utc))

    def test_rejects_early_late_naive_and_invalid_policy(self):
        c = contract()
        with self.assertRaisesRegex(ValueError, "after close"):
            v2.origin_commit_window(c, 0, RealDateTime(2026, 9, 30, 23, tzinfo=timezone.utc))
        with self.assertRaisesRegex(ValueError, "after close"):
            v2.origin_commit_window(c, 0, RealDateTime(2026, 10, 4, tzinfo=timezone.utc))
        with self.assertRaisesRegex(ValueError, "aware UTC"):
            v2.origin_commit_window(c, 0, RealDateTime(2026, 10, 1, 9))
        c["data_commitment"]["origin_commit_deadline_hours"] = 0
        with self.assertRaisesRegex(ValueError, "1 to 72"):
            v2.origin_commit_window(c, 0, FIXED_NOW)

    def test_exclusive_commit_roundtrip_and_revision_detection(self):
        c = contract()
        with tempfile.TemporaryDirectory(prefix="primus-v2-") as dirname:
            root = Path(dirname)
            config = root / "contract.json"
            config.write_text(json.dumps(c), encoding="utf-8")
            parquet = root / "origin_00.parquet"
            data = frame(RealDateTime(2026, 9, 27, tzinfo=timezone.utc), 24, c["universe"])
            data.to_parquet(parquet, index=False)
            target = root / "manifest.json"
            with patch.object(v2, "datetime", FixedDateTime):
                manifest = v2.commit_origin_manifest_exclusive(
                    target, [parquet], root=root, contract_path=config,
                    contract=c, origin_index=0)
            self.assertEqual(manifest["schema_version"], v2.SCHEMA)
            self.assertFalse(manifest["data_already_revealed"])
            verified, panel = v2.load_verified_origin(
                target, root=root, contract_path=config, contract=c, origin_index=0)
            self.assertEqual(verified["snapshot_sha256"], manifest["snapshot_sha256"])
            self.assertEqual(len(panel), 192)
            started = root / "origin_00_started.json"
            v2.reserve_origin_evaluation(
                started, root / "origin_00_result.json", target,
                root=root, contract_path=config, contract=c, origin_index=0)
            marker = json.loads(started.read_text(encoding="utf-8"))
            self.assertEqual(marker["origin_index"], 0)
            self.assertEqual(marker["manifest_sha256"], v2.file_sha256(target))
            with self.assertRaisesRegex(ValueError, "already attempted"):
                v2.reserve_origin_evaluation(
                    started, root / "origin_00_result.json", target,
                    root=root, contract_path=config, contract=c, origin_index=0)
            with self.assertRaises(FileExistsError):
                # The exclusive target may never silently replace a commitment.
                target.open("x")
            data.loc[0, "close"] = 100.5
            data.to_parquet(parquet, index=False)
            with self.assertRaisesRegex(ValueError, "snapshot file changed"):
                v2.load_verified_origin(
                    target, root=root, contract_path=config, contract=c, origin_index=0)

    def test_rejects_future_bar_and_v1_manifest(self):
        c = contract()
        with tempfile.TemporaryDirectory(prefix="primus-v2-") as dirname:
            root = Path(dirname)
            config = root / "contract.json"
            config.write_text(json.dumps(c), encoding="utf-8")
            parquet = root / "future.parquet"
            frame(RealDateTime(2026, 9, 27, tzinfo=timezone.utc), 25,
                  c["universe"]).to_parquet(parquet, index=False)
            with patch.object(v2, "datetime", FixedDateTime):
                with self.assertRaisesRegex(ValueError, "no future bars"):
                    v2.build_origin_manifest(
                        [parquet], root=root, contract_path=config,
                        contract=c, origin_index=0)
            bad = frame(RealDateTime(2026, 9, 27, tzinfo=timezone.utc), 24, c["universe"])
            bad.loc[0, "high"] = 98.0
            bad.to_parquet(parquet, index=False)
            with patch.object(v2, "datetime", FixedDateTime):
                with self.assertRaisesRegex(ValueError, "OHLC range"):
                    v2.build_origin_manifest(
                        [parquet], root=root, contract_path=config,
                        contract=c, origin_index=0)
            old_manifest = root / "old.json"
            old_manifest.write_text(json.dumps({"schema_version": "exia.genetic_primus.prospective_data_manifest/1",
                                                "immutable": True, "origin_index": 0}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "wrong origin"):
                v2.load_verified_origin(
                    old_manifest, root=root, contract_path=config,
                    contract=c, origin_index=0)

    def test_append_only_chain_and_previous_hash(self):
        c = contract()
        c["frozen_at_utc"] = "2026-09-24T00:00:00Z"
        c["outer"].update(start_utc="2026-09-25T00:00:00Z",
                          end_exclusive_utc="2026-10-25T00:00:00Z")
        c["training"]["initial_history_start_utc"] = "2026-09-24T00:00:00Z"
        with tempfile.TemporaryDirectory(prefix="primus-v2-") as dirname:
            root = Path(dirname)
            config = root / "contract.json"
            config.write_text(json.dumps(c), encoding="utf-8")
            first, second = root / "first.parquet", root / "second.parquet"
            frame(RealDateTime(2026, 9, 24, tzinfo=timezone.utc), 24,
                  c["universe"]).to_parquet(first, index=False)
            frame(RealDateTime(2026, 9, 28, tzinfo=timezone.utc), 18,
                  c["universe"]).to_parquet(second, index=False)
            m0, m1 = root / "manifest0.json", root / "manifest1.json"
            with patch.object(v2, "datetime", clock_at(RealDateTime(2026, 9, 28, 9, tzinfo=timezone.utc))):
                v2.commit_origin_manifest_exclusive(
                    m0, [first], root=root, contract_path=config,
                    contract=c, origin_index=0)
            with patch.object(v2, "datetime", clock_at(FIXED_NOW)):
                v2.commit_origin_manifest_exclusive(
                    m1, [first, second], root=root, contract_path=config,
                    contract=c, origin_index=1, previous_manifest_path=m0)
            verified, panel = v2.load_verified_origin(
                m1, root=root, contract_path=config, contract=c,
                origin_index=1, previous_manifest_path=m0)
            self.assertEqual(verified["previous_manifest_sha256"], v2.file_sha256(m0))
            self.assertEqual(len(panel), 42 * len(c["universe"]))
            old = json.loads(m0.read_text(encoding="utf-8"))
            old["unrelated_edit"] = True
            m0.write_text(json.dumps(old), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "previous manifest hash mismatch"):
                v2.load_verified_origin(
                    m1, root=root, contract_path=config, contract=c,
                    origin_index=1, previous_manifest_path=m0)


if __name__ == "__main__":
    unittest.main()
