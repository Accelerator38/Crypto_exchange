"""Tests for conservative retrotest-derived memory priors."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from panteon_v2.app.migration import save_v2_snapshot
from panteon_v2.app.startup import _load_or_migrate_state
from panteon_v2.domain.types import Regime
from panteon_v2.memory import PerformanceMemory
from panteon_v2.memory.retro_prior import (
    build_retro_prior_snapshot_from_breakdown_files,
    merge_retro_prior_into_memory,
)


def _write_breakdown(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "actor_type",
        "label",
        "regime",
        "pnl_usd",
        "pnl_pct",
        "closed_trades",
        "wins",
        "losses",
        "win_rate_pct",
        "pnl_per_trade_usd",
        "entries",
        "signals",
        "bars_seen",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _state(snapshot: dict, label: str, regime: Regime) -> dict:
    return snapshot["state"][f"{label}|{regime.label}"]


class TestRetroPriorMemory(unittest.TestCase):
    def test_build_prior_scales_and_caps_retro_breakdown_rows(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "agent_regime_breakdown.csv"
            _write_breakdown(
                path,
                [
                    {
                        "actor_type": "agent",
                        "label": "StrongAgent",
                        "regime": "range_low_vol",
                        "pnl_usd": 100.0,
                        "pnl_pct": 10.0,
                        "closed_trades": 100,
                        "wins": 60,
                        "losses": 40,
                        "win_rate_pct": 60.0,
                        "pnl_per_trade_usd": 1.0,
                        "entries": 80,
                        "signals": 140,
                        "bars_seen": 1000,
                    },
                    {
                        "actor_type": "agent",
                        "label": "TinySample",
                        "regime": "bullish",
                        "pnl_usd": 5.0,
                        "pnl_pct": 50.0,
                        "closed_trades": 1,
                        "wins": 1,
                        "losses": 0,
                        "win_rate_pct": 100.0,
                        "pnl_per_trade_usd": 5.0,
                        "entries": 1,
                        "signals": 1,
                        "bars_seen": 10,
                    },
                ],
            )

            snapshot = build_retro_prior_snapshot_from_breakdown_files(
                [path],
                exchange="MEXC",
                source_run_id="unit-retro",
                prior_weight=0.25,
                max_prior_closed_trades=12,
                max_abs_prior_pnl_pct=3.0,
                min_closed_trades=5,
            )

        strong = _state(snapshot, "StrongAgent", Regime.RANGE_LOW_VOL)
        self.assertEqual(snapshot["exchange_scope"], "MEXC")
        self.assertEqual(strong["closed_trades"], 12)
        self.assertEqual(strong["wins"], 7)
        self.assertEqual(strong["losses"], 5)
        self.assertAlmostEqual(strong["pnl_pct"], 2.5)
        self.assertGreaterEqual(strong["signals"], strong["closed_trades"])
        self.assertNotIn("TinySample|bullish", snapshot["state"])
        self.assertEqual(snapshot["_retro_prior"]["source_run_id"], "unit-retro")

    def test_merge_prior_fills_missing_virtual_pairs_without_overwriting_live_memory(self):
        perf = PerformanceMemory(trade_fraction=1.0, exchange_scope="MEXC")
        perf.restore(
            {
                "trade_fraction": 1.0,
                "exchange_scope": "MEXC",
                "state": {
                    "StrongAgent|range_low_vol": {
                        "closed_trades": 3,
                        "entries": 3,
                        "signals": 3,
                        "wins": 1,
                        "losses": 2,
                        "pnl_pct": -1.2,
                    }
                },
            }
        )
        prior = {
            "trade_fraction": 1.0,
            "exchange_scope": "MEXC",
            "state": {
                "StrongAgent|range_low_vol": {
                    "closed_trades": 12,
                    "entries": 12,
                    "signals": 12,
                    "wins": 7,
                    "losses": 5,
                    "pnl_pct": 2.5,
                },
                "NewAgent|bearish": {
                    "closed_trades": 5,
                    "entries": 5,
                    "signals": 5,
                    "wins": 4,
                    "losses": 1,
                    "pnl_pct": 1.0,
                },
            },
        }

        report = merge_retro_prior_into_memory(perf, prior)

        self.assertEqual(report.applied_pairs, 1)
        self.assertEqual(report.skipped_existing_pairs, 1)
        self.assertAlmostEqual(
            perf.get("StrongAgent", Regime.RANGE_LOW_VOL).pnl_pct,
            -1.2,
        )
        self.assertAlmostEqual(perf.get("NewAgent", Regime.BEARISH).pnl_pct, 1.0)

    def test_startup_loads_mexc_retro_prior_next_to_snapshot_without_touching_real_perf(self):
        with tempfile.TemporaryDirectory() as td:
            snapshot_path = os.path.join(td, "mexc_snapshot.json")
            virtual = PerformanceMemory(trade_fraction=1.0, exchange_scope="MEXC")
            real = PerformanceMemory(trade_fraction=1.0, exchange_scope="MEXC")
            real.restore(
                {
                    "trade_fraction": 1.0,
                    "exchange_scope": "MEXC",
                    "state": {
                        "RealAgent|bearish": {
                            "closed_trades": 2,
                            "entries": 2,
                            "signals": 2,
                            "wins": 2,
                            "losses": 0,
                            "pnl_pct": 0.8,
                        }
                    },
                }
            )
            save_v2_snapshot(virtual, snapshot_path, real_perf=real)
            prior_path = Path(td) / "mexc_retro_prior_memory.json"
            prior_path.write_text(
                json.dumps(
                    {
                        "trade_fraction": 1.0,
                        "exchange_scope": "MEXC",
                        "state": {
                            "PriorAgent|range_low_vol": {
                                "closed_trades": 6,
                                "entries": 6,
                                "signals": 6,
                                "wins": 4,
                                "losses": 2,
                                "pnl_pct": 1.1,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            loaded_virtual = PerformanceMemory(trade_fraction=1.0, exchange_scope="MEXC")
            loaded_real = PerformanceMemory(trade_fraction=1.0, exchange_scope="MEXC")
            _load_or_migrate_state(
                perf=loaded_virtual,
                real_perf=loaded_real,
                snapshot_path=snapshot_path,
                migrate_from_v1=[],
                exchange_name="MEXC",
            )

        self.assertAlmostEqual(
            loaded_virtual.get("PriorAgent", Regime.RANGE_LOW_VOL).pnl_pct,
            1.1,
        )
        self.assertFalse(
            loaded_real.get("PriorAgent", Regime.RANGE_LOW_VOL).has_data
        )
        self.assertAlmostEqual(loaded_real.get("RealAgent", Regime.BEARISH).pnl_pct, 0.8)

