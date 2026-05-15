import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src" / "panteon_runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))


class VirtualPanteonMirrorTests(unittest.TestCase):
    def test_name_detection_accepts_internal_and_display_labels(self):
        from virtual_panteon import is_virtual_panteon_name

        self.assertTrue(is_virtual_panteon_name("V_Virtual_Panteon"))
        self.assertTrue(is_virtual_panteon_name("Virtual_Panteon"))
        self.assertFalse(is_virtual_panteon_name("V_Panteon_shadow"))

    def test_mirror_replays_source_actions_only_for_matching_bar(self):
        from virtual_panteon import VirtualPanteonMirror

        class Source:
            _last_actions_bar = 42
            _last_actions = {"BTC": 1, "ETH": 0}
            _last_risk_multipliers = {"BTC": 1.75}
            _last_regime = "bullish"

        mirror = VirtualPanteonMirror(Source())

        self.assertEqual(
            mirror.act(
                prices={"BTC": 100.0, "ETH": 10.0},
                volumes={},
                month=5,
                portfolio_value=50.0,
                bar_index=42,
            ),
            {"BTC": 1, "ETH": 0},
        )
        self.assertEqual(mirror._last_risk_multipliers, {"BTC": 1.75})
        self.assertEqual(mirror._last_regime, "bullish")

        self.assertEqual(
            mirror.act(
                prices={"BTC": 100.0, "ETH": 10.0},
                volumes={},
                month=5,
                portfolio_value=50.0,
                bar_index=43,
            ),
            {},
        )


if __name__ == "__main__":
    unittest.main()
