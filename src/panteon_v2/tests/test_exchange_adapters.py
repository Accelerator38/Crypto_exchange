"""Contract tests for live v1 exchange adapters."""

from __future__ import annotations

import unittest

from panteon_v2.domain import Action, Regime, Signal
from panteon_v2.execution import Exchange, OrderStatus


def _signal(action: Action, *, sym: str = "BTC", price: float = 100.0) -> Signal:
    return Signal(
        id=42,
        bar=7,
        sym=sym,
        action=action,
        price=price,
        regime=Regime.NEUTRAL,
        by_player="TestPlayer",
    )


class FakeMexcFuturesClient:
    def __init__(self):
        self.orders = []
        self.positions = [
            {
                "symbol": "BTC_USDT",
                "positionType": 1,
                "holdVol": 2,
                "openAvgPrice": 95.0,
                "leverage": 2,
                "unrealizedPnl": 1.5,
            }
        ]

    def _get_contract_meta(self, symbol: str) -> dict:
        return {
            "symbol": f"{symbol}_USDT",
            "contractSize": 0.001,
            "minVol": 1,
            "volUnit": 1,
            "takerFeeRate": 0.0002,
        }

    def _req(self, method: str, path: str):
        self.orders.append(("req", method, path))
        return {"data": list(self.positions)}

    def place_order(self, symbol: str, side: int, vol: int, leverage: int = 2) -> dict:
        self.orders.append(("place_order", symbol, side, vol, leverage))
        return {
            "success": True,
            "order_id": f"MEXC-{side}-{vol}",
            "amount": vol * self._get_contract_meta(symbol)["contractSize"],
            "fee": 0.001,
            "contracts": vol,
            "contractSize": self._get_contract_meta(symbol)["contractSize"],
        }


class FakeBitgetFuturesClient:
    BITGET_MIN_NOTIONAL_USDT = 5.10

    def __init__(self):
        self.orders = []
        self.positions = [
            {
                "symbol": "ETH",
                "side": "short",
                "qty": 0.04,
                "entry": 2100.0,
                "leverage": 2,
                "unrealized_pnl": -0.5,
            }
        ]

    def _get_contract_meta(self, symbol: str) -> dict:
        return {
            "symbol": f"{symbol}/USDT:USDT",
            "amountStep": 0.01,
            "minVol": 1,
            "volUnit": 1,
        }

    def get_futures_positions(self):
        return list(self.positions)

    def place_order(self, symbol: str, side: int, vol: int, leverage: int = 2) -> dict:
        self.orders.append(("place_order", symbol, side, vol, leverage))
        return {
            "success": True,
            "order_id": f"BITGET-{side}-{vol}",
            "amount": vol * self._get_contract_meta(symbol)["amountStep"],
        }

    def close_all(self, symbol: str):
        self.orders.append(("close_all", symbol))
        return {"success": True, "order_id": f"BITGET-CLOSE-{symbol}"}


class TestMexcExchangeAdapter(unittest.TestCase):
    def test_open_short_maps_to_mexc_futures_order(self):
        from panteon_v2.app.mexc_adapter import MexcExchangeAdapter

        client = FakeMexcFuturesClient()
        adapter = MexcExchangeAdapter(order_client=client, read_client=client, leverage=2)

        self.assertIsInstance(adapter, Exchange)
        result = adapter.send_order(_signal(Action.FUT_SHORT_FULL), qty=0.003)

        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertEqual(client.orders[-1], ("place_order", "BTC", 3, 3, 2))
        self.assertEqual(result.exchange_order_id, "MEXC-3-3")
        self.assertEqual(result.trade.side, "short")
        self.assertAlmostEqual(result.trade.qty, 0.003)

    def test_close_long_maps_to_mexc_close_side(self):
        from panteon_v2.app.mexc_adapter import MexcExchangeAdapter

        client = FakeMexcFuturesClient()
        adapter = MexcExchangeAdapter(order_client=client, read_client=client, leverage=2)

        result = adapter.send_order(_signal(Action.FUT_CLOSE_ALL), qty=0.002)

        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertEqual(client.orders[-1], ("place_order", "BTC", 4, 2, 2))
        self.assertEqual(result.trade.side, "long")

    def test_reads_mexc_position_shape(self):
        from panteon_v2.app.mexc_adapter import MexcExchangeAdapter

        client = FakeMexcFuturesClient()
        adapter = MexcExchangeAdapter(order_client=client, read_client=client, leverage=2)

        pos = adapter.get_position("BTC")

        self.assertEqual(pos.sym, "BTC")
        self.assertEqual(pos.side, "long")
        self.assertAlmostEqual(pos.qty, 0.002)
        self.assertEqual(adapter.get_min_notional("BTC"), 5.0)


class TestBitgetExchangeAdapter(unittest.TestCase):
    def test_open_long_maps_to_bitget_futures_order(self):
        from panteon_v2.app.bitget_adapter import BitgetExchangeAdapter

        client = FakeBitgetFuturesClient()
        adapter = BitgetExchangeAdapter(order_client=client, read_client=client, leverage=2)

        self.assertIsInstance(adapter, Exchange)
        result = adapter.send_order(_signal(Action.FUT_LONG_FULL, sym="ETH", price=2000.0), qty=0.03)

        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertEqual(client.orders[-1], ("place_order", "ETH", 1, 3, 2))
        self.assertEqual(result.exchange_order_id, "BITGET-1-3")
        self.assertEqual(result.trade.side, "long")
        self.assertAlmostEqual(result.trade.qty, 0.03)

    def test_close_uses_bitget_close_all(self):
        from panteon_v2.app.bitget_adapter import BitgetExchangeAdapter

        client = FakeBitgetFuturesClient()
        adapter = BitgetExchangeAdapter(order_client=client, read_client=client, leverage=2)

        result = adapter.send_order(_signal(Action.FUT_CLOSE_ALL, sym="ETH", price=2000.0), qty=0.04)

        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertEqual(client.orders[-1], ("close_all", "ETH"))
        self.assertEqual(result.exchange_order_id, "BITGET-CLOSE-ETH")
        self.assertEqual(result.trade.side, "short")

    def test_reads_bitget_position_shape_and_min_notional(self):
        from panteon_v2.app.bitget_adapter import BitgetExchangeAdapter

        client = FakeBitgetFuturesClient()
        adapter = BitgetExchangeAdapter(order_client=client, read_client=client, leverage=2)

        pos = adapter.get_position("ETH")

        self.assertEqual(pos.sym, "ETH")
        self.assertEqual(pos.side, "short")
        self.assertAlmostEqual(pos.qty, 0.04)
        self.assertEqual(adapter.get_min_notional("ETH"), 5.10)


if __name__ == "__main__":
    unittest.main()
