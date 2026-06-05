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
        self.assets = {"USDT": 120.5, "USDT_AVAIL": 118.0}
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

    def account_assets(self):
        return dict(self.assets)

    def place_order(self, symbol: str, side: int, vol: int, leverage: int = 2) -> dict:
        self.orders.append(("place_order", symbol, side, vol, leverage))
        return {
            "success": True,
            "order_id": f"MEXC-{side}-{vol}",
            "status": "filled",
            "avgPrice": 100.0,
            "amount": vol * self._get_contract_meta(symbol)["contractSize"],
            "fee": 0.001,
            "contracts": vol,
            "contractSize": self._get_contract_meta(symbol)["contractSize"],
        }


class FakeBitgetFuturesClient:
    BITGET_MIN_NOTIONAL_USDT = 5.10

    def __init__(self):
        self.orders = []
        self.assets = {"USDT": 40.25, "USDT_AVAIL": 39.0}
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

    def account_assets(self):
        return dict(self.assets)

    def place_order(self, symbol: str, side: int, vol: int, leverage: int = 2) -> dict:
        self.orders.append(("place_order", symbol, side, vol, leverage))
        return {
            "success": True,
            "order_id": f"BITGET-{side}-{vol}",
            "status": "filled",
            "avgPrice": 2000.0,
            "amount": vol * self._get_contract_meta(symbol)["amountStep"],
        }

    def close_all(self, symbol: str):
        self.orders.append(("close_all", symbol))
        return {
            "success": True,
            "order_id": f"BITGET-CLOSE-{symbol}",
            "status": "filled",
            "avgPrice": 2000.0,
            "amount": 0.04,
        }


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

    def test_open_bumps_to_mexc_contract_minimum(self):
        from panteon_v2.app.mexc_adapter import MexcExchangeAdapter

        client = FakeMexcFuturesClient()
        adapter = MexcExchangeAdapter(order_client=client, read_client=client, leverage=2)

        result = adapter.send_order(_signal(Action.FUT_LONG_FULL), qty=0.0004)

        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertEqual(client.orders[-1], ("place_order", "BTC", 1, 1, 2))
        self.assertEqual(result.exchange_order_id, "MEXC-1-1")
        self.assertAlmostEqual(result.trade.qty, 0.001)

    def test_ack_only_mexc_order_is_pending_until_fill_is_confirmed(self):
        from panteon_v2.app.mexc_adapter import MexcExchangeAdapter

        class AckOnlyClient(FakeMexcFuturesClient):
            def place_order(self, symbol: str, side: int, vol: int, leverage: int = 2) -> dict:
                self.orders.append(("place_order", symbol, side, vol, leverage))
                return {"success": True, "order_id": "ACK-ONLY"}

        client = AckOnlyClient()
        adapter = MexcExchangeAdapter(order_client=client, read_client=client, leverage=2)

        result = adapter.send_order(_signal(Action.FUT_LONG_FULL), qty=0.003)

        self.assertEqual(result.status, OrderStatus.PENDING)
        self.assertEqual(result.exchange_order_id, "ACK-ONLY")
        self.assertIsNone(result.trade)

    def test_mexc_ack_with_submitted_amount_is_pending_until_fill_is_confirmed(self):
        from panteon_v2.app.mexc_adapter import MexcExchangeAdapter

        class AckWithSubmittedAmountClient(FakeMexcFuturesClient):
            def place_order(self, symbol: str, side: int, vol: int, leverage: int = 2) -> dict:
                meta = self._get_contract_meta(symbol)
                self.orders.append(("place_order", symbol, side, vol, leverage))
                return {
                    "success": True,
                    "order_id": "MEXC-ACK-1",
                    "data": "MEXC-ACK-1",
                    "contracts": vol,
                    "contractSize": meta["contractSize"],
                    "amount": vol * meta["contractSize"],
                }

        client = AckWithSubmittedAmountClient()
        adapter = MexcExchangeAdapter(order_client=client, read_client=client, leverage=2)

        result = adapter.send_order(_signal(Action.FUT_LONG_FULL), qty=0.003)

        self.assertEqual(result.status, OrderStatus.PENDING)
        self.assertEqual(result.exchange_order_id, "MEXC-ACK-1")
        self.assertIsNone(result.trade)

    def test_mexc_open_fails_closed_on_fallback_contract_metadata(self):
        from panteon_v2.app.mexc_adapter import MexcExchangeAdapter

        class FallbackMetaClient(FakeMexcFuturesClient):
            def _get_contract_meta(self, symbol: str) -> dict:
                meta = super()._get_contract_meta(symbol)
                meta["metadataFallback"] = True
                meta["metadataSource"] = "fallback"
                return meta

        client = FallbackMetaClient()
        adapter = MexcExchangeAdapter(order_client=client, read_client=client, leverage=2)

        result = adapter.send_order(_signal(Action.FUT_LONG_FULL), qty=0.003)

        self.assertEqual(result.status, OrderStatus.REJECTED)
        self.assertIn("metadata", result.message.lower())
        self.assertFalse(any(order[0] == "place_order" for order in client.orders))

    def test_mexc_close_all_allows_fallback_contract_metadata(self):
        from panteon_v2.app.mexc_adapter import MexcExchangeAdapter

        class FallbackMetaClient(FakeMexcFuturesClient):
            def _get_contract_meta(self, symbol: str) -> dict:
                meta = super()._get_contract_meta(symbol)
                meta["metadataFallback"] = True
                meta["metadataSource"] = "fallback"
                return meta

        client = FallbackMetaClient()
        adapter = MexcExchangeAdapter(order_client=client, read_client=client, leverage=2)

        result = adapter.send_order(_signal(Action.FUT_CLOSE_ALL), qty=0.002)

        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertEqual(client.orders[-1], ("place_order", "BTC", 4, 2, 2))

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

    def test_reads_mexc_account_equity(self):
        from panteon_v2.app.mexc_adapter import MexcExchangeAdapter

        client = FakeMexcFuturesClient()
        adapter = MexcExchangeAdapter(order_client=client, read_client=client, leverage=2)

        self.assertAlmostEqual(adapter.get_account_equity(), 120.5)

    def test_empty_positions_with_recent_data_error_are_not_reliable(self):
        from panteon_v2.app.v1_futures_adapter import V1FuturesExchangeAdapter

        class ReadClient:
            def get_futures_positions(self):
                return []

            def recent_data_error(self, max_age_sec=120.0):
                return "cached_futures_assets"

        class OrderClient:
            def account_assets(self):
                return {"USDT": 120.5, "USDT_AVAIL": 118.0}

            def _get_contract_meta(self, symbol: str) -> dict:
                return {
                    "symbol": f"{symbol}_USDT",
                    "contractSize": 0.001,
                    "minVol": 1,
                    "volUnit": 1,
                }

        adapter = V1FuturesExchangeAdapter(
            name="MEXC",
            order_client=OrderClient(),
            read_client=ReadClient(),
            leverage=2,
        )

        self.assertEqual(adapter.get_all_positions(), {})
        self.assertFalse(adapter.positions_snapshot_reliable())
        self.assertEqual(adapter.positions_snapshot_error(), "cached_futures_assets")

    def test_zero_balance_full_snapshot_preserves_data_health(self):
        from panteon_v2.app.v1_futures_adapter import V1FuturesExchangeAdapter

        class ReadClient:
            def get_full_snapshot(self):
                return {
                    "futures": {
                        "equity": 0.0,
                        "available": 0.0,
                        "unrealized": 0.0,
                        "positions": [],
                    },
                    "spot": {"total_value": 0.0},
                    "total_equity": 0.0,
                    "data_health": {
                        "last_data_error_reason": "cached_futures_assets",
                        "snapshot_healthy": False,
                    },
                }

        class OrderClient:
            def account_assets(self):
                return {}

            def _get_contract_meta(self, symbol: str) -> dict:
                return {
                    "symbol": f"{symbol}_USDT",
                    "contractSize": 0.001,
                    "minVol": 1,
                    "volUnit": 1,
                }

        adapter = V1FuturesExchangeAdapter(
            name="MEXC",
            order_client=OrderClient(),
            read_client=ReadClient(),
            leverage=2,
        )

        snapshot = adapter.get_account_snapshot()

        self.assertEqual(snapshot["current_balance"], 0.0)
        self.assertEqual(
            snapshot["data_health"]["last_data_error_reason"],
            "cached_futures_assets",
        )


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

    def test_bitget_open_ack_status_with_submitted_amount_is_pending(self):
        from panteon_v2.app.bitget_adapter import BitgetExchangeAdapter

        class AckOpenClient(FakeBitgetFuturesClient):
            def place_order(self, symbol: str, side: int, vol: int, leverage: int = 2) -> dict:
                self.orders.append(("place_order", symbol, side, vol, leverage))
                return {
                    "success": True,
                    "order_id": "BITGET-ACK-1",
                    "status": "open",
                    "amount": vol * self._get_contract_meta(symbol)["amountStep"],
                }

        client = AckOpenClient()
        adapter = BitgetExchangeAdapter(order_client=client, read_client=client, leverage=2)

        result = adapter.send_order(_signal(Action.FUT_LONG_FULL, sym="ETH", price=2000.0), qty=0.03)

        self.assertEqual(result.status, OrderStatus.PENDING)
        self.assertEqual(result.exchange_order_id, "BITGET-ACK-1")
        self.assertIsNone(result.trade)

    def test_reads_bitget_position_shape_and_min_notional(self):
        from panteon_v2.app.bitget_adapter import BitgetExchangeAdapter

        client = FakeBitgetFuturesClient()
        adapter = BitgetExchangeAdapter(order_client=client, read_client=client, leverage=2)

        pos = adapter.get_position("ETH")

        self.assertEqual(pos.sym, "ETH")
        self.assertEqual(pos.side, "short")
        self.assertAlmostEqual(pos.qty, 0.04)
        self.assertEqual(adapter.get_min_notional("ETH"), 5.10)

    def test_reads_bitget_account_equity(self):
        from panteon_v2.app.bitget_adapter import BitgetExchangeAdapter

        client = FakeBitgetFuturesClient()
        adapter = BitgetExchangeAdapter(order_client=client, read_client=client, leverage=2)

        self.assertAlmostEqual(adapter.get_account_equity(), 40.25)


if __name__ == "__main__":
    unittest.main()
