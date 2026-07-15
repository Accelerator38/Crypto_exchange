from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "src" / "panteon_runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))


def test_bitget_funding_fetcher_converts_open_interest_to_usdt():
    from bitget_funding import BitgetFundingDataFetcher

    class Exchange:
        def __init__(self):
            self.loaded = False

        def load_markets(self):
            self.loaded = True

        def fetch_ticker(self, market):
            assert market == "BTC/USDT:USDT"
            return {
                "last": 100.0,
                "markPrice": 101.0,
                "quoteVolume": 5000.0,
                "baseVolume": 50.0,
                "info": {"holdingAmount": "3", "fundingRate": "0.0001"},
            }

        def fetch_funding_rate(self, market):
            assert market == "BTC/USDT:USDT"
            return {
                "fundingRate": 0.0002,
                "fundingTimestamp": 4_000_000_000_000,
            }

        def fetch_open_interest(self, market):
            assert market == "BTC/USDT:USDT"
            return {"openInterestAmount": 2.0, "openInterestValue": None}

        def publicMixGetV2MixMarketAccountLongShort(self, params):
            assert params == {"symbol": "BTCUSDT", "period": "5m"}
            return {
                "data": [
                    {
                        "longAccountRatio": "0.65",
                        "shortAccountRatio": "0.35",
                        "ts": "1714020600000",
                    }
                ]
            }

    exchange = Exchange()
    fetcher = BitgetFundingDataFetcher(symbols=["BTC"], exchange=exchange)

    cache = fetcher.fetch_all()

    assert exchange.loaded is True
    assert cache["BTC"]["funding_rate"] == 0.0002
    assert cache["BTC"]["open_interest_base"] == 2.0
    assert cache["BTC"]["open_interest_usdt"] == 202.0
    assert cache["BTC"]["long_ratio"] == 0.65
    assert cache["BTC"]["short_ratio"] == 0.35
    assert cache["BTC"]["long_short_source"] == "account_long_short_v2"
    assert cache["BTC"]["context_complete"] is False  # index price absent in fixture
    assert fetcher.get("BTC")["open_interest_usdt"] == 202.0
    assert fetcher.get_global()["total_oi_usdt"] == 202.0


def test_bitget_funding_fetcher_falls_back_to_v3_account_ratios(monkeypatch):
    import bitget_funding
    from bitget_funding import BitgetFundingDataFetcher

    monkeypatch.setattr(bitget_funding.time, "time", lambda: 1_783_858_800.0)

    class Exchange:
        def load_markets(self):
            return None

        def fetch_ticker(self, _market):
            return {"last": 100.0, "markPrice": 101.0}

        def fetch_funding_rate(self, _market):
            return {
                "fundingRate": 0.0002,
                "markPrice": 101.0,
                "indexPrice": 100.0,
            }

        def fetch_open_interest(self, _market):
            return {"openInterestAmount": 2.0}

        def publicMixGetV2MixMarketAccountLongShort(self, _params):
            raise RuntimeError("40054 empty")

        def request(self, path, api, method, params):
            assert path == "v3/market/futures-account-long-short"
            assert api == ["public", "common"]
            assert method == "GET"
            assert params == {"symbol": "BNBUSDT", "period": "5m"}
            return {
                "data": [
                    {
                        "longAccountRatio": "0.69",
                        "shortAccountRatio": "0.31",
                        "ts": "1783858800000",
                    }
                ]
            }

    fetcher = BitgetFundingDataFetcher(symbols=["BNB"], exchange=Exchange())
    cache = fetcher.fetch_all()

    assert cache["BNB"]["long_ratio"] == 0.69
    assert cache["BNB"]["short_ratio"] == 0.31
    assert cache["BNB"]["long_short_source"] == "futures_account_long_short_v3"
    assert cache["BNB"]["long_short_ratio_age_sec"] == 0.0
    assert cache["BNB"]["context_complete"] is True
    assert fetcher.get_global()["ratio_source_counts"] == {
        "futures_account_long_short_v3": 1
    }


def test_bitget_bridge_registers_funding_fetcher_before_market_fetch(monkeypatch, tmp_path):
    import bitget_connector

    class FakeSpotExchange:
        def load_markets(self):
            return None

        def fetch_tickers(self, symbols):
            return {
                symbol: {"last": 100.0, "quoteVolume": 10_000.0}
                for symbol in symbols
            }

    class FakeFuturesClient:
        def __init__(self, api_key, api_secret, api_passphrase):
            self.exchange = SimpleNamespace(
                markets={
                    "BTC/USDT:USDT": {},
                    "ETH/USDT:USDT": {},
                }
            )
            self._bad_symbols = set()

        def _market_symbol(self, symbol):
            return f"{symbol}/USDT:USDT"

        def ticker_price(self, symbol):
            return 100.0

    class FakeFetcher:
        instances = []

        def __init__(self, symbols):
            self.symbols = list(symbols)
            self.started = False
            FakeFetcher.instances.append(self)

        def fetch_all(self):
            return {
                symbol: {"open_interest_usdt": 1000.0}
                for symbol in self.symbols
            }

        def start(self):
            self.started = True

    registered = []
    monkeypatch.setattr(bitget_connector, "_BITGET_SETTINGS_RAW", {})
    monkeypatch.setattr(bitget_connector, "_public_client", lambda default_type="spot": FakeSpotExchange())
    monkeypatch.setattr(bitget_connector, "BitgetFuturesClient", FakeFuturesClient)
    monkeypatch.setattr(bitget_connector, "BitgetFundingDataFetcher", FakeFetcher)
    monkeypatch.setattr(bitget_connector, "_HAS_BITGET_FUNDING", True)
    monkeypatch.setattr(bitget_connector, "_set_panteon_fetcher", registered.append)

    cfg = {
        "initial_capital": 100.0,
        "trade_fraction": 0.10,
        "leverage": 2,
        "poll_interval": 60,
        "liquidity_min_adv": 0.0,
        "spot_fee": 0.001,
        "futures_fee": 0.0002,
        "slippage": 0.0001,
        "tf_kline": "1m",
        "bar": 60,
        "symbols": ["BTC", "ETH"],
    }

    bridge = bitget_connector.AgentBitgetBridge(
        {},
        cfg,
        mode="live_futures",
        api_key="key",
        api_secret="secret",
        api_passphrase="pass",
        output_dir=str(tmp_path),
    )
    try:
        prices, volumes = bridge._fetch_market()
    finally:
        tmp_root = str(tmp_path.resolve())
        root_logger = logging.getLogger()
        for handler in list(root_logger.handlers):
            if not isinstance(handler, logging.FileHandler):
                continue
            if str(getattr(handler, "baseFilename", "")).startswith(tmp_root):
                root_logger.removeHandler(handler)
                handler.close()

    assert prices == {"BTC": 100.0, "ETH": 100.0}
    assert set(volumes) == {"BTC", "ETH"}
    assert bridge.funding is FakeFetcher.instances[0]
    assert bridge.funding.started is True
    assert bridge.funding.symbols == ["BTC", "ETH"]
    assert registered == [bridge.funding]
