from __future__ import annotations

import importlib


def test_mexc_contract_info_normalizes_min_size_and_precision():
    tool = importlib.import_module("tools.check_exchange_futures_rules")

    payload = {
        "data": [
            {
                "symbol": "BTC_USDT",
                "minVol": "1",
                "maxVol": "10000",
                "priceUnit": "0.1",
                "volUnit": "1",
                "contractSize": "0.001",
                "priceScale": 1,
                "volScale": 0,
                "amountScale": 3,
                "apiAllowed": True,
                "state": 0,
            }
        ]
    }

    report = tool.normalize_mexc_contract_info(payload, symbols=("BTC",))

    assert report["exchange"] == "MEXC"
    assert report["rules"][0]["symbol"] == "BTC"
    assert report["rules"][0]["market_symbol"] == "BTC_USDT"
    assert report["rules"][0]["min_order_size"] == 1.0
    assert report["rules"][0]["price_tick"] == 0.1
    assert report["rules"][0]["quantity_step"] == 1.0
    assert report["rules"][0]["contract_size"] == 0.001
    assert report["rules"][0]["active"] is True
    assert report["passed"] is True


def test_bitget_market_normalizes_min_size_and_cost_limits():
    tool = importlib.import_module("tools.check_exchange_futures_rules")

    markets = {
        "BTC/USDT:USDT": {
            "symbol": "BTC/USDT:USDT",
            "id": "BTCUSDT",
            "active": True,
            "contractSize": 0.001,
            "precision": {"price": 0.1, "amount": 0.001},
            "limits": {"amount": {"min": 0.001, "max": 100.0}, "cost": {"min": 5.0}},
            "info": {
                "sizeMultiplier": "0.001",
                "minTradeNum": "0.001",
                "maxTradeAmount": "100",
            },
        }
    }

    report = tool.normalize_bitget_markets(markets, symbols=("BTC",))

    assert report["exchange"] == "BITGET"
    assert report["rules"][0]["symbol"] == "BTC"
    assert report["rules"][0]["market_symbol"] == "BTC/USDT:USDT"
    assert report["rules"][0]["min_order_size"] == 0.001
    assert report["rules"][0]["min_notional_usd"] == 5.0
    assert report["rules"][0]["quantity_step"] == 0.001
    assert report["rules"][0]["active"] is True
    assert report["passed"] is True


def test_bitget_contract_config_normalizes_public_rest_payload():
    tool = importlib.import_module("tools.check_exchange_futures_rules")

    payload = {
        "code": "00000",
        "data": [
            {
                "symbol": "BTCUSDT",
                "baseCoin": "BTC",
                "quoteCoin": "USDT",
                "symbolStatus": "normal",
                "minTradeNum": "0.001",
                "maxTradeNum": "100",
                "sizeMultiplier": "0.001",
                "minTradeUSDT": "5",
                "pricePlace": "1",
                "priceEndStep": "1",
                "volumePlace": "3",
            }
        ],
    }

    report = tool.normalize_bitget_contracts(payload, symbols=("BTC",))

    assert report["exchange"] == "BITGET"
    assert report["rules"][0]["symbol"] == "BTC"
    assert report["rules"][0]["market_symbol"] == "BTCUSDT"
    assert report["rules"][0]["min_order_size"] == 0.001
    assert report["rules"][0]["min_notional_usd"] == 5.0
    assert report["rules"][0]["quantity_step"] == 0.001
    assert report["rules"][0]["price_tick"] == 0.1
    assert report["rules"][0]["active"] is True
    assert report["passed"] is True
