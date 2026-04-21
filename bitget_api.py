"""Bitget direct client and compatibility layer for combo trading."""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Dict, List


def _bootstrap_project_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_site = os.path.join(base_dir, ".venv", "Lib", "site-packages")
    if os.path.isdir(venv_site) and venv_site not in sys.path:
        sys.path.insert(0, venv_site)


_bootstrap_project_paths()

import ccxt  # type: ignore

from exchange_api_runtime import (
    TradingStats,
    inject_live_state_into_player,
    save_dashboard,
    save_summary_json,
)


log = logging.getLogger("bitget_api")

API_KEY = os.getenv("BITGET_API_KEY", "")
API_SECRET = os.getenv("BITGET_SECRET_KEY", "")
API_PASSPHRASE = os.getenv("BITGET_PASSPHRASE", "")
TRADING_MODE = os.getenv("BITGET_TRADING_MODE", "live_futures")


def _build_client(
    default_type: str,
    api_key: str = "",
    api_secret: str = "",
    api_passphrase: str = "",
) -> ccxt.bitget:
    options = {
        "defaultType": default_type,
        "defaultSubType": "linear",
    }
    config = {
        "enableRateLimit": True,
        "options": options,
    }
    if api_key:
        config["apiKey"] = api_key
    if api_secret:
        config["secret"] = api_secret
    if api_passphrase:
        config["password"] = api_passphrase
    return ccxt.bitget(config)


def _normalize_base_symbol(symbol: str) -> str:
    if not symbol:
        return ""
    sym = symbol.upper()
    if ":" in sym:
        sym = sym.split(":", 1)[0]
    if "/" in sym:
        sym = sym.split("/", 1)[0]
    if sym.endswith("_USDT"):
        sym = sym[:-5]
    if sym.endswith("USDT") and len(sym) > 4:
        sym = sym[:-4]
    return sym


class BitgetDirectClient:
    """Compatibility client that returns the same snapshot shape as MexcDirectClient."""

    def __init__(self, api_key: str, api_secret: str, api_passphrase: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        if api_key and api_secret and not api_passphrase:
            raise ValueError(
                "Bitget API requires BITGET_PASSPHRASE in addition to API key and secret."
            )
        self.public = _build_client("spot")
        self.spot = _build_client("spot", api_key, api_secret, api_passphrase)
        self.swap = _build_client("swap", api_key, api_secret, api_passphrase)
        self._last_good_futures_snapshot: Dict[str, float] | None = None
        self._last_good_futures_snapshot_at = 0.0

    def get_all_balances(self) -> Dict[str, dict]:
        result: Dict[str, dict] = {}
        if not self.api_key:
            return result
        bal = self.spot.fetch_balance({"type": "spot"})
        for asset, payload in (bal.get("total") or {}).items():
            total = float(payload or 0.0)
            free = float((bal.get("free") or {}).get(asset, 0.0) or 0.0)
            used = float((bal.get("used") or {}).get(asset, 0.0) or 0.0)
            if total > 0 or free > 0 or used > 0:
                result[asset.upper()] = {
                    "free": free,
                    "locked": used,
                    "total": total,
                }
        return result

    def get_usdt_equivalent(self, balances: Dict[str, dict]) -> float:
        stables = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "FDUSD"}
        return sum(v["total"] for k, v in balances.items() if k in stables)

    def get_open_orders(self) -> List[dict]:
        if not self.api_key:
            return []
        try:
            return self.spot.fetch_open_orders()
        except Exception:
            return []

    def get_recent_trades(self, limit: int = 20) -> List[dict]:
        if not self.api_key:
            return []
        try:
            return self.spot.fetch_my_trades(limit=limit)
        except Exception:
            return []

    def get_futures_account(self) -> dict:
        if not self.api_key:
            return {}
        balance = self.swap.fetch_balance(
            {"type": "swap", "productType": "USDT-FUTURES"}
        )
        usdt = balance.get("USDT", {}) or {}
        return {
            "equity": float(usdt.get("total", 0.0) or 0.0),
            "available": float(usdt.get("free", 0.0) or 0.0),
        }

    def get_futures_positions(self) -> List[dict]:
        if not self.api_key:
            return []
        positions = self.swap.fetch_positions(
            params={"productType": "USDT-FUTURES", "marginCoin": "USDT"}
        )
        result = []
        for pos in positions:
            contracts = float(pos.get("contracts", 0.0) or 0.0)
            if contracts <= 0:
                continue
            info = pos.get("info", {}) or {}
            result.append(
                {
                    "symbol": _normalize_base_symbol(pos.get("symbol", "")),
                    "side": str(pos.get("side", "") or info.get("holdSide", "long")).lower(),
                    "qty": contracts,
                    "leverage": int(float(pos.get("leverage", 1) or 1)),
                    "entry": float(pos.get("entryPrice", 0.0) or 0.0),
                    "unrealized_pnl": float(pos.get("unrealizedPnl", 0.0) or 0.0),
                    "margin": float(pos.get("initialMargin", 0.0) or 0.0),
                }
            )
        return result

    def get_full_snapshot(self) -> dict:
        snap = {
            "futures": {"equity": 0.0, "available": 0.0, "unrealized": 0.0, "positions": []},
            "spot": {"usdt": 0.0, "crypto": {}, "total_value": 0.0},
            "total_equity": 0.0,
            "primary_capital": 0.0,
        }

        try:
            fut = self.get_futures_account()
            snap["futures"]["equity"] = float(fut.get("equity", 0.0) or 0.0)
            snap["futures"]["available"] = float(fut.get("available", 0.0) or 0.0)
        except Exception as e:
            log.debug("bitget futures account snapshot: %s", e)

        try:
            positions = self.get_futures_positions()
            snap["futures"]["positions"] = positions
            snap["futures"]["unrealized"] = sum(
                float(p.get("unrealized_pnl", 0.0) or 0.0) for p in positions
            )
        except Exception as e:
            log.debug("bitget futures positions snapshot: %s", e)

        zero_assets = (
            float(snap["futures"]["equity"] or 0.0) <= 0.0
            and float(snap["futures"]["available"] or 0.0) <= 0.0
        )
        if (
            zero_assets
            and self._last_good_futures_snapshot
            and (time.time() - self._last_good_futures_snapshot_at) <= 300
        ):
            cached = dict(self._last_good_futures_snapshot)
            current_unrealized = float(snap["futures"]["unrealized"] or 0.0)
            if current_unrealized == 0.0 and not snap["futures"]["positions"]:
                current_unrealized = float(cached.get("unrealized", 0.0) or 0.0)
            realized_base = max(
                float(cached.get("equity", 0.0) or 0.0) - float(cached.get("unrealized", 0.0) or 0.0),
                0.0,
            )
            snap["futures"]["equity"] = max(realized_base + current_unrealized, 0.0)
            snap["futures"]["available"] = max(float(cached.get("available", 0.0) or 0.0), 0.0)
            snap["futures"]["unrealized"] = current_unrealized
            log.warning(
                "bitget snapshot: futures equity/available returned zero -> reuse cached core "
                "(age=%.1fs, positions=%d)",
                time.time() - self._last_good_futures_snapshot_at,
                len(snap["futures"]["positions"]),
            )

        if snap["futures"]["equity"] > 0 or snap["futures"]["available"] > 0:
            self._last_good_futures_snapshot = {
                "equity": float(snap["futures"]["equity"] or 0.0),
                "available": float(snap["futures"]["available"] or 0.0),
                "unrealized": float(snap["futures"]["unrealized"] or 0.0),
            }
            self._last_good_futures_snapshot_at = time.time()

        balances = self.get_all_balances()
        non_stables: List[str] = []
        for asset, payload in balances.items():
            total = float(payload.get("total", 0.0) or 0.0)
            if total <= 0:
                continue
            if asset in {"USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "FDUSD"}:
                snap["spot"]["usdt"] += total
            else:
                snap["spot"]["crypto"][asset] = {"qty": total, "price": 0.0, "value": 0.0}
                non_stables.append(asset)

        if non_stables:
            try:
                tickers = self.public.fetch_tickers([f"{sym}/USDT" for sym in non_stables])
                for asset in non_stables:
                    ticker = tickers.get(f"{asset}/USDT", {}) or {}
                    price = float(ticker.get("last", 0.0) or 0.0)
                    qty = snap["spot"]["crypto"][asset]["qty"]
                    snap["spot"]["crypto"][asset]["price"] = price
                    snap["spot"]["crypto"][asset]["value"] = qty * price
            except Exception as e:
                log.debug("bitget spot prices snapshot: %s", e)

        crypto_value = sum(v["value"] for v in snap["spot"]["crypto"].values())
        snap["spot"]["total_value"] = snap["spot"]["usdt"] + crypto_value
        snap["total_equity"] = snap["futures"]["equity"] + snap["spot"]["total_value"]
        if TRADING_MODE == "live_futures":
            snap["primary_capital"] = snap["futures"]["equity"]
        else:
            snap["primary_capital"] = snap["spot"]["total_value"]
        return snap

    def log_snapshot(self, snapshot: dict) -> None:
        fut = snapshot.get("futures", {})
        spot = snapshot.get("spot", {})
        log.info(
            "  [BITGET] futures_equity=%.4f  futures_available=%.4f  spot_total=%.4f  total=%.4f",
            float(fut.get("equity", 0.0) or 0.0),
            float(fut.get("available", 0.0) or 0.0),
            float(spot.get("total_value", 0.0) or 0.0),
            float(snapshot.get("total_equity", 0.0) or 0.0),
        )
        for pos in fut.get("positions", [])[:10]:
            log.info(
                "  [BITGET] position %s %s lev=%dx entry=%.6f pnl=%+.4f",
                pos.get("symbol", "?"),
                str(pos.get("side", "")).upper(),
                int(pos.get("leverage", 1) or 1),
                float(pos.get("entry", 0.0) or 0.0),
                float(pos.get("unrealized_pnl", 0.0) or 0.0),
            )


def create_direct_client(
    api_key: str,
    api_secret: str,
    api_passphrase: str = "",
) -> BitgetDirectClient:
    return BitgetDirectClient(api_key, api_secret, api_passphrase)


def _check_futures_api_permission(direct_client: BitgetDirectClient) -> bool:
    try:
        direct_client.get_futures_account()
        direct_client.get_futures_positions()
        return True
    except ccxt.AuthenticationError as e:  # type: ignore[attr-defined]
        log.error("Bitget authentication error: %s", e)
        return False
    except Exception as e:
        log.warning("Bitget futures permission probe: %s", e)
        return False
