"""MEXC live futures adapter for Panteon v2."""

from __future__ import annotations

import os
from typing import Any, Optional

from .v1_futures_adapter import V1FuturesExchangeAdapter, load_runtime_leverage


class MexcExchangeAdapter(V1FuturesExchangeAdapter):
    """Wrap the existing v1 MEXC futures client as a v2 Exchange."""

    def __init__(
        self,
        *,
        order_client: Optional[Any] = None,
        read_client: Optional[Any] = None,
        leverage: Optional[int] = None,
    ) -> None:
        if order_client is None:
            api_key = os.getenv("MEXC_API_KEY", "")
            api_secret = os.getenv("MEXC_SECRET_KEY", "")
            if not api_key or not api_secret:
                raise RuntimeError("MEXC_API_KEY and MEXC_SECRET_KEY are required")
            from mexc_connector import MexcFuturesClient  # type: ignore

            order_client = MexcFuturesClient(api_key, api_secret, testnet=False)

        if read_client is None:
            try:
                api_key = os.getenv("MEXC_API_KEY", "")
                api_secret = os.getenv("MEXC_SECRET_KEY", "")
                from exchange_api_runtime import MexcDirectClient  # type: ignore

                read_client = MexcDirectClient(api_key, api_secret)
            except Exception:
                read_client = order_client

        super().__init__(
            name="MEXC",
            order_client=order_client,
            read_client=read_client,
            leverage=leverage if leverage is not None else load_runtime_leverage("MEXC"),
            default_min_notional=5.0,
            default_fee_rate=0.0002,
            close_via_place_order=True,
        )
