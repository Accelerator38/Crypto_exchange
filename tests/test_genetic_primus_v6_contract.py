from __future__ import annotations

from pathlib import Path

from exia.genetic_primus.contracts_v6 import load_v6_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/genetic_primus_v6_contract.json"


def test_v6_contract_pins_full_history_kline_flow() -> None:
    contract = load_v6_contract(CONTRACT, root=ROOT)
    assert contract.flow_source["start_month"] == "2022-01"
    assert contract.flow_source["end_month"] == "2024-12"
    assert contract.flow_source["derived_flow_feature"] == "kline_taker_imbalance_4h"
    assert contract.payload["execution_source"]["holding_bars"] == 6
    assert contract.payload["evolution"]["fallback_selection_allowed"] is False
