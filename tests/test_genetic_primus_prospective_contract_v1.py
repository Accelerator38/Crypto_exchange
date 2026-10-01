from copy import deepcopy
from datetime import timedelta
from pathlib import Path

from exia.genetic_primus.prospective_contract_v1 import load_and_validate, validate_contract


CONFIG = Path(__file__).parents[1] / "configs" / "genetic_primus_prospective_3d_v1.json"


def _contract():
    return load_and_validate(CONFIG)[0]


def _rejected(contract):
    try:
        validate_contract(contract)
    except ValueError:
        return True
    return False


def test_frozen_bounds_produce_ten_contiguous_three_day_origins():
    _, origins = load_and_validate(CONFIG)
    assert len(origins) == 10
    assert all(end - start == timedelta(days=3) for start, end in origins)
    assert all(origins[index][1] == origins[index + 1][0] for index in range(9))


def test_safety_and_fixed_candidate_set():
    contract = _contract()
    assert all(value is False for value in contract["safety"].values())
    assert [item["candidate_id"] for item in contract["candidates"]] == [
        "NoTrade", "EMA_TrendConsensus", "GA_overlay_f7a4ae11"
    ]


def test_rejects_outer_tuning_and_candidate_replacement():
    tuned = deepcopy(_contract())
    tuned["measurement"]["selection_or_tuning_on_outer"] = True
    assert _rejected(tuned)
    replaced = deepcopy(_contract())
    replaced["candidates"][2]["genome"] = [1, 0, 0, 0, 0]
    assert _rejected(replaced)


def test_rejects_bad_purge_and_runtime_authority():
    leaked = deepcopy(_contract())
    leaked["training"]["purge_bars"] = 0
    assert _rejected(leaked)
    live = deepcopy(_contract())
    live["safety"]["orders_enabled"] = True
    assert _rejected(live)
