"""Registered future-only contract for continuous offline origin evaluation."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from .measurement_lab_v2 import CATALOG, FEATURES
from .prospective_contract_v1 import (
    SAFETY_KEYS, _utc, derive_origins, validate_contract as validate_v1,
)

SCHEMA = "exia.genetic_primus.prospective_3d/2"


def validate_contract(contract: dict[str, Any]) -> tuple:
    """Validate the fixed design without claiming that provenance is true.

    The runner separately hashes the selection report and registry ancestry.
    V1 remains accepted only for historical manifest compatibility.
    """
    if contract.get("schema_version") == "exia.genetic_primus.prospective_3d/1":
        return validate_v1(contract)
    if contract.get("schema_version") != SCHEMA:
        raise ValueError("unexpected prospective contract schema")
    origins = derive_origins(contract)
    if len(origins) != 10 or (origins[-1][1] - origins[0][0]) != timedelta(days=30):
        raise ValueError("V2 requires ten three-day origins")
    if _utc(contract["frozen_at_utc"]) >= origins[0][0]:
        raise ValueError("contract must be frozen before outer start")
    if contract["outer"]["outcomes_available_at_registration"]:
        raise ValueError("outer outcomes already available")
    training = contract["training"]
    if (training["purge_bars"] != 7 or training["label_horizon_bars"] != 7
            or training["timeframe_minutes"] != 240):
        raise ValueError("four-hour training and seven-bar purge required")
    if training["window"] != "expanding" or training["genetic_search_during_outer"] \
            or training["genome_updates_during_outer"]:
        raise ValueError("fixed expanding policy required during outer period")
    selection = contract["model_selection"]
    if selection["window"] != training["window"] or selection["scaler"] != training["scaler"]:
        raise ValueError("selection and deployment preparation differ")
    if not isinstance(selection["report_path"], str) or not selection["report_path"]:
        raise ValueError("model selection report path required")
    if not isinstance(selection["report_sha256"], str) or len(selection["report_sha256"]) != 64:
        raise ValueError("model selection report SHA required")
    if tuple(contract["features"]) != FEATURES:
        raise ValueError("registered feature order changed")
    candidates = contract["candidates"]
    if len(candidates) != 3 or tuple(x["candidate_id"] for x in candidates[:2]) != (
            "NoTrade", "EMA_TrendConsensus"):
        raise ValueError("NoTrade, EMA and one overlay required")
    overlay = candidates[2]
    if (not isinstance(overlay["candidate_id"], str) or not overlay["candidate_id"].startswith("GA_")
            or overlay["candidate_id"] in ("NoTrade", "EMA_TrendConsensus")):
        raise ValueError("registered overlay candidate ID required")
    genome = overlay["genome"]
    if (not isinstance(genome, list) or len(genome) != 5
            or any(type(weight) is not int for weight in genome)
            or tuple(genome) not in CATALOG):
        raise ValueError("overlay genome is outside the sparse candidate catalog")
    accounting = contract["accounting"]
    if accounting["terminal_exit"] != "completed_prefix_final_candle_close":
        raise ValueError("continuous terminal exit required")
    if (accounting["normal_round_trip_bps"] < 0
            or accounting["stress_round_trip_bps"] < accounting["normal_round_trip_bps"]):
        raise ValueError("invalid normal/stress cost order")
    measurement = contract["measurement"]
    if measurement["daily_observations"] != 30 or measurement["selection_or_tuning_on_outer"]:
        raise ValueError("outer period may not select or tune")
    commitment = contract["data_commitment"]
    if (commitment["manifest_version"] != 2
            or commitment["origin_snapshot_commit_rule"]
            != "after_origin_complete_before_evaluation"
            or type(commitment["origin_commit_deadline_hours"]) is not int
            or not 0 < commitment["origin_commit_deadline_hours"] <= 72
            or commitment["future_outer_data_hash_known_at_registration"]):
        raise ValueError("invalid V2 origin commitment")
    if any(contract["safety"].get(key) is not False for key in SAFETY_KEYS):
        raise ValueError("runtime and network permissions must be false")
    if contract["decision"]["automatic_promotion"]:
        raise ValueError("automatic promotion is forbidden")
    return origins
