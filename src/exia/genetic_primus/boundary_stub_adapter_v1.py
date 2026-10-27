"""Restricted synthetic call-protocol adapter. No market/evaluator imports."""
from __future__ import annotations

import copy
import math

INJECTED_LOCAL_DEPENDENCIES = ("src/exia/genetic_primus/boundary_contract_v1.py",)
CASES = ("carry", "retarget", "abstain")
SYMBOLS = ("SYNTH_A", "SYNTH_B")
CALL_KEYS = ("bars", "targets", "symbols", "start_timestamp", "end_timestamp",
             "timeframe_ms", "round_trip_cost_bps", "force_flat_timestamps", "record_details")


def fixture(api, contract, case):
    api["validate_contract"](contract)
    if case not in CASES:
        raise ValueError("unregistered synthetic fixture")
    day, bar = api["DAY_MS"], api["BAR_MS"]
    start, end = 3 * day, 12 * day
    models = []
    for origin in range(3):
        payload = {"synthetic_only": True, "case": case, "origin": origin}
        models.append({"sha256": api["digest"](payload), "payload": payload,
                       "available_ms": start + origin * 3 * day})
    bars, targets = [], []
    for timestamp in range(start, end, bar):
        origin = (timestamp - start) // (3 * day)
        for column, symbol in enumerate(SYMBOLS):
            # Open differs from the previous close: a boundary gap is intentional.
            price = 100.0 + column * 10.0 + origin * 5.0
            bars.append({"timestamp": timestamp, "symbol": symbol,
                         "open": price, "close": price + 1.0})
            direction = 1 if case == "carry" else ((1, 0, -1)[origin] if case == "retarget" else 0)
            targets.append({"execution_timestamp": timestamp, "symbol": symbol,
                            "target": direction if column == 0 else -direction,
                            "signal_timestamp": timestamp,
                            "model_sha256": models[origin]["sha256"]})
    return {"schema": "exia.genetic_primus.synthetic_adapter_input/1",
            "case": case, "synthetic_only": True, "start_ms": start, "end_ms": end,
            "symbols": list(SYMBOLS), "models": models, "bars": bars, "targets": targets}


def validate_fixture(api, contract, value):
    api["validate_contract"](contract)
    if not isinstance(value, dict) or set(value) != {
        "schema", "case", "synthetic_only", "start_ms", "end_ms", "symbols", "models", "bars", "targets"
    }:
        raise ValueError("invalid fixture envelope")
    if value["schema"] != "exia.genetic_primus.synthetic_adapter_input/1" or value["synthetic_only"] is not True:
        raise ValueError("synthetic envelope required")
    if value["case"] not in CASES or value["symbols"] != list(SYMBOLS):
        raise ValueError("registered synthetic case/universe required")
    expected = fixture(api, contract, value["case"])
    # This adapter deliberately cannot accept arbitrary market data or trained models.
    if api["canonical"](value) != api["canonical"](expected):
        raise ValueError("only exact registered synthetic payloads accepted")
    plan = api["boundary_plan"](contract, value["start_ms"], value["end_ms"])
    if len(value["models"]) != len(plan["origins"]):
        raise ValueError("incomplete model-origin bank")
    for model, origin in zip(value["models"], plan["origins"]):
        api["model_sha"](model["sha256"])
        if api["digest"](model["payload"]) != model["sha256"]:
            raise ValueError("synthetic model payload identity mismatch")
        if model["available_ms"] > origin["origin_open_ms"]:
            raise ValueError("synthetic model late")
    return plan


class RecordingStub:
    """Only permitted callee: records parameters, computes no fills or PnL."""
    def __init__(self, api):
        self.api = api
        self.calls = []

    def __call__(self, *, bars, targets, symbols, start_timestamp, end_timestamp,
                 timeframe_ms, round_trip_cost_bps, force_flat_timestamps, record_details):
        parameters = {"symbols": list(symbols), "start_timestamp": start_timestamp,
                      "end_timestamp": end_timestamp, "timeframe_ms": timeframe_ms,
                      "round_trip_cost_bps": round_trip_cost_bps,
                      "force_flat_timestamps": list(force_flat_timestamps),
                      "record_details": record_details}
        captured = {"parameters": parameters,
                    "bars_semantic_sha256": self.api["digest"](bars),
                    "targets_semantic_sha256": self.api["digest"](targets),
                    "bar_rows": len(bars), "target_rows": len(targets),
                    "force_flat_argument_was_tuple": type(force_flat_timestamps) is tuple,
                    "symbols_argument_was_tuple": type(symbols) is tuple,
                    "orders_emitted": False, "fills_generated": False,
                    "financial_result": False}
        self.calls.append(copy.deepcopy(captured))
        return copy.deepcopy(captured)


def invoke_fixture(api, contract, value, scenario, stub):
    if type(stub) is not RecordingStub or stub.api is not api:
        raise ValueError("only the injected recording stub is allowed, not a real evaluator")
    if stub.calls:
        raise ValueError("fresh single-use recording stub required")
    plan = validate_fixture(api, contract, value)
    if scenario not in plan["round_trip_cost_bps_by_scenario"]:
        raise ValueError("unregistered reference cost scenario")
    before = api["digest"](value)
    parameters = plan["explicit_accounting_parameters"]
    captured = stub(
        bars=copy.deepcopy(value["bars"]), targets=copy.deepcopy(value["targets"]),
        symbols=tuple(value["symbols"]), start_timestamp=parameters["start_timestamp"],
        end_timestamp=parameters["end_timestamp"], timeframe_ms=parameters["timeframe_ms"],
        round_trip_cost_bps=plan["round_trip_cost_bps_by_scenario"][scenario],
        force_flat_timestamps=tuple(parameters["force_flat_timestamps"]),
        record_details=parameters["record_details"],
    )
    positions = {symbol: 0 for symbol in value["symbols"]}
    entry_models = {symbol: None for symbol in value["symbols"]}
    ledger = []
    for row in value["targets"]:
        symbol, timestamp = row["symbol"], row["execution_timestamp"]
        index = (timestamp - value["start_ms"]) // contract["time"]["retrain_cadence_ms"]
        model = value["models"][index]
        if row["model_sha256"] != model["sha256"]:
            raise ValueError("active model mismatch")
        transition = api["boundary_transition"](
            contract, position=positions[symbol], target=row["target"],
            entry_model_sha256=entry_models[symbol], decision_model_sha256=row["model_sha256"],
            origin_open_ms=timestamp, signal_close_ms=row["signal_timestamp"],
            model_available_ms=model["available_ms"], interval_end_ms=value["end_ms"])
        ledger.append({"symbol": symbol, "execution_timestamp": timestamp,
                       "signal_timestamp": row["signal_timestamp"], **transition})
        positions[symbol] = transition["position_after"]
        entry_models[symbol] = transition["position_entry_model_sha256"]
    last_close = {row["symbol"]: row["close"] for row in value["bars"]
                  if row["timestamp"] == value["end_ms"] - api["BAR_MS"]}
    terminal = [{"symbol": symbol, "timestamp": value["end_ms"],
                 "phase": "expected_final_close_liquidation_not_executed",
                 "position_before": positions[symbol], "position_after_expected": 0,
                 "position_entry_model_sha256": entry_models[symbol],
                 "expected_semantic_events": ["exit"] if positions[symbol] else [],
                 "expected_price": last_close[symbol], "fees_required_if_position": bool(positions[symbol]),
                 "actual_fill_verified": False} for symbol in value["symbols"]]
    if api["digest"](value) != before:
        raise ValueError("synthetic source mutated")
    if not math.isfinite(captured["parameters"]["round_trip_cost_bps"]):
        raise ValueError("nonfinite synthetic cost")
    return {"schema": "exia.genetic_primus.boundary_stub_adapter_result/1",
            "case": value["case"], "scenario": scenario, "input_semantic_sha256": before,
            "captured_call": captured, "decision_ledger": ledger,
            "terminal_expectations": terminal, "input_unchanged": True,
            "stub_call_count": len(stub.calls), "protocol_stub_verified": True,
            "real_evaluator_adapter_verified": False,
            "runtime_dependency_graph_verified": False, "financial_result": False,
            "safety": dict(api["SAFETY"])}
