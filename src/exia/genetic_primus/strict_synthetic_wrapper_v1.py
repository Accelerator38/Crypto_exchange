"""Strict artificial-input adapter and trade-bound entry-identity sidecar."""
from __future__ import annotations

import copy
import math
import sys
from hashlib import sha256
from pathlib import Path
from types import CodeType, ModuleType

ACCOUNTING_SHA = "f9294a2e4b2220aa7552ebed7d2480e7e7647744fe5a62330888acba569af5bb"
REGISTERED_INPUTS = {
    "carry": "49701562278ac43447f91c7c98b05f7a9e8fb7ddaae0676b56b2db3fa2ce750d",
    "retarget": "6bc49639c7711480b3eb6c9984650356e9a2bf7b4032108ed820f5d2f0b0c00f",
    "abstain": "e57b4b376244fe1e2ce1bdce20436b9c0dcf5df173da2193bb36277937d539dd",
}
TOLERANCE = 1e-10


def plain_json(value, depth=0):
    if depth > 12:
        raise ValueError("excessive fixture nesting")
    if type(value) in (str, int, bool, type(None)):
        if type(value) is str and len(value) > 200:
            raise ValueError("oversized string")
        if type(value) is int and abs(value) > 2**53 - 1:
            raise ValueError("oversized integer")
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite number")
        return
    if type(value) is list:
        if len(value) > 108:
            raise ValueError("oversized array")
        for item in value:
            plain_json(item, depth + 1)
        return
    if type(value) is dict:
        if len(value) > 12 or any(type(key) is not str for key in value):
            raise ValueError("invalid fixture object")
        for item in value.values():
            plain_json(item, depth + 1)
        return
    raise ValueError("only plain bounded JSON values accepted")


def validate_inputs(api, contract, value, scenario):
    api["validate_contract"](contract)
    plain_json(value)
    if type(value) is not dict or set(value) != {
        "schema", "case", "synthetic_only", "start_ms", "end_ms", "symbols", "models", "bars", "targets"
    }:
        raise ValueError("invalid artificial envelope")
    if value["schema"] != "exia.genetic_primus.synthetic_adapter_input/1" or value["synthetic_only"] is not True:
        raise ValueError("artificial fixture required")
    if value["case"] not in REGISTERED_INPUTS or value["symbols"] != ["SYNTH_A", "SYNTH_B"]:
        raise ValueError("unregistered case/universe")
    if scenario not in ("normal", "stress"):
        raise ValueError("unregistered reference cost scenario")
    plan = api["boundary_plan"](contract, value["start_ms"], value["end_ms"])
    if len(plan["origins"]) != 3 or len(value["models"]) != 3:
        raise ValueError("registered three-origin model bank required")
    for index, (model, origin) in enumerate(zip(value["models"], plan["origins"])):
        if type(model) is not dict or set(model) != {"sha256", "payload", "available_ms"}:
            raise ValueError("invalid model envelope")
        api["model_sha"](model["sha256"])
        if model["payload"] != {"synthetic_only": True, "case": value["case"], "origin": index}:
            raise ValueError("not a registered artificial model payload")
        if api["digest"](model["payload"]) != model["sha256"]:
            raise ValueError("model payload hash mismatch")
        if type(model["available_ms"]) is not int or not 0 <= model["available_ms"] <= origin["origin_open_ms"]:
            raise ValueError("late/invalid model availability")
    times = range(value["start_ms"], value["end_ms"], api["BAR_MS"])
    grid = [(timestamp, symbol) for timestamp in times for symbol in value["symbols"]]
    if len(grid) != 108 or len(value["bars"]) != len(grid) or len(value["targets"]) != len(grid):
        raise ValueError("incomplete/extraneous fixture grid")
    for key, bar, target in zip(grid, value["bars"], value["targets"]):
        if type(bar) is not dict or set(bar) != {"timestamp", "symbol", "open", "close"}:
            raise ValueError("invalid candle schema")
        if type(target) is not dict or set(target) != {
            "execution_timestamp", "symbol", "target", "signal_timestamp", "model_sha256"
        }:
            raise ValueError("invalid target schema")
        if type(bar["timestamp"]) is not int or (bar["timestamp"], bar["symbol"]) != key:
            raise ValueError("unordered/duplicate candle grid")
        for field in ("open", "close"):
            if type(bar[field]) not in (int, float) or not 0 < bar[field] <= 1_000_000:
                raise ValueError("invalid artificial price")
        if type(target["execution_timestamp"]) is not int or (target["execution_timestamp"], target["symbol"]) != key:
            raise ValueError("unordered/duplicate target grid")
        if type(target["target"]) is not int or target["target"] not in (-1, 0, 1):
            raise ValueError("boolean/non-discrete direction rejected")
        if type(target["signal_timestamp"]) is not int or not 0 <= target["signal_timestamp"] <= key[0]:
            raise ValueError("future/invalid signal")
        index = (key[0] - value["start_ms"]) // contract["time"]["retrain_cadence_ms"]
        model = value["models"][index]
        api["model_sha"](target["model_sha256"])
        if target["model_sha256"] != model["sha256"] or model["available_ms"] > target["signal_timestamp"]:
            raise ValueError("active model/payload/availability mismatch")
    identity = api["digest"](value)
    if identity != REGISTERED_INPUTS[value["case"]]:
        raise ValueError("fixture bytes differ from previously registered synthetic input")
    return {"input_semantic_sha256": identity, "model_bank_semantic_sha256": api["digest"](value["models"]),
            "boundary_plan_semantic_sha256": api["digest"](plan),
            "validated_before_callee": True, "synthetic_only": True,
            "round_trip_cost_bps": plan["round_trip_cost_bps_by_scenario"][scenario]}


def code_identity(api, code):
    def constant(value):
        if isinstance(value, CodeType):
            return {"code": code_identity(api, value)}
        if isinstance(value, tuple):
            return {"tuple": [constant(x) for x in value]}
        if isinstance(value, frozenset):
            return {"frozenset": sorted((constant(x) for x in value), key=lambda x: api["canonical"](x))}
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if type(value) in (str, int, float, bool, type(None)):
            return {"type": type(value).__name__, "value": value}
        raise ValueError("unexpected compiled constant")
    return api["digest"]({"name": code.co_name, "filename": code.co_filename,
                          "bytecode": code.co_code.hex(), "constants": [constant(x) for x in code.co_consts],
                          "names": list(code.co_names), "vars": list(code.co_varnames),
                          "freevars": list(code.co_freevars), "cellvars": list(code.co_cellvars),
                          "argcount": code.co_argcount, "posonly": code.co_posonlyargcount,
                          "kwonly": code.co_kwonlyargcount, "flags": code.co_flags,
                          "linetable": code.co_linetable.hex(), "exceptions": code.co_exceptiontable.hex()})


def checked_callee(api, namespace):
    if type(namespace) is not ModuleType or namespace.__name__ != "_gp_synthetic_accounting_probe_v1":
        raise ValueError("only the verified synthetic accounting namespace is allowed")
    if sys.modules.get(namespace.__name__) is not namespace:
        raise ValueError("unregistered namespace")
    path = Path(namespace.__file__).resolve()
    if path.parts[-4:] != ("src", "exia", "genetic_primus", "accounting_v2.py"):
        raise ValueError("unexpected accounting path")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != ACCOUNTING_SHA:
        raise ValueError("unreviewed accounting bytes")
    compiled = compile(raw, str(path), "exec", dont_inherit=True)
    codes = {x.co_name: x for x in compiled.co_consts if isinstance(x, CodeType)}
    if namespace.simulate_cash_targets.__globals__ is not namespace.__dict__:
        raise ValueError("simulator globals are not the verified namespace")
    if code_identity(api, namespace.simulate_cash_targets.__code__) != code_identity(api, codes["simulate_cash_targets"]):
        raise ValueError("loaded simulator function changed")
    methods = {x.co_name: x for x in codes["CashSleeve"].co_consts if isinstance(x, CodeType)}
    for name, expected in methods.items():
        actual = getattr(namespace.CashSleeve, name)
        if isinstance(actual, property):
            actual = actual.fget
        if actual.__globals__ is not namespace.__dict__:
            raise ValueError("accounting method globals changed")
        if code_identity(api, actual.__code__) != code_identity(api, expected):
            raise ValueError("loaded accounting method changed")
    return {"accounting_source_byte_sha256": ACCOUNTING_SHA,
            "simulator_code_identity": code_identity(api, namespace.simulate_cash_targets.__code__),
            "source_function_codes_matched": True, "runtime_dependency_graph_verified": False}


def entry_sidecar(api, value, actual):
    decisions, trades = actual.decisions.to_dict("records"), actual.trades.to_dict("records")
    if len(decisions) != len(value["targets"]):
        raise ValueError("missing/extraneous actual decisions")
    prices = {(x["timestamp"], x["symbol"]): x for x in value["bars"]}
    positions = {symbol: None for symbol in value["symbols"]}
    consumed, closing_refs, ledger = set(), {}, []
    def numeric(left, right):
        if type(left) not in (int, float) or not math.isfinite(left) or abs(left - right) > TOLERANCE:
            raise ValueError("quantity/price mismatch in actual entry attribution")
    def consume(symbol, state, timestamp, price, reason):
        indices = [index for index, trade in enumerate(trades)
                   if trade["symbol"] == symbol and trade["exit_timestamp"] == timestamp and trade["exit_reason"] == reason]
        if len(indices) != 1 or indices[0] in consumed:
            raise ValueError("missing/duplicate actual exit trade")
        index, trade = indices[0], trades[indices[0]]
        if (trade["entry_model_sha256"], trade["entry_timestamp"], trade["signal_timestamp"]) != (
            state["sha"], state["timestamp"], state["signal"]):
            raise ValueError("actual trade entry identity mismatch")
        numeric(trade["quantity"], state["quantity"])
        numeric(trade["entry_price"], state["price"])
        numeric(trade["exit_price"], price)
        if trade["direction"] != (1 if state["quantity"] > 0 else -1):
            raise ValueError("actual trade entry direction mismatch")
        consumed.add(index); closing_refs[state["ref"]] = index
        return index
    for source, row in zip(value["targets"], decisions):
        symbol, timestamp = source["symbol"], source["execution_timestamp"]
        if (row["symbol"], row["execution_timestamp"], row["signal_timestamp"], row["model_sha256"], row["desired_position"]) != (
            symbol, timestamp, source["signal_timestamp"], source["model_sha256"], source["target"]):
            raise ValueError("actual decision/input model mismatch")
        if row["forced_reset"] is not False:
            raise ValueError("forward carry policy cannot accept forced-reset decisions")
        before = positions[symbol]
        direction = 1 if before and before["quantity"] > 0 else -1 if before else 0
        if row["position_before"] != direction:
            raise ValueError("actual position-before mismatch")
        target = source["target"]
        if direction != target and before:
            consume(symbol, before, timestamp, prices[timestamp, symbol]["open"], "target_change")
        if target == 0:
            numeric(row["quantity_after"], 0.0)
            after = None
        elif direction == target:
            numeric(row["quantity_after"], before["quantity"])
            after = before
        else:
            quantity = row["quantity_after"]
            if type(quantity) not in (int, float) or not math.isfinite(quantity) or quantity * target <= 0:
                raise ValueError("invalid actual entry quantity")
            after = {"sha": row["model_sha256"], "timestamp": timestamp, "signal": row["signal_timestamp"],
                     "quantity": quantity, "price": prices[timestamp, symbol]["open"],
                     "ref": api["digest"]({"symbol": symbol, "entry_timestamp": timestamp, "model": row["model_sha256"]})}
        positions[symbol] = after
        ledger.append({"symbol": symbol, "execution_timestamp": timestamp,
                       "decision_model_sha256": row["model_sha256"],
                       "position_entry_model_sha256_before": before["sha"] if before else None,
                       "position_entry_model_sha256_after": after["sha"] if after else None,
                       "entry_ref_before": before["ref"] if before else None,
                       "entry_ref_after": after["ref"] if after else None,
                       "quantity_after": row["quantity_after"], "position_after": target,
                       "field_origin": "wrapper_derived_bound_to_actual_closed_trade"})
    terminal = []
    for symbol, state in positions.items():
        index = consume(symbol, state, value["end_ms"],
                        prices[value["end_ms"] - 14_400_000, symbol]["close"], "interval_end") if state else None
        terminal.append({"symbol": symbol, "timestamp": value["end_ms"],
                         "entry_model_sha256_before": state["sha"] if state else None,
                         "actual_terminal_trade_index": index, "flat_after_expected_and_trade_bound": True})
    if len(consumed) != len(trades):
        raise ValueError("unattributed actual trades")
    for row in ledger:
        for suffix in ("before", "after"):
            ref = row["entry_ref_" + suffix]
            row["actual_closed_trade_index_" + suffix] = closing_refs[ref] if ref else None
    return {"decision_ledger": ledger, "terminal_ledger": terminal,
            "actual_trades_attributed": len(consumed), "all_entries_bound_to_actual_closed_trades": True,
            "entry_sha_is_original_decision_field": False,
            "coverage": "offline_complete_interval_not_online_open_positions"}


def run_wrapper(api, contract, value, scenario, namespace):
    proof = validate_inputs(api, contract, value, scenario)
    callee = checked_callee(api, namespace)
    start, end = value["start_ms"], value["end_ms"]
    actual = namespace.simulate_cash_targets(
        bars=namespace.pd.DataFrame(copy.deepcopy(value["bars"])),
        targets=namespace.pd.DataFrame(copy.deepcopy(value["targets"])), symbols=tuple(value["symbols"]),
        start_timestamp=start, end_timestamp=end, timeframe_ms=contract["time"]["bar_ms"],
        round_trip_cost_bps=proof["round_trip_cost_bps"],
        force_flat_timestamps=(), record_details=True)
    sidecar = entry_sidecar(api, value, actual)
    if api["digest"](value) != proof["input_semantic_sha256"]:
        raise ValueError("input mutated during wrapper call")
    return {"schema": "exia.genetic_primus.strict_synthetic_wrapper_result/1",
            "case": value["case"], "scenario": scenario, "input_validation": proof,
            "callee_identity": callee, "entry_identity_sidecar": sidecar,
            "actual_metrics": actual.metrics,
            "actual": {key: getattr(actual, key).to_dict("records") for key in ("decisions", "trades", "events", "portfolio")},
            "synthetic_only": True, "financial_strategy_result": False,
            "full_contract_integration_verified": False, "safety": dict(api["SAFETY"])}
