"""Bounded artificial accounting checks with an independent rational oracle."""
from __future__ import annotations

import ast
import math
import os
import sys
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from types import ModuleType

ACCOUNTING_SHA = "f9294a2e4b2220aa7552ebed7d2480e7e7647744fe5a62330888acba569af5bb"
TOLERANCE = 1e-10


def checked_source(root):
    path = Path(root) / "src/exia/genetic_primus/accounting_v2.py"
    raw = path.read_bytes()
    if len(raw) > 100_000 or sha256(raw).hexdigest() != ACCOUNTING_SHA:
        raise ValueError("unreviewed accounting source")
    tree = ast.parse(raw.decode("utf-8-sig"))
    allowed_from = {"__future__", "dataclasses", "typing"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(x.name not in {"numpy", "pandas"} for x in node.names):
            raise ValueError("unreviewed import")
        if isinstance(node, ast.ImportFrom) and (node.level or node.module not in allowed_from):
            raise ValueError("unreviewed from-import")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {
            "open", "eval", "exec", "compile", "__import__"
        }:
            raise ValueError("dynamic/file operation in accounting source")
    if any(not isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))
           and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))
           for n in tree.body):
        raise ValueError("unexpected top-level operation")
    return path, raw


def load_verified_accounting(root, allowed_output):
    """Executes pinned offline source only, bypassing package init and .pyc caches."""
    root, allowed_output = Path(root).resolve(), Path(allowed_output).resolve()
    allowed_output.relative_to(root / "Reports/Exia/Genetic_Primus")
    if Path(sys.executable).resolve() != root / ".venv/Scripts/python.exe":
        raise ValueError("only the workspace Python is allowed")
    path, raw = checked_source(root)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    sys.dont_write_bytecode = True
    denied = []
    def audit(event, args):
        if event.startswith("socket.") or event in {
            "subprocess.Popen", "os.system", "os.spawn", "os.posix_spawn", "os.fork",
            "os.remove", "os.rename", "os.rmdir"
        }:
            denied.append(event)
            raise PermissionError("probe blocks networking/process/destructive operations")
        if event == "open":
            target, mode, flags = args
            writing = (isinstance(mode, str) and any(x in mode for x in "wax+")) or (
                isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
            if writing:
                if not isinstance(target, (str, bytes, os.PathLike)):
                    denied.append("unscoped_write")
                    raise PermissionError("probe blocks unscoped writes")
                try:
                    Path(os.fsdecode(target)).resolve().relative_to(allowed_output)
                except ValueError:
                    denied.append("unscoped_write")
                    raise PermissionError("probe blocks writes outside new artifact directory")
        if event == "os.mkdir":
            target = Path(os.fsdecode(args[0])).resolve()
            if target != allowed_output:
                try:
                    target.relative_to(allowed_output)
                except ValueError:
                    denied.append("unscoped_mkdir")
                    raise PermissionError("probe blocks unscoped directories")
    sys.addaudithook(audit)
    namespace = ModuleType("_gp_synthetic_accounting_probe_v1")
    namespace.__file__ = str(path)
    sys.modules[namespace.__name__] = namespace
    exec(compile(raw, str(path), "exec"), namespace.__dict__)
    np, pd = namespace.np, namespace.pd
    dependencies = []
    for name, module in (("numpy", np), ("pandas", pd)):
        source = Path(module.__file__).resolve()
        dependencies.append({"name": name, "version_observed": module.__version__,
                             "package_init_path_observed": str(source),
                             "package_init_file_byte_sha256": sha256(source.read_bytes()).hexdigest(),
                             "package_loaded_bytecode_identity_verified": False})
    if any(event != "socket.gethostname" for event in denied):
        raise PermissionError("unexpected blocked operation during dependency load")
    environment = {"python_executable": sys.executable, "python_version": sys.version,
                   "accounting_source_byte_sha256": sha256(raw).hexdigest(),
                   "accounting_loading": "compile_verified_source_no_package_init_no_pyc",
                   "direct_packages_observed": dependencies, "native_thread_limits_requested": 1,
                   "source_identity_verified": True, "runtime_dependency_graph_verified": False,
                   "compiled_extensions_and_transitive_packages_verified": False,
                   "network_process_guard_installed": True,
                   "blocked_dependency_import_events": list(denied),
                   "hostname_request_allowed": False}
    return namespace, environment, denied


def rational_oracle(value, cost_bps, bar_ms, resets=()):
    """No NumPy/pandas or accounting-module calls. Exact cash conservation equations."""
    symbols = tuple(value["symbols"])
    times = list(range(value["start_ms"], value["end_ms"], bar_ms))
    if not symbols or not times or len(times) > 54 or len(symbols) > 2:
        raise ValueError("oracle fixture size limit")
    rate = Fraction(str(cost_bps)) / 20_000
    prices = {(x["timestamp"], x["symbol"]): x for x in value["bars"]}
    targets = {(x["execution_timestamp"], x["symbol"]): x for x in value["targets"]}
    states = {s: {"cash": Fraction(1, len(symbols)), "qty": Fraction(0), "fees": Fraction(0),
                  "realized": Fraction(0), "entry": None} for s in symbols}
    trades, decisions, events, portfolio = [], [], [], []
    peak, max_dd = Fraction(1), Fraction(0)
    def f(number):
        return Fraction(str(number))
    def snapshot(timestamp, phase, marks):
        nonlocal peak, max_dd
        equities = [states[s]["cash"] + states[s]["qty"] * marks[s] for s in symbols]
        if min(equities) <= 0:
            raise ValueError("insolvent artificial sleeve")
        equity = sum(equities)
        peak = max(peak, equity); max_dd = max(max_dd, 1 - equity / peak)
        unrealized = sum((states[s]["qty"] * (marks[s] - states[s]["entry"]["price"]) - states[s]["entry"]["fee"])
                         if states[s]["entry"] is not None else Fraction(0) for s in symbols)
        events.append({"event_index": len(events), "timestamp": timestamp, "phase": phase,
                       "equity": float(equity), "cash": float(sum(states[s]["cash"] for s in symbols)),
                       "drawdown": float(equity / peak - 1),
                       "realized_pnl": float(sum(states[s]["realized"] for s in symbols)),
                       "unrealized_pnl": float(unrealized)})
        return equity
    def close(symbol, price, timestamp, reason):
        state = states[symbol]
        if not state["qty"]:
            return
        qty, entry = state["qty"], state["entry"]
        exit_fee = abs(qty) * price * rate
        gross = qty * (price - entry["price"])
        net = gross - entry["fee"] - exit_fee
        trades.append({"symbol": symbol, "direction": 1 if qty > 0 else -1,
                       "quantity": float(qty), "signal_timestamp": entry["signal"],
                       "entry_timestamp": entry["timestamp"], "exit_timestamp": timestamp,
                       "entry_price": float(entry["price"]), "exit_price": float(price),
                       "entry_fee": float(entry["fee"]), "exit_fee": float(exit_fee),
                       "gross_pnl": float(gross), "net_pnl": float(net),
                       "entry_model_sha256": entry["sha"], "exit_reason": reason})
        state["cash"] += qty * price - exit_fee
        state["fees"] += exit_fee; state["realized"] += net
        state["qty"] = Fraction(0); state["entry"] = None
    initial_marks = {s: f(prices[times[0], s]["open"]) for s in symbols}
    previous = snapshot(value["start_ms"], "initial", initial_marks)
    for timestamp in times:
        opens = {s: f(prices[timestamp, s]["open"]) for s in symbols}
        closes = {s: f(prices[timestamp, s]["close"]) for s in symbols}
        snapshot(timestamp, "open_mark", opens)
        for symbol in symbols:
            state, row = states[symbol], targets[timestamp, symbol]
            before = 1 if state["qty"] > 0 else -1 if state["qty"] < 0 else 0
            desired, reset = row["target"], timestamp in resets
            if state["qty"] and (reset or before != desired):
                close(symbol, opens[symbol], timestamp, "retrain_forced_exit" if reset else "target_change")
            if not state["qty"] and desired:
                # Solve |q|*price + entry_fee = available sleeve equity.
                quantity = desired * state["cash"] / (opens[symbol] * (1 + rate))
                fee = abs(quantity) * opens[symbol] * rate
                state["cash"] -= quantity * opens[symbol] + fee
                state["qty"] = quantity; state["fees"] += fee
                state["entry"] = {"price": opens[symbol], "fee": fee, "timestamp": timestamp,
                                  "signal": row["signal_timestamp"], "sha": row["model_sha256"]}
            decisions.append({"symbol": symbol, "execution_timestamp": timestamp,
                              "signal_timestamp": row["signal_timestamp"], "model_sha256": row["model_sha256"],
                              "position_before": before, "desired_position": desired,
                              "quantity_after": float(state["qty"]), "forced_reset": reset,
                              "derived_entry_model_sha256": state["entry"]["sha"] if state["entry"] else None})
        snapshot(timestamp, "after_fills", opens)
        close_time = timestamp + bar_ms
        equity = snapshot(close_time, "close_mark", closes)
        if close_time == value["end_ms"]:
            for symbol in symbols:
                close(symbol, closes[symbol], close_time, "interval_end")
            equity = snapshot(close_time, "terminal_fills", closes)
        portfolio.append({"timestamp": close_time, "net_return": float(equity / previous - 1), "equity": float(equity)})
        previous = equity
    return {"trades": trades, "decisions": decisions, "events": events, "portfolio": portfolio,
            "metrics": {"final_equity": float(previous), "net_return": float(previous - 1),
                        "max_drawdown": float(max_dd),
                        "net_realized_pnl": float(sum(x["realized"] for x in states.values())),
                        "total_fees": float(sum(x["fees"] for x in states.values())),
                        "trades": len(trades), "fills": 2 * len(trades), "decision_count": len(decisions)},
            "final_flat": all(not x["qty"] for x in states.values())}


def compare_result(actual, expected):
    count, maximum = 0, 0.0
    def equal(left, right):
        nonlocal count, maximum
        if isinstance(right, float):
            if isinstance(left, bool) or not isinstance(left, (int, float)) or not math.isfinite(left):
                raise AssertionError("nonfinite/non-numeric accounting output")
            error = abs(left - right); maximum = max(maximum, error)
            if error > TOLERANCE:
                raise AssertionError("accounting/oracle numeric mismatch")
        elif left != right:
            raise AssertionError("accounting/oracle discrete mismatch")
        count += 1
    serialized = {key: getattr(actual, key).to_dict("records") for key in ("trades", "decisions", "events", "portfolio")}
    for key, rows in serialized.items():
        equal(len(rows), len(expected[key]))
        for left, right in zip(rows, expected[key]):
            for field, value in right.items():
                if field == "derived_entry_model_sha256":
                    continue  # Not an original decision field; cannot certify it as one.
                equal(left[field], value)
    for key, value in expected["metrics"].items():
        equal(actual.metrics[key], value)
    equal(actual.metrics["market_readiness"], False)
    equal(actual.metrics["orders_enabled"], False)
    equal(actual.metrics["promotion_authority"], False)
    if actual.metrics["maximum_reconciliation_error"] > TOLERANCE:
        raise AssertionError("unexpected actual reconciliation error")
    return {"comparison_checks": count, "maximum_absolute_error": maximum,
            "tolerance": TOLERANCE, "consistent": True,
            "per_decision_entry_sha_directly_available": False}, serialized


def run_artificial_case(module, value, cost_bps, bar_ms, resets=()):
    if len(value["bars"]) > 108 or len(value["targets"]) > 108 or value["synthetic_only"] is not True:
        raise ValueError("bounded synthetic data only")
    oracle = rational_oracle(value, cost_bps, bar_ms, resets)
    result = module.simulate_cash_targets(
        module.pd.DataFrame(value["bars"]), module.pd.DataFrame(value["targets"]),
        symbols=tuple(value["symbols"]), start_timestamp=value["start_ms"], end_timestamp=value["end_ms"],
        timeframe_ms=bar_ms, round_trip_cost_bps=cost_bps,
        force_flat_timestamps=tuple(resets), record_details=True)
    checks, serialized = compare_result(result, oracle)
    return {"case": value["case"], "cost_bps": cost_bps, "forced_open_resets": list(resets),
            "checks": checks, "actual_metrics": result.metrics, "actual": serialized,
            "oracle_decision_entry_sha_derived_not_original": [x["derived_entry_model_sha256"] for x in oracle["decisions"]],
            "oracle_final_flat": oracle["final_flat"],
            "synthetic_only": True, "financial_strategy_result": False}
