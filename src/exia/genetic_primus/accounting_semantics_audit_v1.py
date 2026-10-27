"""Static code/provenance and stored-number checks; never execute a simulator."""

import ast
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math


SCHEMA = "exia.genetic_primus.accounting_semantics_audit/1"
TOLERANCE = 1e-10


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return sha256(canonical(value)).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


@dataclass(frozen=True)
class Source:
    path: str
    byte_sha256: str
    historical_binding: str
    text: str


def source(path, raw, expected_sha, *, historically_bound=False):
    hashed = sha256(raw).hexdigest()
    require(hashed == expected_sha, "static source fingerprint mismatch")
    text = raw.decode("utf-8-sig")
    ast.parse(text)  # Parse only. No import, compile-to-execution or eval.
    return Source(path, hashed, "result_recorded_file_hash" if historically_bound else "reviewed_current_source_only", text)


def function(tree, name):
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(matches) == 1, "ambiguous or missing static function: " + name)
    return matches[0]


def proof(blob, node, rule):
    return {"path": blob.path, "byte_sha256": blob.byte_sha256, "binding": blob.historical_binding,
            "start_line": node.lineno, "end_line": node.end_lineno, "rule": rule,
            "ast_semantic_sha256": sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()}


def simulator_call(blob, function_name):
    tree = ast.parse(blob.text)
    node = function(tree, function_name)
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "simulate_cash_targets"]
    require(len(calls) == 1, "unsupported simulator call topology")
    call = calls[0]
    require(all(keyword.arg is not None for keyword in call.keywords), "opaque simulator kwargs")
    kwargs = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
    require("start_timestamp" in kwargs and "end_timestamp" in kwargs and "round_trip_cost_bps" in kwargs,
            "simulator interval/cost arguments missing")
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
               and n.module == "exia.genetic_primus.accounting_v2"
               and any(a.name == "simulate_cash_targets" and a.asname is None for a in n.names)]
    require(len(imports) == 1, "unsupported simulator import binding")
    return {"function": function_name, "kwargs_expressions_not_evaluated": kwargs,
            "explicit_force_flat_argument": "force_flat_timestamps" in kwargs,
            "proof": proof(blob, call, "historically_hash_bound_callsite_static_only"),
            "dependency_runtime_identity_verified": False}


def current_simulator_profile(blob):
    tree = ast.parse(blob.text)
    simulate = function(tree, "simulate_cash_targets")
    defaults = dict(zip((a.arg for a in simulate.args.kwonlyargs), simulate.args.kw_defaults))
    require(isinstance(defaults.get("force_flat_timestamps"), ast.Tuple)
            and not defaults["force_flat_timestamps"].elts, "reviewed force-flat default changed")
    entries = [n for n in ast.walk(function(tree, "enter")) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "notional" for t in n.targets)]
    require(len(entries) == 1 and ast.unparse(entries[0].value) == "self.cash / (1.0 + self.fee_rate)",
            "reviewed entry sizing changed")
    sleeves = [n for n in ast.walk(simulate) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "CashSleeve"]
    require(len(sleeves) == 1 and [ast.unparse(a) for a in sleeves[0].args] == [
        "1.0 / len(symbols)", "round_trip_cost_bps / 20000.0"], "reviewed sleeve/cost basis changed")
    endings = [n for n in ast.walk(simulate) if isinstance(n, ast.If)
               and ast.unparse(n.test) == "close_time == end_timestamp"]
    require(len(endings) == 1, "reviewed terminal condition changed")
    snapshots = sorted({n.args[1].value for n in ast.walk(simulate)
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "snapshot"
                        and len(n.args) > 1 and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str)})
    require(set(snapshots) == {"initial", "open_mark", "after_fills", "close_mark", "terminal_fills"},
            "reviewed valuation phases changed")
    return {"source_path": blob.path, "byte_sha256": blob.byte_sha256,
            "historical_dependency_identity_verified": False,
            "observed_current_rules": {
                "initial_portfolio_equity": 1.0, "sleeves": "equal_initial_capital_no_cross_sleeve_rebalancing",
                "entry_notional": "sleeve_cash/(1+per_side_fee_rate)",
                "held_quantity": "unchanged_until_explicit_exit",
                "per_side_fee_rate": "round_trip_bps/20000", "fee_basis": "actual_simulated_fill_notional",
                "mark": "cash+signed_quantity*mark_price", "position_carry": "permitted_if_target_unchanged_and_no_reset",
                "force_flat_default": "empty_tuple", "mandatory_terminal": "whole_simulation_interval_final_close",
                "drawdown_sampling": snapshots, "intrabar_high_low_risk_verified": False,
                "funding_borrow_real_fills_verified": False},
            "proofs": [proof(blob, simulate, "reviewed_current_static_simulator"),
                       proof(blob, entries[0], "reviewed_current_sizing_formula"),
                       proof(blob, function(tree, "close"), "reviewed_current_fill_fee_and_realized_pnl"),
                       proof(blob, function(tree, "snapshot"), "reviewed_current_mark_and_drawdown")],
            "historical_accounting_verified": False}


def current_daily_observation(blob, function_name):
    tree = ast.parse(blob.text)
    node = function(tree, function_name)
    products = [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "np" and n.func.attr == "prod"]
    divisions = [n for n in ast.walk(node) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.FloorDiv)]
    require(products and divisions, "reviewed daily compounding/bucket pattern missing")
    return {"current_observed_rule": "compound_bar_returns_in_equal_24_hour_blocks_from_first_portfolio_close",
            "calendar_day_alignment": "requires_verified_complete_bar_grid_and_midnight_interval_start",
            "zero_return_day_completeness_verified": False, "historical_helper_identity_verified": False,
            "proof": proof(blob, node, "reviewed_current_daily_aggregation_static_only")}


def finite(value):
    if type(value) not in (int, float):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


def compound(returns):
    if not isinstance(returns, list) or not returns:
        return None
    values = [finite(value) for value in returns]
    if any(value is None or value <= -1 for value in values):
        return None
    try:
        result = math.prod(1.0 + value for value in values) - 1.0
        return result if math.isfinite(result) else None
    except OverflowError:
        return None


def equality(name, lhs, rhs, fields):
    lhs, rhs = finite(lhs), finite(rhs)
    delta = lhs - rhs if lhs is not None and rhs is not None else None
    return {"check": name, "status": "unknown" if delta is None else "consistent" if abs(delta) <= TOLERANCE else "inconsistent",
            "lhs": lhs, "rhs": rhs, "residual": delta, "absolute_tolerance": TOLERANCE,
            "source_fields": fields, "historical_execution_verified": False}


def scaled(value, scale=1):
    value = finite(value)
    return value / scale if value is not None else None


def sum_fields(collection, key, *, scale=1):
    values = list(collection.values()) if isinstance(collection, dict) else collection
    if not isinstance(values, list) or not values:
        return None
    extracted = [scaled(v.get(key), scale) if isinstance(v, dict) else None for v in values]
    return math.fsum(extracted) if all(v is not None for v in extracted) else None


def stored_checks(metrics, *, percent=False, economics=None, expected_days=None):
    """Arithmetic of persisted summaries, never a new strategy evaluation."""
    net = scaled(metrics.get("net_return_pct" if percent else "net_return"), 100 if percent else 1)
    returns = metrics.get("daily_returns")
    checks = [equality("daily_compounding_matches_reported_net", compound(returns), net,
                       ["daily_returns", "net_return_pct" if percent else "net_return"])]
    if expected_days is not None:
        checks.append(equality("reported_daily_count_matches_interval", len(returns) if isinstance(returns, list) else None,
                               expected_days, ["daily_returns", "bound_interval"]))
    if percent:
        gross, fees = scaled(metrics.get("gross_return_pct"), 100), scaled(metrics.get("fees_pct"), 100)
        checks.append(equality("same_trajectory_gross_minus_fees_matches_net",
                               gross - fees if gross is not None and fees is not None else None, net,
                               ["gross_return_pct", "fees_pct", "net_return_pct"]))
    else:
        # Do not substitute another scenario's gross_zero_cost net for this trajectory's gross PnL.
        fees = scaled(metrics.get("fees"))
        checks.append({"check": "same_trajectory_gross_minus_fees_matches_net", "status": "unknown",
                       "unknown_reason": "gross_pnl_not_reported_for_this_trajectory_no_cross_scenario_substitution",
                       "source_fields": [], "historical_execution_verified": False})
    if economics is not None:
        checks += [equality("economics_net_matches_summary_net", scaled(economics.get("net_pct_initial_capital"), 100), net,
                            ["economics.net_pct_initial_capital", "metrics.net_return_pct"]),
                   equality("economics_fees_matches_summary_fees", scaled(economics.get("fees_pct_initial_capital"), 100), fees,
                            ["economics.fees_pct_initial_capital", "metrics.fees_pct"]),
                   equality("symbol_net_contributions_match_portfolio_net",
                            sum_fields(economics.get("by_symbol"), "net_pnl_pct_initial_capital", scale=100), net,
                            ["economics.by_symbol.*.net_pnl_pct_initial_capital", "metrics.net_return_pct"])]
        origins = economics.get("by_origin")
        origin_returns = [scaled(v.get("net_return_pct"), 100) for v in origins] if isinstance(origins, list) and all(isinstance(v, dict) for v in origins) else None
        checks.append(equality("continuous_origin_block_compounding_matches_net", compound(origin_returns), net,
                               ["economics.by_origin[*].net_return_pct", "metrics.net_return_pct"]))
    return checks


def terminal_finding(declared, call, profile):
    mismatch = (declared == "origin_final_candle_close" and call["explicit_force_flat_argument"] is False
                and profile["observed_current_rules"]["mandatory_terminal"] == "whole_simulation_interval_final_close")
    return {"finding": "terminal_scope_declaration_vs_current_default",
            "status": "potential_mismatch" if mismatch else "not_established",
            "declared": declared, "runner_explicit_force_flat_argument": call["explicit_force_flat_argument"],
            "observed_current_default": profile["observed_current_rules"]["mandatory_terminal"],
            "historical_execution_mismatch_proven": False,
            "cannot_infer_every_position_actually_carried": True,
            "action": "register_forward_terminal_and_retrain_carry_rules_explicitly_do_not_rewrite_old_results"}


def observation_profile(result_path, result_sha, contract_path, contract_sha, declared, call, current_profile):
    value = {"result_path": result_path, "result_byte_sha256": result_sha,
             "contract_path": contract_path, "contract_byte_sha256": contract_sha,
             "declared_accounting": declared, "producer_callsite": call,
             "current_simulator_byte_sha256": current_profile["byte_sha256"],
             "findings": [terminal_finding(declared.get("terminal_exit"), call, current_profile),
                          {"finding": "historical_dependency_fingerprint_missing", "status": "unresolved",
                           "dependency": "accounting_v2.py", "runtime_identity_verified": False}],
             "strict_historical_accounting_verified": False, "cross_source_comparison_authority": False,
             "promotion_authority": False}
    value["profile_sha256"] = digest(value)
    return value
