"""Pure metadata/AST audit. Never imports or executes audited research code."""
from __future__ import annotations

import ast
import hashlib
import json

SAFETY_KEYS = (
    "credentials_allowed", "orders_enabled", "execution_allowed", "paper",
    "live", "promotion_authority", "training_allowed",
    "financial_evaluation_allowed", "network_allowed",
)
SOURCE_NAMES = (
    "rolling_linear_retrain_v1.py", "rolling_linear_retrain_v2.py",
    "measurement_lab_v2.py", "features.py", "prospective_continuous_v2.py",
)
PROPOSAL_SHA = "a5abd987351f2ef60ad359fd396f6c2879e4c45ee166e08fde7ec2807b94e637"


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("nonfinite JSON number")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def semantic_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def validate_proposal(proposal):
    """Accept only this exact non-executable design, not launch authority."""
    if proposal.get("schema") != "exia.genetic_primus.forward_research_proposal/1":
        raise ValueError("proposal schema")
    if (proposal.get("status") != "draft_not_executable"
            or proposal.get("scope") != "separate_proposal_not_amendment_of_frozen_contracts"):
        raise ValueError("not a separate draft")
    safety = proposal.get("safety", {})
    if set(safety) != set(SAFETY_KEYS) or any(safety[k] is not False for k in SAFETY_KEYS):
        raise ValueError("authority must remain literally false")
    # The entire design is pinned by the publisher. Check essential invariants
    # here too, so a false flag cannot mask internally enabled search or timing.
    clock, models = proposal["clock"], proposal["models"]
    expected = {"bar_hours": 4, "deployment_days": 3, "retrain_days": 3, "purge_bars": 7}
    if any(type(clock[k]) is not int or clock[k] != v for k, v in expected.items()):
        raise ValueError("chronological design changed")
    if (models["features"] != ["return_12", "efficiency_12"]
            or type(models["coefficient_count"]) is not int or models["coefficient_count"] != 3
            or models["hidden_layers"] != 0 or models["ga_enabled"] is not False
            or models["variants"] != ["uniform", "half_life_90d_anchor_20pct"]
            or models["full_coefficients_and_scaler_refit_each_origin"] is not True
            or models["recency"]["weight_all_outcomes_including_losses"] is not True
            or models["minimum_trades_forcing"] is not False):
        raise ValueError("candidate design changed")
    for key in ("training_latency_budget_seconds", "execution_delay_seconds"):
        if clock[key] is not None:
            raise ValueError("timing not registered in this draft")
    if (proposal["labels"]["new_executable_label_definition"] is not None
            or proposal["accounting"]["cost_adapter"] is not None
            or proposal["accounting"]["normal_and_stress_cost_parameters"] is not None
            or proposal["accounting"]["synthetic_results_are_profit_evidence"] is not False):
        raise ValueError("unresolved execution is not evidence")
    budget, sealed = proposal["testing_budget"], proposal["sealed"]
    if (type(budget["authorized_market_runs"]) is not int or budget["authorized_market_runs"] != 0
            or budget["planned_candidate_pipelines"] != 2
            or budget["planned_hyperparameter_search_trials"] != 0
            or budget["historical_global_trial_count"] is not None
            or budget["historical_dsr_pbo_verified"] is not False
            or budget["registered_dates_and_origin_count"] is not None
            or budget["power_minimum_effect_and_block_design"] is not None):
        raise ValueError("unregistered testing budget")
    if (sealed["automatic_old_interval_reuse"] is not False or sealed["peek_allowed"] is not False
            or sealed["existing_frozen_contracts_unchanged"] is not True
            or sealed["new_interval_start_utc"] is not None
            or sealed["new_interval_end_exclusive_utc"] is not None):
        raise ValueError("sealed authority changed")
    if (proposal["lineage"]["all_bindings_verified_before_launch"] is not False
            or proposal["readiness"]["market_research_launch_ready"] is not False
            or proposal["readiness"]["real_money_ready"] is not False
            or len(set(proposal["readiness"]["blocking_requirements"])) != 10):
        raise ValueError("draft cannot earn readiness")
    if (type(proposal["measurement"]["economic_floor_bps_per_day"]) not in (int, float)
            or proposal["measurement"]["economic_floor_bps_per_day"] != 0.0
            or proposal["measurement"]["user_allows_drawdowns"] is not True):
        raise ValueError("user economic criterion changed")
    sha = semantic_sha(proposal)
    if sha != PROPOSAL_SHA:
        raise ValueError("pinned draft design changed; create a separate version")
    return sha


def function_node(tree, name):
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(nodes) != 1:
        raise ValueError("missing or ambiguous source function: " + name)
    return nodes[0]


def anchor(source, name, expression):
    """Require an exact AST expression/statement; no behavioral proof implied."""
    tree = ast.parse(source)
    function = function_node(tree, name)
    expected = ast.parse(expression).body[0]
    if isinstance(expected, ast.Expr):
        expected = expected.value
    dump = ast.dump(expected, include_attributes=False)
    matches = [node for node in ast.walk(function)
               if ast.dump(node, include_attributes=False) == dump]
    if len(matches) != 1:
        raise ValueError("source anchor changed: " + name + ": " + expression)
    node = matches[0]
    return {"function": name, "line": node.lineno, "end_line": node.end_lineno,
            "ast_sha256": hashlib.sha256(dump.encode()).hexdigest()}


def audit(proposal, sources):
    if set(sources) != set(SOURCE_NAMES):
        raise ValueError("explicit source scope required")
    proposal_sha = validate_proposal(proposal)
    observations = []

    def add(identifier, file, function, expression, conclusion, limitation):
        observations.append({"id": identifier, "file": file,
                             "status": "source_anchor_observed_not_execution_verified",
                             "anchor": anchor(sources[file], function, expression),
                             "observation": conclusion, "limitation": limitation})

    v1, v2, lab, features, continuous = SOURCE_NAMES
    add("label_cutoff", v1, "fit_at_origin",
        "examples.origin_end_exclusive.le(cutoff)",
        "Training selects completed label ends before the purged cutoff.",
        "No independent label_available_at or data revision availability guard.")
    add("purge", v1, "fit_at_origin", "cutoff = deploy_start_ms - purge_bars * DT",
        "Purge is measured in bars relative to deployment start.",
        "No separate training_asof or measured fit latency in this path.")
    add("label_reference", v1, "build_origin_examples", "returns = closes / opens - 1.0",
        "Labels describe gross open-to-last-close return for a three-day origin.",
        "Execution delays, fills, funding and intrabar risk are not label semantics.")
    add("examples_require_future_prices", v1, "build_origin_examples",
        "closes = prices.reindex(close_index).close.to_numpy(float)",
        "Building the shared examples table requires the future origin terminal close.",
        "Legitimate for completed training labels/offline backtests, not an online deploy-feature builder; split payloads.")
    add("prior_features", v1, "build_origin_examples", "previous = start - DT",
        "Feature lookup uses the previous bar timestamp.",
        "Timestamp-as-open, exact grid and close availability require data verification.")
    add("scaler_train_only", v1, "fit_at_origin", "x = train[list(FEATURES)].to_numpy(float)",
        "Scaler input is selected training feature values, not the deploy frame.",
        "Static dataflow observation, not prefix-invariance or actual payload verification.")
    add("scaler_refit", v1, "fit_at_origin", "median = np.median(x, axis=0)",
        "Median is recomputed in the per-origin fit function.",
        "Rows/payload hashes do not alone attest source data or fit completion time.")
    add("all_outcome_weights", v1, "fit_at_origin", "weights = 0.20 / n + 0.80 * recent / recent.sum()",
        "Recency weights cover all selected proposal outcomes, including losses.",
        "Proposal-conditioned sample is not every universe row; ESS is not independent sample count.")
    add("gross_not_fixed_net", v2, "fit_gross_at_origin",
        "model = fit_at_origin(examples, deploy_start_ms, variant, purge_bars=purge_bars, ridge_alpha=ridge_alpha, normal_round_trip_bps=0.0, minimum_opportunities=minimum_opportunities, minimum_ess=minimum_ess)",
        "V2 fits gross labels rather than subtracting a fixed round-trip cost per label.",
        "Decision cost approximation still needs execution-aligned validation.")
    add("full_linear_retrain", v2, "build_rolling_schedule",
        "model = fit_gross_at_origin(examples, origin, variant, purge_bars=purge_bars, ridge_alpha=ridge_alpha, round_trip_cost_bps=round_trip_cost_bps, minimum_opportunities=minimum_opportunities, minimum_ess=minimum_ess)",
        "The linear schedule builder calls full fitting at each origin.",
        "No fit_started/completed/available timestamps or end-to-end market contract binding.")
    add("decision_model_sha", v2, "schedule_with_position", 'schedule["model_sha256"] = model["model_sha256"]',
        "Schedule rows carry the active model SHA.",
        "Entry model identity and open-position state require the accounting integration.")
    add("same_examples_deploy", v2, "schedule_with_position",
        'deploy = examples.loc[examples.origin_start.eq(deploy_start_ms)].set_index("symbol")',
        "Deploy input comes from the examples table that also contains future labels.",
        "Inspected prediction reads features/proposal, not labels; shared schema is a leakage hazard, not proof of leakage.")
    add("trailing_features", features, "_per_symbol_features", "ema_12 = close.ewm(span=12, adjust=False).mean()",
        "EMA reference uses a trailing recursive calculation.",
        "Exact feature prefix invariance and missing/revised-bar behavior remain untested here.")
    add("prior_bar_alignment", lab, "from_frame", 'prior["timestamp"] += timeframe_ms',
        "Feature timestamps are shifted by one bar for execution alignment.",
        "This does not establish nonzero decision/compute/submit/fill latency.")
    add("zero_latency_reference", lab, "from_frame", 'base["signal_timestamp"] = base.execution_timestamp',
        "PreparedWindow labels signal and execution with the same timestamp.",
        "Zero-latency engineering reference is not a realistic execution contract.")
    add("prospective_fixed_genomes", continuous, "evaluate_committed_chain",
        "schedules[name].append(deploy.schedule(genome, scaler))",
        "Continuous prospective schedules the supplied genomes while refitting a scaler.",
        "This is not full coefficient retraining; do not mix these algorithms in ranking.")
    return {"schema": "exia.genetic_primus.forward_research_static_audit/1",
            "proposal_semantic_sha256": proposal_sha,
            "source_scope": list(SOURCE_NAMES), "observations": observations,
            "observation_count": len(observations),
            "scope": "selected_source_ast_and_metadata_not_full_repository_or_behavioral_audit",
            "market_data_read": False, "audited_code_executed": False,
            "new_model_fits": 0, "new_market_strategy_evaluations": 0,
            "market_research_launch_ready": False, "real_money_ready": False,
            "safety": dict(proposal["safety"]),
            "next_action": "isolated_causal_origin_payload_contract_and_synthetic_metadata_guards_before_training"}
