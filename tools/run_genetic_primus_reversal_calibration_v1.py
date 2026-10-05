"""One explicit local step per invocation; no network, subprocesses or runtime."""

import argparse
from hashlib import sha256
import json
from math import isclose, prod
import os
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Reports/Exia/Genetic_Primus/reversal_calibration_pilot_v1_20261005"
PILOT = "configs/genetic_primus_reversal_calibration_pilot_v1.json"
PILOT_SHA = "beb3327263492ea13da28f77c8ff53ef665f6a43da1d5369323497b23528fb02"
DRAFT = "configs/genetic_primus_market_relative_reversal_draft_v1.json"
DRAFT_SHA = "4779b762c65c814d25278e7774a3288d87ce9aaa183932bc735a2f27981a1224"
SOURCES = {
    "builder": "src/exia/genetic_primus/market_relative_reversal_builder_v1.py",
    "fitter": "src/exia/genetic_primus/market_relative_reversal_linear_v1.py",
    "evaluator": "src/exia/genetic_primus/market_relative_reversal_rolling_v1.py",
    "reader": "src/exia/genetic_primus/reversal_calibration_reader_v1.py",
    "statistics": "src/exia/genetic_primus/reversal_calibration_v1.py",
    "runner": "tools/run_genetic_primus_reversal_calibration_v1.py",
    "safety_checker": "tools/check_genetic_primus_reversal_pilot_synthetic_v1.py",
}
FORBIDDEN = ("credentials_allowed", "orders_enabled", "paper", "live", "promotion_authority")


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def load(name):
    return runpy.run_path(str(ROOT / SOURCES[name]))


def write_new(path, value):
    """Complete temp + exclusive hard link: never replace an existing artifact.

    No fsync loop. A power failure is detected on resume by hashes/JSON parsing.
    """
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode()
    pending = path.with_suffix(path.suffix + ".pending")
    if path.exists() or pending.exists():
        raise ValueError("artifact already exists or incomplete prior write; no overwrite")
    with pending.open("xb") as stream:
        stream.write(data)
        stream.flush()
    os.link(pending, path)  # Fails, without overwrite, if destination exists.
    pending.unlink()  # Only the temporary file just created by this invocation.


def source_hashes():
    return {name: file_sha(ROOT / path) for name, path in SOURCES.items()}


def frozen_inputs():
    if file_sha(ROOT / PILOT) != PILOT_SHA or file_sha(ROOT / DRAFT) != DRAFT_SHA:
        raise ValueError("frozen draft changed")
    return json.loads((ROOT / PILOT).read_bytes())


def register():
    pilot = frozen_inputs()
    reader, stats = load("reader"), load("statistics")
    entries = reader["verify_snapshot"](ROOT)
    folds = stats["calibration_folds"]()
    if folds["folds_sha256"] != pilot["folds_sha256"]:
        raise ValueError("fold plan changed")
    prior_registry = "configs/genetic_primus_objective_registry_v9.json"
    r = {"schema": "exia.genetic_primus.reversal_calibration_registration/1",
         "objective_id": pilot["objective_id"], "registration_date_utc": "2026-10-05",
         "status": "REGISTERED_REVEALED_CALIBRATION_ONLY",
         "evidence_class": pilot["evidence_class"], "candidate_budget": 1, "origin_count": 20,
         "user_authorized_one_offline_pilot": True,
         "authorization": "Current human user confirmed one local offline pilot; no trading/network authority",
         "start_ms": reader["START_MS"], "end_ms": reader["END_MS"],
         "train_anchor_ms": reader["TRAIN_ANCHOR_MS"], "input_lower_ms": reader["LOWER_MS"],
         "pilot_draft_path": PILOT, "pilot_draft_sha256": PILOT_SHA,
         "model_draft_path": DRAFT, "model_draft_sha256": DRAFT_SHA,
         "data_use_override": "Only this fixed revealed calibration, despite engineering-only model draft; no outer access",
         "source_manifest_path": reader["MANIFEST"], "source_manifest_sha256": reader["MANIFEST_SHA"],
         "snapshot_sha256": reader["SNAPSHOT_SHA"], "snapshot_files": entries,
         "implementation_paths": SOURCES, "sources_sha256": source_hashes(), **folds,
         "previous_registry_path": prior_registry, "previous_registry_sha256": file_sha(ROOT / prior_registry),
         "registry_coverage": "Hash-linked local registration; historical adaptive trials still incomplete",
         "analysis": pilot["analysis"], "no_retuning": True, "no_network": True,
         "funding_verified": False, "real_execution_verified": False,
         "global_dsr_pbo_calibrated": False, **{k: False for k in FORBIDDEN}}
    r["registration_sha256"] = digest(r)
    OUT.mkdir(parents=True, exist_ok=True)
    write_new(OUT / "registration.json", r)
    print(json.dumps({"registered": True, "snapshot_files_verified": len(entries),
                      "candidate_budget": 1, "origins": 20, "registration_sha256": r["registration_sha256"]}))


def registration():
    frozen_inputs()
    r = json.loads((OUT / "registration.json").read_bytes())
    if (digest({k: v for k, v in r.items() if k != "registration_sha256"}) != r["registration_sha256"]
            or r["sources_sha256"] != source_hashes()
            or r["model_draft_sha256"] != DRAFT_SHA or r["pilot_draft_sha256"] != PILOT_SHA
            or r["implementation_paths"] != SOURCES or any(r[k] is not False for k in FORBIDDEN)
            or file_sha(ROOT / r["previous_registry_path"]) != r["previous_registry_sha256"]
            or load("statistics")["calibration_folds"]()["folds_sha256"] != r["folds_sha256"]):
        raise ValueError("registered implementation/contract chain changed")
    return r


def step_path(index):
    return OUT / f"origin_{index:02d}.json"


def read_step(index, r):
    value = json.loads(step_path(index).read_bytes())
    if (digest({k: v for k, v in value.items() if k != "step_sha256"}) != value["step_sha256"]
            or value["registration_sha256"] != r["registration_sha256"]
            or value["index"] != index or value["result"]["origin_ms"] != r["folds"][index-1]["origin_ms"]):
        raise ValueError("step hash/index/registration mismatch")
    return value


def terminal_accounting(scenario, transition_count):
    state = scenario["state"]
    fee_rate = scenario["round_trip_cost_bps"]/20_000
    terminal_fee = sum(p["quantity"]*p["last_mark_price"]*fee_rate
                       for p in state["positions"].values())/8
    return {"marked_net_return": state["last_nav"]-1,
        "hypothetical_terminal_liquidation_fee": terminal_fee,
        "net_after_terminal_liquidation": state["last_nav"]-terminal_fee-1,
        "max_drawdown_close": state["max_drawdown_close"], "transition_count": transition_count,
        "cumulative_fees_initial_nav": state["cumulative_fees"]}


def step(index):
    if not 1 <= index <= 20:
        raise ValueError("one origin index 1..20 required")
    r = registration()
    if any(step_path(i).exists() for i in range(index, 21)):
        raise ValueError("already evaluated/out-of-order: no repeat")
    previous, previous_sha = None, None
    for i in range(1, index):
        value = read_step(i, r)
        if (value["previous_step_sha256"] != previous_sha
                or value["previous_state_sha256"] != (previous["state_sha256"] if previous else None)):
            raise ValueError("state/result chain broken")
        previous, previous_sha = value["result"]["state"], value["step_sha256"]
    b, f, e, reader = (load(name) for name in ("builder", "fitter", "evaluator", "reader"))
    fold = r["folds"][index-1]
    input_start = fold["first_train_origin_ms"]-13*reader["BAR_MS"]
    candles = reader["read_bounded"](ROOT, start_ms=input_start,
        end_ms=fold["deployment_end_exclusive_ms"], candle_type=b["Candle"])
    primitives = e["Primitives"](b["build_features"], b["build_training_origin"], f["Observation"],
        f["fit_origin"], f["predict_gross"], r["sources_sha256"]["builder"], r["sources_sha256"]["fitter"])
    result = e["evaluate_origin"](candles, origin_ms=fold["origin_ms"],
        train_start_origin_ms=r["train_anchor_ms"], previous_state=previous, primitives=primitives,
        evaluator_sha256=r["sources_sha256"]["evaluator"], draft_sha256=DRAFT_SHA,
        data_kind="revealed_calibration", calibration_registration=r)
    value = {"index": index, "registration_sha256": r["registration_sha256"],
             "input_start_ms": input_start, "input_end_ms": fold["deployment_end_exclusive_ms"],
             "input_rows": len(candles), "previous_step_sha256": previous_sha,
             "previous_state_sha256": previous["state_sha256"] if previous else None, "result": result}
    value["step_sha256"] = digest(value)
    write_new(step_path(index), value)
    print(json.dumps({"origin": index, "rows": len(candles), "decisions": len(result["decisions"]),
                      "fallback": result["model"]["fallback"], "saved": True}))


def summarize():
    r = registration()
    stats = load("statistics")
    daily = {name: [] for name in ("normal", "stress")}
    signals, origins, previous_sha, previous_state_sha = [], [], None, None
    event_count = {name: 0 for name in daily}
    for i in range(1, 21):
        v = read_step(i, r)
        if v["previous_step_sha256"] != previous_sha or v["previous_state_sha256"] != previous_state_sha:
            raise ValueError("summary chain broken")
        result = v["result"]
        model = result["model"]
        if len(result["decisions"]) != 8:
            raise ValueError("decision count mismatch")
        for d in result["decisions"]:
            if (d["model_sha256"] != model["model_sha256"] or d["train_cutoff"] != r["folds"][i-1]["train_cutoff_ms"]
                    or any(d[k] != model[k] for k in ("scaler_sha256", "folds_sha256", "weights_sha256"))):
                raise ValueError("decision/model lineage mismatch")
        origins.append(result["origin_ms"])
        signals.append(any(d["target"] == 1 for d in result["decisions"]))
        for name in daily:
            scenario = result["scenarios"][name]
            daily[name].extend(scenario["daily_returns"])
            event_count[name] += len(scenario["events"])
            if not isclose(prod(1+x for x in daily[name]), scenario["state"]["last_nav"], abs_tol=1e-12):
                raise ValueError("calendar-return NAV reconciliation failed")
        previous_sha, previous_state_sha = v["step_sha256"], result["state"]["state_sha256"]
    summary = stats["summarize_calibration"](daily_returns=daily, origins=origins, signal_origins=signals)
    accounting = {}
    for name, scenario in result["scenarios"].items():
        accounting[name] = terminal_accounting(scenario, event_count[name])
    final = {"schema": "exia.genetic_primus.reversal_calibration_result/1",
        "registration_sha256": r["registration_sha256"], "last_step_sha256": previous_sha,
        "summary": summary, "accounting": accounting, "daily_returns": daily,
        "signal_origins": signals, "NoTrade_net_return": 0.0,
        "independent_validation": False, "funding_verified": False, "real_execution_verified": False,
        "nonnegative_final_net_all_costs_proven": False, **{k: False for k in FORBIDDEN}}
    final["result_sha256"] = digest(final)
    write_new(OUT / "result.json", final)
    print(json.dumps({"saved": True, "origins": 20, "days": len(daily["normal"]),
                      "signal_origins": sum(signals), "accounting": accounting}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "step", "summarize"))
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.action == "step":
        if args.index is None:
            parser.error("step requires --index")
        step(args.index)
    elif args.index is not None:
        parser.error("--index only allowed with step")
    elif args.action == "register":
        register()
    else:
        summarize()


if __name__ == "__main__":
    main()
