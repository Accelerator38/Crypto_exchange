"""Single prespecified entry rule on the disclosed nine September origins."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.ablation_v1 import summarize_simulation
from exia.genetic_primus.accounting_v2 import simulate_cash_targets
from exia.genetic_primus.entry_hypothesis_v1 import (
    EFFICIENCY_MIN, LOOKBACK_BARS, POLICY_ID, STRENGTH_MIN,
    causal_entry_features, entry_schedule,
)
from exia.genetic_primus.prospective_contract_v1 import file_sha256, load_and_validate
from exia.genetic_primus.prospective_evaluator_v1 import _frame_sha
from exia.genetic_primus.prospective_manifest_v1 import load_verified_snapshot
from tools.run_genetic_primus_ablation_v1 import (
    ARCHIVE, CONFIG, DEFAULT_OUTPUT as ABLATION_RESULT,
    _archived_decisions, _market,
)

OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/entry_hypothesis_v1_20261001/result.json"
DT = 240 * 60_000


def _evaluate() -> dict:
    contract, _ = load_and_validate(CONFIG)
    base, lineage, origin_ends = _archived_decisions(contract)
    start = int(base["execution_timestamp"].min())
    end = origin_ends[-1]
    market, market_sha = _market(contract, start, end)
    ablation = json.loads(ABLATION_RESULT.read_text(encoding="utf-8"))
    if (ablation["contract_sha256"] != file_sha256(CONFIG)
            or ablation["market_rows_sha256"] != market_sha
            or ablation["archive_lineage"] != lineage
            or ablation["interval_start"] != start
            or ablation["interval_end_exclusive"] != end):
        raise ValueError("stage-two baseline provenance changed")
    _, panel = load_verified_snapshot(
        ARCHIVE / "data_manifest_origin_09.json", root=ROOT,
        contract_path=CONFIG, contract=contract)
    first_signal_history = start - (LOOKBACK_BARS + 1) * DT
    source = panel.loc[panel["timestamp"].ge(first_signal_history)
                       & panel["timestamp"].lt(end),
                       ["timestamp", "symbol", "close"]]
    features = causal_entry_features(source, DT)
    schedule, diagnostics = entry_schedule(base, features)
    if len(schedule) != len(market) or schedule["target"].isin((-1,)).any():
        raise ValueError("hypothesis is not a complete long-only schedule")
    costs = {"normal": float(contract["accounting"]["normal_round_trip_bps"]),
             "stress": float(contract["accounting"]["stress_round_trip_bps"])}
    evaluations = {}
    for label, cost_bps in costs.items():
        replay = simulate_cash_targets(
            market, schedule, symbols=tuple(contract["universe"]),
            start_timestamp=start, end_timestamp=end, timeframe_ms=DT,
            round_trip_cost_bps=cost_bps, record_details=True)
        evaluations[label] = summarize_simulation(
            replay, schedule, base, tuple(origin_ends), cost_bps)
    reference_names = ("NoTrade", "EMA_long_only", "GA_target_gate", "GA_entry_only")
    references = {name: ablation["policies"][name][0]["evaluations"]
                  for name in reference_names}
    return {
        "schema_version": "exia.genetic_primus.disclosed_entry_hypothesis/1",
        "evidence_class": "retrospective_disclosed_engineering_hypothesis_only",
        "policy_id": POLICY_ID,
        "rule": {"side": "long_only", "strength": "return_12/(sqrt(12)*volatility_6)",
                 "strength_min": STRENGTH_MIN, "efficiency_12_min": EFFICIENCY_MIN,
                 "entry_only": True, "exit": "EMA_proposal_changes",
                 "zero_or_nonfinite_volatility": "abstain"},
        "interval_start": start, "interval_end_exclusive": end,
        "origin_count": len(origin_ends),
        "contract_sha256": file_sha256(CONFIG),
        "ablation_result_sha256": file_sha256(ABLATION_RESULT),
        "script_sha256": file_sha256(Path(__file__)),
        "market_rows_sha256": market_sha,
        "feature_rows_sha256": _frame_sha(features),
        "schedule_sha256": _frame_sha(schedule),
        "archive_lineage": lineage,
        "long_proposal_opportunities": int(diagnostics["proposal"].eq(1).sum()),
        "qualifying_entry_opportunities": int(diagnostics["can_enter"].sum()),
        "evaluations": evaluations, "references": references,
        "safety": dict(contract["safety"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    expected = (ROOT / ".venv/Scripts/python.exe").resolve()
    if Path(sys.executable).resolve() != expected:
        raise ValueError("only the project .venv Python is authorized")
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "Reports/Exia/Genetic_Primus").resolve()):
        raise ValueError("output must stay inside Genetic Primus reports")
    if output.exists():
        raise FileExistsError("hypothesis result already exists")
    result = _evaluate()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "DISCLOSED_ENTRY_HYPOTHESIS_COMPLETE_NO_PROMOTION",
                      "qualifying_opportunities": result["qualifying_entry_opportunities"],
                      "gross_pct": round(result["evaluations"]["normal"]["gross_return_pct"], 3),
                      "net_pct": round(result["evaluations"]["normal"]["net_return_pct"], 3),
                      "stress_net_pct": round(result["evaluations"]["stress"]["net_return_pct"], 3)},
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
