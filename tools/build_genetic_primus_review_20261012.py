"""Bounded serial audit/export of existing frozen evidence. No new backtest.

Run explicit stages: audit, export, check, vectors, charts, seal. Exclusive writes.
No exchange access, collector, runtime import, fitting, pytest or background work.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["MPLBACKEND"] = "Agg"
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "Reports/Exia/Genetic_Primus"
OUT = BASE / "research_review_20261012"
BUNDLE = OUT / "model_bundle"
CHAINS = ("cadence_matrix_v1_20261009", "label_phase_v1_20261012",
          "robust_linear_v1_20261012", "causal_capital_controls_v1_20261012",
          "position_policy_v1_20261012")
POLICY = BASE / CHAINS[-1]
CADENCE = BASE / CHAINS[0]
HUBER = BASE / CHAINS[2]
DATA = BASE / "compact_postclose_research_v1_20261008/data_bank.npz"
FLAGS = {k: False for k in ("credentials_allowed", "orders_enabled", "execution_allowed",
    "paper", "live", "promotion_authority", "training_allowed", "financial_evaluation_allowed", "network_allowed")}


def identity(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(p, sealed=True):
    if p.stat().st_size > 16_000_000:
        raise ValueError("bounded JSON read exceeded")
    v = json.loads(p.read_text(encoding="utf-8"))
    if sealed and v.get("sha256") != identity({k: x for k, x in v.items() if k != "sha256"}):
        raise ValueError(f"body SHA mismatch: {p.name}")
    return v


def pin(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1_048_576), b""):
            h.update(b)
    return {"sha256": h.hexdigest(), "bytes": p.stat().st_size}


def write(p, v):
    v = copy.deepcopy(v)
    v["sha256"] = identity({k: x for k, x in v.items() if k != "sha256"})
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(v, f, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def verify_pin(p, expected):
    actual = pin(p)
    if actual["sha256"] != expected["sha256"] or ("bytes" in expected and actual["bytes"] != expected["bytes"]):
        raise ValueError(f"file SHA mismatch: {p}")


def audit():
    checks = []
    for name in CHAINS:
        directory = BASE / name
        receipt, reg = read(directory / "receipt.json"), read(directory / "registration.json")
        count = 0
        for filename, expected in receipt["artifact_pins"].items():
            verify_pin(directory / filename, expected)
            count += 1
        source_count = 0
        for field in ("code_pins", "input_pins", "source_pins"):
            for filename, expected in reg.get(field, {}).items():
                verify_pin(ROOT / filename, expected)
                source_count += 1
        if reg.get("safety") != FLAGS or receipt.get("safety") != FLAGS:
            raise ValueError("authority flags differ")
        checks.append({"chain": name, "artifacts_verified": count, "source_pins_verified": source_count,
                       "registration": pin(directory / "registration.json"), "receipt": pin(directory / "receipt.json")})
    completion = read(POLICY / "completion_receipt.json")
    independent = read(POLICY / "independent_audit.json")
    metrics, paths = {}, {}
    for period in ("development", "report", "q2024_1", "q2024_3", "q2025_1", "q2025_3"):
        analysis = read(POLICY / f"analysis_{period}.json")
        metrics[period], paths[period] = analysis["policies"], {}
        for policy in ("C", "C_H", "C_V", "EMA_V", "Passive_V", "EMA_M", "Passive_M"):
            item = read(POLICY / f"replay_{period}_{policy}_stress.json")
            paths[period][policy] = {"equity": item["equity"], "daily_log_returns": item["daily_log_returns"],
                                     "source": str((POLICY / f"replay_{period}_{policy}_stress.json").relative_to(ROOT)),
                                     "pin": pin(POLICY / f"replay_{period}_{policy}_stress.json")}
        if period in ("development", "report"):
            item = read(HUBER / f"replay_{period}_H_stress.json")
            metrics[period]["Huber_H3_stress"] = {"metrics": item["metrics"]}
            paths[period]["Huber_H3"] = {"equity": item["equity"], "daily_log_returns": item["daily_log_returns"],
                                        "source": str((HUBER / f"replay_{period}_H_stress.json").relative_to(ROOT)),
                                        "pin": pin(HUBER / f"replay_{period}_H_stress.json")}
    cadence = {}
    for policy in "ABCDEFG":
        cadence[policy] = {period: read(CADENCE / f"replay_{period}_{policy}_stress.json")["metrics"]
                           for period in ("development", "report")}
    compact = read(BASE / "compact_postclose_research_v1_20261008/comparison.json")
    with np.load(DATA, allow_pickle=False) as data:
        t = data["times"]
        data_info = {"source": str(DATA.relative_to(ROOT)), "pin": pin(DATA), "bars": len(t),
            "first_utc": datetime.fromtimestamp(int(t[0]) / 1000, timezone.utc).isoformat(),
            "last_utc": datetime.fromtimestamp(int(t[-1]) / 1000, timezone.utc).isoformat(),
            "features_shape": list(data["features"].shape), "regular_4h": bool(np.all(np.diff(t) == 14_400_000))}
    evidence = {"schema": "genetic_primus_review/1", "asof": "2026-10-12", "chains": checks,
        "data": data_info, "metrics": metrics, "cadence": cadence, "compact": compact["rows"],
        "pooled": read(POLICY / "summary.json"), "prior_independent_audit": independent,
        "completion_receipt": completion, "new_fits": 0, "new_financial_replays": 0,
        "new_prospective_claim": False, "safety": FLAGS}
    write(OUT / "evidence.json", evidence)
    for period in paths.values():
        for item in period.values():
            item["equity"] = [{"timestamp_ms": r["timestamp_ms"], "nav": r["nav"]} for r in item["equity"]]
    write(OUT / "equity_paths.json", {"schema": "read_only_ledger_extract/1", "periods": paths, "safety": FLAGS})
    print(json.dumps({"chains": len(checks), "artifact_hashes_verified": sum(c["artifacts_verified"] for c in checks),
                      "data_last": data_info["last_utc"], "fits": 0, "replays": 0}))


def export():
    evidence = read(OUT / "evidence.json")
    snapshots = {}
    for name, p, policy, cadence in (
        ("Huber_H3", HUBER / "fit_report_index_020.json", "long when gross forecast >20bps; retain units", 3),
        ("Ridge_C", CADENCE / "fit_report_s1_day_060.json", "C: daily binary; C_V: shared risk controller", 1)):
        fit = read(p)
        model = fit["model"]
        write(BUNDLE / f"{name}.json", fit)
        snapshots[name] = {"source": str(p.relative_to(ROOT)), "source_pin": pin(p),
            "model_sha256": model["sha256"], "scaler_sha256": fit["scaler_sha256"],
            "fit_utc": datetime.fromtimestamp(fit["fit_ms"] / 1000, timezone.utc).isoformat(),
            "latest_mature_label_utc": datetime.fromtimestamp(fit["latest_mature_label_ms"] / 1000, timezone.utc).isoformat(),
            "training_identity": fit["training_identity"], "feature_prefix_sha256": fit["feature_prefix_sha256"],
            "parameters": 7, "training_origin_count": fit["train_origins"], "policy": policy,
            "decision_days": cadence, "retrain_days": 3, "forecast_days": 3, "label_stride_days": 1,
            "status": "research_candidate_limited_evidence" if name == "Huber_H3" else "reference_not_demonstrated_alpha"}
    write(BUNDLE / "release_metadata.json", {"schema": "offline_research_shortlist/1", "asof": "2026-10-12",
        "selection_is_retrospective": True, "trading_release": False, "snapshots": snapshots,
        "independent_baselines": ["Passive_V", "EMA_V"], "data": evidence["data"],
        "deployment_default": "NoTrade", "safety": FLAGS})
    print(json.dumps({"exported_frozen_models": list(snapshots), "new_training": False,
                      "fit_utc": {k: v["fit_utc"] for k, v in snapshots.items()}}))


def check():
    completion = read(POLICY / "completion_receipt.json")
    for name, expected in completion["linked_artifact_pins"].items():
        verify_pin(POLICY / name, expected)
    verify_pin(ROOT / completion["audit_source_path"], completion["audit_source_pin"])
    api = module(BUNDLE / "offline_models.py", "review_offline_models")
    legacy = module(ROOT / "src/exia/genetic_primus/compact_forecast_v1.py", "review_legacy_predictor")
    with np.load(DATA, allow_pickle=False) as data:
        t, p, f = data["times"], data["closes"], data["features"]
    assertions = 3
    for name, source in (("Huber_H3", HUBER / "replay_report_H_stress.json"),
                         ("Ridge_C", CADENCE / "replay_report_C_stress.json")):
        fit = read(BUNDLE / f"{name}.json")
        model = fit["model"]
        for x in (f[-1], f[-120], np.zeros((8, 6)), np.full((8, 6), 100.)):
            np.testing.assert_allclose(api.predict(model, x), legacy.predict(model, x), rtol=0, atol=1e-14)
            assertions += 1
        decisions = read(source)["decisions"]
        matches = [d for d in decisions if d["model_sha256"] == model["sha256"]]
        assert matches
        for d in matches:
            idx = int(np.searchsorted(t, d["origin_ms"]))
            np.testing.assert_allclose(api.predict(model, f[idx]), d["prediction"], rtol=0, atol=1e-14)
            assertions += 1
        corrupted = copy.deepcopy(model)
        corrupted["parameters"][0] += .1
        try:
            api.predict(corrupted, f[-1])
        except ValueError:
            assertions += 1
        else:
            raise AssertionError("tampered weights accepted")
    for idx in (400, 3000, len(t) - 1):
        actual = api.latest_features(t[:idx + 1], p[:idx + 1], asof_ms=int(t[idx]),
                                    history_anchor_ms=int(t[0]), symbols=api.SYMBOLS)
        np.testing.assert_allclose(actual, f[idx], rtol=0, atol=1e-13)
        assertions += 1
    for period in ("report", "development", "q2024_1", "q2024_3", "q2025_1", "q2025_3"):
        for policy in ("EMA_V", "Passive_V"):
            rows = read(POLICY / f"replay_{period}_{policy}_stress.json")["decisions"]
            for d in rows:
                origin = d["origin_ms"]
                grid = np.arange(origin - 61 * api.DAY_MS, origin, api.DAY_MS)
                cov = api.lagged_covariance(grid, p[np.searchsorted(t, grid)], decision_ms=origin)
                weights = api.research_targets(policy, f[np.searchsorted(t, origin)], cov)
                np.testing.assert_allclose(weights, d["targets"], rtol=0, atol=1e-13)
                assert weights.sum() <= .8 + 1e-12 and np.max(weights) <= .1 + 1e-12
                assert np.sqrt(weights @ cov @ weights) <= .005 + 1e-12
                assertions += 3
    invalid_calls = [
        lambda: api.latest_features(t[:-1], p[:-1], asof_ms=int(t[-1]), history_anchor_ms=int(t[0]), symbols=api.SYMBOLS),
        lambda: api.latest_features(t[1:], p[1:], asof_ms=int(t[-1]), history_anchor_ms=int(t[0]), symbols=api.SYMBOLS),
        lambda: api.latest_features(t, p, asof_ms=int(t[-1]), history_anchor_ms=int(t[0]), symbols=tuple(reversed(api.SYMBOLS))),
        lambda: api.lagged_covariance(np.arange(61) * api.DAY_MS, np.ones((61, 8)), decision_ms=60 * api.DAY_MS),
        lambda: api.research_targets("live", f[-1], np.eye(8)),
        lambda: api.research_targets("EMA_V", f[-1], -np.eye(8)),
        lambda: api.predict(read(BUNDLE / "Huber_H3.json")["model"], np.full((8, 6), np.nan)),
    ]
    for invalid in invalid_calls:
        try:
            invalid()
        except ValueError:
            assertions += 1
        else:
            raise AssertionError("invalid input accepted")
    write(OUT / "verification.json", {"passed": assertions, "numpy_version": np.__version__,
        "new_financial_replay": False, "pytest": False, "collector": False,
        "method": "frozen prediction and target parity plus pure invalid-input checks",
        "code_pin": pin(BUNDLE / "offline_models.py"), "safety": FLAGS})
    print(json.dumps({"direct_checks_passed": assertions, "fits": 0, "financial_replays": 0}))


def charts():
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    # Read four bounded source ledgers directly; publish only plotted NAV points.
    v = {"report": {}}
    for name in ("Passive_V", "EMA_V", "C_V", "Huber_H3"):
        source = HUBER / "replay_report_H_stress.json" if name == "Huber_H3" else POLICY / f"replay_report_{name}_stress.json"
        item = read(source)
        v["report"][name] = {"equity": [{"timestamp_ms": r["timestamp_ms"], "nav": r["nav"]} for r in item["equity"]],
                             "source": str(source.relative_to(ROOT)), "source_pin": pin(source)}
    write(OUT / "chart_data.json", {"periods": v, "safety": FLAGS})
    colors = {"Passive_V": "#165D43", "EMA_V": "#2870B8", "C_V": "#C77A13", "Huber_H3": "#9255A0"}
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for k, color in colors.items():
        e = v["report"][k]["equity"]
        times = [datetime.fromtimestamp(r["timestamp_ms"] / 1000, timezone.utc) for r in e]
        nav = np.array([r["nav"] for r in e])
        axes[0].plot(times, nav, label=k, color=color, lw=1.8)
        axes[1].plot(times, 100 * (nav / np.maximum.accumulate(nav) - 1), color=color, lw=1.5)
    axes[0].axhline(10000, color="#666666", ls="--", label="NoTrade")
    axes[0].set(ylabel="Капитал, USDT", title="Август–сентябрь 2026   Исследовательское OHLCV исполнение, stress costs")
    axes[0].legend(ncol=5, fontsize=9)
    axes[1].set(ylabel="Просадка, %", xlabel="Дата UTC")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    for ax in axes:
        ax.grid(alpha=.2)
    fig.text(.5, .01, "Huber_H3 имеет другую экспозицию и решения раз в 3 дня; кривые не доказывают превосходство сигналов.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, 1))
    dest = OUT / "capital_and_drawdown.png"
    if dest.exists():
        raise FileExistsError(dest)
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    evidence = read(OUT / "evidence.json")
    periods = ("q2024_1", "q2024_3", "q2025_1", "q2025_3", "development", "report")
    labels = ("I кв. 2024", "III кв. 2024", "I кв. 2025", "III кв. 2025", "Май–июль 2026", "Авг.–сент. 2026")
    policies = ("C", "C_H", "C_V", "EMA_V", "Passive_V")
    values = np.array([[evidence["metrics"][period][policy + "_stress"]["metrics"]["net_return"] * 100
                        for period in periods] for policy in policies])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    img = ax.imshow(values, cmap="RdYlGn", vmin=-40, vmax=40, aspect="auto")
    ax.set(xticks=np.arange(6), xticklabels=labels, yticks=np.arange(5), yticklabels=policies,
           title="Доходность отдельных режимов после stress costs, %")
    for i in range(5):
        for j in range(6):
            ax.text(j, i, f"{values[i,j]:+.2f}%", ha="center", va="center", color="black", fontsize=11,
                    bbox={"facecolor": "white", "alpha": .6, "edgecolor": "none", "pad": 3})
    fig.colorbar(img, ax=ax, label="Чистая доходность, %")
    fig.text(.5, .015, "Начальный капитал каждого режима 10 000 USDT. Старые кварталы используют синтетический funding.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    dest = OUT / "regime_returns.png"
    if dest.exists():
        raise FileExistsError(dest)
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print("2 charts generated from existing ledgers; no new strategy calculations")


def vectors():
    api = module(BUNDLE / "offline_models.py", "review_vector_api")
    with np.load(DATA, allow_pickle=False) as data:
        t, prices, features = data["times"], data["closes"], data["features"]
    models = {}
    for name, source in (("Huber_H3", HUBER / "replay_report_H_stress.json"),
                         ("Ridge_C", CADENCE / "replay_report_C_stress.json")):
        fit = read(BUNDLE / f"{name}.json")
        row = next(r for r in read(source)["decisions"] if r["model_sha256"] == fit["model"]["sha256"])
        models[name] = {"model_sha256": row["model_sha256"], "features": features[np.searchsorted(t, row["origin_ms"])].tolist(),
                        "prediction": row["prediction"], "origin_ms": row["origin_ms"], "source_pin": pin(source)}
    policies = {}
    for name in ("EMA_V", "Passive_V"):
        source = POLICY / f"replay_report_{name}_stress.json"
        row = read(source)["decisions"][0]
        origin = row["origin_ms"]
        grid = np.arange(origin - 61 * api.DAY_MS, origin, api.DAY_MS)
        cov = api.lagged_covariance(grid, prices[np.searchsorted(t, grid)], decision_ms=origin)
        policies[name] = {"covariance": cov.tolist(), "features": features[np.searchsorted(t, origin)].tolist(),
                          "targets": row["targets"], "source_pin": pin(source)}
    write(BUNDLE / "test_vectors.json", {"models": models, "policies": policies,
          "purpose": "serialization and numerical parity only; not a new financial test", "safety": FLAGS})
    print("4 frozen-output vectors exported; no raw market archive included")


def sources():
    mapping = {
        "legacy_README.md": "README.md",
        "legacy_live_freeze.py": "src/panteon_v2/app/live_freeze.py",
        "legacy_genetics.md": "docs/PANTEON_GENETICS_AGENTS.md",
        "V4.md": "docs/GENETIC_PRIMUS_V4.md",
        "V5.md": "Reports/Exia/Genetic_Primus/V5_SEALED_OUTER60_FINAL_REPORT_RU.md",
        "V6.md": "Reports/Exia/Genetic_Primus/V6_SEALED_OUTER60_FINAL_REPORT_RU.md",
        "V7.md": "Reports/Exia/Genetic_Primus/V7_PROPOSAL_MATRIX_FINAL_REPORT_RU.md",
        "V8.md": "docs/GENETIC_PRIMUS_V8_CLEAN_SEPTEMBER_CLOSEOUT.md",
        "price_audit.md": "Reports/Exia/Genetic_Primus/price_execution_audit_v1_20261008/REPORT_FINAL.md",
        "execution_audit.md": "Reports/Exia/Genetic_Primus/execution_input_audit_v1_20261008/REPORT_RU.md",
        "compact_postclose.md": "Reports/Exia/Genetic_Primus/compact_postclose_research_v1_20261008/REPORT_RU.md",
        "label_phase.md": "Reports/Exia/Genetic_Primus/label_phase_v1_20261012/REPORT_RU.md",
        "robust_linear.md": "Reports/Exia/Genetic_Primus/robust_linear_v1_20261012/REPORT_RU.md",
        "position_policy.md": "Reports/Exia/Genetic_Primus/position_policy_v1_20261012/REPORT_RU.md",
        "prospective_NOT_ACTIVATED.json": "Reports/Exia/Genetic_Primus/position_policy_v1_20261012/prospective_design_NOT_ACTIVATED.json",
    }
    for filename in ("position_policy_v1.py", "cadence_matrix_v1.py", "compact_forecast_v1.py",
                     "compact_forecast_data_v1.py", "robust_linear_v1.py"):
        mapping[filename] = "src/exia/genetic_primus/" + filename
    destination = OUT / "references"
    destination.mkdir(exist_ok=True)
    pins = {}
    for name, source in mapping.items():
        p = ROOT / source
        if p.stat().st_size > 100_000:
            raise ValueError("oversized source snapshot")
        with (destination / name).open("xb") as f:
            f.write(p.read_bytes())
        pins[name] = {"source": source, "source_pin": pin(p)}
    write(destination / "index.json", {"verbatim_reference_snapshots": pins,
          "not_an_importable_package": True, "inherited_links_require_original_workspace": True,
          "safety": FLAGS})
    print(json.dumps({"read_only_reference_copies": len(pins)}))


def seal():
    read(OUT / "evidence.json")
    read(OUT / "verification.json")
    files = {str(p.relative_to(OUT)): pin(p) for p in sorted(OUT.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc",)
             and p.name != "equity_paths.json"}
    if "release_manifest.json" in files:
        raise FileExistsError("already sealed")
    write(OUT / "release_manifest.json", {"schema": "research_publication_manifest/1", "files": files,
        "builder": pin(Path(__file__)), "artifact_count": len(files), "safety": FLAGS})
    print(json.dumps({"sealed_artifacts": len(files), "runtime_enabled": False}))


def prepublish():
    problems = []
    files = [p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    forbidden = (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
                 re.compile(r"\b(?:ghp|github_pat|sk_live)_[A-Za-z0-9_]{20,}"),
                 re.compile(r'''(?i)["'](?:api_key|api_secret|password|passphrase|private_key)["']\s*[:=]\s*["'][^"']{8,}["']'''))
    for p in files:
        if p.suffix in (".md", ".json", ".py"):
            content = p.read_text(encoding="utf-8")
            if any(pattern.search(content) for pattern in forbidden):
                problems.append("possible_secret:" + p.name)
    links = 0
    for name in ("README.md", "REPORT_1_CURRENT_STATE_RU.md", "REPORT_2_IMPROVEMENT_RESEARCH_RU.md", "model_bundle/README.md"):
        p = OUT / name
        for target in re.findall(r'\]\(([^)]+)\)', p.read_text(encoding="utf-8")):
            if target.startswith(("http:", "https:", "#")):
                continue
            if target == "release_manifest.json":
                continue
            if not (p.parent / target).is_file():
                problems.append("broken_link:" + target)
            links += 1
    if problems:
        raise ValueError(",".join(problems))
    result = {"publication_files_checked": len(files), "local_links_checked": links,
              "heuristic_secret_findings": 0, "not_a_complete_security_audit": True,
              "builder_pin": pin(Path(__file__)), "safety": FLAGS}
    write(OUT / "publication_checks.json", result)
    print(json.dumps({k: result[k] for k in ("publication_files_checked", "local_links_checked", "heuristic_secret_findings")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "export", "check", "vectors", "sources", "charts", "prepublish", "seal"))
    globals()[parser.parse_args().stage]()
