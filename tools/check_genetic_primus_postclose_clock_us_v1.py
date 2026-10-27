"""One-shot exact-clock engineering evidence; no market replay/network/runtime."""
import hashlib
import io
import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from contextlib import redirect_stderr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from exia.genetic_primus.postclose_clock_us_v1 import (
    SAFETY, adapt_recorded_books, lift_contract, lossless_legacy_window, lossless_ms,
    resolve_endpoint_us, seal, validate_contract, verify, window_requests_us,
)
from exia.genetic_primus.postclose_label_plan_v1 import resolve_endpoint
from exia.genetic_primus.market_fragment_v1 import object_sha
import test_genetic_primus_postclose_clock_us_v1 as tests

CONFIG = "configs/genetic_primus_postclose_clock_us_v1.json"
PARENT = "configs/genetic_primus_postclose_label_plan_v1.json"
OUT = ROOT / "Reports/Exia/Genetic_Primus/postclose_clock_us_smoke_v1_20261008"
EVIDENCE = ("postclose_label_preflight_v1_20261008", "l2_format_smoke_v1_20261008",
            "market_timebase_smoke_v1_20261008")
SOURCE_NAMES = (CONFIG, PARENT, "src/exia/genetic_primus/postclose_clock_us_v1.py",
                "tests/test_genetic_primus_postclose_clock_us_v1.py",
                "tools/check_genetic_primus_postclose_clock_us_v1.py",
                "docs/GENETIC_PRIMUS_POSTCLOSE_CLOCK_US_V1.md",
                "src/exia/genetic_primus/postclose_label_plan_v1.py",
                "src/exia/genetic_primus/unified_measurement_v1.py",
                "src/exia/genetic_primus/market_fragment_v1.py",
                "src/exia/genetic_primus/market_timebase_fragment_v1.py")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha(path):
    # Small source/evidence files only. Input archives are never opened.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, body):
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(body, output, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        output.write("\n")


def verify_prior():
    result = {}
    for name in EVIDENCE:
        prior = (ROOT / "Reports/Exia/Genetic_Primus" / name).resolve()
        receipt = read_json(prior / "receipt.json")
        verify(receipt)
        manifest = read_json(prior / "artifact_hashes.json")
        if object_sha(manifest["files"]) != manifest["sha256"]:
            raise ValueError("prior manifest seal mismatch: " + name)
        for source, expected in receipt["source_hashes"].items():
            path = (ROOT / source).resolve()
            if not path.is_relative_to(ROOT) or file_sha(path) != expected:
                raise ValueError("prior source changed: " + source)
        for artifact, expected in manifest["files"].items():
            path = (prior / artifact).resolve()
            if not path.is_relative_to(prior) or file_sha(path) != expected:
                raise ValueError("prior artifact changed: " + artifact)
        result[name] = {"sources_verified": len(receipt["source_hashes"]),
                        "artifacts_verified": len(manifest["files"])}
    return result


def synthetic_cases(contract, parent):
    """All inputs are artificial clock observations, no raw-market price fields."""
    ready = tests.BASE
    cases = {"aligned_resolved": tests.happy(), "fractional_resolved": tests.happy(submillisecond=True)}
    w = tests.happy()
    w["observations"][0]["exchange_us"] = ready
    cases["signal_equality"] = w
    w = tests.happy()
    ref = w["observations"][0]
    w["observations"].insert(1, tests.observation("peer", ref["exchange_us"], ref["received_us"]))
    cases["reference_batch"] = w
    for delta, name in ((-1, "arrival_minus1"), (0, "arrival_equal"), (1, "arrival_plus1")):
        w = tests.happy()
        arrival = w["observations"][0]["available_us"] + contract["order_latency_us"]
        received = arrival + 1000 if delta == -1 else arrival + delta
        w["observations"][1] = tests.observation("end", arrival + delta, received)
        cases[name] = w
    for delta, name in ((-1, "expiry_minus1"), (0, "expiry_equal"), (1, "expiry_plus1")):
        w = tests.happy()
        expires = w["observations"][0]["available_us"] + contract["order_ttl_us"]
        w["observations"][1] = tests.observation("end", expires + delta)
        cases[name] = w
    w = tests.happy()
    endpoint = w["observations"][1]
    w["observations"].append(tests.observation("peer", endpoint["exchange_us"], endpoint["received_us"]))
    cases["endpoint_batch"] = w
    w = tests.happy()
    w["observations"][1]["snapshot_sha256"] = w["observations"][0]["snapshot_sha256"]
    cases["aliased_snapshot"] = w
    expected = {"aligned_resolved": "CLOCK_RESOLVED_NOT_FILL_QUALIFIED",
                "fractional_resolved": "CLOCK_RESOLVED_NOT_FILL_QUALIFIED",
                "signal_equality": "AMBIGUOUS_REFERENCE_BOUNDARY",
                "reference_batch": "AMBIGUOUS_REFERENCE_BATCH",
                "arrival_minus1": "EXPIRED_NO_POST_ARRIVAL",
                "arrival_equal": "AMBIGUOUS_POST_ARRIVAL_BOUNDARY",
                "arrival_plus1": "CLOCK_RESOLVED_NOT_FILL_QUALIFIED",
                "expiry_minus1": "CLOCK_RESOLVED_NOT_FILL_QUALIFIED",
                "expiry_equal": "EXPIRED_NO_POST_ARRIVAL", "expiry_plus1": "EXPIRED_NO_POST_ARRIVAL",
                "endpoint_batch": "AMBIGUOUS_POST_ARRIVAL_BATCH", "aliased_snapshot": "INPUT_REJECTED"}
    results, old_agreements = {}, 0
    for name, w in cases.items():
        try:
            resolution = resolve_endpoint_us(ready, w, contract)
        except ValueError as exc:
            if name != "aliased_snapshot":
                raise
            resolution = {"status": "INPUT_REJECTED", "error": str(exc)}
        if resolution["status"] != expected[name]:
            raise ValueError("unexpected clock fixture status: " + name)
        try:
            bridge = lossless_legacy_window(ready, w, contract)
        except ValueError as exc:
            compatibility = {"accepted": False, "reason": str(exc)}
        else:
            old = resolve_endpoint(bridge["signal_available_ms"], bridge["window"], parent)
            if old["status"] != resolution["status"]:
                raise ValueError("legacy status disagreement")
            for side in ("reference", "endpoint"):
                if (old[side]["snapshot_id"] != resolution[side]["snapshot_id"]
                        or old[side]["available_ms"] * 1000 != resolution[side]["available_us"]):
                    raise ValueError("legacy endpoint disagreement")
            if old["arrival_ms"] * 1000 != resolution["arrival_us"] or old["expires_ms_exclusive"] * 1000 != resolution["expires_us_exclusive"]:
                raise ValueError("legacy timing disagreement")
            compatibility = {"accepted": True, "legacy_exact_agreement": True, "legacy_resolution": old}
            old_agreements += 1
        results[name] = {"basis": "synthetic_fixture", "signal_available_us": ready,
                         "window": deepcopy(w), "resolution": resolution, "legacy_bridge": compatibility}
    if old_agreements != 1:
        raise ValueError("expected exactly one aligned legacy comparison")
    return seal({"schema": "synthetic_clock_cases/1", "cases": results,
                 "legacy_exact_agreements": old_agreements,
                 "financial_replay_ready": False, "safety": SAFETY})


def plot_boundaries(cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = {"CLOCK_RESOLVED_NOT_FILL_QUALIFIED": "#287d46",
               "AMBIGUOUS_POST_ARRIVAL_BOUNDARY": "#ae7417", "EXPIRED_NO_POST_ARRIVAL": "#ad3939"}
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    for axis, family, title in ((axes[0], "arrival", "Post-arrival: equal clock does NOT establish venue order"),
                                (axes[1], "expiry", "TTL is exclusive: no floor, ceil or rounding")):
        names = [family + suffix for suffix in ("_minus1", "_equal", "_plus1")]
        labels = []
        for row, name in enumerate(names):
            case = cases["cases"][name]
            result = case["resolution"]
            endpoint = case["window"]["observations"][1]
            cutoff = result["arrival_us"] if family == "arrival" else result["expires_us_exclusive"]
            coordinate = endpoint["exchange_us" if family == "arrival" else "available_us"] - cutoff
            axis.scatter(coordinate, row, s=90, color=palette[result["status"]])
            status = ("Clock resolved (NOT fill)" if result["status"] == "CLOCK_RESOLVED_NOT_FILL_QUALIFIED"
                      else "Ambiguous" if result["status"].startswith("AMBIGUOUS") else "Expired / no candidate")
            labels.append(name.replace("_", " ") + " : " + status)
        axis.set_yticks(range(3), labels)
        axis.set_xticks((-1, 0, 1))
        axis.set_xlim(-1.5, 1.5)
        axis.set_ylim(-0.5, 2.5)
        axis.axvline(0, color="black", alpha=0.35, linestyle="--")
        axis.grid(alpha=0.2)
        axis.set_title(title)
        axis.set_xlabel(("Endpoint exchange - simulated arrival" if family == "arrival"
                         else "Endpoint available - simulated expiry") + " (integer microseconds)")
    fig.suptitle("SYNTHETIC clock boundary checks — not market validation, fills, NAV or profits")
    fig.tight_layout()
    fig.savefig(OUT / "synthetic_clock_boundaries.png", dpi=140)
    fig.savefig(OUT / "synthetic_clock_boundaries.svg")
    plt.close(fig)


def main():
    if OUT.exists():
        raise ValueError("existing evidence root must not be overwritten")
    prior_integrity = verify_prior()
    config, parent = read_json(ROOT / CONFIG), read_json(ROOT / PARENT)
    if file_sha(ROOT / PARENT) != config["parent_file_sha256"]:
        raise ValueError("registered parent file hash mismatch")
    if lift_contract(parent, file_sha(ROOT / PARENT)) != config:
        raise ValueError("native contract must be exact registered parent lift")
    validate_contract(config)
    sources = {name: file_sha(ROOT / name) for name in SOURCE_NAMES}
    for name in EVIDENCE:
        for artifact in ("receipt.json", "artifact_hashes.json"):
            relative = "Reports/Exia/Genetic_Primus/" + name + "/" + artifact
            sources[relative] = file_sha(ROOT / relative)
    recorded = "Reports/Exia/Genetic_Primus/l2_format_smoke_v1_20261008/reconstruction.json"
    sources[recorded] = file_sha(ROOT / recorded)
    checks = io.StringIO()
    result = unittest.TextTestRunner(stream=checks, verbosity=0).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(tests.ClockUsTests))
    if not result.wasSuccessful() or result.testsRun != 45:
        raise ValueError("bounded unit checks failed: " + checks.getvalue()[:600])
    reconstruction = read_json(ROOT / recorded)
    verify(reconstruction)
    books = reconstruction["events"]
    adapter = adapt_recorded_books(books)
    first, last = books[0]["received_ts_us"], books[-1]["received_ts_us"]
    real_window = {"coverage_start_us": first, "coverage_end_us_exclusive": last + 1,
                   "clock_basis": "recorded_unknown", "observations": adapter["observations"]}
    real_resolution = resolve_endpoint_us(first + config["signal_compute_latency_us"], real_window, config)
    if real_resolution["status"] != "BLOCKED_UNKNOWN_STRATEGY_AVAILABILITY":
        raise ValueError("recorded availability unexpectedly qualified")
    if adapter != adapt_recorded_books(books):
        raise ValueError("nondeterministic recorded adapter")
    refusals = {"exchange_us": 0, "received_us": 0}
    for observation in adapter["observations"]:
        for key in refusals:
            try:
                lossless_ms(observation[key])
            except ValueError:
                refusals[key] += 1
    cases = synthetic_cases(config, parent)
    if cases != synthetic_cases(config, parent):
        raise ValueError("nondeterministic synthetic cases")
    requests = window_requests_us(parent["validation_start_ms"] * 1000,
                                  parent["validation_start_ms"] * 1000 + 17, config)
    summary = {"tests_passed": result.testsRun, "recorded_book_observations": len(books),
               "missing_strategy_available_us": adapter["missing_strategy_availability_count"],
               "recorded_status": real_resolution["status"], "lossless_ms_refusals": refusals,
               "synthetic_cases": len(cases["cases"]), "legacy_exact_agreements": cases["legacy_exact_agreements"],
               "legacy_bridge_refusals": sum(c["legacy_bridge"]["accepted"] is False for c in cases["cases"].values()),
               "timestamp_precision_loss_us": 0, "financial_replay_ready": False}
    verify_prior()
    if any(file_sha(ROOT / name) != expected for name, expected in sources.items()):
        raise ValueError("source/evidence changed during checks")
    OUT.mkdir(parents=True, exist_ok=False)
    write_json(OUT / "recorded_clock_adapter.json", adapter)
    write_json(OUT / "recorded_window.json", real_window)
    write_json(OUT / "recorded_clock_resolution.json", real_resolution)
    write_json(OUT / "synthetic_clock_cases.json", cases)
    write_json(OUT / "synthetic_window_requests.json", {"basis": "synthetic_fixture", "injected_feature_delay_us": 17,
                                                        "requests": requests})
    write_json(OUT / "summary.json", summary)
    with (OUT / "checks.txt").open("x", encoding="utf-8") as output:
        output.write(checks.getvalue())
    plot_log = io.StringIO()
    with redirect_stderr(plot_log):
        plot_boundaries(cases)
    if plot_log.getvalue():
        with (OUT / "plot_diagnostics.txt").open("x", encoding="utf-8") as output:
            output.write(plot_log.getvalue()[:4096])
    table = "\n".join("| " + name + " | " + case["resolution"]["status"] + " | "
                      + ("accepted: exact agreement" if case["legacy_bridge"]["accepted"] else "rejected") + " |"
                      for name, case in cases["cases"].items())
    report = f"""# Post-close clock: точные микросекунды и fail-closed availability

Дата запуска: 2026-10-08. Engineering only, не financial replay.

## Итоги

- {result.testsRun}/{result.testsRun} bounded deterministic unittest проверок прошли.
- Новый clock contract точно наследует pinned ms parent, без изменения старого кода/результатов.
- {len(books)} сохранённых BTCUSDT L2 observations (2024-12-01) адаптированы без потери us precision.
- Strategy available time отсутствует в {adapter['missing_strategy_availability_count']} observations:
  available_us=null, статус {real_resolution['status']}, reference/endpoint отсутствуют.
- Provider timestamps с неделимой на 1000 частью: exchange={refusals['exchange_us']}, receive={refusals['received_us']}.
  Ни один timestamp не округлён и receive не объявлен strategy availability.
- {len(cases['cases'])} synthetic clock cases; один aligned case полностью согласен с legacy resolver.
  {summary['legacy_bridge_refusals']} остальных bridge attempts корректно отклонены по precision/ambiguity/invalid input.
- Native requests сохраняют injected 17 us feature delay и scheduled exit через 3 дня,
  а не entry fill + 3 дня. Этот injected delay не является измерением реальной модели.

## Сценарии clocks

| Синтетический случай | Native статус | Legacy bridge |
| --- | --- | --- |
{table}

TTL exclusive: endpoint за 1 us до expiry допустим, ровно на expiry/после — нет.
Ровно на signal/arrival без порядка событий — ambiguous; несколько снимков
одного available batch также ambiguous. Reference и endpoint обязаны быть
distinct по ID и evidence hash. Aliased snapshot не может стать новым событием.
Задержанная receipt pre-arrival snapshot не превращается в post-arrival.

Это **консервативное уточнение** старой ms семантики: старый resolver допускал
equality и выбирал первый элемент списка. Он не изменён. Compatibility statement
относится только к aligned/unambiguous synthetic case, не ко всем старым endpoint outcomes.
Lossless bridge принимает только resolved synthetic metadata и сохраняет unqualified flags.

## Происхождение и пределы

Проверены source/artifact hashes трёх прошлых evidence roots:
`{json.dumps(prior_integrity, sort_keys=True)}`.
Исходные gzip не открывались повторно; book observations взяты из pinned reconstruction.json.
Clock observations содержат hashes и timestamps, без цен/объёмов. Reported ticker
funding не может стать endpoint, settlement, final rate или cashflow.
Provider receive, время скачивания и exchange time не доказывают consumer availability.
Actual availability input API намеренно не реализован до независимого evidence contract.

Все 9 authority flags false. 0 fits, 0 labels, 0 market strategy replays,
0 network requests. Real availability, coverage, fills и financial replay не квалифицированы.
График — synthetic clock boundaries, не капитал или доходность.

## Следующее действие

Exact-clock инженерный блок завершён. Следующий содержательный этап — проверка
наличия/происхождения execution данных целевого августа–сентября 2026 для full8:
depth/sequence/coverage, strategy availability, marks/funding settlements,
historical rules и account fees. BTC 2024 не заменяет этот bundle.
Не запускать новый training/replay на старых next-open labels и не подменять
недоказанные availability/costs synthetic данными. Сеть требует отдельного разрешения.
"""
    with (OUT / "REPORT_RU.md").open("x", encoding="utf-8", newline="\n") as output:
        output.write(report)
    receipt = seal({"schema": "postclose_clock_us_smoke_receipt/1", "source_hashes": sources,
                    "parent_file_sha256": config["parent_file_sha256"], "contract_sha256": object_sha(config),
                    "adapter_sha256": adapter["sha256"], "resolution_sha256": real_resolution["sha256"],
                    "synthetic_cases_sha256": cases["sha256"], "summary_sha256": object_sha(summary),
                    "prior_evidence_integrity": prior_integrity, "unit_checks_passed": result.testsRun,
                    "financial_replay_ready": False, "economic_labels_constructed": 0,
                    "new_model_fits": 0, "market_strategy_replays": 0, "network_requests": 0,
                    "safety": SAFETY})
    write_json(OUT / "receipt.json", receipt)
    artifacts = {path.name: file_sha(path) for path in sorted(OUT.iterdir()) if path.is_file()}
    write_json(OUT / "artifact_hashes.json", {"files": artifacts, "sha256": object_sha(artifacts)})
    print(json.dumps({"ok": True, "tests": result.testsRun, "recorded_observations": len(books),
                      "recorded_status": real_resolution["status"], "synthetic_cases": len(cases["cases"]),
                      "financial_replay_ready": False, "out": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)[:650]}, ensure_ascii=False)
              .encode("utf-8")[:920].decode("utf-8", "ignore"))
        sys.exit(1)
