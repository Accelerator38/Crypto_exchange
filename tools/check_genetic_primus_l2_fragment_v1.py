"""Single bounded local prefix + named regressions; never collector or training."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exia.genetic_primus.bounded_gzip_csv_v1 import read_prefix
from exia.genetic_primus.l2_fragment_reconstruction_v1 import L2Fragment, HEADER
from exia.genetic_primus.market_fragment_v1 import book_snapshots
from exia.genetic_primus.unified_measurement_v1 import SAFETY, identity

OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/l2_format_smoke_v1_20261008"
CONFIG = "configs/genetic_primus_l2_fragment_v1.json"
TESTS = "tests/test_genetic_primus_l2_fragment_v1.py"
PREVIOUS = "Reports/Exia/Genetic_Primus/postclose_label_preflight_v1_20261008"
SOURCES = (CONFIG, TESTS, "src/exia/genetic_primus/bounded_gzip_csv_v1.py",
           "src/exia/genetic_primus/l2_fragment_reconstruction_v1.py", "src/exia/genetic_primus/market_fragment_v1.py",
           "src/exia/genetic_primus/unified_measurement_v1.py", "tools/check_genetic_primus_l2_fragment_v1.py",
           "docs/GENETIC_PRIMUS_L2_FORMAT_SMOKE_V1.md", PREVIOUS + "/receipt.json", PREVIOUS + "/artifact_hashes.json")


def file_sha(path):
    if path.stat().st_size > 2_000_000:
        raise ValueError("code/metadata artifact bound exceeded")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    file_sha(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, body):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(body, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def prior_integrity():
    receipt = read_json(ROOT / PREVIOUS / "receipt.json")
    manifest = read_json(ROOT / PREVIOUS / "artifact_hashes.json")
    for name, sha in receipt["source_hashes"].items():
        if file_sha(ROOT / name) != sha:
            raise ValueError("prior label-plan source changed")
    for name, sha in manifest["files"].items():
        if file_sha(ROOT / PREVIOUS / name) != sha:
            raise ValueError("prior label-plan evidence changed")
    return {"sources_verified": len(receipt["source_hashes"]), "artifacts_verified": len(manifest["files"])}


def graph(events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"
    base = events[0]["received_ts_us"]
    times = [(e["received_ts_us"] - base) / 1_000_000 for e in events]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    axes[0].step(times, [float(e["bids"][0][0]) for e in events], where="post", label="Best bid")
    axes[0].step(times, [float(e["asks"][0][0]) for e in events], where="post", label="Best ask")
    axes[0].set(title="BTC 2024-12-01: ограниченный market fragment, без торговли", ylabel="Наблюдаемая цена, USDT")
    axes[0].legend()
    axes[1].plot(times, [(e["received_ts_us"] - e["exchange_ts_us"]) / 1000 for e in events], marker=".", color="tab:orange")
    axes[1].set(xlabel="Receive time от первого полного snapshot, секунды", ylabel="Provider lag, ms")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.savefig(OUTPUT / "book_fragment.png", dpi=140)
    fig.savefig(OUTPUT / "book_fragment.svg")
    plt.close(fig)


def main():
    if OUTPUT.exists():
        raise FileExistsError("immutable L2 evidence root already exists")
    sources = {name: file_sha(ROOT / name) for name in SOURCES}
    c = read_json(ROOT / CONFIG)
    if c["schema"] != "bounded_l2_format_smoke/1" or c["safety"] != SAFETY or c["financial_replay_ready"] is not False:
        raise ValueError("offline-only format-smoke contract required")
    prior = prior_integrity()
    spec = importlib.util.spec_from_file_location("explicit_l2_checks", ROOT / TESTS)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    log = io.StringIO()
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(checks.PrefixReaderTests),
                               unittest.defaultTestLoader.loadTestsFromTestCase(checks.ReconstructionTests)])
    tests = unittest.TextTestRunner(stream=log, verbosity=1).run(suite)
    if not tests.wasSuccessful() or tests.testsRun != 31:
        raise ValueError("named L2 checks failed: " + log.getvalue()[:500])
    source = Path(c["source_path"])
    before = source.stat()
    with source.open("rb") as raw:
        prefix = read_prefix(raw, **{key: c[key] for key in
                             ("max_rows", "max_compressed_bytes", "max_uncompressed_bytes", "max_line_bytes")})
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("source archive stat changed during prefix read")
    if tuple(prefix["header"]) != HEADER or not prefix["rows"]:
        raise ValueError("expected nonempty BTC L2 source format")
    args = {"source_prefix_sha256": prefix["uncompressed_prefix_sha256"], **{key: c[key] for key in
            ("snapshot_levels_per_side", "max_state_levels_per_side", "max_rows_per_message", "max_rows", "max_events", "max_receive_gap_us")}}
    engine = L2Fragment(**args)
    half = len(prefix["rows"]) // 2
    for row in prefix["rows"][:half]:
        engine.ingest(row)
    checkpoint = engine.checkpoint()
    resumed = L2Fragment.restore(checkpoint, **args)
    for row in prefix["rows"][half:]:
        engine.ingest(row)
        resumed.ingest(row)
    engine.finish()
    resumed.finish()
    result = engine.result()
    if result != resumed.result() or result["faults"] or not result["events"]:
        raise ValueError("real prefix integrity/restart/reconstruction check failed")
    groups = [(key, list(rows)) for key, rows in itertools.groupby(prefix["rows"], key=lambda r:
              (r["local_timestamp"], r["timestamp"], r["is_snapshot"]))]
    # Last group is deliberately unpublished, irrespective of apparent depth.
    expected = [book_snapshots(rows, args["source_prefix_sha256"], c["snapshot_levels_per_side"])[0]
                for key, rows in groups[:-1] if key[2] == "true"]
    actual = [event for event in result["events"] if event["message_type"] == "snapshot"]
    fields = ("received_ts_us", "exchange_ts_us", "bids", "asks")
    if len(actual) != len(expected) or any(any(a[key] != b[key] for key in fields) for a, b in zip(actual, expected)):
        raise ValueError("legacy observed-snapshot validation disagrees")
    fault_demo = {}
    for name, messages in {
        "explicit_crossed_delta_then_resnapshot": [checks.snapshot(), checks.delta(200, [("bid", "100.5", "2")]), checks.snapshot(300)],
        "with_delta": [checks.snapshot(), checks.delta(200, [("ask", "100.1", "9")]), checks.delta(300, [("bid", "99.9", "1")])],
        "silently_lost_delta": [checks.snapshot(), checks.delta(300, [("bid", "99.9", "1")])],
    }.items():
        fault_demo[name] = checks.drive(messages).result()
    if sources != {name: file_sha(ROOT / name) for name in sources}:
        raise ValueError("code/metadata changed during check")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(OUTPUT / "recorded_prefix.json", prefix)
    write_json(OUTPUT / "reconstruction.json", result)
    write_json(OUTPUT / "checkpoint.json", checkpoint)
    write_json(OUTPUT / "resumed_result.json", resumed.result())
    write_json(OUTPUT / "synthetic_fault_demonstration.json", fault_demo)
    graph(result["events"])
    with (OUTPUT / "checks.txt").open("x", encoding="utf-8") as stream:
        stream.write(log.getvalue())
    first, last = result["events"][0], result["events"][-1]
    span = (last["received_ts_us"] - first["received_ts_us"]) / 1_000_000
    report = f"""# L2 format smoke: 8 октября 2026

**31/31** bounded offline checks прошли. Реальный источник — предоставленный
BTC incremental_book_L2 2024-12-01; он прочитан read-only, не переписан.
Старый snapshot-only валидатор повторно использован и согласуется с новым
state по всем complete observed snapshots. Это не финансовый ретротест.

## Реальный ограниченный фрагмент

Parsed rows: **{prefix['row_count']}**. Compressed bytes read: **{prefix['compressed_bytes_read']}**;
inflated bytes: **{prefix['uncompressed_bytes_inflated']}**. Жёсткие caps:
{c['max_compressed_bytes']} compressed / {c['max_uncompressed_bytes']} inflated bytes,
{c['max_rows']} parsed rows. Whole archive size по stat: {before.st_size} bytes.
Полный архив не прочитан; full archive SHA и весь gzip CRC не проверены.
Сохранены отдельные hashes прочитанного compressed/uncompressed prefix и rows.

Полных опубликованных snapshots: **{result['snapshot_count']}**, реальных delta
groups: **{result['delta_count']}**, reconstruction faults: **{len(result['faults'])}**.
Последние {result['dropped_final_rows']} rows unclosed group отброшены.
Длина опубликованного receive-фрагмента: **{span:.6f} s**, depth по контракту
15 уровней на сторону, не полный depth venue. Source clocks сохранены в us.
Restore из checkpoint внутри prefix точно совпал с непрерывным расчётом.

![Observed BBO и provider lag](book_fragment.png)

Provider lag — local receipt minus exchange timestamp. Это не задержка
отправки/исполнения нашей заявки и не latency calibration для production.
Это короткий фрагмент 2024 года, не equity curve стратегии за 2026 год.

## Синтетические проверки и ограничения

Absolute delta replacement, zero delete, complete snapshot reset, atomic
updates, crossed/empty/duplicate levels, row/depth/event caps, повреждённый
gzip/CSV, clock regression, gaps и восстановление snapshot проверены отдельно.
Если real prefix не содержит deltas, их реальная семантика остаётся
неподтверждённой: синтетическая проверка не заменяет data evidence.

Без venue sequence ID тихо потерянный delta не обнаруживается надёжно.
В synthetic_fault_demonstration обе версии с/без одного delta формально
проходят, но дают разные asks. Поэтому venue_sequence_verified=false,
full_depth_verified=false, strategy_availability_verified=false.
Provider local timestamp не автоматически равен доступности нашему агенту.
Нельзя без отдельного адаптера округлять us в ms и терять ordering.

Источники и старое post-close evidence сохранены: {prior['sources_verified']}
source hashes и {prior['artifacts_verified']} artifact hashes сверены.
Financial replay/training/full8-2026 qualification остаются **False**.
Новых fits, strategy-profit replays и network requests: **0**; все 9 flags false.

## Далее

Проверить небольшие fragments trades/derivative и общий receive/exchange clock,
особенно funding: predicted funding rate не равен фактическому settlement.
Затем собрать реальный full8 bundle с coverage/sequence/availability,
historical rules/fees и observed settlement marks для нужного validation и
causal training scope. BTC-файл 2024 года не подставляется в 2026 labels.
"""
    with (OUTPUT / "REPORT_RU.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report)
    receipt = {"schema": "bounded_l2_smoke_receipt/1", "source_hashes": sources,
        "input_archive": {"path": str(source), "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
                          "stat_unchanged_during_read": True, "full_sha256_verified": False},
        "prefix_hashes": {k: prefix[k] for k in ("compressed_prefix_sha256", "uncompressed_prefix_sha256", "row_bank_sha256")},
        "bounds": prefix["limits"], "unit_checks_passed": tests.testsRun,
        "real_snapshot_count": result["snapshot_count"], "real_delta_count": result["delta_count"],
        "legacy_snapshot_agreement": True, "checkpoint_resume_exact": True,
        "reconstruction_sha256": result["sha256"], "prior_evidence_integrity": prior,
        "financial_replay_ready": False, "economic_labels_constructed": 0, "new_model_fits": 0,
        "market_strategy_replays": 0, "network_requests": 0, "safety": SAFETY}
    receipt["sha256"] = identity(receipt)
    write_json(OUTPUT / "receipt.json", receipt)
    artifacts = {path.name: file_sha(path) for path in sorted(OUTPUT.iterdir())}
    write_json(OUTPUT / "artifact_hashes.json", {"files": artifacts, "sha256": identity(artifacts)})
    print(json.dumps({"checks_passed": tests.testsRun, "rows": prefix["row_count"],
        "compressed_bytes": prefix["compressed_bytes_read"], "inflated_bytes": prefix["uncompressed_bytes_inflated"],
        "snapshots": result["snapshot_count"], "real_deltas": result["delta_count"],
        "dropped_final_rows": result["dropped_final_rows"], "resume_exact": True,
        "financial_replay_ready": False, "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc)[:700]}, ensure_ascii=False))
        sys.exit(1)
