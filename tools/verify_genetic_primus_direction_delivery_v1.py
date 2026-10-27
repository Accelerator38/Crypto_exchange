"""Verify immutable results and report transcription; write delivery receipt only."""
from __future__ import annotations
import re
from pathlib import Path
import run_genetic_primus_direction_benchmark_v1 as b


def main():
    reg = b.registration()
    original = b.sealed(b.OUT / "verification.json")
    for name, expected in original["artifact_pins"].items():
        assert b.pin(b.OUT / name) == expected
    forecast = b.sealed(b.OUT / "forecast_comparison.json")["models"]
    financial = b.sealed(b.OUT / "financial_comparison.json")["models"]
    b.sealed(b.OUT / "descriptive_attribution.json")
    checks = b.sealed(b.OUT / "independent_checks.json")
    assert checks["checker_pin"] == b.pin(b.ROOT / "tools/check_genetic_primus_direction_benchmark_v1.py")
    assert checks["checks"]["saved_models"] == 252
    assert b.sealed(b.NEXT_REGISTRY)["objectives"][-1]["registration_sha256"] == reg["sha256"]
    friendly = {"Ridge6": "ridge6", "Logistic6": "logistic6", "Logistic12": "logistic12", "GaussianNB12": "gaussian_nb12",
                "LDA12": "lda12", "QDA12": "qda12", "Tree12": "tree12", "Forest12": "forest12",
                "ExtraTrees12": "extra_trees12", "HistBoost12": "hist_boost12", "SGDLogistic12": "sgd_logistic12", "MLP12": "mlp12"}
    report = (b.OUT / "REPORT_RU.md").read_text(encoding="utf-8")
    table_checks = 0
    for line in report.splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 7 and cells[0] in friendly:
            name = friendly[cells[0]]
            actual = [float(cell.replace(",", ".").replace("−", "-")) for cell in cells[1:]]
            expected = [forecast[name]["accuracy"] * 100, forecast[name]["balanced_accuracy"] * 100, forecast[name]["auc"],
                        financial[name]["normal"]["metrics"]["net_return"] * 100,
                        financial[name]["stress"]["metrics"]["net_return"] * 100,
                        financial[name]["stress"]["metrics"]["max_drawdown"] * 100]
            for i, (left, right) in enumerate(zip(actual, expected)):
                assert abs(left - right) <= (0.00501 if i < 2 else 0.000501), (name, i, left, right)
                table_checks += 1
    assert table_checks == 72
    links = 0
    for name in ("REPORT_RU.md", "SUMMARY_RU.md"):
        text = (b.OUT / name).read_text(encoding="utf-8")
        for path in re.findall(r"\]\((E:/[^)]+)\)", text):
            assert Path(path).is_file(), path
            links += 1
    receipt = b.write(b.OUT / "delivery_receipt.json", {
        "registration_sha256": reg["sha256"], "original_verification_sha256": original["sha256"],
        "original_artifacts_verified": len(original["artifact_pins"]), "report_table_checks": table_checks,
        "local_report_links_checked": links, "historical_fits": 252, "financial_replays": 30,
        "new_fits_or_replays_in_delivery": 0, "charts_visually_inspected": ["direction_accuracy.png", "stress_capital_drawdown.png"],
        "delivery_source_pin": b.pin(Path(__file__)),
        "artifact_pins": {path.name: b.pin(path) for path in sorted(b.OUT.iterdir()) if path.is_file()},
        "registry_pin": b.pin(b.NEXT_REGISTRY), "safety": b.read(b.CONFIG)["safety"],
        "promotion": False, "independent_validation": False})
    print({"original_pins": len(original["artifact_pins"]), "delivery_pins": len(receipt["artifact_pins"]),
           "table_checks": table_checks, "links": links, "verified": True, "promotion": False})


if __name__ == "__main__":
    main()
