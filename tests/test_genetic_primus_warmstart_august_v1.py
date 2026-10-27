"""Operator parity and anchor-budget regressions, without market execution."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
API = runpy.run_path(str(ROOT / "tools/run_genetic_primus_warmstart_august_v1.py"))


def test_two_anchor_mode_matches_original_search_for_all_registered_seeds():
    def objective(g):
        return sum(x * y for x, y in zip(g, (3.1, 0.4, -2.9, 0.2, 0.05)))

    for seed in range(20260907, 20260917):
        original = API["cold"].search("genetic_search", objective, 24, seed, 6)
        extracted = API["anchored_search"](
            objective, 24, seed, 6, anchors=(API["NO_TRADE"], API["ALLOW_ALL"])
        )
        assert original == extracted


def test_anchor_counts_inside_budget_and_is_not_forced_winner():
    calls = []

    def objective(g):
        calls.append(g)
        return -float(sum(abs(x) for x in g[:4]))

    result = API["anchored_search"](objective, 24, 20260907, 6)
    assert tuple(calls[:3]) == API["ANCHORS"]
    assert len(calls) == len(set(calls)) == result["unique_evaluations"] == 24
    assert calls.count(API["cold"].FIXED) == 1
    assert result["winner"] != API["cold"].FIXED


def test_invalid_anchors_fail_before_fitness_evaluation():
    def forbidden(_):
        raise AssertionError("fitness must not run")

    for anchors in ((API["NO_TRADE"], API["NO_TRADE"]),
                    (API["NO_TRADE"], (1, 1, 1, 1, 1))):
        try:
            API["anchored_search"](forbidden, 24, 20260907, 6, anchors=anchors)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid anchors accepted")


def test_frozen_config_isolated_from_cold_start():
    config, base, output, cold_output = API["load_config"]()
    assert output != cold_output
    assert config["unique_candidates_per_origin"] == base["search"]["unique_candidates_per_origin"] == 24
    assert config["origins"] == base["origins"] == 10
    assert base["search"]["seed_base"] == 20260907
    assert not any(config["safety"].values())
