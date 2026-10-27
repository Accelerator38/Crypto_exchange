"""Immutable pre-data amendment: explicit strict purge boundary for ternary v1."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_genetic_primus_direction_ternary_v1 as run


class StrictCausal:
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def split(self, times, origin, horizon_ms, **kwargs):
        train, calibration = self.original.split(times, origin, horizon_ms, **kwargs)
        cal_days, purge = kwargs.get("calibration_days", 7), kwargs.get("purge", 7)
        train = train[times[train+1] < origin-cal_days*run.core.DAY-purge*horizon_ms]
        calibration = calibration[times[calibration+1] < origin-purge*horizon_ms]
        if not len(train) or not len(calibration): raise ValueError("empty strict causal fold")
        return train, calibration


run.causal = StrictCausal(run.causal)
run.CONFIG = run.ROOT / "configs/genetic_primus_direction_ternary_v2.json"
run.OUT = run.ROOT / "Reports/Exia/Genetic_Primus/direction_ternary_v2_20261013"
run.REGISTRY = run.ROOT / "configs/genetic_primus_objective_registry_v20.json"
run.CODE = [Path(__file__), Path(run.__file__), run.CONFIG,
            run.ROOT / "src/exia/genetic_primus/direction_ternary_v1.py",
            run.ROOT / "src/exia/genetic_primus/direction_selective_v1.py", Path(run.parent.__file__)]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "checks":
        run.checks()
        times = run.np.arange(60*1440, dtype=run.np.int64)*60000
        train, calibration = run.causal.split(times, 30*run.core.DAY, 60000, seed=17)
        assert run.np.all(times[train+1] < 23*run.core.DAY-7*60000)
        assert run.np.all(times[calibration+1] < 30*run.core.DAY-7*60000)
        run.parent.write(run.OUT / "strict_boundary_check.json", {"status": "PASS", "code_pins": run.pins(),
                         "historical_fits": 0, "test_outcomes_used_for_amendment": False})
    elif len(sys.argv) > 1 and sys.argv[1] == "init":
        old_out = run.ROOT / "Reports/Exia/Genetic_Primus/direction_ternary_v1_20261013"
        assert not list(old_out.glob("origin_*/*.joblib")) and not list(old_out.glob("origin_*/*.npz"))
        run.parent.write(run.OUT / "v1_preflight_stop.json", {"old_registration_sha256": run.parent.digest(old_out/"registration.json"),
            "reason": "parent includes exact purge cutoff; v1 preflight asserts strict cutoff",
            "historical_estimator_fits": 0, "historical_forecasts_written": 0,
            "old_synthetic_fits": 2, "performance_used_for_amendment": False})
        with redirect_stdout(io.StringIO()):
            run.init()
        print(run.json.dumps({"registry": "v20", "fit_attempt_budget": 756, "amendment_before_historical_fitting": True}))
    else:
        run.main()


if __name__ == "__main__": main()
