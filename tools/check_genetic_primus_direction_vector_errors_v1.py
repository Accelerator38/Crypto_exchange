"""Independent raw-return boolean check, without confusion-matrix helpers or fits."""
from pathlib import Path
import json
import numpy as np
import analyze_genetic_primus_direction_vector_errors_v1 as analysis

if __name__ == "__main__":
    c, reg = analysis.registered(); run = analysis.run; pc = run.config(); checked = 0
    rows = run.read(analysis.OUT / "results.json")["rows"]
    for h in pc["timeframes_minutes"]:
        names = run.methods(pc, h)
        counts = np.zeros((len(names), 8, 4), np.int64)
        for origin in range(21):
            with np.load(run.prefix(run.OUT, origin, h).with_suffix(".npz"), allow_pickle=False) as a, np.load(run.prefix(run.baseout(pc), origin, h).with_suffix(".npz"), allow_pickle=False) as b:
                pp = [np.array([-1, 0, 1])[p.argmax(axis=-1)] for p in a["probability"]]+run.fixed.predictions(b)[:9]
                returns = b["return_pct"]; band = c["neutral_bands_pct"][str(h)]
                for k, pred in enumerate(pp):
                    counts[k, :, 0] += (((pred == 1) & (returns < 0)) | ((pred == -1) & (returns > 0))).sum(axis=0)
                    counts[k, :, 1] += (((pred == 1) & (returns < -band)) | ((pred == -1) & (returns > band))).sum(axis=0)
                    counts[k, :, 2] += (pred != 0).sum(axis=0)
                    counts[k, :, 3] += (((pred == 1) & (returns >= -band) & (returns <= band)) | ((pred == -1) & (returns >= -band) & (returns <= band))).sum(axis=0)
        for k, method in enumerate(names):
            for asset, coin in enumerate(pc["symbols"]):
                row = next(r for r in rows if r["minutes"] == h and r["method"] == method and r["symbol"] == coin and r["band_factor"] == 1)
                assert np.array_equal(counts[k, asset], [row[key] for key in ("opposite_raw_sign_count", "opposite_direction_count", "directional_calls", "neutral_outcome_on_directional_call_count")])
                checked += 1
    run.parent.write(analysis.OUT / "independent_boolean_audit.json", dict(checked_current_band_metric_cells=checked,
        counts_per_cell=4, confusion_helpers_used=False, estimator_fits=0,
        checker_sha256=run.parent.digest(Path(__file__))))
    print(json.dumps(dict(checked_metric_cells=checked, independent_boolean_counts_per_cell=4, fits=0)))
