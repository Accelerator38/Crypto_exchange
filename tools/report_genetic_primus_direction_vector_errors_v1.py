"""Report-only axes-shape adapter; frozen diagnostic calculation is unchanged."""
from contextlib import contextmanager
from pathlib import Path
import analyze_genetic_primus_direction_vector_errors_v1 as analysis
import numpy as np


@contextmanager
def flat_subplot_axes():
    """Support the frozen reporter's flat axes[0] legend access on 2x2 grids."""
    import matplotlib.pyplot as plt
    original = plt.subplots
    def flat(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        if isinstance(axes, np.ndarray) and axes.ndim > 1:
            axes = axes.ravel()
        return fig, axes
    plt.subplots = flat
    try:
        yield
    finally:
        plt.subplots = original


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["report", "finalize"])
    args = parser.parse_args()
    if args.stage == "report":
        with flat_subplot_axes():
            analysis.report()
        analysis.run.parent.write(analysis.OUT / "report_build.json", dict(
            report_only_adapter_sha256=analysis.run.parent.digest(Path(__file__)),
            correction="Flatten matplotlib axes for legend indexing; no changes to data, labels, predictions, metrics or registration",
            estimator_fits=0))
    else:
        analysis.finalize()
        analysis.run.parent.write(analysis.OUT / "report_adapter_manifest.json", dict(
            artifact_pins={analysis.run.rel(Path(__file__)): analysis.run.parent.digest(Path(__file__)),
                           analysis.run.rel(analysis.OUT / "delivery_manifest.json"): analysis.run.parent.digest(analysis.OUT / "delivery_manifest.json")},
            estimator_fits=0))
