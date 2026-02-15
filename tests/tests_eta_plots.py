import importlib
import math
import os
import random
import sys


def load_plot_module():
    root = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, root)
    try:
        return importlib.import_module("examples.plot_eta_estimators")
    finally:
        sys.path.pop(0)


def test_simulation_returns_valid_structure():
    module = load_plot_module()
    rng = random.Random(123)
    durations = module.simulate_case("lognormal", rng, 30)
    results = module.run_simulation(durations, workers=4)

    assert results["completed"][-1] == len(durations)
    assert results["times"] == sorted(results["times"])
    assert results["true_eta"][-1] == 0.0
    assert len(results["times"]) == len(results["mean_eta"])
    assert any(math.isfinite(value) for value in results["logcauchy_eta"])
