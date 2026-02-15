from examples import plot_eta_estimators as plot_module
import math
import random


def test_simulation_returns_valid_structure():
    for seed, case_name in enumerate(plot_module.CASE_NAMES, start=123):
        rng = random.Random(seed)
        durations = plot_module.simulate_case(case_name, rng, 30)
        results = plot_module.run_simulation(durations, workers=4)

        assert results["completed"][-1] == len(durations)
        assert results["times"] == sorted(results["times"])
        assert results["true_eta"][-1] == 0.0
        assert len(results["times"]) == len(results["mean_eta"])
        assert any(math.isfinite(value) for value in results["mean_eta"])
        assert any(math.isfinite(value) for value in results["smooth_eta"])
        assert any(math.isfinite(value) for value in results["logcauchy_eta"])
