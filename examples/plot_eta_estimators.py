"""
Plot ETA estimators on simulated workloads.

Requires matplotlib to render the plots.
"""
import argparse
import math
import random
from heapq import heappop, heappush, heapify

from tqdm.std import EMA, LogCauchyETA

CASE_NAMES = (
    "lognormal",
    "lognormal_outlier",
    "logcauchy",
    "lognormal_mixture",
    "lognormal_falling_mean",
)


def sample_lognormal(rng, mu, sigma):
    return rng.lognormvariate(mu, sigma)


def sample_logcauchy(rng, mu, sigma):
    u = rng.random()
    return math.exp(mu + sigma * math.tan(math.pi * (u - 0.5)))


def simulate_case(case_name, rng, total_tasks):
    if case_name == "lognormal":
        return [sample_lognormal(rng, 0.0, 0.6) for _ in range(total_tasks)]
    if case_name == "lognormal_outlier":
        durations = [sample_lognormal(rng, 0.0, 0.6) for _ in range(total_tasks)]
        outlier_index = rng.randrange(total_tasks)
        durations[outlier_index] = max(durations) * 25
        return durations
    if case_name == "logcauchy":
        return [sample_logcauchy(rng, 0.0, 0.8) for _ in range(total_tasks)]
    if case_name == "lognormal_mixture":
        durations = []
        for _ in range(total_tasks):
            if rng.random() < 0.25:
                durations.append(sample_lognormal(rng, 1.3, 0.4))
            else:
                durations.append(sample_lognormal(rng, 0.0, 0.35))
        return durations
    if case_name == "lognormal_falling_mean":
        durations = []
        for i in range(total_tasks):
            frac = 1.0 if total_tasks <= 1 else i / (total_tasks - 1)
            mu = 0.6 + (-0.2 - 0.6) * frac
            durations.append(sample_lognormal(rng, mu, 0.4))
        return durations
    raise ValueError(f"Unknown case: {case_name}")


def estimate_parallel_completion_time(running_remaining, pending_durations, workers):
    if not running_remaining and not pending_durations:
        return 0.0
    availability = list(running_remaining)
    availability.extend([0.0] * max(workers - len(availability), 0))
    heapify(availability)
    for duration in pending_durations:
        start_time = heappop(availability)
        heappush(availability, start_time + duration)
    return max(availability)


class LogCauchyCensoredEstimator(LogCauchyETA):
    """Log-Cauchy estimator that handles multiple right-censored workers."""
    def _estimate_params_multi(self, censored_logs):
        if self.sample_count < 2:
            return None
        mu, log_sigma = self._initial_params()
        if mu is None or log_sigma is None:
            return None
        for _ in range(self.max_iter):
            sigma = max(self.min_scale, math.exp(log_sigma))
            grad_mu = 0.0
            grad_log_sigma = 0.0
            for log_value, count in self.samples:
                u = (log_value - mu) / sigma
                denom = 1 + u * u
                grad_mu += count * (2 * u / (sigma * denom))
                grad_log_sigma += count * ((u * u - 1) / denom)
            for censored_log in censored_logs:
                u_c = (censored_log - mu) / sigma
                denom = 1 + u_c * u_c
                survival_prob = 0.5 - math.atan(u_c) * LogCauchyETA._INV_PI
                if survival_prob > 1e-12:
                    denom_survival = denom * survival_prob
                    scale = LogCauchyETA._INV_PI / denom_survival
                    grad_mu += scale / sigma
                    grad_log_sigma += u_c * LogCauchyETA._INV_PI / denom_survival
            total_weight = self.sample_count + len(censored_logs)
            if total_weight:
                grad_mu /= total_weight
                grad_log_sigma /= total_weight
            mu += self.learning_rate * grad_mu
            log_sigma = max(
                math.log(self.min_scale),
                log_sigma + self.learning_rate * grad_log_sigma)
        self.mu = mu
        self.log_sigma = log_sigma
        return mu, max(self.min_scale, math.exp(log_sigma))

    def _conditional_median_remaining(self, elapsed, mu, sigma):
        if elapsed <= 0:
            return math.exp(mu)
        log_elapsed = math.log(elapsed)
        cdf_elapsed = self._cdf(log_elapsed, mu, sigma)
        # Conditional median keeps half of the remaining probability mass above the elapsed time.
        target = 0.5 * (1 + cdf_elapsed)
        target = min(max(target, 1e-12), 1 - 1e-12)
        duration = math.exp(self._inv_cdf(target, mu, sigma))
        return max(0.0, duration - elapsed)

    def estimate_remaining_parallel(self, running_elapsed, remaining_tasks, workers):
        if remaining_tasks <= 0:
            return 0.0
        censored_logs = [math.log(e) for e in running_elapsed if e > 0]
        params = self._estimate_params_multi(censored_logs)
        if not params:
            return None
        mu, sigma = params
        running_remaining = [
            self._conditional_median_remaining(elapsed, mu, sigma) for elapsed in running_elapsed
        ]
        pending_count = max(remaining_tasks - len(running_remaining), 0)
        median_duration = math.exp(mu)
        pending_durations = [median_duration] * pending_count
        return estimate_parallel_completion_time(
            running_remaining,
            pending_durations,
            workers)


def run_simulation(durations, workers, smoothing=0.3):
    """Simulate parallel task execution and estimate remaining time series."""
    total_tasks = len(durations)
    index = 0
    heap = []
    for _ in range(min(workers, total_tasks)):
        duration = durations[index]
        index += 1
        heappush(heap, (duration, 0.0, duration))
    completed = 0
    ema_dn = EMA(smoothing)
    ema_dt = EMA(smoothing)
    logcauchy = LogCauchyCensoredEstimator()
    last_time = 0.0
    times = []
    completed_counts = []
    true_eta = []
    mean_eta = []
    smooth_eta = []
    logcauchy_eta = []

    while heap:
        current_time, start_time, duration = heappop(heap)
        finished = [(current_time, start_time, duration)]
        while heap and heap[0][0] == current_time:
            finished.append(heappop(heap))
        for _, _, finished_duration in finished:
            completed += 1
            logcauchy.add_completed(finished_duration)
            if index < total_tasks:
                next_duration = durations[index]
                index += 1
                heappush(heap, (current_time + next_duration, current_time, next_duration))

        running_remaining = [end - current_time for end, _, _ in heap]
        pending_durations = durations[index:]
        true_remaining = estimate_parallel_completion_time(
            running_remaining,
            pending_durations,
            workers)
        elapsed = current_time
        dt = elapsed - last_time
        dn = len(finished)
        ema_dn(dn)
        ema_dt(dt)

        mean_rate = completed / elapsed if elapsed > 0 else None
        smooth_rate = ema_dn() / ema_dt() if ema_dt() else None
        remaining_tasks = total_tasks - completed
        running_elapsed = [current_time - start for _, start, _ in heap]
        logcauchy_remaining = logcauchy.estimate_remaining_parallel(
            running_elapsed, remaining_tasks, workers)

        times.append(elapsed)
        completed_counts.append(completed)
        true_eta.append(true_remaining)
        mean_eta.append((remaining_tasks / mean_rate) if mean_rate else math.nan)
        smooth_eta.append((remaining_tasks / smooth_rate) if smooth_rate else math.nan)
        logcauchy_eta.append(
            logcauchy_remaining if logcauchy_remaining is not None else math.nan)
        last_time = elapsed

    return {
        "times": times,
        "completed": completed_counts,
        "true_eta": true_eta,
        "mean_eta": mean_eta,
        "smooth_eta": smooth_eta,
        "logcauchy_eta": logcauchy_eta,
    }


def plot_cases(results, output_path):
    """Plot ETA estimator series for a list of simulation result dictionaries."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        nrows=len(results),
        ncols=1,
        sharex=True,
        squeeze=False,
        figsize=(12, 2.8 * len(results)))
    for ax, result in zip(axes[:, 0], results):
        times = result["times"]
        ax.plot(times, result["true_eta"], label="true ETA", color="black", linewidth=1.5)
        ax.plot(times, result["mean_eta"], label="unsmoothed mean", color="#1f77b4")
        ax.plot(times, result["smooth_eta"], label="smoothed mean", color="#ff7f0e")
        ax.plot(times, result["logcauchy_eta"], label="log-cauchy", color="#2ca02c")
        ax.set_title(result["name"].replace("_", " ").title())
        ax.set_ylabel("Remaining time (s)")
        ax.grid(alpha=0.2)
        ax_tasks = ax.twinx()
        ax_tasks.step(
            times,
            result["completed"],
            where="post",
            color="gray",
            alpha=0.4,
            label="finished tasks")
        ax_tasks.set_ylabel("Tasks completed")
        ax_tasks.set_ylim(0, result["total"])
        handles, labels = ax.get_legend_handles_labels()
        if ax is axes[0, 0]:
            task_handles, task_labels = ax_tasks.get_legend_handles_labels()
            ax.legend(handles + task_handles, labels + task_labels,
                      loc="upper right", fontsize="small")
    axes[-1, 0].set_xlabel("Elapsed time (s)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description="Plot ETA estimator comparisons.")
    parser.add_argument(
        "--output",
        default="eta_estimators.png",
        help="Output PNG path for the generated plot.")
    parser.add_argument("--tasks", type=int, default=200, help="Total tasks per case.")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    results = []
    for case_name in CASE_NAMES:
        durations = simulate_case(case_name, rng, args.tasks)
        series = run_simulation(durations, args.workers)
        series["name"] = case_name
        series["total"] = args.tasks
        results.append(series)
    plot_cases(results, args.output)


if __name__ == "__main__":
    main()
