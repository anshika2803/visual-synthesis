"""The statistics of the E1-E7 reports (section 3.7 of docs/e1_e6_research_plan.md).

    wilson(k, n)                  95% interval of a proportion
    mcnemar(before, after)        exact two-sided test on paired yes/no outcomes
    bootstrap_ci(values)          95% interval of a mean, resampling prompts
    bootstrap_diff_ci(a, b)       95% interval of mean(a) - mean(b), independent samples
    holm(p_values)                Holm-adjusted p-values for a family of tests
"""

import math

import numpy as np

RESAMPLES = 1000
SEED = 42


def wilson(k, n, z=1.959964):
    """Wilson score interval; ``(nan, nan)`` for n = 0."""
    if not n:
        return math.nan, math.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar(before, after):
    """Exact McNemar p-value for paired booleans: a binomial test on the discordant pairs."""
    before, after = np.asarray(before, bool), np.asarray(after, bool)
    b, c = int((before & ~after).sum()), int((~before & after).sum())
    n = b + c
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)


def bootstrap_ci(values, resamples=RESAMPLES, seed=SEED):
    """Percentile interval of the mean. For paired data pass the per-prompt differences."""
    values = np.asarray(values, float)
    if not len(values):
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (resamples, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_diff_ci(a, b, resamples=RESAMPLES, seed=SEED):
    """Percentile interval of mean(a) - mean(b) for two independent samples (e.g. L6 vs L1)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    diffs = (a[rng.integers(0, len(a), (resamples, len(a)))].mean(axis=1)
             - b[rng.integers(0, len(b), (resamples, len(b)))].mean(axis=1))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def holm(p_values):
    """Holm step-down adjustment; NaN p-values stay NaN and are not counted."""
    p = np.asarray(p_values, float)
    adjusted = np.full(len(p), math.nan)
    order = [i for i in np.argsort(p) if not math.isnan(p[i])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()
