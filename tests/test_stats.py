import math

import pytest

from visual_synthesis.stats import bootstrap_ci, bootstrap_diff_ci, holm, mcnemar, wilson


def test_wilson_interval_matches_the_textbook_value():
    low, high = wilson(10, 100)
    assert (round(low, 4), round(high, 4)) == (0.0552, 0.1744)
    assert wilson(0, 20)[0] == 0.0
    assert all(math.isnan(v) for v in wilson(0, 0))


def test_mcnemar_is_exact_on_discordant_pairs():
    # 10 pairs changed one way, 0 the other: p = 2 * 0.5**10.
    assert mcnemar([True] * 10 + [False] * 5, [False] * 10 + [False] * 5) == pytest.approx(2 * 0.5**10)
    assert mcnemar([True, False], [True, False]) == 1.0  # no discordant pair


def test_bootstrap_intervals_cover_the_mean():
    low, high = bootstrap_ci([0.0, 1.0] * 50)
    assert low < 0.5 < high
    low, high = bootstrap_diff_ci([1.0] * 30, [0.0] * 30)
    assert (low, high) == (1.0, 1.0)


def test_holm_adjusts_in_step_down_order_and_skips_nan():
    adjusted = holm([0.01, 0.04, 0.03, float("nan")])
    assert adjusted[:3] == pytest.approx([0.03, 0.06, 0.06])
    assert math.isnan(adjusted[3])
