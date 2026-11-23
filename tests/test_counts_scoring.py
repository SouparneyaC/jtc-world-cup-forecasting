import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import poisson, skellam

from jtc_forecast.counts import poisson_at_least, poisson_sum_at_least, skellam_more_than
from jtc_forecast.scoring import brier_score, expected_crowd, fit_crowd_compression, mean_brier

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_poisson_at_least_matches_the_tail_sum():
    tail = sum(poisson.pmf(k, 3.2) for k in range(4, 80))
    assert poisson_at_least(3.2, 4) == pytest.approx(tail)
    assert poisson_at_least(3.2, 0) == 1.0


def test_sum_of_poissons_is_poisson_of_the_sum():
    lam_a, lam_b = 1.2, 0.8
    by_convolution = sum(poisson.pmf(i, lam_a) * poisson.pmf(j, lam_b)
                         for i in range(40) for j in range(40) if i + j >= 3)
    assert poisson_sum_at_least([lam_a, lam_b], 3) == pytest.approx(by_convolution)


def test_skellam_probabilities_partition_the_outcomes():
    lam_a, lam_b = 5.1, 3.9
    tie = float(skellam.pmf(0, lam_a, lam_b))
    total = skellam_more_than(lam_a, lam_b) + skellam_more_than(lam_b, lam_a) + tie
    assert total == pytest.approx(1.0)
    assert skellam_more_than(4.0, 4.0) < 0.5
    assert skellam_more_than(6.0, 3.0) > 0.5


def test_brier_score():
    assert brier_score(0.7, 1) == pytest.approx(0.09)
    assert brier_score(0.7, 0) == pytest.approx(0.49)
    assert mean_brier([0.5, 0.5], [1, 0]) == pytest.approx(0.25)


def test_crowd_compression_recovers_a_known_line():
    mine = np.linspace(0.05, 0.95, 50)
    crowd = 0.50 + 0.6 * (mine - 0.5)
    fit = fit_crowd_compression(mine, crowd)
    assert fit["slope"] == pytest.approx(0.6)
    assert fit["intercept"] == pytest.approx(0.50)
    assert fit["r"] == pytest.approx(1.0)
    assert expected_crowd(0.9, fit["slope"], fit["intercept"]) == pytest.approx(0.74)


def test_crowd_compression_needs_enough_pairs():
    with pytest.raises(ValueError):
        fit_crowd_compression([0.4, 0.6], [0.45, 0.55])


def test_recorded_results_reproduce_the_published_fit():
    """The committed results table gives the slope and correlation quoted in the paper."""
    mine, crowd = [], []
    with open(RESULTS / "questions_group_to_r16.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pair = (float(row["our_estimate"]), float(row["crowd_estimate"]))
            except ValueError:
                continue
            mine.append(pair[0])
            crowd.append(pair[1])
    fit = fit_crowd_compression(mine, crowd)
    assert fit["n"] == 792
    assert fit["slope"] == pytest.approx(0.605, abs=0.005)
    assert fit["intercept"] == pytest.approx(0.503, abs=0.005)
    assert fit["r"] == pytest.approx(0.827, abs=0.005)
