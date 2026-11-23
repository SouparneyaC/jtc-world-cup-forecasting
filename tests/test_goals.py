import numpy as np
import pytest
from scipy.stats import poisson

from jtc_forecast.goals import (
    btts_prob,
    fit_rho,
    match_result_probs,
    nb_pmf,
    scoreline_grid,
    scoreline_grid_nb,
    tau,
    team_score_at_least,
    total_goals_over_under,
)


def test_grid_sums_to_one():
    for rho in (-0.15, 0.0, 0.05):
        assert sum(scoreline_grid(1.4, 1.1, rho).values()) == pytest.approx(1.0)
    assert sum(scoreline_grid_nb(1.4, 1.1, 0.1, -0.05).values()) == pytest.approx(1.0)


def test_result_probabilities_sum_to_one():
    probs = match_result_probs(scoreline_grid(1.6, 0.9, -0.05))
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["home_win"] > probs["away_win"]


def test_zero_rho_is_independent_poisson():
    grid = scoreline_grid(1.3, 1.0, 0.0, max_goals=20)
    assert grid[(2, 1)] == pytest.approx(poisson.pmf(2, 1.3) * poisson.pmf(1, 1.0), rel=1e-6)


def test_tau_changes_only_the_four_low_cells():
    assert tau(0, 0, 1.2, 1.0, -0.1) == pytest.approx(1 + 1.2 * 1.0 * 0.1)
    assert tau(0, 1, 1.2, 1.0, -0.1) == pytest.approx(1 - 0.12)
    assert tau(1, 0, 1.2, 1.0, -0.1) == pytest.approx(1 - 0.1)
    assert tau(1, 1, 1.2, 1.0, -0.1) == pytest.approx(1.1)
    assert tau(2, 0, 1.2, 1.0, -0.1) == 1.0 and tau(3, 3, 1.2, 1.0, -0.1) == 1.0


def test_negative_rho_adds_probability_to_goalless_draws():
    plain = scoreline_grid(1.2, 1.0, 0.0)
    corrected = scoreline_grid(1.2, 1.0, -0.1)
    assert corrected[(0, 0)] > plain[(0, 0)]
    assert corrected[(1, 0)] < plain[(1, 0)]


def test_negative_binomial_reduces_to_poisson_and_sums_to_one():
    assert nb_pmf(3, 1.7, 0.0) == pytest.approx(poisson.pmf(3, 1.7))
    assert sum(nb_pmf(k, 1.7, 0.2) for k in range(200)) == pytest.approx(1.0, abs=1e-6)
    assert nb_pmf(8, 1.7, 0.3) > poisson.pmf(8, 1.7)  # fatter tail


def test_market_helpers_are_consistent():
    grid = scoreline_grid(1.5, 1.2, -0.05)
    ou = total_goals_over_under(grid, 2)
    assert ou["over"] + ou["under_or_equal"] == pytest.approx(1.0)
    no_home = 1 - team_score_at_least(grid, "home", 1)
    no_away = 1 - team_score_at_least(grid, "away", 1)
    assert btts_prob(grid) == pytest.approx(1 - no_home - no_away + grid[(0, 0)])
    assert team_score_at_least(grid, "home", 0) == pytest.approx(1.0)


def test_fit_rho_recovers_the_sign_and_size():
    lam_h, lam_a, true_rho = 1.3, 1.1, -0.15
    grid = scoreline_grid(lam_h, lam_a, true_rho)
    cells = list(grid)
    probs = np.array([grid[c] for c in cells])
    rng = np.random.default_rng(7)
    draws = rng.choice(len(cells), size=40_000, p=probs)
    data = [(cells[k][0], cells[k][1], lam_h, lam_a) for k in draws]
    candidates = [r / 100 for r in range(-30, 11, 5)]
    assert fit_rho(data, rho_grid=candidates) == pytest.approx(true_rho, abs=0.05)
