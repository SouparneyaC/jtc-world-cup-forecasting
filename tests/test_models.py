"""Fit the goal and outcome models on simulated matches with known coefficients."""

import math

import numpy as np
import pytest

from jtc_forecast.counts import CountModel, fit_count_model
from jtc_forecast.glm import GoalModel, build_design_matrix, fit_goal_model
from jtc_forecast.outcome import fit_ordered_logit, outcome_probs

B0, B1, B2 = 0.20, 0.30, 0.0015


def simulate_matches(n=6000, neutral_share=0.3, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        diff = rng.normal(0, 150)
        neutral = rng.random() < neutral_share
        home_adv = 0.0 if neutral else 1.0
        lam_home = math.exp(B0 + B1 * home_adv + B2 * diff)
        lam_away = math.exp(B0 - B2 * diff)
        rows.append({
            "date": f"2020-{1 + i % 12:02d}-{1 + i % 27:02d}",
            "elo_home_pre": 1500 + diff / 2, "elo_away_pre": 1500 - diff / 2,
            "neutral": "TRUE" if neutral else "FALSE",
            "home_score": int(rng.poisson(lam_home)), "away_score": int(rng.poisson(lam_away)),
        })
    return rows


def test_goal_model_recovers_coefficients():
    model = fit_goal_model(simulate_matches(), xi=0.0)
    assert model.intercept == pytest.approx(B0, abs=0.04)
    assert model.home_advantage == pytest.approx(B1, abs=0.06)
    assert model.elo_diff_coef == pytest.approx(B2, abs=0.0004)


def test_neutral_matches_get_no_home_advantage_in_training():
    rows = [{"date": "2020-01-01", "elo_home_pre": 1500, "elo_away_pre": 1500,
             "neutral": "TRUE", "home_score": 1, "away_score": 1}]
    _, X, _ = build_design_matrix(rows, xi=0.0)
    assert X[:, 1].tolist() == [0.0, 0.0]


def test_future_fixtures_are_not_used_for_fitting():
    rows = simulate_matches(500) + [{"date": "2030-01-01", "elo_home_pre": 1500,
                                     "elo_away_pre": 1500, "neutral": "TRUE",
                                     "home_score": "", "away_score": ""}]
    assert fit_goal_model(rows, xi=0.0).n_matches == 500


def test_lambdas_are_equal_for_equal_teams_on_neutral_ground():
    model = GoalModel(0.2, 0.3, 0.002, 0.0, 10)
    lam_home, lam_away = model.lambdas(1800, 1800, neutral=True)
    assert lam_home == pytest.approx(lam_away) == pytest.approx(math.exp(0.2))
    home_edge, _ = model.lambdas(1800, 1800, neutral=False)
    assert home_edge == pytest.approx(math.exp(0.5))


def test_ordered_logit_probabilities_and_ordering():
    probs = outcome_probs(120.0, 1.0, 0.004, 0.3, -0.8, 0.4)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert all(p > 0 for p in probs.values())


def test_ordered_logit_fit_recovers_structure():
    rng = np.random.default_rng(3)
    b_elo, b_home, c1, c2 = 0.004, 0.3, -0.8, 0.5
    rows = []
    for _ in range(8000):
        diff = rng.normal(0, 150)
        neutral = rng.random() < 0.3
        p = outcome_probs(diff, 0.0 if neutral else 1.0, b_elo, b_home, c1, c2)
        result = rng.choice(["away_win", "draw", "home_win"],
                            p=[p["away_win"], p["draw"], p["home_win"]])
        hs, as_ = {"away_win": (0, 1), "draw": (1, 1), "home_win": (1, 0)}[result]
        rows.append({"date": "2020-01-01", "elo_home_pre": 1500 + diff / 2,
                     "elo_away_pre": 1500 - diff / 2,
                     "neutral": "TRUE" if neutral else "FALSE",
                     "home_score": hs, "away_score": as_})
    model = fit_ordered_logit(rows, xi=0.0)
    assert model.c1 < model.c2
    assert model.b_elo == pytest.approx(b_elo, abs=0.0008)
    assert model.b_home == pytest.approx(b_home, abs=0.08)
    even = model.probs(1500, 1500, neutral=True)
    assert sum(even.values()) == pytest.approx(1.0)


def test_count_model_recovers_coefficients_and_source_shift():
    rng = np.random.default_rng(5)
    rows = []
    for _ in range(6000):
        diff = rng.normal(0, 150)
        source = "statsbomb" if rng.random() < 0.4 else "espn_boxscore"
        is_home = int(rng.random() < 0.3)
        eta = 1.2 + 0.001 * diff + 0.1 * is_home + (0.25 if source == "statsbomb" else 0.0)
        rows.append({"corners": rng.poisson(math.exp(eta)), "elo_diff_pre": diff,
                     "is_home": is_home, "data_source": source})
    model = fit_count_model(rows, "corners")
    assert isinstance(model, CountModel)
    assert model.baseline_source == "espn_boxscore"
    assert model.intercept == pytest.approx(1.2, abs=0.05)
    assert model.elo_diff_coef == pytest.approx(0.001, abs=0.0003)
    assert model.source_coefs["statsbomb"] == pytest.approx(0.25, abs=0.06)
    assert model.expected(0.0) == pytest.approx(math.exp(model.intercept))
