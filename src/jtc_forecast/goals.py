"""Scoreline probabilities with the Dixon-Coles low-score correction.

Given an expected goal count for each team, build the grid P(home = i, away = j)
and read market probabilities off it: match result, total goals, both teams
to score, and a team scoring at least n.

Goals are modelled as independent Poisson (or Negative-Binomial) counts. The
Dixon-Coles factor tau then adjusts the four lowest-scoring cells, where
independence is a known poor fit. The grid is renormalised to sum to 1, since
the correction and the cut-off at max_goals move the raw total slightly.

References: Dixon and Coles (1997); Karlis and Ntzoufras (2003).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

from scipy.stats import poisson

Grid = dict[tuple[int, int], float]


def nb_pmf(k: int, mu: float, alpha: float) -> float:
    """Negative-Binomial (NB2) pmf with mean mu and variance mu + alpha * mu**2.

    alpha = 0 gives Poisson(mu); larger alpha fattens the tails.
    """
    if alpha <= 1e-10:
        return float(poisson.pmf(k, mu))
    r = 1.0 / alpha
    p = r / (r + mu)
    log_pmf = (
        math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
        + r * math.log(p) + k * math.log(1 - p)
    )
    return math.exp(log_pmf)


def tau(i: int, j: int, lam_home: float, lam_away: float, rho: float) -> float:
    """Dixon-Coles correction factor for the cell (i, j)."""
    if i == 0 and j == 0:
        return 1 - lam_home * lam_away * rho
    if i == 0 and j == 1:
        return 1 + lam_home * rho
    if i == 1 and j == 0:
        return 1 + lam_away * rho
    if i == 1 and j == 1:
        return 1 - rho
    return 1.0


def _grid(pmf_home: Callable[[int], float], pmf_away: Callable[[int], float],
          lam_home: float, lam_away: float, rho: float, max_goals: int) -> Grid:
    grid: Grid = {}
    total = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = pmf_home(i) * pmf_away(j) * tau(i, j, lam_home, lam_away, rho)
            grid[(i, j)] = p
            total += p
    return {cell: p / total for cell, p in grid.items()}


def scoreline_grid(lam_home: float, lam_away: float, rho: float = 0.0,
                   max_goals: int = 10) -> Grid:
    """Poisson scoreline grid with the Dixon-Coles correction."""
    return _grid(lambda k: float(poisson.pmf(k, lam_home)),
                 lambda k: float(poisson.pmf(k, lam_away)),
                 lam_home, lam_away, rho, max_goals)


def scoreline_grid_nb(lam_home: float, lam_away: float, alpha: float,
                      rho: float = 0.0, max_goals: int = 10) -> Grid:
    """Same grid with Negative-Binomial marginals, for overdispersed goal counts."""
    return _grid(lambda k: nb_pmf(k, lam_home, alpha),
                 lambda k: nb_pmf(k, lam_away, alpha),
                 lam_home, lam_away, rho, max_goals)


def match_result_probs(grid: Grid) -> dict[str, float]:
    """Home win, draw and away win."""
    return {
        "home_win": sum(p for (i, j), p in grid.items() if i > j),
        "draw": sum(p for (i, j), p in grid.items() if i == j),
        "away_win": sum(p for (i, j), p in grid.items() if i < j),
    }


def total_goals_over_under(grid: Grid, threshold: int) -> dict[str, float]:
    """P(total goals > threshold) and P(total goals <= threshold)."""
    over = sum(p for (i, j), p in grid.items() if i + j > threshold)
    return {"over": over, "under_or_equal": 1 - over}


def btts_prob(grid: Grid) -> float:
    """Both teams score at least once."""
    return sum(p for (i, j), p in grid.items() if i >= 1 and j >= 1)


def team_score_at_least(grid: Grid, team: str, n: int) -> float:
    """P(team scores n or more goals); team is "home" or "away"."""
    if team == "home":
        return sum(p for (i, _), p in grid.items() if i >= n)
    return sum(p for (_, j), p in grid.items() if j >= n)


def fit_rho(matches: Iterable[tuple[int, int, float, float]],
            rho_grid: Iterable[float] | None = None,
            pmf: Callable[[int, float], float] | None = None) -> float:
    """Pick rho by maximum likelihood over a grid of candidates.

    matches holds (home_goals, away_goals, lam_home, lam_away). pmf is the
    marginal pmf(k, lam), Poisson by default.
    """
    candidates = list(rho_grid) if rho_grid is not None else [r / 100 for r in range(-30, 11)]
    marginal = pmf or (lambda k, lam: float(poisson.pmf(k, lam)))
    data = list(matches)

    best_rho, best_ll = 0.0, float("-inf")
    for rho in candidates:
        ll = 0.0
        for home_goals, away_goals, lam_h, lam_a in data:
            p = (marginal(home_goals, lam_h) * marginal(away_goals, lam_a)
                 * tau(home_goals, away_goals, lam_h, lam_a, rho))
            ll += math.log(p) if p > 0 else 0.0
        if ll > best_ll:
            best_ll, best_rho = ll, rho
    return best_rho
