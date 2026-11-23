"""Scoring forecasts, and measuring how the crowd's forecasts relate to ours.

Brier score is the squared error of a probability against a 0/1 outcome; lower
is better, and it is a proper scoring rule, so the best strategy is to report
your honest probability.

The crowd-compression fit regresses the crowd's probability on ours:

    crowd = a + b * (mine - 0.5)

A slope b below 1 means the crowd's forecasts are pulled toward 50% relative to
ours. The intercept a is the crowd's probability when ours is 50%.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def brier_score(forecast: float, outcome: int) -> float:
    """Squared error of one probability forecast against a 0 or 1 outcome."""
    return (forecast - outcome) ** 2


def mean_brier(forecasts: Sequence[float], outcomes: Sequence[int]) -> float:
    """Average Brier score over many forecasts."""
    f, o = np.asarray(forecasts, dtype=float), np.asarray(outcomes, dtype=float)
    return float(np.mean((f - o) ** 2))


def fit_crowd_compression(mine: Sequence[float], crowd: Sequence[float]) -> dict[str, float]:
    """Least-squares fit of crowd = a + b * (mine - 0.5).

    Returns the slope, the intercept a, the correlation r, r squared and n.
    """
    x = np.asarray(mine, dtype=float) - 0.5
    y = np.asarray(crowd, dtype=float)
    if len(x) < 3:
        raise ValueError("need at least 3 paired forecasts")
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return {"slope": float(slope), "intercept": float(intercept), "r": r,
            "r_squared": r * r, "n": int(len(x))}


def expected_crowd(mine: float, slope: float, intercept: float) -> float:
    """The crowd's probability the fit expects when ours is `mine`."""
    return intercept + slope * (mine - 0.5)
