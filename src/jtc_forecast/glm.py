"""Poisson regression of goals on the Elo difference.

Each match gives two observations, one per team:

    log(lambda) = b0 + b1 * home_adv + b2 * elo_diff

elo_diff is the scoring team's pre-match rating minus its opponent's.
home_adv is 1 for the listed home team on a non-neutral venue and 0 otherwise.
Neutral venues give neither side the bonus. An earlier version of the model
applied the bonus to the "home" label on neutral venues, about a quarter of the
history, which biased the home-advantage coefficient downward.

Older matches are down-weighted by exp(-xi * days_before_latest_match).

Reference: Hvattum and Arntzen (2010).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import statsmodels.api as sm

XI_DEFAULT = 0.0008  # decay per day; a half-life of about 2.4 years


def _to_date(value: Any) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def _is_neutral(value: Any) -> bool:
    return value is True or str(value).upper() == "TRUE"


def _played(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [r for r in rows if r.get("home_score") not in ("", "NA", None)]


def build_design_matrix(rows: Iterable[Mapping[str, Any]], xi: float = XI_DEFAULT):
    """Return (goals, design matrix, weights) for the played matches in rows."""
    played = _played(rows)
    latest = max(_to_date(r["date"]) for r in played)

    y, X, w = [], [], []
    for r in played:
        weight = math.exp(-xi * (latest - _to_date(r["date"])).days)
        elo_home, elo_away = float(r["elo_home_pre"]), float(r["elo_away_pre"])
        home_adv = 0.0 if _is_neutral(r.get("neutral")) else 1.0

        y.append(int(r["home_score"]))
        X.append([1.0, home_adv, elo_home - elo_away])
        w.append(weight)

        y.append(int(r["away_score"]))
        X.append([1.0, 0.0, elo_away - elo_home])
        w.append(weight)

    return np.array(y), np.array(X), np.array(w)


@dataclass(frozen=True)
class GoalModel:
    intercept: float
    home_advantage: float
    elo_diff_coef: float
    xi_decay: float
    n_matches: int

    def lambdas(self, elo_home: float, elo_away: float,
                neutral: bool = True) -> tuple[float, float]:
        """Expected goals for the home and away team."""
        bonus = 0.0 if neutral else self.home_advantage
        diff = elo_home - elo_away
        lam_home = math.exp(self.intercept + bonus + self.elo_diff_coef * diff)
        lam_away = math.exp(self.intercept - self.elo_diff_coef * diff)
        return lam_home, lam_away


def fit_goal_model(rows: Iterable[Mapping[str, Any]], xi: float = XI_DEFAULT) -> GoalModel:
    """Fit the goal model by weighted maximum likelihood."""
    rows = list(rows)
    y, X, w = build_design_matrix(rows, xi)
    result = sm.GLM(y, X, family=sm.families.Poisson(), freq_weights=w).fit()
    b0, b1, b2 = (float(p) for p in result.params)
    return GoalModel(b0, b1, b2, xi, len(_played(rows)))
